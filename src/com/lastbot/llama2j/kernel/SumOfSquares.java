package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class SumOfSquares extends Kernel {
    private static final int SMALL_KERNEL = 1024;
    private static final int LARGE_KERNEL = 1024 * 1024;

    private final CUfunction smallKernel;
    private final CUfunction largeLocalSumKernel;
    private final CUfunction largeReductionKernel;

    public SumOfSquares(ContextCUDA cuda) {
        super(cuda, "sumOfSquares");
        smallKernel = createSmall();
        largeLocalSumKernel = createLargeLocalSum();
        largeReductionKernel = createLargeReduction();
    }

    public static void call(float[] sum, float[] x, int size) {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            ss += x[j] * x[j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / (float) Math.sqrt(ss);
        sum[0] = ss;
    }

    public void test(float[] sum, float[] x, int size) {
        float[] copyOfSum = Arrays.copyOf(sum, sum.length);
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer pSum = cuda.allocateAndCopyToDevice(x, false);
        Pointer px = cuda.allocateAndCopyToDevice(x, false);
        cuda.synchronizeTransfer();
        call(0, pSum, px, size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pSum, sum);
        cuda.copyFromDeviceToHost(px, x);
        cuda.free(pSum);
        cuda.free(px);

        call(copyOfSum, copyOfx, size);

        compareWithThreshold("SumOfSquares.call (" + (size <= SMALL_KERNEL ? "small" : "large") +
                        ") sum ",
                sum, copyOfSum, 1e-2f);
    }

    private void call(int kernelStreamId, Pointer sum, Pointer x, int size) {
        CUstream stream = cuda.getCUKernelStream(kernelStreamId);
        if (size <= SMALL_KERNEL) {
            int blockSizeX = findNextPowerOf2(size);
            int gridSizeX = (int) Math.ceil((double) size / blockSizeX);
            int sharedMemory = blockSizeX * Float.BYTES;

//            __global__ void sumOfSquares(float* sum, float* x, int size) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(sum),
                    Pointer.to(x),
                    Pointer.to(new int[]{size})
            );

            cuLaunchKernel(smallKernel,
                    gridSizeX, 1, 1,          // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    sharedMemory, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } else if (size <= LARGE_KERNEL) {
            int numberOfThreads = 1024;
            int numBlocks = (int) Math.ceil((double) size / numberOfThreads);
            Pointer blockSum = cuda.allocate((long) numBlocks * Float.BYTES);
            // exp and sum
            {
                int sharedMemory = numberOfThreads * Float.BYTES;
                int blockSizeX = 1024;
                int gridSizeX = numBlocks;

//                __global__ void localSumOfSquares(float* blockSum, float* x, int size) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(blockSum),
                        Pointer.to(x),
                        Pointer.to(new int[]{size})
                );
                cuLaunchKernel(largeLocalSumKernel,
                        gridSizeX, 1, 1,          // Grid dimension
                        blockSizeX, 1, 1,      // Block dimension
                        sharedMemory, stream,  // Shared memory size and stream
                        kernelParameters, null // Kernel- and extra parameters
                );
            }
//            cuda.synchronizeKernel(kernelStreamId);
            // reduction
            {
                int blockSizeX = findNextPowerOf2(numBlocks);
                int gridSizeX = 1;
                int sharedMemory = blockSizeX * Float.BYTES;

//                __global__ void sumReduction(float* sum, float* blockSum, int numBlocks) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(sum),
                        Pointer.to(blockSum),
                        Pointer.to(new int[]{numBlocks}),
                        Pointer.to(new int[]{size})
                );
                cuLaunchKernel(largeReductionKernel,
                        gridSizeX, 1, 1,          // Grid dimension
                        blockSizeX, 1, 1,      // Block dimension
                        sharedMemory, stream,  // Shared memory size and stream
                        kernelParameters, null // Kernel- and extra parameters
                );
            }
            cuda.free(blockSum);
        } else {
            throw new RuntimeException("ExpAndSum.call with too large size" + size);
        }
    }

    private CUfunction createSmall() {
        String code =
                """
                    extern "C"
                    __global__ void sumOfSquares(float* sum, float* x, int size) {
                        int tid = blockIdx.x * blockDim.x + threadIdx.x;

                        extern __shared__ float sdata[];

                        if (tid < size) {
                            sdata[threadIdx.x] = x[tid] * x[tid];
                        } else {
                            sdata[threadIdx.x] = 0.0f;
                        }
                        __syncthreads();  // Ensure all threads in block have stored their values

                        // Block-wise reduction
                        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                            if (threadIdx.x < stride) {
                                sdata[tid] += sdata[tid + stride];
                            }
                            __syncthreads();  // Ensure all threads in block are in sync after each step
                        }

                        // Only thread 0 in the block writes the block result to global memory
                        if (tid == 0) {
                            double result = sdata[0] / size + 1e-5f;
                            sum[0] = 1.0f / sqrtf(result);
                        }
                    }
                """;
        return loadFromCode(code, "sumOfSquares");
    }

    private CUfunction createLargeLocalSum() {
        String code =
                """
                            extern "C"
                            // First kernel: Calculate the exponential values and perform block-wise reduction.
                            __global__ void localSumOfSquares(float* blockSum, float* x, int size) {
                                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                                    
                                // Shared memory for block-wise summation
                                extern __shared__ float sdata[];
                                    
                                // Ensure the thread is within bounds
                                if (tid < size) {
                                    sdata[threadIdx.x] = x[tid] * x[tid];
                                } else {
                                    sdata[threadIdx.x] = 0.0f;
                                }
                                __syncthreads();  // Ensure all threads in block have stored their values
                                    
                                __syncthreads();
                                    
                                // Block-wise reduction
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (threadIdx.x < stride) {
                                        sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                                    }
                                    __syncthreads();
                                }
                                    
                                // First thread of each block writes its result to the global array
                                if (threadIdx.x == 0) {
                                    blockSum[blockIdx.x] = sdata[0];
                                }
                            }
                        """;
        return loadFromCode(code, "localSumOfSquares");
    }

    private CUfunction createLargeReduction() {
        String code =
                """
                            extern "C"
                            // Second kernel: Sums up the partial sums
                            __global__ void sumReduction(float* sum, float* blockSum, int numBlocks, int size) {
                                extern __shared__ float sdata[];
                            
                                int tid = threadIdx.x;
                                if (tid < numBlocks) {
                                    sdata[tid] = blockSum[tid];
                                } else {
                                    sdata[tid] = 0.0f;
                                }
                                    
                                __syncthreads();
                                    
                                // Block-wise reduction
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (tid < stride) {
                                        sdata[tid] += sdata[tid + stride];
                                    }
                                    __syncthreads();
                                }
                                    
                                // First thread writes the final result
                                if (tid == 0) {
                                    double result = sdata[0] / size + 1e-5f;
                                    sum[0] = 1.0f / sqrtf(result);
                                }
                            }
                        """;
        return loadFromCode(code, "sumReduction");
    }
}