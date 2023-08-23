package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class ExpAndSum extends Kernel {
    public static final int BLOCK_SIZE = 128;

    private static final int SMALL_KERNEL = BLOCK_SIZE;
    private static final int LARGE_KERNEL = 1024 * 1024;

    private final CUfunction smallKernel;
    private final CUfunction largeLocalSumKernel;
    private final CUfunction largeReductionKernel;

    public ExpAndSum(ContextCUDA cuda) {
        super(cuda, "expAndSum");
        smallKernel = createSmall();
        largeLocalSumKernel = createLargeLocalSum();
        largeReductionKernel = createLargeReduction();
    }

    public static void call(float[] sum, float[] x, float[] maxValue, int index, int size) {
        float s = 0.0f;
        float max_val = maxValue[0];

        for (int i = 0; i < size; i++) {
            // zzz consider expm1
            x[index + i] = (float) Math.exp(x[index + i] - max_val);
            s += x[index + i];
        }
        sum[0] = s;
    }

    public void test(float[] sum, float[] x, float[] maxValue, int index, int size) {
        int threadsPerBlock = Math.min(findNextPowerOf2(size), BLOCK_SIZE);
        int blocksPerGrid = (int) Math.ceil((double) size / threadsPerBlock);

        float[] copyOfSum = Arrays.copyOf(sum, sum.length);
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer pSum = cuda.allocateAndCopyToDevice(TEST_STREAM, sum, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        Pointer pMaxValue = cuda.allocateAndCopyToDevice(TEST_STREAM, maxValue, false);
        Pointer tmp = cuda.allocate((long) blocksPerGrid * Float.BYTES);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pSum, px, pMaxValue, tmp, index, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pSum, sum.length, sum);
        cuda.copyFromDeviceToHost(TEST_STREAM, px, x.length, x);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pSum);
        cuda.free(px);
        cuda.free(pMaxValue);
        cuda.free(tmp);

        call(copyOfSum, copyOfx, maxValue, index, size);

        compareWithThreshold("ExpAndSum.call (" + (size <= SMALL_KERNEL ? "small" : "large") +
                        ") sum ",
                sum, copyOfSum, 1e-2f);
        compareWithThreshold("ExpAndSum.call (" + (size <= SMALL_KERNEL ? "small" : "large") +
                        ") x ",
                x, copyOfx, 1e-5f);
    }

    public void call(int streamId, Pointer sum, Pointer x, Pointer maxValue, Pointer tmp, int index, int size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        if (size <= SMALL_KERNEL) {
            int blockSizeX = findNextPowerOf2(size);
            int gridSizeX = (int) Math.ceil((double) size / blockSizeX);
            int sharedMemory = blockSizeX * Float.BYTES;

//            __global__ void expAndSum(float* sum, float* x, float* maxValue, int index, int size) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(sum),
                    Pointer.to(x),
                    Pointer.to(maxValue),
                    Pointer.to(new int[]{index}),
                    Pointer.to(new int[]{size})
            );

            isError(cuLaunchKernel(smallKernel,
                    gridSizeX, 1, 1,          // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    sharedMemory, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            if (SYNC_KERNEL_CALLS) {
                cuda.synchronizeStream(streamId);
            }
        } else if (size <= LARGE_KERNEL) {
            int threadsPerBlock = Math.min(findNextPowerOf2(size), BLOCK_SIZE);
            int blocksPerGrid = (int) Math.ceil((double) size / threadsPerBlock);
            // exp and sum
            {
                int sharedMemory = threadsPerBlock * Float.BYTES;
                int blockSizeX = threadsPerBlock;
                int gridSizeX = blocksPerGrid;

//            __global__ void expAndSum(float* blockSum, float* x, float* maxValue, int index, int size) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(tmp),
                        Pointer.to(x),
                        Pointer.to(maxValue),
                        Pointer.to(new int[]{index}),
                        Pointer.to(new int[]{size})
                );
                isError(cuLaunchKernel(largeLocalSumKernel,
                        gridSizeX, 1, 1,          // Grid dimension
                        blockSizeX, 1, 1,      // Block dimension
                        sharedMemory, stream,  // Shared memory size and stream
                        kernelParameters, null // Kernel- and extra parameters
                ));
                if (SYNC_KERNEL_CALLS) {
                    cuda.synchronizeStream(streamId);
                }
            }
            // reduction
            {
                int blockSizeX = Math.min(findNextPowerOf2(blocksPerGrid), BLOCK_SIZE);
                int gridSizeX = (int) Math.ceil((double) blocksPerGrid / blockSizeX);
                int sharedMemory = blockSizeX * Float.BYTES;

//                __global__ void sumReduction(float* sum, float* blockSum, int blocksPerGrid) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(sum),
                        Pointer.to(tmp),
                        Pointer.to(new int[]{blocksPerGrid})
                );
                isError(cuLaunchKernel(largeReductionKernel,
                        gridSizeX, 1, 1,          // Grid dimension
                        blockSizeX, 1, 1,      // Block dimension
                        sharedMemory, stream,  // Shared memory size and stream
                        kernelParameters, null // Kernel- and extra parameters
                ));
                if (SYNC_KERNEL_CALLS) {
                    cuda.synchronizeStream(streamId);
                }
            }
        } else {
            throw new RuntimeException("ExpAndSum.call with too large size" + size);
        }
    }

    private CUfunction createSmall() {
        String code =
                """
                            extern "C"
                            __global__ void expAndSumSmall(float* sum, float* x, float* maxValue, int index, int size) {
                                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                                extern __shared__ float sdata[];

                                if (tid < size) {
                                    float max_val = maxValue[0];

                                    x[index + tid] = expf(x[index + tid] - max_val);
                                    
                                    // Store the value in shared memory for reduction
                                    sdata[threadIdx.x] = x[index + tid];
                                  } else {
                                    sdata[threadIdx.x] = 0.0f;
                                }
                                __syncthreads();  // Ensure all threads in block have stored their values

                                // Block-wise reduction
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (tid < stride && (threadIdx.x + stride) < blockDim.x) {
                                        sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                                    }
                                    __syncthreads();  // Ensure all threads in block are in sync after each step
                                }

                                if (tid == 0) {
                                    sum[0] = sdata[0];
                                }
                            }
                        """;
        return loadFromCode(code, "expAndSumSmall");
    }

    private CUfunction createLargeLocalSum() {
        String code =
                """
                            extern "C"
                            // First kernel: Calculate the exponential values and perform block-wise reduction.
                            __global__ void localExpAndSum(float* blockSum, float* x, float* maxValue, int index, int size) {
                                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                                    
                                // Shared memory for block-wise summation
                                extern __shared__ float sdata[];
                                    
                                // Ensure the thread is within bounds
                                if (tid < size) {
                                    float max_val = maxValue[0];
                                    x[index + tid] = expf(x[index + tid] - max_val);
                                    sdata[threadIdx.x] = x[index + tid];
                                } else {
                                    sdata[threadIdx.x] = 0.0f;
                                }
                                    
                                __syncthreads();
                                    
                                // Block-wise reduction
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x) {
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
        return loadFromCode(code, "localExpAndSum");
    }

    private CUfunction createLargeReduction() {
        String code =
                """
                            extern "C"
                            // Second kernel: Sums up the partial sums
                            __global__ void sumReduction(float* sum, float* blockSum, int blocksPerGrid) {
                                extern __shared__ float sdata[];
                            
                                int tid = threadIdx.x;
                                if (tid < blocksPerGrid) {
                                    sdata[tid] = blockSum[tid];
                                } else {
                                    sdata[tid] = 0.0f;
                                }
                                    
                                __syncthreads();

                                // Block-wise reduction
                                #pragma unroll 8
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (tid < stride && (threadIdx.x + stride) < blockDim.x) {
                                        sdata[tid] += sdata[tid + stride];
                                    }
                                    __syncthreads();
                                }
                                    
                                // First thread writes the final result
                                if (tid == 0) {
                                    *sum = sdata[0];
                                }
                            }
                        """;
        return loadFromCode(code, "sumReduction");
    }
}