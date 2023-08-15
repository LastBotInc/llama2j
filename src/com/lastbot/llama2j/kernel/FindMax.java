package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class FindMax extends Kernel {
    private static final int SMALL_KERNEL = 1024;
    private static final int LARGE_KERNEL = 1024 * 1024;

    private final ContextCUDA cuda;

    private final CUfunction smallKernel;
    private final CUfunction largeLocalMaxKernel;
    private final CUfunction largeReductionKernel;

    public FindMax(ContextCUDA cuda) {
        super(cuda, "findMax");
        this.cuda = cuda;
        smallKernel = createSmall();
        largeLocalMaxKernel = createLargeLocalMax();
        largeReductionKernel = createLargeReduction();
    }

    public static void call(float[] max, float[] x, int index, int size) {
        float max_val = x[index]; // index + 0
        for (int i = 1; i < size; i++) {
            if (x[index + i] > max_val) {
                max_val = x[index + i];
            }
        }
        max[0] = max_val;
    }

    public void test(float[] max, float[] x, int index, int size) {
        float[] copyOfMax = Arrays.copyOf(max, max.length);
        Pointer pMax = cuda.allocateAndCopyToDevice(max, false);
        Pointer px = cuda.allocateAndCopyToDevice(x, false);
        cuda.synchronizeTransfer();
        call(0, pMax, px, index, size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pMax, max);
        cuda.free(pMax);
        cuda.free(px);

        call(copyOfMax, x, index, size);

        compareWithThreshold("FindMax.call (" + (size <= SMALL_KERNEL ? "small" : "large") +
                        ") max ",
                max, copyOfMax, 1e-2f);
    }

    private void call(int kernelStreamId, Pointer max, Pointer x, int index, int size) {
        CUstream stream = cuda.getCUKernelStream(kernelStreamId);
        if (size <= SMALL_KERNEL) {
            int blockSizeX = findNextPowerOf2(size);
            int gridSizeX = (int) Math.ceil((double) size / blockSizeX);
            int sharedMemory = blockSizeX * Float.BYTES;

//            __global__ void findMax(float *max, float *x, int index, int size) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(max),
                    Pointer.to(x),
                    Pointer.to(new int[]{index}),
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
            Pointer blockMax = cuda.allocate((long) numBlocks * Float.BYTES);
            // exp and sum
            {
                int sharedMemory = numberOfThreads * Float.BYTES;
                int blockSizeX = 1024;
                int gridSizeX = numBlocks;

//                __global__ void localMax(float *blockMax, float *x, int size) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(blockMax),
                        Pointer.to(x),
                        Pointer.to(new int[]{size})
                );
                cuLaunchKernel(largeLocalMaxKernel,
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

//                __global__ void maxReduction(float *max, float *blockMax, int numBlocks) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(max),
                        Pointer.to(blockMax),
                        Pointer.to(new int[]{numBlocks})
                );
                cuLaunchKernel(largeReductionKernel,
                        gridSizeX, 1, 1,          // Grid dimension
                        blockSizeX, 1, 1,      // Block dimension
                        sharedMemory, stream,  // Shared memory size and stream
                        kernelParameters, null // Kernel- and extra parameters
                );
            }
            cuda.free(blockMax);
        } else {
            throw new RuntimeException("ExpAndSum.call with too large size" + size);
        }
    }

    private CUfunction createSmall() {
        String code =
                """
                        #include <cfloat>
                        extern "C"
                        __global__ void findMax(float *max, float *x, int index, int size) {
                            int tid = threadIdx.x;
                            int blockSize = blockDim.x;
                            extern __shared__ float sdata[];

                            if(tid < size) {
                                sdata[tid] = x[index + tid];
                            } else {
                                // Fill remaining shared memory with lowest possible value
                                sdata[tid] = -FLT_MAX;
                            }
                            __syncthreads(); // Make sure the shared memory is populated

                            // Binary tree reduction
                            for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
                                if (tid < stride && tid + stride < size) {
                                    sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
                                }
                                __syncthreads();
                            }

                            // Write result to global memory
                            if(tid == 0) {
                                max[0] = sdata[0];
                            }
                        }
                 """;
        return loadFromCode(code, "findMax");
    }

    private CUfunction createLargeLocalMax() {
        String code =
                """
                        #include <cfloat>
                        extern "C"
                        __global__ void localMax(float *blockMax, float *x, int size) {
                            int tid = threadIdx.x;
                            int blockSize = blockDim.x;
                            int globalId = blockIdx.x * blockSize + tid;
                            extern __shared__ float sdata[];

                            if(globalId < size) {
                                sdata[tid] = x[globalId];
                            } else {
                                // Fill remaining shared memory with lowest possible value
                                sdata[tid] = -FLT_MAX;
                            }
                            __syncthreads(); // Make sure the shared memory is populated

                            // Binary tree reduction
                            for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
                                if (tid < stride) {
                                    sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
                                }
                                __syncthreads();
                            }

                            // Write the local maximum of each block to global memory
                            if(tid == 0) {
                                blockMax[blockIdx.x] = sdata[0];
                            }
                        }
                """;
        return loadFromCode(code, "localMax");
    }

    private CUfunction createLargeReduction() {
        String code =
                """
                        #include <cfloat>
                        extern "C"
                        __global__ void maxReduction(float *max, float *blockMax, int numBlocks) {
                            extern __shared__ float sdata[];
                        
                            int tid = threadIdx.x;
                        
                            if(tid < numBlocks) {
                                sdata[tid] = blockMax[tid];
                            } else {
                                sdata[tid] = -FLT_MAX;
                            }
                            __syncthreads();
                        
                            // Binary tree reduction
                            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                if (tid < stride && tid + stride < numBlocks) {
                                    sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
                                }
                                __syncthreads();
                            }
                        
                            // Write the global maximum to global memory
                            if(tid == 0) {
                                max[0] = sdata[0];
                            }
                        }
                """;
        return loadFromCode(code, "maxReduction");
    }
}
