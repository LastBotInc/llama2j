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
        Pointer pMax = cuda.allocateAndCopyToDevice(TEST_STREAM, max, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        call(TEST_STREAM, pMax, px, index, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pMax, max);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pMax);
        cuda.free(px);

        call(copyOfMax, x, index, size);

        compareWithThreshold("FindMax.call (" + (size <= SMALL_KERNEL ? "small" : "large") +
                        ") max ",
                max, copyOfMax, 1e-2f);
    }

    public void call(int streamId, Pointer max, Pointer x, int index, int size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
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

            isError(cuLaunchKernel(smallKernel,
                    gridSizeX, 1, 1,         // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    sharedMemory, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            if (SYNC_KERNEL_CALLS) {
                cuda.synchronizeStream(streamId);
            }
        } else if (size <= LARGE_KERNEL) {
            int threadsPerBlock = Math.min(findNextPowerOf2(size), MAX_THREADS_PER_BLOCK);
            int blocksPerGrid = (int) Math.ceil((double) size / threadsPerBlock);
            Pointer blockMax = cuda.allocate((long) blocksPerGrid * Float.BYTES);
            // exp and sum
            {
                int sharedMemory = threadsPerBlock * Float.BYTES;
                int blockSizeX = threadsPerBlock;
                int gridSizeX = blocksPerGrid;

//                __global__ void localMax(float *blockMax, float *x, int size) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(blockMax),
                        Pointer.to(x),
                        Pointer.to(new int[]{size})
                );
                isError(cuLaunchKernel(largeLocalMaxKernel,
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
                int blockSizeX = findNextPowerOf2(blocksPerGrid);
                int gridSizeX = (int) Math.ceil((double) blocksPerGrid / blockSizeX);
                int sharedMemory = blockSizeX * Float.BYTES;

//                __global__ void maxReduction(float *max, float *blockMax, int blocksPerGrid) {
                Pointer kernelParameters = Pointer.to(
                        Pointer.to(max),
                        Pointer.to(blockMax),
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
                                if (tid < stride && (threadIdx.x + stride) < blockDim.x) {
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
                        __global__ void maxReduction(float *max, float *blockMax, int blocksPerGrid) {
                            extern __shared__ float sdata[];
                        
                            int tid = threadIdx.x;
                        
                            if(tid < blocksPerGrid) {
                                sdata[tid] = blockMax[tid];
                            } else {
                                sdata[tid] = -FLT_MAX;
                            }
                            __syncthreads();
                        
                            // Binary tree reduction
                            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                if (tid < stride && tid + stride < blocksPerGrid) {
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
