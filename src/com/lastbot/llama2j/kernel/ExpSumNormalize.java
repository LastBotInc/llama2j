package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class ExpSumNormalize extends Kernel {
    public static final int BLOCK_SIZE = 16;

    private final CUfunction smallKernel;

    public ExpSumNormalize(ContextCUDA cuda) {
        super(cuda, "expSumNormalize");
        smallKernel = createSmall();
    }

    public static void call(float[] x, float[] maxValue, int index, int size) {
        float sum = 0.0f;
        float max_val = maxValue[0];

        for (int i = 0; i < size; i++) {
            x[index + i] = (float) Math.exp(x[index + i] - max_val);
            sum += x[index + i];
        }
        if (sum != 0.0f) {
            for (int i = 0; i < size; i++) {
                x[index + i] /= sum;
            }
        }
    }

    public void test(float[] x, float[] maxValue, int index, int size) {
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        Pointer pMaxValue = cuda.allocateAndCopyToDevice(TEST_STREAM, maxValue, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, px, pMaxValue, index, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, px, x.length, x);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(px);
        cuda.free(pMaxValue);

        call(copyOfx, maxValue, index, size);

        compareWithThreshold("ExpAndSum.call x ",
                x, copyOfx, 1e-5f);
    }

    public void call(int streamId, Pointer x, Pointer maxValue, int index, int size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int sharedMemory = BLOCK_SIZE * Float.BYTES;
        // __global__ void expAndSumSmall(float* x, float* maxValue, int index, int size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(x),
                Pointer.to(maxValue),
                Pointer.to(new int[]{index}),
                Pointer.to(new int[]{size})
        );

        isError(cuLaunchKernel(smallKernel,
                1, 1, 1,          // Grid dimension
                BLOCK_SIZE, 1, 1,      // Block dimension
                sharedMemory, stream,  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
    }

    private CUfunction createSmall() {
        String code =
                """
                        extern "C"
                        __global__ void expAndSumSmall(float* x, float* maxValue, int index, int size) {
                            extern __shared__ float sdata[];

                            int itemsPerThread = size / blockDim.x + 1;
                            int start = threadIdx.x * itemsPerThread;
                            int end = start + itemsPerThread;

                            //printf("S-E: size %d  threadIdx.x %d  itemsPerThread %d  start %d  end %d\\n",
                            //size, threadIdx.x, itemsPerThread, start, end);

                            float max_val = maxValue[0];

                            float value;
                            float localSum = 0.0f;
                            for (int i = start; i < end; i++) {
                                if (i < size) {
                                    value = expf(x[index + i] - max_val);
                                    if (!isfinite(value)) {
                                        printf("!VALUE: threadIdx.x %d i %d  value %.5f  x[index + i] %.5f  max_val %.5f\\n",
                                        threadIdx.x, i, value, x[index + i], max_val);
                                    }
                                    assert(isfinite(value));

                                    x[index + i] = value;
                                    localSum += value;
                                    // printf("CALC: threadIdx.x %d i %d  value %.5f  localSum %.5f\\n",
                                    // threadIdx.x, i, value, localSum);
                                }
                            }
                            
                            __syncthreads();  // Ensure all threads in block have stored their values

                            // Store the localSum in shared memory for reduction
                            sdata[threadIdx.x] = localSum;

                            __syncthreads();  // Ensure all threads in block have stored their values

                            // debug
                            if (threadIdx.x == 0) {
                                float sum0 = 0.0f;
                                for (int i = 0; i < blockDim.x; i++) {
                                    sum0 += sdata[i];
                                }
                                //printf("sum0  %.5f\\n", sum0);
                            }

                            // Block-wise reduction
                            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x) {
                                    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                                }
                                __syncthreads();  // Ensure all threads in block are in sync after each step
                            }

                            __syncthreads();

                            float finalSum = sdata[0];

                            // debug
                            //if (threadIdx.x == 0) {
                            //    printf("finalSum  %.5f\\n", finalSum);
                            //    for (int i = 0; i < blockDim.x; i++) {
                            //        printf("BLK i %d  sdata %.5f\\n", i, sdata[i]);
                            //    }
                            //}

                            for (int i = start; i < end; i++) {
                                if (i < size) {
                                    if (finalSum != 0.0f) {
                                        x[index + i] /= finalSum;
                                    }
                                    assert(isfinite(x[index + i]));
                                    //printf("NORM: i %d  result %.5f  size %d  threadIdx.x %d  start %d  end %d\\n",
                                    //                i, x[index + i], size, threadIdx.x, start, end);
                                    // assert(isfinite(x[index + i]));
                                }
                            }
                        }
                        """;
        return loadFromCode(code, "expAndSumSmall");
    }
}