package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Kernel: Implements softmax as a single kernel
 */
public class ExpSumNormalize extends Kernel {
    public static final int BLOCK_SIZE = 128;

    private final CUfunction smallKernel;

    public ExpSumNormalize(ContextCUDA cuda) {
        super(cuda, "expSumNormalize");
        smallKernel = createSmall();
    }

    public static void call(float[] x, float[] maxValue, int maxIndex, int index, int size) {
        float sum = 0.0f;
        float max_val = maxValue[maxIndex];

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

    public void test(float[] x, float[] maxValue, int maxIndex, int index, int size) {
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        Pointer pMaxValue = cuda.allocateAndCopyToDevice(TEST_STREAM, maxValue, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, px, pMaxValue, maxIndex, index, size);

        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, px, x.length, x);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(px);
        cuda.free(pMaxValue);

        call(copyOfx, maxValue, maxIndex, index, size);

        compareWithThreshold("ExpSumNormalize.call x ",
                x, copyOfx, 1e-5f);
    }

    public void call(int streamId, Pointer x, Pointer maxValue, int maxIndex, int index, int size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int sharedMemory = BLOCK_SIZE * Float.BYTES;
//        __global__ void expAndSumSmall(float* x, float* maxValue, int index, int maxIndex, int size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(x),
                Pointer.to(maxValue),
                Pointer.to(new int[]{index}),
                Pointer.to(new int[]{maxIndex}),
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
                        __global__ void expAndSumSmall(float* x, float* maxValue, int index, int maxIndex, int size) {
                            extern __shared__ float sdata[];

                            int itemsPerThread = size / blockDim.x + 1;
                            int start = threadIdx.x * itemsPerThread;
                            int end = min(start + itemsPerThread, size);
                            
                            float* xPointer = x + index;

                            float max_val = maxValue[maxIndex];

                            float value;
                            float localSum = 0.0f;
                            #pragma unroll 2
                            for (int i = start; i < end; i++) {
                                value = expf(xPointer[i] - max_val);

                                xPointer[i] = value;
                                localSum += value;
                            }
                            
                            // Store the localSum in shared memory for reduction
                            sdata[threadIdx.x] = localSum;

                            __syncthreads();  // Ensure all threads in block have stored their values

                            // Block-wise reduction
                            for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                if (threadIdx.x < stride && (threadIdx.x + stride) < blockDim.x) {
                                    sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                                }
                                __syncthreads();  // Ensure all threads in block are in sync after each step
                            }

                            float finalSum = sdata[0];

                            if (finalSum != 0.0f) {
                                #pragma unroll 4
                                for (int i = start; i < end; i++) {
                                    xPointer[i]  /= finalSum;
                                }
                            }
                        }
                        """;
        return loadFromCode(code, "expAndSumSmall");
    }
}