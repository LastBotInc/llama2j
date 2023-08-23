package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class SumOfSquares extends Kernel {
    public static final int BLOCK_SIZE = 256;

    private final CUfunction smallKernel;

    public SumOfSquares(ContextCUDA cuda) {
        super(cuda, "sumOfSquares");
        smallKernel = createSmall();
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
        Pointer pSum = cuda.allocateAndCopyToDevice(TEST_STREAM, sum, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pSum, px, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pSum, sum.length, sum);
        cuda.copyFromDeviceToHost(TEST_STREAM, px, x.length, x);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pSum);
        cuda.free(px);

        call(copyOfSum, copyOfx, size);

        compareWithThreshold("SumOfSquares.call size " + size + " sum ",
                sum, copyOfSum, 1e-2f);
    }

    public void call(int streamId, Pointer sum, Pointer x, int size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int sharedMemory = BLOCK_SIZE * Float.BYTES;
        // __global__ void expAndSumSmall(float* x, float* maxValue, int index, int size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(sum),
                Pointer.to(x),
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
                    __global__ void sumOfSquares(float* sum, float* x, int size) {
                            extern __shared__ float sdata[];

                            int itemsPerThread = size / blockDim.x + 1;
                            int start = threadIdx.x * itemsPerThread;
                            int end = start + itemsPerThread;

                            float value;
                            float localSum = 0.0f;
                            for (int i = start; i < end; i++) {
                                if (i < size) {
                                    value = x[i] * x[i];
                                    localSum += value;
                                }
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

                            if (threadIdx.x == 0) {
                                float ss = sdata[0];
                                ss /= size;
                                ss += 1e-5f;
                                ss = sqrtf(ss);
                                sum[0] = ss;
                            }
                    }
                """;
        return loadFromCode(code, "sumOfSquares");
    }
}