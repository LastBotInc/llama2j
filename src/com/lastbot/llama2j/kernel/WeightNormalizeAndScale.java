package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.LLogger;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class WeightNormalizeAndScale extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public WeightNormalizeAndScale(ContextCUDA cuda) {
        super(cuda, "weightNormalizeAndScale");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] out, float[] x, float[] weight, int weightIndex,
                             float[] sumOfSquares, int size) {
        float ss = sumOfSquares[0];
        for (int j = 0; j < size; j++) {
            out[j] = weight[weightIndex + j] * (ss * x[j]);
        }
    }

    public void test(float[] out, float[] x, float[] weight, int weightIndex,
                     float[] sumOfSquares, int size) {
        float[] copyOfOut = Arrays.copyOf(out, out.length);
        float[] copyOfx = Arrays.copyOf(x, x.length); // x can point to out!
        Pointer pOut = cuda.allocateAndCopyToDevice(out, false);
        Pointer px = cuda.allocateAndCopyToDevice(x, false);
        Pointer pWeight = cuda.allocateAndCopyToDevice(weight, false);
        Pointer pSumOfSquares = cuda.allocateAndCopyToDevice(sumOfSquares, false);
        cuda.synchronizeTransfer();
        call(0, pOut, px, pWeight, weightIndex, pSumOfSquares, size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pOut, out);
        cuda.synchronizeTransfer();
        cuda.free(pOut);
        cuda.free(px);
        cuda.free(pWeight);
        cuda.free(pSumOfSquares);

        call(copyOfOut, copyOfx, weight, weightIndex, sumOfSquares, size);

        compareWithThreshold("WeightNormalizeAndScale.call", out, copyOfOut, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer out, Pointer x, Pointer weight, int weightIndex,
                     Pointer sumOfSquares, int size) {
//        __global__ void weightNormalizeAndScale(float *out, float *x, float *weight,
//        int weightIndex, float* sumOfSquares, int size)
        Pointer kernelParameters = Pointer.to(
                Pointer.to(out),
                Pointer.to(x),
                Pointer.to(weight),
                Pointer.to(new int[]{weightIndex}),
                Pointer.to(sumOfSquares),
                Pointer.to(new int[]{size})
        );

        // Set up the kernel launch parameters.
        int blockSizeX = Math.min(findNextPowerOf2(size), 1024);
        int gridSizeX = (int) Math.ceil((double) size / blockSizeX);

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, cuda.getCUKernelStream(kernelStreamId),  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
    }

    private CUfunction create() {
        String code =
                """
                    extern "C"
                    __global__ void weightNormalizeAndScale(float *out, const float *x, const float *weight,
                    const int weightIndex, const float* sumOfSquares, const int size)
                    {
                        int i = blockIdx.x * blockDim.x + threadIdx.x;
                        if (i < size) {
                            out[i] = weight[weightIndex + i] * (sumOfSquares[0] * x[i]);
                        }
                    }
                """;
        return loadFromCode(code, "weightNormalizeAndScale");
    }
}
