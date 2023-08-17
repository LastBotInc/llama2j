package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class MatMul extends Kernel {
    private final CUfunction kernel;

    public MatMul(ContextCUDA cuda) {
        super(cuda, "matMul");
        cuda.setDevice();
        kernel = create();
    }

    public static void call(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        int i;
        float val;
        int weightPos;
        for (i = 0; i < d; i++) {
            weightPos = weightIndex + i * n;
            val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[weightPos + j] * x[j];
            }
            xout[i] = val;
        }
    }

    public void test(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        call(xout, x, w, weightIndex, n, d);
        float[] copyOfXout = Arrays.copyOf(xout, xout.length);
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer pXout = cuda.allocateAndCopyToDevice(xout, false);
        Pointer px = cuda.allocateAndCopyToDevice(x, false);
        Pointer pw = cuda.allocateAndCopyToDevice(w, false);
        cuda.synchronizeTransfer();
        call(0, pXout, px, pw, weightIndex, n, d);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pXout, xout);
        cuda.synchronizeTransfer();
        cuda.free(pXout);
        cuda.free(px);
        cuda.free(pw);

        call(copyOfXout, copyOfx, w, weightIndex, n, d);

        compareWithThreshold("MatMul.call xout ",
                xout, copyOfXout, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer xout, Pointer x, Pointer w, int weightIndex, int n, int d) {
        CUstream stream = cuda.getCUKernelStream(kernelStreamId);
        int blockSizeX = Math.min(findNextPowerOf2(d), MAX_THREADS_PER_BLOCK);
        int gridSizeX = (int) Math.ceil((double) d / blockSizeX);

//        __global__ void matMul(float* xout, float* x, float* w, int n, int d) {
        Pointer wIndexed = w.withByteOffset((long) weightIndex * Sizeof.FLOAT);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(xout),
                Pointer.to(x),
                Pointer.to(wIndexed),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{d})
        );

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeKernel(kernelStreamId);
        }
    }

    private CUfunction create() {
        String code =
                """
                            extern "C"
                            __global__ void matMul(float* xout, float* x, float* w, int n, int d) {
                                int i = blockIdx.x * blockDim.x + threadIdx.x;

                                if (i < d) {
                                    float* weightPos = w + i * n;
                                    float val = 0.0f;
                                    for (int j = 0; j < n; j++) {
                                        val += weightPos[j] * x[j];
                                    }
                                    xout[i] = val;
                                }
                            }
                        """;
        return loadFromCode(code, "matMul");
    }
}