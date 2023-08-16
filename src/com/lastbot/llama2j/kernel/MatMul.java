package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;

import java.util.Arrays;

public class MatMul extends Kernel {

    private final cublasHandle[] cublasHandles;
    private static final float[] alphaArray = new float[]{1.0f};
    private static final float[] betaArray = new float[]{0.0f};
    private final Pointer pAlpha;
    private final Pointer pBeta;

    public MatMul(ContextCUDA cuda) {
        super(cuda, "matMul");

        this.cublasHandles = cuda.createCublasHandles();
        cuda.setDevice();
        this.pAlpha = Pointer.to(alphaArray);
        this.pBeta = Pointer.to(betaArray);
        cuda.synchronizeTransfer();
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
                xout, copyOfXout, 1e-2f);
    }

    public void call(int kernelStreamId, Pointer xout, Pointer x, Pointer w, int weightIndex, int n, int d) {
        Pointer wWithIndex = w.withByteOffset((long) weightIndex * Float.BYTES);
        cuda.setDevice();
        JCublas2.cublasSgemv(cublasHandles[kernelStreamId], cublasOperation.CUBLAS_OP_T,
                n, d, pAlpha, wWithIndex, n, x, 1, pBeta, xout, 1);
        //        JCublas.cublasSgemv('t', n, d, 1.0f, wWithIndex, n, x, 1, 0.0f, xout, 1);
//        JCublas.cublasSgemv('n', n, d, 1.0f, w, n, x, 1, 0.0f, xout, 1);
    }
}
