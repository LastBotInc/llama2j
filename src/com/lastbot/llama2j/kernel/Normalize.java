package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class Normalize extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public Normalize(ContextCUDA cuda) {
        super(cuda, "normalize");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] x, float[] divider, int index, int size) {
        float d = divider[0];
        for (int i = 0; i < size; i++) {
            x[index + i] /= d;
        }
    }

    public void test(float[] x, float[] divider, int index, int size) {
        float[] copyOfX = Arrays.copyOf(x, x.length);
        Pointer px = cuda.allocateAndCopyToDevice(x, false);
        Pointer pDivider = cuda.allocateAndCopyToDevice(divider, false);
        cuda.synchronizeTransfer();
        call(0, px, pDivider, index, size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(px, x);
        cuda.synchronizeTransfer();
        cuda.free(px);
        cuda.free(pDivider);

        call(copyOfX, divider, index, size);

        compareWithThreshold("Normalize.call", x, copyOfX, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer x, Pointer divider, int index, int size) {
//        __global__ void normalize(float *x, float *divider, int index, int size)
        Pointer kernelParameters = Pointer.to(
                Pointer.to(x),
                Pointer.to(divider),
                Pointer.to(new int[]{index}),
                Pointer.to(new int[]{size})
        );

        int blockSizeX = MAX_THREADS_PER_BLOCK;
        int gridSizeX = (int) Math.ceil((double) size / blockSizeX);

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, cuda.getCUKernelStream(kernelStreamId),  // Shared memory size and stream
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
                    __global__ void normalize(float *x, float *divider, int index, int size)
                    {
                        int i = blockIdx.x * blockDim.x + threadIdx.x;
                        if (i < size) {
                            x[index + i] /= divider[0];
                        }
                    }
                """;
        return loadFromCode(code, "normalize");
    }
}
