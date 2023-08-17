package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class Accum extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public Accum(ContextCUDA cuda) {
        super(cuda, "accum");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] a, float[] b, int size) {
        for (int i = 0; i < size; i++) {
            a[i] += b[i];
        }
    }

    public void test(float[] a, float[] b, int size) {
        float[] copyOfA = Arrays.copyOf(a, a.length);
        Pointer pa = cuda.allocateAndCopyToDevice(a, false);
        Pointer pb = cuda.allocateAndCopyToDevice(b, false);
        cuda.synchronizeTransfer();
        call(0, pa, pb, size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pa, a);
        cuda.synchronizeTransfer();
        cuda.free(pa);
        cuda.free(pb);

        call(copyOfA, b, size);

        compareWithThreshold("Accum.call", a, copyOfA, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer a, Pointer b, int size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(new int[]{size})
        );

        int blockSizeX = Math.min(findNextPowerOf2(size), MAX_THREADS_PER_BLOCK);
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
                            __global__ void accum(float *a, float *b, int size)
                            {
                                int i = blockIdx.x * blockDim.x + threadIdx.x;
                                if (i < size) {
                                    a[i] += b[i];
                                }
                            }
                        """;
        return loadFromCode(code, "accum");
    }
}
