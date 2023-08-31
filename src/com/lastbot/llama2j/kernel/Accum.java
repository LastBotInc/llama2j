package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Kernel: Adds a vector to another.
 */
public class Accum extends Kernel {
    public static final int BLOCK_SIZE = 64;

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
        Pointer pa = cuda.allocateAndCopyToDevice(TEST_STREAM, a, false);
        Pointer pb = cuda.allocateAndCopyToDevice(TEST_STREAM, b, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pa, pb, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pa, a.length, a);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pa);
        cuda.free(pb);

        call(copyOfA, b, size);

        compareWithThreshold("Accum.call", a, copyOfA, 1e-5f);
    }

    public void call(int streamId, Pointer a, Pointer b, int size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(new int[]{size})
        );

        int blockSizeX = Math.min(findNextPowerOf2(size), BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) size / blockSizeX);

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, cuda.getCUKernelStream(streamId),  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
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
