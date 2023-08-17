package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class MemSetFloat extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public MemSetFloat(ContextCUDA cuda) {
        super(cuda, "memSetFloat");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] a, float value, int size) {
        for (int i = 0; i < size; i++) {
            a[i] = value;
        }
    }

    public void test(float[] a, float value, int size) {
        float[] copyOfA = Arrays.copyOf(a, a.length);
        Pointer pa = cuda.allocateAndCopyToDevice(a, false);
        cuda.synchronizeTransfer();
        call(0, pa, value, size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pa, a);
        cuda.synchronizeTransfer();
        cuda.free(pa);

        call(copyOfA, value, size);

        compareWithThreshold("MemSetFloat.call", a, copyOfA, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer a, float value, int size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(new float[]{value}),
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
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeKernel(kernelStreamId);
        }
    }

    private CUfunction create() {
        String code =
                """
                            extern "C"
                            __global__ void memSetFloat(float *a, float value, int size)
                            {
                                volatile int tidx = threadIdx.x + blockIdx.x * blockDim.x;
                                
                                volatile int stride = gridDim.x * blockDim.x;
                                for (int i = tidx; i < size; i += stride) {
                                    a[i] = value;
                                }
                            }
                        """;
        return loadFromCode(code, "memSetFloat");
    }
}
