package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class Silu extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public Silu(ContextCUDA cuda) {
        super(cuda, "silu");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] hb, float[] hb2, int hidden_dim) {
        for (int i = 0; i < hidden_dim; i++) {
            hb[i] *= (float) (1.0f / (1.0f + Math.exp((-hb[i])))) * hb2[i];
        }
    }

    public void test(float[] hb, float[] hb2, int hidden_dim) {
        float[] copyOfHb = Arrays.copyOf(hb, hb.length);
        Pointer pHb = cuda.allocateAndCopyToDevice(TEST_STREAM, hb, false);
        Pointer pHb2 = cuda.allocateAndCopyToDevice(TEST_STREAM, hb2, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pHb, pHb2, hidden_dim);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pHb, hb);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pHb);
        cuda.free(pHb2);

        call(copyOfHb, hb2, hidden_dim);

        compareWithThreshold("Silu.call", hb, copyOfHb, 1e-5f);
    }

    public void call(int streamId, Pointer hb, Pointer hb2, int hidden_dim) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(hb),
                Pointer.to(hb2),
                Pointer.to(new int[]{hidden_dim})
        );

        int blockSizeX = Math.min(findNextPowerOf2(hidden_dim), MAX_THREADS_PER_BLOCK);
        int gridSizeX = (int) Math.ceil((double) hidden_dim / blockSizeX);

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
                            __global__ void accum(float *hb, float *hb2, int hidden_dim)
                            {
                                int i = blockIdx.x * blockDim.x + threadIdx.x;
                                if (i < hidden_dim) {
                                    float value = hb[i];
                                    //float sigmoid = 1.0f / (1.0f + __expf(-value));
                                    // __frcp_rz is faster but less accurate
                                    float sigmoid = __frcp_rz(1.0f + __expf(-value));
                                    hb[i] = value * sigmoid * hb2[i];
                                }
                            }
                        """;
        return loadFromCode(code, "accum");
    }
}
