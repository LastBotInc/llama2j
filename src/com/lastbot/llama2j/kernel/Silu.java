package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.LLogger;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Kernel: implements activation as SwiGLU
 * <p>
 *     for (int i = 0; i < hidden_dim; i++) {
 *          float val = s->hb[i];
 *          // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
 *          val *= (1.0f / (1.0f + expf(-val)));
 *          // elementwise multiply with w3(x)
 *          val *= s->hb2[i];
 *          s->hb[i] = val;
 *     }
 * <p>
 * CPU version implements partial lookup table with liner interpolation
 * for EXP values for performance.
 *
 */
public class Silu extends Kernel {
    public static final int BLOCK_SIZE = 64;
    public static final boolean LOOKUP_EXP = true;

    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public Silu(ContextCUDA cuda) {
        super(cuda, "silu");
        this.cuda = cuda;
        this.kernel = create();
    }

    private static final int MAX_LOOK_UP_ABS_VALUE = 5;
    private static final int TABLE_SIZE = 8192;
    private static final int LOOP_UP_MULTIPLIER = (TABLE_SIZE / (2 * MAX_LOOK_UP_ABS_VALUE));
    private static final float[] expTable = new float[TABLE_SIZE];
    private static boolean isInitialized = false;

    public static void init() {
        Thread.ofVirtual().start(() ->
        {
            for (int i = 0; i < TABLE_SIZE; i++) {
                double val = (i / (double) TABLE_SIZE) * (2.0 * MAX_LOOK_UP_ABS_VALUE) - MAX_LOOK_UP_ABS_VALUE;
                expTable[i] = (float) (1.0f / (1.0f + Math.exp((-val))));
            }
            isInitialized = true;
            LLogger.info("Silu exp() lookup created");
        });
    }

    public static float lookupExp(float val) {
        if (LOOKUP_EXP && isInitialized) {
            float position = ((val + MAX_LOOK_UP_ABS_VALUE) * LOOP_UP_MULTIPLIER);
            int index = (int) position;
            if (position > 0 && position < TABLE_SIZE - 1) {
                float valueAtLowerIndex = expTable[index];
                return valueAtLowerIndex + (position - index) * (expTable[index + 1] - valueAtLowerIndex);
            }
        }
        return (float) (1.0f / (1.0f + Math.exp((-val))));
    }

    public static void call(float[] hb, float[] hb2, int hidden_dim) {
        for (int i = 0; i < hidden_dim; i++) {
            hb[i] *= lookupExp(hb[i]) * hb2[i];
        }
        // for (int i = 0; i < hidden_dim; i++) {
        // hb[i] *= (float) (1.0f / (1.0f + Math.exp((-hb[i])))) * hb2[i];
        // }
    }

    public void test(float[] hb, float[] hb2, int hidden_dim) {
        float[] copyOfHb = Arrays.copyOf(hb, hb.length);
        Pointer pHb = cuda.allocateAndCopyToDevice(TEST_STREAM, hb, false);
        Pointer pHb2 = cuda.allocateAndCopyToDevice(TEST_STREAM, hb2, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pHb, pHb2, hidden_dim);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pHb, hb.length, hb);
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

        int blockSizeX = Math.min(findNextPowerOf2(hidden_dim), BLOCK_SIZE);
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
                            __global__ void silu(float *hb, float *hb2, int hidden_dim)
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
        return loadFromCode(code, "silu");
    }
}
