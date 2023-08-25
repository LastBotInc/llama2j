package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.LLogger;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class ApplyRope extends Kernel {
    public static final int BLOCK_SIZE = 64;

    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public ApplyRope(ContextCUDA cuda) {
        super(cuda, "applyRope");
        this.cuda = cuda;
        this.kernel = create();
    }

    private static float[] cosCache = null;
    private static float[] sinCache = null;

    public static void init(int dim, int head_size, int seqLen) {
        Thread.ofVirtual().start(() -> {
            if (cosCache != null) {
                return;
            }
            int cacheSize = (dim / 2) * seqLen;
            cosCache = new float[cacheSize];
            sinCache = new float[cacheSize];

            int i;
            int pos;
            float val;
            int index;
            for (i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = (float) (1.0f / Math.pow(10000.0f, head_dim / (float) head_size));
                for (pos = 0; pos < seqLen; pos++) {
                    val = pos * freq;
                    index = (pos * dim / 2) + i / 2;
                    cosCache[index] = (float) Math.cos(val);
                    sinCache[index] = (float) Math.sin(val);
                }
            }
            LLogger.info("Rope cache initialized with  " + String.format("%,d", cacheSize) + " values");
        });
    }

    public static void call(float[] q, float[] k, int pos, int dim, int kv_dim, int head_size) {
        // if not initialized, accessing cache will throw an exception

        // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        int i;
        int index;
        float fcr;
        float fci;
        float v0;
        float v1;

        // keys
        for (i = 0; i < dim; i += 2) {
            index = (pos * dim / 2) + i / 2;
            fcr = cosCache[index];
            fci = sinCache[index];
            v0 = q[i];
            v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        // values
        for (i = 0; i < kv_dim; i += 2) {
            index = (pos * dim / 2) + i / 2;
            fcr = cosCache[index];
            fci = sinCache[index];
            v0 = k[i];
            v1 = k[i + 1];
            k[i] = v0 * fcr - v1 * fci;
            k[i + 1] = v0 * fci + v1 * fcr;
        }
    }

    public void test(float[] q, float[] k, int pos, int dim, int kv_dim, int head_size) {
        float[] copyOfQ = Arrays.copyOf(q, q.length);
        float[] copyOfK = Arrays.copyOf(k, k.length);
        Pointer pq = cuda.allocateAndCopyToDevice(TEST_STREAM, q, false);
        Pointer pk = cuda.allocateAndCopyToDevice(TEST_STREAM, k, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pq, pk, pos, dim, kv_dim, head_size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pk, k.length, k);
        cuda.copyFromDeviceToHost(TEST_STREAM, pq, q.length, q);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pq);
        cuda.free(pk);

        call(copyOfQ, copyOfK, pos, dim, kv_dim, head_size);

        compareWithThreshold("ApplyRope.call q", q, copyOfQ, 1e-5f);
        compareWithThreshold("ApplyRope.call k", k, copyOfK, 1e-5f);
    }

    public void call(int streamId, Pointer q, Pointer k, int pos, int dim, int kv_dim, int head_size) {
//        __global__ void applyRope(float *q, float *k, int pos, int dim, int kv_dim, int head_size)
        Pointer kernelParameters = Pointer.to(
                Pointer.to(q),
                Pointer.to(k),
                Pointer.to(new int[]{pos}),
                Pointer.to(new int[]{dim}),
                Pointer.to(new int[]{kv_dim}),
                Pointer.to(new int[]{head_size})
        );

        // choose larger dimension
        int maxDim = Math.max(dim, kv_dim);

        int blockSizeX = Math.min(findNextPowerOf2(maxDim), BLOCK_SIZE);

        int gridSizeX = (int) Math.ceil((double) maxDim / blockSizeX);

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
                            __global__ void applyRope(float *q, float *k, int pos, int dim, int kv_dim, int head_size)
                            {
                                // process elements in steps of 2
                                int i = blockIdx.x * blockDim.x + threadIdx.x;

                                if (i % 2 != 0) {
                                    return;
                                }

                                int head_dim = i % head_size;
                                float freq = (float) (1.0f / powf(10000.0f, head_dim / (float)head_size));
                                float val = pos * freq;
                                float fcr = (float) cosf(val);
                                float fci = (float) sinf(val);

                                // Ensure we don't go out of bounds
                                if (i < dim - 1) {
                                    float q0 = q[i];
                                    float q1 = q[i + 1];
                                    q[i] = q0 * fcr - q1 * fci;
                                    q[i + 1] = q0 * fci + q1 * fcr;
                                }

                                // Ensure we don't go out of bounds
                                if (i < kv_dim - 1) {
                                    float k0 = k[i];
                                    float k1 = k[i + 1];
                                    k[i] = k0 * fcr - k1 * fci;
                                    k[i + 1] = k0 * fci + k1 * fcr;
                                }
                            }
                        """;

        return loadFromCode(code, "applyRope");
    }
}
