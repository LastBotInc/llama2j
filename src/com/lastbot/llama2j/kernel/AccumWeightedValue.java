package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.SlicePointer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AccumWeightedValue extends Kernel {
    public static final int BLOCK_SIZE = 64;

    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public AccumWeightedValue(ContextCUDA cuda) {
        super(cuda, "accumWeightedValue");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] xb, float[] att, float[] l_value_cache, int pos, int xbIndex,
                            int valueBase, int head_size, int kv_dim, int attentionIndex) {
        // weighted sum of the values, store back into xb
        int vIndex = valueBase;
        float a;
        int t;
        int i;

        for (i = 0; i < head_size; i++) {
            xb[xbIndex + i] = 0f;
        }
        for (t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            // get the attention weight for this timestep
            a = att[attentionIndex + t];
            // accumulate the weighted value into xb
            for (i = 0; i < head_size; i++) {
                xb[xbIndex + i] += a * l_value_cache[vIndex + i];
            }
            vIndex += kv_dim;
        }
    }

    public static void callParallel(float[] xb, float[] att, float[] l_value_cache, int pos, int xbIndex,
                                    int valueBase, int head_size, int kv_dim, int attentionIndex) {
        CountDownLatch latch = new CountDownLatch(pos);
        // weighted sum of the values, store back into xb
        for (int t = 0; t <= pos; t++) {
            int finalT = t;
            Thread.ofVirtual().start(() -> {
                // get the value vector for this head and at this timestep
                int vIndex = valueBase + finalT * kv_dim;
                // get the attention weight for this timestep
                float a = att[attentionIndex + finalT];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[xbIndex + i] += a * l_value_cache[vIndex + i];
                }
                latch.countDown();
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public void test(float[] xb, float[] att, float[] l_value_cache, int pos, int xbIndex,
                     int valueBase, int head_size, int kv_dim, int attentionIndex) {
        float[] copyOfXb = Arrays.copyOf(xb, xb.length);
        Pointer pXb = cuda.allocateAndCopyToDevice(TEST_STREAM, xb, false);
        Pointer pAtt = cuda.allocateAndCopyToDevice(TEST_STREAM, att, false);
        Pointer pl_value_cache = cuda.allocateAndCopyToDevice(TEST_STREAM, l_value_cache, false);
        SlicePointer slidedL_value_cache = new SlicePointer(pl_value_cache, 0, 0,
                (long) l_value_cache.length * Sizeof.FLOAT);

        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pXb, pAtt, slidedL_value_cache, pos, xbIndex, valueBase, head_size, kv_dim, attentionIndex);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pXb, xb.length, xb);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pXb);
        cuda.free(pAtt);
        cuda.free(pl_value_cache);

        call(copyOfXb, att, l_value_cache, pos, xbIndex, valueBase, head_size, kv_dim, attentionIndex);

        compareWithThreshold("AccumWeightedValue.call", xb, copyOfXb, 1e-5f);
    }

    public void call(int streamId, Pointer xb, Pointer att, SlicePointer slidedL_value_cache, int pos, int xbIndex,
                     int valueBase, int head_size, int kv_dim, int attentionIndex) {

        //        __global__ void accumWeightedValue(float* xb, float* att, float* l_value_cache,
        //        int pos, int xbIndex, int valueBase, int head_size,
        //        int kv_dim, int attentionIndex)
        int offsetValueBase = valueBase - (int) slidedL_value_cache.floatOffset(); // note conversion
        Pointer l_value_cache = slidedL_value_cache.pointer();

        Pointer kernelParameters = Pointer.to(
                Pointer.to(xb),
                Pointer.to(att),
                Pointer.to(l_value_cache),
                Pointer.to(new int[]{pos}),
                Pointer.to(new int[]{xbIndex}),
                Pointer.to(new int[]{offsetValueBase}),
                Pointer.to(new int[]{head_size}),
                Pointer.to(new int[]{kv_dim}),
                Pointer.to(new int[]{attentionIndex})
        );

        int blockSizeX = Math.min(findNextPowerOf2(head_size), BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) head_size / blockSizeX);

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
                            __global__ void accumWeightedValue(float* xb, float* att, float* l_value_cache,
                                                                int pos, int xbIndex, int valueBase, int head_size,
                                                                int kv_dim, int attentionIndex)
                            {
                                int i = blockIdx.x * blockDim.x + threadIdx.x;

                                if (i < head_size) {
                                    float sum = 0.0f;
                                    int vIndex;
                                    float a;

                                    for (int t = 0; t <= pos; t++) {
                                        // get the value vector for this head and at this timestep
                                        vIndex = valueBase + t * kv_dim;
                                        // get the attention weight for this timestep
                                        a = att[attentionIndex + t];
                                        // accumulate the weighted value into xb
                                        sum += a * l_value_cache[vIndex + i];
                                    }
                                    xb[xbIndex + i] = sum;
                                }
                            }
                        """;
        return loadFromCode(code, "accumWeightedValue");
    }
}
