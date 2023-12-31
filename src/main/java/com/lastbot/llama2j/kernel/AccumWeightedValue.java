package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.SlicePointer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

/**
 * Kernel: Calculates accumulated attention values to be stored back to xb
 * <p>
 * See: callNoUnroll() for details.
 *
 */
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
        if (head_size % 4 == 0) {
            callUnroll4(xb, att, l_value_cache, pos, xbIndex, valueBase, head_size, kv_dim, attentionIndex);
        } else {
            callNoUnroll(xb, att, l_value_cache, pos, xbIndex, valueBase, head_size, kv_dim, attentionIndex);
        }
    }

    public static void callUnroll4(float[] xb, float[] att, float[] l_value_cache, int pos,
                                   int xbIndex, int valueBase, int head_size, int kv_dim, int attentionIndex) {
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
            // unroll by 4
            for (i = 0; i < head_size; i += 4) {
                xb[xbIndex + i] += a * l_value_cache[vIndex + i];
                xb[xbIndex + i + 1] += a * l_value_cache[vIndex + i + 1];
                xb[xbIndex + i + 2] += a * l_value_cache[vIndex + i + 2];
                xb[xbIndex + i + 3] += a * l_value_cache[vIndex + i + 3];
            }
            vIndex += kv_dim;
        }
    }

    public static void callNoUnroll(float[] xb, float[] att, float[] l_value_cache, int pos, int xbIndex,
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
        int offsetValueBase = valueBase - Math.toIntExact(slidedL_value_cache.floatOffset()); // note conversion
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

                                    float* attPointer = att + attentionIndex;
                                    float* valuePointer = &(l_value_cache[valueBase]) + i;

                                    #pragma unroll 8
                                    for (int t = 0; t <= pos; t++) {
                                        sum += attPointer[t] * valuePointer[t * kv_dim];
                                    }
                                    xb[xbIndex + i] = sum;
                                }
                            }
                        """;
        return loadFromCode(code, "accumWeightedValue");
    }
}
