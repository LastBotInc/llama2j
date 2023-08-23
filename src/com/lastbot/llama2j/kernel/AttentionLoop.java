package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AttentionLoop extends Kernel {
    public static final int BLOCK_SIZE = 64;

    private final CUfunction kernel;

    public AttentionLoop(ContextCUDA cuda) {
        super(cuda, "attentionLoop");
        cuda.setDevice();
        kernel = create();
    }

    public static void call(float[] q, float[] l_key_cache, float[] att, int attentionIndex,
                            int keybase, int kv_dim, int queryIndex, int pos, int head_size) {
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            // calculate the attention score as the dot product of q and k
            int keyIndex = keybase + t * kv_dim;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[queryIndex + i] * l_key_cache[keyIndex + i];
            }
            // save the score to the attention buffer
            att[attentionIndex + t] = score / (float) Math.sqrt(head_size);
        }
    }

    public void test(float[] q, float[] l_key_cache, float[] att, int attentionIndex,
                     int keybase, int kv_dim, int queryIndex, int pos, int head_size) {
        float[] copyOfAtt = Arrays.copyOf(att, att.length);
        Pointer pq = cuda.allocateAndCopyToDevice(TEST_STREAM, q, false);
        Pointer pL_key_cache = cuda.allocateAndCopyToDevice(
                TEST_STREAM, l_key_cache, false);
        Pointer pAtt = cuda.allocateAndCopyToDevice(TEST_STREAM, att, false);
        Pointer tmp = cuda.allocate(Sizeof.FLOAT);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pq, pL_key_cache, pAtt, tmp, attentionIndex, keybase, kv_dim, queryIndex, pos, head_size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pAtt, att.length, att);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pq);
        cuda.free(pL_key_cache);
        cuda.free(tmp);

        call(q, l_key_cache, copyOfAtt, attentionIndex, keybase, kv_dim, queryIndex, pos, head_size);

        compareWithThreshold("AttentionLoop.call att ",
                att, copyOfAtt, 1e-5f);
    }

    public void call(int streamId, Pointer q, Pointer l_key_cache, Pointer att, Pointer max, int attentionIndex,
                     int keybase, int kv_dim, int queryIndex, int pos, int head_size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int size = pos + 1;
        int blockSizeX = findNextPowerOf2(size);
        int gridSizeX = (int) Math.ceil((double) size / blockSizeX);

//        __global__ void attentionLoop(float* q, float* l_key_cache, float* att, float* max,
//        int attentionIndex, int keybase, int kv_dim, int pos, int head_size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(q.withByteOffset((long) queryIndex * Sizeof.FLOAT)),
                Pointer.to(l_key_cache),
                Pointer.to(att),
                Pointer.to(max),
                Pointer.to(new int[]{attentionIndex}),
                Pointer.to(new int[]{keybase}),
                Pointer.to(new int[]{kv_dim}),
                Pointer.to(new int[]{pos}),
                Pointer.to(new int[]{head_size})
        );

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
    }

    private CUfunction create() {
        String code =
                """
                            #include <cfloat>
                            extern "C"
                            __global__ void attentionLoop(float* q, float* l_key_cache, float* att, float* max,
                            int attentionIndex, int keybase, int kv_dim, int pos, int head_size) {
                                    int t = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (t <= pos) // NOTE: including current position
                                    {
                                        // get the key vector for this head and at this timestep
                                        // calculate the attention score as the dot product of q and k
                                        int keyIndex = keybase + t * kv_dim;
                                        float value;
                                        float score = 0.0f;
                                        float headMax = -FLT_MAX;
                                        for (int i = 0; i < head_size; i++) {
                                            value = q[i] * l_key_cache[keyIndex + i];
                                            score += value;
                                            headMax = fmaxf(headMax, value);
                                        }
                                        // save the score to the attention buffer
                                        att[attentionIndex + t] = score / (float) sqrtf(head_size);
                                        max[0] = headMax;
                                    }
                                }
                        """;
        return loadFromCode(code, "attentionLoop");
    }
}