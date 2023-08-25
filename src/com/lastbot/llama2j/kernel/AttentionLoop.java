package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.LLogger;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AttentionLoop extends Kernel {
    public static final int BLOCK_SIZE = 32;

    private final CUfunction kernel;

    public AttentionLoop(ContextCUDA cuda) {
        super(cuda, "attentionLoop");
        cuda.setDevice();
        kernel = create();
    }

    public static void call(float[] q, float[] l_key_cache, float[] att, float[] max, int maxIndex, int attentionIndex,
                            int keybase, int kv_dim, int queryIndex, int pos, int head_size) {
        // iterate over all timesteps, including the current one
        float headMax = -Float.MAX_VALUE;
        int t;
        int i;
        float value;
        float score;
        for (t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            // calculate the attention score as the dot product of q and k
            int keyIndex = keybase + t * kv_dim;
            score = 0.0f;
            for (i = 0; i < head_size; i++) {
                value = q[queryIndex + i] * l_key_cache[keyIndex + i];
                score += value;
                headMax = Math.max(headMax, value);
            }
            // save the score to the attention buffer
            att[attentionIndex + t] = score / (float) Math.sqrt(head_size);
            // store max if needed
            if (max != null) {
                max[maxIndex] = Math.max(max[maxIndex], headMax);
            }
        }
    }

    public void test(float[] q, float[] l_key_cache, float[] att, float[] max, int maxIndex, int attentionIndex,
                     int keybase, int kv_dim, int queryIndex, int pos, int head_size) {
        float[] copyOfAtt = Arrays.copyOf(att, att.length);
        float[] copyOfMax = Arrays.copyOf(max, max.length);
        Pointer pq = cuda.allocateAndCopyToDevice(TEST_STREAM, q, false);
        Pointer pL_key_cache = cuda.allocateAndCopyToDevice(
                TEST_STREAM, l_key_cache, false);
        Pointer pAtt = cuda.allocateAndCopyToDevice(TEST_STREAM, att, false);
        Pointer pMax = cuda.allocateAndCopyToDevice(TEST_STREAM, max, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pq, pL_key_cache, pAtt, pMax, maxIndex, attentionIndex, keybase, kv_dim, queryIndex, pos, head_size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pAtt, att.length, att);
        cuda.copyFromDeviceToHost(TEST_STREAM, pMax, max.length, max);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pq);
        cuda.free(pL_key_cache);
        cuda.free(pMax);

        call(q, l_key_cache, copyOfAtt, copyOfMax, maxIndex, attentionIndex, keybase, kv_dim, queryIndex, pos, head_size);

        compareWithThreshold("AttentionLoop.call att ",
                att, copyOfAtt, 1e-5f);
        compareWithThreshold("AttentionLoop.call max ",
                max, copyOfMax, 1e-5f);
    }

    public void call(int streamId, Pointer q, Pointer l_key_cache, Pointer att, Pointer max, int maxIndex,
                     int attentionIndex, int keybase, int kv_dim, int queryIndex, int pos, int head_size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int size = pos + 1;
        int blockSizeX = BLOCK_SIZE;
        int gridSizeX = (int) Math.ceil((double) size / blockSizeX);

//        __global__ void attentionLoop(float* q, float* l_key_cache, float* att, float* max,
//        int maxIndex, int attentionIndex, int keybase, int kv_dim, int pos, int head_size) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(q.withByteOffset((long) queryIndex * Sizeof.FLOAT)),
                Pointer.to(l_key_cache),
                Pointer.to(att),
                Pointer.to(max),
                Pointer.to(new int[]{maxIndex}),
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

                            __device__ float atomicMax(float* address, float value) {
                                float old = *address, assumed;
                                do {
                                    assumed = old;
                                    if (assumed >= value) break; // No need to swap if the stored value is already larger
                                    old = __int_as_float(atomicCAS((unsigned int*)address,
                                                                    __float_as_int(assumed),
                                                                    __float_as_int(value)));
                                } while (assumed != old);
                                return old;
                            }

                            extern "C"
                            __global__ void attentionLoop(float* q, float* l_key_cache, float* att, float* max,
                            int maxIndex, int attentionIndex, int keybase, int kv_dim, int pos, int head_size) {
                                    int t = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (t <= pos) // NOTE: including current position
                                    {
                                        // get the key vector for this head and at this timestep
                                        // calculate the attention score as the dot product of q and k
                                        int keyIndex = keybase + t * kv_dim;
                                        float score = 0.0f;
                                        float headMax = -FLT_MAX;
                                        float value;
                                        float* keyPointer = (&(l_key_cache[keyIndex]));

                                        #pragma unroll 16
                                        for (int i = 0; i < head_size; i++) {
                                            value = q[i] * keyPointer[i];
                                            score += value;
                                            headMax = fmaxf(headMax, value);
                                        }
                                        // save the score to the attention buffer
                                        att[attentionIndex + t] = score / (float) sqrtf(head_size);
                                        // save max
                                        if (max != NULL) {
                                            float *maxAddress = max + maxIndex;
                                            atomicMax(maxAddress, headMax);
                                        }
                                    }
                                }
                        """;
        return loadFromCode(code, "attentionLoop");
    }
}