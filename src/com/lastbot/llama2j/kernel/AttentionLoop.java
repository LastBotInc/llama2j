package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AttentionLoop extends Kernel {
    private static final int SMALL_KERNEL = 1024;

    private final CUfunction smallKernel;

    public AttentionLoop(ContextCUDA cuda) {
        super(cuda, "attentionLoop");
        cuda.setDevice();
        smallKernel = createSmall();
    }

    public static void call(float[] q, float[] l_key_cache, float[] att, int attentionIndex, int keybase,
                            int kv_dim, int queryIndex, int pos, int head_size) {
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

//        // get the query vector for this head
//        float* q = s->q + h * head_size;
//        // attention scores for this head
//        float* att = s->att + h * p->seq_len;
//        // iterate over all timesteps, including the current one
//        for (int t = 0; t <= pos; t++) {
//            // get the key vector for this head and at this timestep
//            float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
//            // calculate the attention score as the dot product of q and k
//            float score = 0.0f;
//            for (int i = 0; i < head_size; i++) {
//                score += q[i] * k[i];
//            }
//            score /= sqrtf(head_size);
//            // save the score to the attention buffer
//            att[t] = score;
//        }

    }

    public void test(float[] score, float[] q, float[] l_key_cache, int queryIndex, int keyIndex, int head_size) {
        float[] copyOfScore = Arrays.copyOf(score, score.length);
        Pointer pScore = cuda.allocateAndCopyToDevice(TEST_STREAM, score, false);
        Pointer pq = cuda.allocateAndCopyToDevice(TEST_STREAM, q, false);
        Pointer pL_key_cache = cuda.allocateAndCopyToDevice(TEST_STREAM, l_key_cache, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pScore, pq, pL_key_cache, queryIndex, keyIndex, head_size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pScore, score);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pScore);
        cuda.free(pq);
        cuda.free(pL_key_cache);

//        call(copyOfScore, q, l_key_cache, queryIndex, keyIndex, head_size);

        compareWithThreshold("Attention.call score ",
                score, copyOfScore, 1e-5f);
    }

    public void call(int streamId, Pointer score, Pointer q, Pointer l_key_cache, int queryIndex, int keyIndex,
                     int head_size) {
        Pointer l_key_cacheWithOffset = l_key_cache.withByteOffset((long) keyIndex * Sizeof.FLOAT);

        call(streamId, score, q, l_key_cacheWithOffset, queryIndex, head_size);
    }

    public void call(int streamId, Pointer score, Pointer q, Pointer l_key_cache, int queryIndex, int head_size) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        if (head_size <= SMALL_KERNEL && head_size % 2 == 0) {
            int blockSizeX = findNextPowerOf2(head_size);
//            int blockSizeX = head_size;
            int gridSizeX = 1;
            int sharedMemory = blockSizeX * Sizeof.FLOAT;

//            __global__ void attentionScore(float[] score, float[] q, float[] l_key_cache, int head_size) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(score),
                    Pointer.to(q.withByteOffset((long) queryIndex * Sizeof.FLOAT)),
                    Pointer.to(l_key_cache),
                    Pointer.to(new int[]{head_size})
            );

            isError(cuLaunchKernel(smallKernel,
                    gridSizeX, 1, 1,          // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    sharedMemory, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            if (SYNC_KERNEL_CALLS) {
                cuda.synchronizeStream(streamId);
            }
        } else {
            throw new RuntimeException("AttentionScore.call invalid head size" + head_size);
        }
    }

    private CUfunction createSmall() {
        String code =
                """
                            extern "C"
                            __global__ void attentionLoop(float* score, float* q, float* k, int head_size) {
                                int tid = threadIdx.x;

                                extern __shared__ float sdata[];

                                if (tid < head_size) {
                                    sdata[tid] = q[tid] * k[tid];
                                } else {
                                    sdata[tid] = 0.0f;
                                }
                                __syncthreads();  // Ensure all threads in block have stored their values

                                // Block-wise reduction
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (tid < stride && (threadIdx.x + stride) < blockDim.x) {
                                        sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                                    }
                                    __syncthreads();  // Ensure all threads in block are in sync after each step
                                }

                                if (tid == 0) {
                                    score[0] = sdata[0] / sqrtf(head_size);
                                }
                            }
                        """;
        return loadFromCode(code, "attentionLoop");
    }
}