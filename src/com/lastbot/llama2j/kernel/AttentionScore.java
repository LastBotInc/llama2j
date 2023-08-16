package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.LLogger;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class AttentionScore extends Kernel {
    private static final int SMALL_KERNEL = 1024;

    private final CUfunction smallKernel;

    public AttentionScore(ContextCUDA cuda) {
        super(cuda, "attentionScore");
        smallKernel = createSmall();
    }

    public static void call(float[] score, float[] q, float[] l_key_cache, int queryIndex, int keyIndex, int head_size) {
        // calculate the attention score as the dot product of q and k
        float sum = 0.0f;
        for (int i = 0; i < head_size; i++) {
            sum += q[queryIndex + i] * l_key_cache[keyIndex + i];
        }
        score[0] = sum / (float) Math.sqrt(head_size);
    }

    public void test(float[] score, float[] q, float[] l_key_cache, int queryIndex, int keyIndex, int head_size) {
        float[] copyOfScore = Arrays.copyOf(score, score.length);
        Pointer pScore = cuda.allocateAndCopyToDevice(score, false);
        Pointer pq = cuda.allocateAndCopyToDevice(q, false);
        Pointer pL_key_cache = cuda.allocateAndCopyToDevice(l_key_cache, false);
        cuda.synchronizeTransfer();

        call(0, pScore, pq, pL_key_cache, queryIndex, keyIndex, head_size);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pScore, score);
        cuda.synchronizeTransfer();
        cuda.free(pScore);
        cuda.free(pq);
        cuda.free(pL_key_cache);

        call(copyOfScore, q, l_key_cache, queryIndex, keyIndex, head_size);

        compareWithThreshold("AttentionScore.call score ",
                score, copyOfScore, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer score, Pointer q, Pointer l_key_cache, int queryIndex, int keyIndex,
                     int head_size) {
        CUstream stream = cuda.getCUKernelStream(kernelStreamId);
        if (head_size <= SMALL_KERNEL && head_size % 2 == 0) {
            int blockSizeX = findNextPowerOf2(head_size);
            int gridSizeX = 1;
            int sharedMemory = blockSizeX * Sizeof.FLOAT;

//            __global__ void attentionScore(float[] score, float[] q, float[] l_key_cache, int head_size) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(score),
                    Pointer.to(q).withByteOffset((long) queryIndex * Sizeof.FLOAT),
                    Pointer.to(l_key_cache).withByteOffset((long) keyIndex * Sizeof.FLOAT),
                    Pointer.to(new int[]{head_size})
            );

            cuLaunchKernel(smallKernel,
                    gridSizeX, 1, 1,          // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    sharedMemory, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        } else {
            throw new RuntimeException("AttentionScore.call invalid head size" + head_size);
        }
    }

    private CUfunction createSmall() {
        String code =
                """
                            extern "C"
                            __global__ void attentionScore(float* score, float* q, float* l_key_cache, int head_size) {
                                int tid = threadIdx.x;

                                extern __shared__ float sdata[];

                                if (tid < head_size) {
                                    // Store the value in shared memory for reduction
                                    printf(">>> 2 %d\\n", tid);
                                    sdata[threadIdx.x] = q[tid] * l_key_cache[tid];
                                } else {
                                    printf(">>> 3 %d\\n", tid);
                                    sdata[threadIdx.x] = 0.0f;
                                }
                                __syncthreads();  // Ensure all threads in block have stored their values

                                if (tid == 0) {
                                    printf(">>> 4");
                                }

                                // Block-wise reduction
                                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                                    if (threadIdx.x < stride) {
                                        sdata[threadIdx.x] += sdata[threadIdx.x + stride];
                                    }
                                    __syncthreads();  // Ensure all threads in block are in sync after each step
                                }

                                if (tid == 0) {
                                    printf(">>> 5");
                                    score[0] = sdata[0] / (float) sqrtf(head_size);
                                }
                                __syncthreads();

                                if (tid == 0) {
                                    printf(">>> 6");
                                }
                            }
                        """;
        return loadFromCode(code, "attentionScore");
    }
}