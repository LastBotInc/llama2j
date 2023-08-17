package com.lastbot.llama2j;

/*
Inference for Llama-2 Transformer model in pure Java and with optional CUDA.

Objectives: reasonable (among the world's fastest) latency, with the absolutely leading
best throughput on systems with one or multiple NVIDIA gaming GPUs such as 4090.

Adapted from and inspired by: :https://github.com/karpathy/llama2.c

See file upstream.txt for details on the commit that this version is synchronized with.

*/

import com.lastbot.llama2j.kernel.*;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;

import static java.lang.Math.abs;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

public class Run {
    private static final boolean USE_CPU = true;
    private static final boolean USE_CUDA = true;

    private static final String MODELS_DIRECTORY = "models";

// ----------------------------------------------------------------------------
// The below commented-out functions show how to implement rmsnorm() or softmax()
// in case the upstream code changes and requires to use them additionally.

// rmsnorm

//    private static void rmsnorm(float[] out, int outIndex, float[] x, float[] weight, int weightIndex, int size) {
//        float[] ss = {0f};
//        sumOfSquares(ss, x, size);
//        normalizeAndScale(out, x, weight, weightIndex, ss, size);
//    }

// softmax

//    private static void softmax(float[] x, int index, int size) {
//        // find max value (for numerical stability)
//        float[] max = {0f};
//        findMax(max, x, index, size);
//
//        // exp and sum
//        float[] sum = {0f};
//        expAndSum(sum, x, max, index, size);
//
//        // normalize
//        normalize(sum, x, index, size);
//    }

    private static final int THREAD_COUNT = 32;

    /**
     * This function is left here as a special case to make the testing with CPU only faster, as 99% of
     * CPU time is spent here.
     */
    private static void matmul(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        matmulParallel(xout, x, w, weightIndex, n, d);
    }

    private static void matmulParallel(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        int sizePerThread = d / THREAD_COUNT;
        CountDownLatch latch = new CountDownLatch(THREAD_COUNT);
        for (int threadId = 0; threadId < THREAD_COUNT; threadId++) {
            // W (d,n) @ x (n,) -> xout (d,)
            final int end = Math.min(d, (threadId + 1) * sizePerThread);
            int finalThreadId = threadId;
            Thread.ofVirtual().start(() -> {
                try {
                    float val;
                    int weightPos;
                    for (int i = finalThreadId * sizePerThread; i < end; i++) {
                        val = 0.0f;
                        weightPos = weightIndex + i * n;
                        for (int j = 0; j < n; j++) {
                            val += w[weightPos + j] * x[j];
                        }
                        xout[i] = val;
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            LLogger.error("fastMatmul was interrupted");
        }
    }

    private static void transformer(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        boolean cpu = context.cpu != null;
        boolean cuda = context.cudas != null && context.cudas.length > 0;
        if (cpu) {
            transformerCPU(token, pos, p, s, w, context);
//            transformerTest(token, pos, p, s, w, context);
        }
//        if (cuda) {
//            transformerCUDA(token, pos, p, s, w, context);
//        }
    }

    private static void transformerCPU(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        int cudaIndex = 0;
        ContextCUDA cuda = context.cudas[cudaIndex];

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freq_cis_imag_row = pos * head_size / 2;

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // hand over to the next device
            if (l > context.layerAllocation.lastLayer[cudaIndex]) {
                cudaIndex++;
                ContextCUDA newCuda = context.cudas[cudaIndex];
                cuda = newCuda;
            }

            // attention rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);
            MemSetFloat.call(s.tmp1, 0f, 1);
            SumOfSquares.call(s.tmp1, s.x, dim);
            WeightNormalizeAndScale.call(s.xb, s.x, w.l_rms_att_weight, l * dim, s.tmp1, dim);

            // qkv matmuls for this position
            matmul(s.q, s.xb, w.l_wq, l * dim * dim, dim, dim);
            matmul(s.k, s.xb, w.l_wk, l * dim * kv_dim, dim, kv_dim);
            matmul(s.v, s.xb, w.l_wv, l * dim * kv_dim, dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            ApplyRope.call(s.q, s.k, w.freq_cis_real, w.freq_cis_imag, dim, kv_dim, head_size, freq_cis_imag_row);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

//            float* key_cache_row = s->key_cache + loff + pos * kv_dim;
//            memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            System.arraycopy(s.k, 0, s.l_key_cache, loff + pos * kv_dim, kv_dim);

//            float* value_cache_row = s->value_cache + loff + pos * kv_dim;
//            memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
            System.arraycopy(s.v, 0, s.l_value_cache, loff + pos * kv_dim, kv_dim);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
//                float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;
//                float*att = s -> att + h * p.seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
//                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyIndex = loff + t * kv_dim + (h / kv_mul) * head_size;

                    MemSetFloat.call(s.tmp2, 0f, 1);
                    Attention.call(s.tmp2, s.q, s.l_key_cache, queryIndex, keyIndex, head_size);
                    // save the score to the attention buffer
                    s.att[attentionIndex + t] = s.tmp2[0];
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
//                softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                MemSetFloat.call(s.tmp1, 0f, 1);
                FindMax.call(s.tmp1, s.att, attentionIndex, pos + 1);

                // exp and sum
                MemSetFloat.call(s.tmp2, 0f, 1);
                ExpAndSum.call(s.tmp2, s.att, s.tmp1, attentionIndex, pos + 1);

                // normalize
                Normalize.call(s.att, s.tmp2, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;
                Arrays.fill(s.xb, xbIndex, xbIndex + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    int vIndex = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attentionIndex + t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbIndex + i] += a * s.l_value_cache[vIndex + i];
                    }
                }
            }

            // final matmul to get the output of the attention
            matmul(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            Accum.call(s.x, s.xb2, dim);

            // ffn rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            MemSetFloat.call(s.tmp1, 0f, 1);
            SumOfSquares.call(s.tmp1, s.x, dim);
            WeightNormalizeAndScale.call(s.xb, s.x, w.l_rms_ffn_weight, l * dim, s.tmp1, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            matmul(s.hb, s.xb, w.l_w1, l * dim * hidden_dim, dim, hidden_dim);
            matmul(s.hb2, s.xb, w.l_w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * (float) (1.0f / (1.0f + Math.exp((-s.hb[i]))));
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            matmul(s.xb, s.hb, w.l_w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            Accum.call(s.x, s.xb, dim);
        } // layers

        // final rmsnorm
//        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        MemSetFloat.call(s.tmp1, 0f, 1);
        SumOfSquares.call(s.tmp1, s.x, dim);
        WeightNormalizeAndScale.call(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        matmul(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
    }

    private static void transformerTest(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        int cudaIndex = 0;
        ContextCUDA cuda = context.cudas[cudaIndex];

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freq_cis_imag_row = pos * head_size / 2;

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // hand RunState over to the next device
            if (l > context.layerAllocation.lastLayer[cudaIndex]) {
                cudaIndex++;
                ContextCUDA newCuda = context.cudas[cudaIndex];
                cuda = newCuda;
            }

            // attention rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);
            cuda.memSetFloat.test(s.tmp1, 0f, 1);
            cuda.sumOfSquares.test(s.tmp1, s.x, dim);
            cuda.weightNormalizeAndScale.test(s.xb, s.x, w.l_rms_att_weight, l * dim, s.tmp1, dim);

            // qkv matmuls for this position
            cuda.matMul.test(s.q, s.xb, w.l_wq, l * dim * dim, dim, dim);
            cuda.matMul.test(s.k, s.xb, w.l_wk, l * dim * kv_dim, dim, kv_dim);
            cuda.matMul.test(s.v, s.xb, w.l_wv, l * dim * kv_dim, dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            cuda.applyRope.test(s.q, s.k, w.freq_cis_real, w.freq_cis_imag, dim, kv_dim, head_size, freq_cis_imag_row);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

//            float* key_cache_row = s->key_cache + loff + pos * kv_dim;
//            memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            System.arraycopy(s.k, 0, s.l_key_cache, loff + pos * kv_dim, kv_dim);

//            float* value_cache_row = s->value_cache + loff + pos * kv_dim;
//            memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
            System.arraycopy(s.v, 0, s.l_value_cache, loff + pos * kv_dim, kv_dim);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
//                float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;
//                float*att = s -> att + h * p.seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
//                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyIndex = loff + t * kv_dim + (h / kv_mul) * head_size;

                    cuda.memSetFloat.test(s.tmp2, 0f, 1);
                    cuda.attention.test(s.tmp2, s.q, s.l_key_cache, queryIndex, keyIndex, head_size);
                    // save the score to the attention buffer
                    s.att[attentionIndex + t] = s.tmp2[0];
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
//                softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                cuda.memSetFloat.test(s.tmp1, 0f, 1);
                cuda.findMax.test(s.tmp1, s.att, attentionIndex, pos + 1);

                // exp and sum
                cuda.memSetFloat.test(s.tmp2, 0f, 1);
                cuda.expAndSum.test(s.tmp2, s.att, s.tmp1, attentionIndex, pos + 1);

                // normalize
                cuda.normalize.test(s.att, s.tmp2, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;
                Arrays.fill(s.xb, xbIndex, xbIndex + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    int vIndex = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attentionIndex + t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbIndex + i] += a * s.l_value_cache[vIndex + i];
                    }
                }
            }

            // final matmul to get the output of the attention
            cuda.matMul.test(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            cuda.accum.test(s.x, s.xb2, dim);

            // ffn rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            cuda.memSetFloat.test(s.tmp1, 0f, 1);
            cuda.sumOfSquares.test(s.tmp1, s.x, dim);
            cuda.weightNormalizeAndScale.test(s.xb, s.x, w.l_rms_ffn_weight, l * dim, s.tmp1, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            cuda.matMul.test(s.hb, s.xb, w.l_w1, l * dim * hidden_dim, dim, hidden_dim);
            cuda.matMul.test(s.hb2, s.xb, w.l_w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * (float) (1.0f / (1.0f + Math.exp((-s.hb[i]))));
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            cuda.matMul.test(s.xb, s.hb, w.l_w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            cuda.accum.test(s.x, s.xb, dim);
        } // layers

        // final rmsnorm
//        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        cuda.memSetFloat.test(s.tmp1, 0f, 1);
        cuda.sumOfSquares.test(s.tmp1, s.x, dim);
        cuda.weightNormalizeAndScale.test(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        cuda.matMul.test(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
    }

    private static void transformerCUDA(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        int cudaIndex = 0;
        ContextCUDA cuda = context.cudas[cudaIndex];
        cuda.setDevice();

        Pointer xCU = s.xCU[cudaIndex];
        Pointer xbCU = s.xbCU[cudaIndex];
        Pointer xb2CU = s.xb2CU[cudaIndex];
        Pointer hbCU = s.hbCU[cudaIndex];
        Pointer hb2CU = s.hb2CU[cudaIndex];
        Pointer qCU = s.qCU[cudaIndex];
        Pointer kCU = s.kCU[cudaIndex];
        Pointer vCU = s.vCU[cudaIndex];
        Pointer attCU = s.attCU[cudaIndex];
        Pointer logitsCU = s.logitsCU[cudaIndex];

        Pointer l_key_cacheCU = s.l_key_cacheCU[cudaIndex];
        Pointer l_value_cacheCU = s.l_value_cacheCU[cudaIndex];

        Pointer tmp1CU = s.tmp1CU[cudaIndex];
        Pointer tmp2CU = s.tmp2CU[cudaIndex];
        Pointer tmp3CU = s.tmp3CU[cudaIndex];

        // copy the token embedding into x
        cudaMemcpy(xCU, w.token_embedding_tableCU.withByteOffset((long) token * dim * Sizeof.FLOAT),
                dim, cudaMemcpyDeviceToDevice);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freq_cis_imag_row = pos * head_size / 2;

        // forward all layers
        for (int l = 0; l < p.n_layers; l++) {
            // hand RunState over to the next device
            // todo zzz check which arrays are actually needed, this version copies all except layer dependent
            if (l > context.layerAllocation.lastLayer[cudaIndex]) {
                cudaIndex++;
                ContextCUDA newCuda = context.cudas[cudaIndex];

                cuda.copyFromDeviceToAnotherDevice(xCU, s.xCU[cudaIndex], newCuda, s.x);
                cuda.copyFromDeviceToAnotherDevice(xbCU, s.xbCU[cudaIndex], newCuda, s.xb);
                cuda.copyFromDeviceToAnotherDevice(xb2CU, s.xb2CU[cudaIndex], newCuda, s.xb2);
                cuda.copyFromDeviceToAnotherDevice(hbCU, s.hbCU[cudaIndex], newCuda, s.hb);
                cuda.copyFromDeviceToAnotherDevice(hb2CU, s.hb2CU[cudaIndex], newCuda, s.hb2);
                cuda.copyFromDeviceToAnotherDevice(qCU, s.qCU[cudaIndex], newCuda, s.q);
                cuda.copyFromDeviceToAnotherDevice(kCU, s.kCU[cudaIndex], newCuda, s.k);
                cuda.copyFromDeviceToAnotherDevice(vCU, s.vCU[cudaIndex], newCuda, s.v);
                cuda.copyFromDeviceToAnotherDevice(attCU, s.attCU[cudaIndex], newCuda, s.att);
                cuda.copyFromDeviceToAnotherDevice(logitsCU, s.logitsCU[cudaIndex], newCuda, s.logits);

                xCU = s.xCU[cudaIndex];
                xbCU = s.xbCU[cudaIndex];
                xb2CU = s.xb2CU[cudaIndex];
                hbCU = s.hbCU[cudaIndex];
                hb2CU = s.hb2CU[cudaIndex];
                qCU = s.qCU[cudaIndex];
                kCU = s.kCU[cudaIndex];
                vCU = s.vCU[cudaIndex];
                attCU = s.attCU[cudaIndex];
                logitsCU = s.logitsCU[cudaIndex];

                l_key_cacheCU = s.l_key_cacheCU[cudaIndex];
                l_value_cacheCU = s.l_value_cacheCU[cudaIndex];

                tmp1CU = s.tmp1CU[cudaIndex];
                tmp2CU = s.tmp2CU[cudaIndex];
                tmp3CU = s.tmp3CU[cudaIndex];

                newCuda.synchronizeTransfer();
                cuda = newCuda;
            }

            // attention rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);

            cuda.memSetFloat.call(0, tmp1CU, 0f, 1);
            cuda.sumOfSquares.call(0, tmp1CU, xCU, dim);
            cuda.weightNormalizeAndScale.call(
                    0, xbCU, xCU, w.l_rms_att_weightCU, l * dim, tmp1CU, dim);

            // qkv matmuls for this position
            cuda.matMul.call(0, qCU, xbCU, w.l_wqCU, l * dim * dim, dim, dim);
            cuda.matMul.call(0, kCU, xbCU, w.l_wkCU, l * dim * kv_dim, dim, kv_dim);
            cuda.matMul.call(0, vCU, xbCU, w.l_wvCU, l * dim * kv_dim, dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            cuda.applyRope.call(0, qCU, kCU, w.freq_cis_realCU, w.freq_cis_imagCU,
                    dim, kv_dim, head_size, freq_cis_imag_row);

            cuda.synchronizeKernel(0);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

            int byteOffset = (loff + pos * kv_dim) * Sizeof.FLOAT;

//            System.arraycopy(s.k, 0, s.l_key_cache, loff + pos * kv_dim, kv_dim);
            cuda.copyFromDeviceToDevice(kCU, l_key_cacheCU.withByteOffset(byteOffset), kv_dim);

//            System.arraycopy(s.v, 0, s.l_value_cache, loff + pos * kv_dim, kv_dim);
            cuda.copyFromDeviceToDevice(vCU, l_value_cacheCU.withByteOffset(byteOffset), kv_dim);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
//                float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;
//                float*att = s -> att + h * p.seq_len;
                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++) {
                    // get the key vector for this head and at this timestep
//                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    int keyIndex = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[queryIndex + i] * s.l_key_cache[keyIndex + i];
                    }
                    score /= (float) Math.sqrt(head_size);
                    // save the score to the attention buffer
                    s.att[attentionIndex + t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
//                softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                float[] max = {0f};
                cuda.findMax.test(max, s.att, attentionIndex, pos + 1);

                // exp and sum
                float[] sum = {0f};
                cuda.expAndSum.test(sum, s.att, max, attentionIndex, pos + 1);

                // normalize
                cuda.normalize.test(s.att, sum, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;
                Arrays.fill(s.xb, xbIndex, xbIndex + head_size, 0f);

                for (int t = 0; t <= pos; t++) {
                    // get the value vector for this head and at this timestep
                    int vIndex = loff + t * kv_dim + (h / kv_mul) * head_size;
                    // get the attention weight for this timestep
                    float a = s.att[attentionIndex + t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[xbIndex + i] += a * s.l_value_cache[vIndex + i];
                    }
                }
            }

            // final matmul to get the output of the attention
            cuda.matMul.test(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            cuda.accum.test(s.x, s.xb2, dim);

            // ffn rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            cuda.sumOfSquares.test(s.tmp1, s.x, dim);
            cuda.weightNormalizeAndScale.test(s.xb, s.x, w.l_rms_ffn_weight, l * dim, s.tmp1, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            cuda.matMul.test(s.hb, s.xb, w.l_w1, l * dim * hidden_dim, dim, hidden_dim);
            cuda.matMul.test(s.hb2, s.xb, w.l_w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * (float) (1.0f / (1.0f + Math.exp((-s.hb[i]))));
            }

            // elementwise multiply with w3(x)
            for (int i = 0; i < hidden_dim; i++) {
                s.hb[i] = s.hb[i] * s.hb2[i];
            }

            // final matmul to get the output of the ffn
            cuda.matMul.test(s.xb, s.hb, w.l_w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            cuda.accum.test(s.x, s.xb, dim);
        } // layers

        // final rmsnorm
//        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        cuda.sumOfSquares.test(s.tmp1, s.x, dim);
        cuda.weightNormalizeAndScale.test(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        cuda.matMul.test(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
    }

// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// utilities

    private static long time() {
        return System.currentTimeMillis();
    }

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

    public static void main(String[] args) {
        CommandLine commandLine = new CommandLine(args);

        Long rngSeed = commandLine.getSeed();
        if (rngSeed == null) {
            rngSeed = time();
        }

        Target target = new Target(USE_CPU, USE_CUDA);

        Config config = new Config();
        TransformerWeights weights = null;

        Sampler sampler = new Sampler(rngSeed);

        // read in the checkpoint file
        long startModelRead = time();
        LLogger.info("Start reading checkpoint " + commandLine.getCheckpoint());

        LayerAllocation layerAllocation;
        Context context = null;

        try (BinFileReader reader =
                     new BinFileReader(MODELS_DIRECTORY + File.separator + commandLine.getCheckpoint())) {
            // read in the config header
            config.dim = reader.nextInt(); // transformer dimension
            config.hidden_dim = reader.nextInt(); // for ffn layers
            config.n_layers = reader.nextInt(); // number of layers
            config.n_heads = reader.nextInt(); // number of query heads
            config.n_kv_heads = reader.nextInt(); // number of key/value heads (can be < query heads because of multiquery)
            config.vocab_size = reader.nextInt(); // vocabulary size, usually 256 (byte-level)
            config.seq_len = reader.nextInt(); // max sequence length

            // negative vocab size is hacky way of signaling unshared weights. bit yikes.
            boolean shared_weights = config.vocab_size > 0;
            config.vocab_size = abs(config.vocab_size);

            LLogger.info(config.toString());

            layerAllocation = new LayerAllocation(commandLine.getGpuMem(), config, target, shared_weights);

            context = new Context(layerAllocation);

            weights = new TransformerWeights(context, reader, config, shared_weights);
        } catch (IOException e) {
            System.exit(1);
        }

        RunState state = new RunState(context, config);

        long endModelRead = time();

        LLogger.info("Read checkpoint in " + String.format("%.2f", (endModelRead - startModelRead) / 1000d) + " s");

        int steps = commandLine.getSteps();
        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > config.seq_len) {
            steps = config.seq_len;
        }

        Tokenizer tokenizer = new Tokenizer(
                MODELS_DIRECTORY + File.separator + commandLine.getTokenizer(), config.vocab_size);

        // process the prompt, if any
        int[] prompt_tokens = tokenizer.bpe_encode(commandLine.getPrompt());

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence

        while (pos < steps) {
            // forward the transformer to get logits for the next token
            transformer(token, pos, config, state, weights, context);

            // advance the state machine
            if (pos < prompt_tokens.length) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos];
            } else {
                // sample the next token
                if (commandLine.getTemperature() == 0.0f) {
                    // greedy argmax sampling: take the token with the highest probability
                    next = sampler.argmax(state.logits, config.vocab_size);
                } else {
                    // apply the temperature to the logits
                    for (int q = 0; q < config.vocab_size; q++) {
                        state.logits[q] /= commandLine.getTemperature();
                    }
                    // apply softmax to the logits to get the probabilities for next token
//                    softmax(state.logits, 0, config.vocab_size);

                    // find max value (for numerical stability)
                    float[] max = {0f};
                    context.lastCuda().findMax.test(max, state.logits, 0, config.vocab_size);

                    // exp and sum
                    float[] sum = {0f};
                    context.lastCuda().expAndSum.test(sum, state.logits, max, 0, config.vocab_size);

                    // normalize
                    context.lastCuda().normalize.test(state.logits, sum, 0, config.vocab_size);

                    if (commandLine.getTopp() == null) {
                        // we sample from this distribution to get the next token
                        next = sampler.sample(state, config.vocab_size);
                    } else {
                        // top-p (nucleus) sampling, clamping the least likely tokens to zero
                        next = sampler.sample_topp(state, config.vocab_size, commandLine.getTopp());
                    }
                }
            }
            pos++;

            // data-dependent terminating condition: the BOS (1) token delimits sequences
            if (next == 1) {
                break;
            }

            String token_str = tokenizer.bpe_decode(token, next);
            if (token_str != null) {
                Output.emit(token_str);
            }

            token = next;

            // init our timer here
            if (start == 0) {
                start = time();
            }
        }
        Output.emit("\n");

        long end = time();

        // cleanup, free memory

        state.close();

        context.close();

        tokenizer.close();

        // report achieved tok/s
        if (pos > 1) {
            LLogger.debug("\nachieved tok/s: " + String.format("%.1f", (pos - 1) / (double) (end - start) * 1000));
        }
    }
}
