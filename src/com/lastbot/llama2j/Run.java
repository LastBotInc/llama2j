package com.lastbot.llama2j;

import com.lastbot.llama2j.kernel.*;
import jcuda.Pointer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.CountDownLatch;
import java.util.zip.CRC32;

import static java.lang.Math.abs;

/*
Inference for Llama-2 Transformer model in pure Java and with optional CUDA.

Objectives: reasonable (among the world's fastest) latency, with the absolutely leading
best throughput on systems with one or multiple NVIDIA gaming GPUs such as 4090.

Adapted from and inspired by: :https://github.com/karpathy/llama2.c

See file upstream.txt for details on the commit that this version is synchronized with.

*/
public class Run {
    public static final int QUANT_GROUP_SIZE = 32;
    public static final int QUANT_BITS = 8;

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

    private static void transformer(int token, int pos, Mode mode, Config p, RunState s,
                                    TransformerWeights w, Context context) {
        switch (mode) {
            case CPU -> transformerCPU(token, pos, p, s, w, context);
            case CUDA -> transformerCUDA(token, pos, p, s, w, context);
            case TEST -> transformerTest(token, pos, p, s, w, context);
        }
    }

    private static void transformerCPU(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freq_cis_imag_row = pos * head_size / 2;

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // attention rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);
//            MemZeroFloat.call(s.tmp1, 0, 1);
            SumOfSquares.call(s.tmp1, s.x, dim);

            WeightNormalizeAndScale.callI8(s.xb, s.x, w.l_rms_att_weight, l * dim, s.tmp1, dim);

            // qkv matmuls for this position
            MatMul.callI8(s.q, s.xb, w.l_wq, l * dim * dim, dim, dim);
            MatMul.callI8(s.k, s.xb, w.l_wk, l * dim * kv_dim, dim, kv_dim);
            MatMul.callI8(s.v, s.xb, w.l_wv, l * dim * kv_dim, dim, kv_dim);

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

            CountDownLatch latch = new CountDownLatch(p.n_heads);
            // multihead attention. iterate over all heads in parallel
            int h;
            for (h = 0; h < p.n_heads; h++) {
                int finalH = h;
                Thread.ofVirtual().start(() -> {
                    // get the query vector for this head
//                float*q = s -> q + h * head_size;
                    int queryIndex = finalH * head_size;
                    // attention scores for this head
                    int attentionIndex = finalH * p.seq_len;

                    int keyBase = loff + (finalH / kv_mul) * head_size;

                    AttentionLoop.call(s.q, s.l_key_cache, s.att, attentionIndex, keyBase,
                            kv_dim, queryIndex, pos, head_size);

////                float*att = s -> att + h * p.seq_len;
//                    // iterate over all timesteps, including the current one
//                    for (int t = 0; t <= pos; t++) {
//                        // get the key vector for this head and at this timestep
////                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
//                        int keyIndex = loff + t * kv_dim + (finalH / kv_mul) * head_size;
//
//                        MemZeroFloat.call(s.tmp2, 0, 1);
//                        Attention.call(s.tmp2, s.q, s.l_key_cache, queryIndex, keyIndex, head_size);
//                        // save the score to the attention buffer
//                        s.att[attentionIndex + t] = s.tmp2[0];
//                    }
                    latch.countDown();
                });
            }
            try {
                latch.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            // multihead attention. iterate over all heads
            for (h = 0; h < p.n_heads; h++) {
                // attention scores for this head
                int attentionIndex = h * p.seq_len;

                // softmax the scores to get attention weights, from 0..pos inclusively
//                softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                // redundant
//                MemZeroFloat.call(s.tmp1, 0, 1);
                FindMax.call(s.tmp1, s.att, attentionIndex, pos + 1);

                // exp and sum
                // redundant
//                MemZeroFloat.call(s.tmp2, 0, 1);
                ExpAndSum.call(s.tmp2, s.att, s.tmp1, attentionIndex, pos + 1);

                // normalize
                Normalize.call(s.att, s.tmp2, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                // todo zzz consider removing zeroing and integrating that into AccumWeightedValue
                // feasibility depends on if the performance will be worse if we switch the inner and outer loops
                int xbIndex = h * head_size;
                MemZeroFloat.call(s.xb, xbIndex, head_size);

                int valueBase = loff + (h / kv_mul) * head_size;
                AccumWeightedValue.call(s.xb, s.att, s.l_value_cache, pos, xbIndex,
                        valueBase, head_size, kv_dim, attentionIndex);
            }

            // final matmul to get the output of the attention
            MatMul.callI8(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            Accum.call(s.x, s.xb2, dim);

            // ffn rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

//            MemZeroFloat.call(s.tmp1, 0, 1);
            SumOfSquares.call(s.tmp1, s.x, dim);
            WeightNormalizeAndScale.callI8(s.xb, s.x, w.l_rms_ffn_weight, l * dim, s.tmp1, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            MatMul.callI8(s.hb, s.xb, w.l_w1, l * dim * hidden_dim, dim, hidden_dim);
            MatMul.callI8(s.hb2, s.xb, w.l_w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            // elementwise multiply with w3(x)
            Silu.call(s.hb, s.hb2, hidden_dim);

//            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
//            for (int i = 0; i < hidden_dim; i++) {
//                s.hb[i] = s.hb[i] * (float) (1.0f / (1.0f + Math.exp((-s.hb[i]))));
//            }
//
//            // elementwise multiply with w3(x)
//            for (int i = 0; i < hidden_dim; i++) {
//                s.hb[i] = s.hb[i] * s.hb2[i];
//            }

            // final matmul to get the output of the ffn
            MatMul.callI8(s.xb, s.hb, w.l_w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            Accum.call(s.x, s.xb, dim);
        } // layers

        // final rmsnorm
//        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
//        MemZeroFloat.call(s.tmp1, 0, 1);
        SumOfSquares.call(s.tmp1, s.x, dim);
        WeightNormalizeAndScale.callI8(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        MatMul.callFP32(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
    }

    private static void transformerTest(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        int dev = 0;
        ContextCUDA cuda = context.cudas[dev];
        cuda.setDevice();

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freq_cis_imag_row = pos * head_size / 2;

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // hand RunState over to the next device
            if (l > context.layerAllocation.lastLayer[dev]) {
                dev++;
                ContextCUDA newCuda = context.cudas[dev];
                cuda = newCuda;
                cuda.setDevice();
            }

            // attention rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);
            cuda.memZeroFloat.test(s.tmp1, 0, 1);
            cuda.sumOfSquares.test(s.tmp1, s.x, dim);

//            if (l == 0) {
//                summarize(pos, "s.x", s.x);
//            }

            cuda.weightNormalizeAndScale.testI8(s.xb, s.x, w.l_rms_att_weight, l * dim, s.tmp1, dim);

            // qkv matmuls for this position
            cuda.matMul.testI8(s.q, s.xb, w.l_wq, l * dim * dim, dim, dim);
            cuda.matMul.testI8(s.k, s.xb, w.l_wk, l * dim * kv_dim, dim, kv_dim);
            cuda.matMul.testI8(s.v, s.xb, w.l_wv, l * dim * kv_dim, dim, kv_dim);

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

                int keyBase = loff + (h / kv_mul) * head_size;

                cuda.attentionLoop.test(s.q, s.l_key_cache, s.att, attentionIndex, keyBase,
                        kv_dim, queryIndex, pos, head_size);

                // softmax the scores to get attention weights, from 0..pos inclusively
//                softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                // redundant
//                cuda.memZeroFloat.test(s.tmp1, 0, 1);
                cuda.findMax.test(s.tmp1, s.att, attentionIndex, pos + 1);

                // exp and sum
                // redundant
//                cuda.memZeroFloat.test(s.tmp2, 0, 1);
                cuda.expAndSum.test(s.tmp2, s.att, s.tmp1, attentionIndex, pos + 1);

                // normalize
                cuda.normalize.test(s.att, s.tmp2, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;
                cuda.memZeroFloat.test(s.xb, xbIndex, head_size);

                int valueBase = loff + (h / kv_mul) * head_size;
                cuda.accumWeightedValue.test(s.xb, s.att, s.l_value_cache, pos, xbIndex,
                        valueBase, head_size, kv_dim, attentionIndex);
            }

            // final matmul to get the output of the attention
            cuda.matMul.testI8(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            cuda.accum.test(s.x, s.xb2, dim);

            // ffn rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            cuda.memZeroFloat.test(s.tmp1, 0, 1);
            cuda.sumOfSquares.test(s.tmp1, s.x, dim);
            cuda.weightNormalizeAndScale.testI8(s.xb, s.x, w.l_rms_ffn_weight, l * dim, s.tmp1, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            cuda.matMul.testI8(s.hb, s.xb, w.l_w1, l * dim * hidden_dim, dim, hidden_dim);
            cuda.matMul.testI8(s.hb2, s.xb, w.l_w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            cuda.silu.test(s.hb, s.hb2, hidden_dim);

            // final matmul to get the output of the ffn
            cuda.matMul.testI8(s.xb, s.hb, w.l_w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            cuda.accum.test(s.x, s.xb, dim);

        } // layers

        // final rmsnorm
//        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        cuda.memZeroFloat.test(s.tmp1, 0, 1);
        cuda.sumOfSquares.test(s.tmp1, s.x, dim);
        cuda.weightNormalizeAndScale.testI8(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        cuda.matMul.testFP32(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
//        log(pos, "s.logits", s.logits);
    }

    private static void transformerCUDA(int token, int pos, Config p, RunState s, TransformerWeights w, Context context) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        int dev = 0;
        ContextCUDA cuda = context.cudas[dev];
        cuda.setDevice();

        // use first device state variables

        Pointer xCU = s.xCU[dev];
        Pointer xbCU = s.xbCU[dev];
        Pointer xb2CU = s.xb2CU[dev];
        Pointer hbCU = s.hbCU[dev];
        Pointer hb2CU = s.hb2CU[dev];
        Pointer qCU = s.qCU[dev];
        Pointer kCU = s.kCU[dev];
        Pointer vCU = s.vCU[dev];
        Pointer attCU = s.attCU[dev];
        Pointer logitsCU = s.logitsCU[dev];

        SlicePointer l_key_cacheCU = s.l_key_cacheCU[dev];
        SlicePointer l_value_cacheCU = s.l_value_cacheCU[dev];

        Pointer tmp1CU = s.tmp1CU[dev];
        Pointer tmp2CU = s.tmp2CU[dev];

        // use first device weight variables

        Pointer token_embedding_tableCU = w.token_embedding_tableCU[dev];
        QuantPointer l_rms_att_weightCU = w.l_rms_att_weightCU[dev];
        QuantPointer l_rms_ffn_weightCU = w.l_rms_ffn_weightCU[dev];
        QuantPointer l_wqCU = w.l_wqCU[dev];
        QuantPointer l_wkCU = w.l_wkCU[dev];
        QuantPointer l_wvCU = w.l_wvCU[dev];
        QuantPointer l_woCU = w.l_woCU[dev];
        QuantPointer l_w1CU = w.l_w1CU[dev];
        QuantPointer l_w2CU = w.l_w2CU[dev];
        QuantPointer l_w3CU = w.l_w3CU[dev];
        QuantPointer rms_final_weightCU = w.rms_final_weightCU[dev];
        Pointer freq_cis_realCU = w.freq_cis_realCU[dev];
        Pointer freq_cis_imagCU = w.freq_cis_imagCU[dev];
        Pointer wclsCU = w.wclsCU[dev];

        // copy the token embedding into x
        cuda.copyFloatsFromDeviceToDevice(0, token_embedding_tableCU, token * dim,
                xCU, 0, dim);

        // pluck out the "pos" row of freq_cis_real and freq_cis_imag
        int freq_cis_imag_row = pos * head_size / 2;

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // hand RunState over to the next device
            if (l > context.layerAllocation.lastLayer[dev]) {
                dev++;
                ContextCUDA newCuda = context.cudas[dev];

                // copy state from the current device to the new device
                // do not copy layer specific state that remains in the current device
                // both source and destination use 0 streamId

                // newCuda.synchronizeStream(0);

                cuda.copyFromDeviceToAnotherDevice(0, xCU, s.xCU[dev], newCuda, 0, dim, s.tmp1);

                // roll over to new device state variables

                xCU = s.xCU[dev];
                xbCU = s.xbCU[dev];
                xb2CU = s.xb2CU[dev];
                hbCU = s.hbCU[dev];
                hb2CU = s.hb2CU[dev];
                qCU = s.qCU[dev];
                kCU = s.kCU[dev];
                vCU = s.vCU[dev];
                attCU = s.attCU[dev];
                logitsCU = s.logitsCU[dev];

                l_key_cacheCU = s.l_key_cacheCU[dev];
                l_value_cacheCU = s.l_value_cacheCU[dev];

                tmp1CU = s.tmp1CU[dev];
                tmp2CU = s.tmp2CU[dev];

                // roll over to new device weight variables (no need to copy anything)

                // token_embedding_tableCU = w.token_embedding_tableCU[dev]; This only needed before the loop
                l_rms_att_weightCU = w.l_rms_att_weightCU[dev];
                l_rms_ffn_weightCU = w.l_rms_ffn_weightCU[dev];
                l_wqCU = w.l_wqCU[dev];
                l_wkCU = w.l_wkCU[dev];
                l_wvCU = w.l_wvCU[dev];
                l_woCU = w.l_woCU[dev];
                l_w1CU = w.l_w1CU[dev];
                l_w2CU = w.l_w2CU[dev];
                l_w3CU = w.l_w3CU[dev];
                rms_final_weightCU = w.rms_final_weightCU[dev];
                freq_cis_realCU = w.freq_cis_realCU[dev];
                freq_cis_imagCU = w.freq_cis_imagCU[dev];
                wclsCU = w.wclsCU[dev];

                newCuda.synchronizeStream(0);
                cuda = newCuda;
            }

            // attention rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);

            cuda.sumOfSquares.call(0, tmp1CU, xCU, dim);

//            log(pos, "tmp1CU", cuda, tmp1CU, 1);
            cuda.weightNormalizeAndScale.callI8(
                    0, xbCU, xCU, l_rms_att_weightCU, l * dim, tmp1CU, dim);

            cuda.synchronizeStream(0);

            // qkv matmuls for this position
            cuda.matMul.callI8(1, qCU, xbCU, l_wqCU, l * dim * dim, dim, dim);
            cuda.matMul.callI8(2, kCU, xbCU, l_wkCU, l * dim * kv_dim, dim, kv_dim);
            cuda.matMul.callI8(0, vCU, xbCU, l_wvCU, l * dim * kv_dim, dim, kv_dim);

            cuda.synchronizeStream(1);
            cuda.synchronizeStream(2);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            cuda.applyRope.call(0, qCU, kCU, freq_cis_realCU, freq_cis_imagCU,
                    dim, kv_dim, head_size, freq_cis_imag_row);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

//            System.arraycopy(s.k, 0, s.l_key_cache, loff + pos * kv_dim, kv_dim);
            cuda.copyFloatsFromDeviceToDevice(0, kCU, 0,
                    l_key_cacheCU.withIndex(loff + pos * kv_dim), 0, kv_dim);

//            System.arraycopy(s.v, 0, s.l_value_cache, loff + pos * kv_dim, kv_dim);
            cuda.copyFloatsFromDeviceToDevice(0, vCU, 0,
                    l_value_cacheCU.withIndex(loff + pos * kv_dim), 0, kv_dim);

            // multihead attention. iterate over all heads
            int maxStream = 0;
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
//                float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;

                int keyBase = loff + (h / kv_mul) * head_size -
                        (int) l_key_cacheCU.floatOffset();

                int streamId = h % ContextCUDA.STREAM_COUNT;
                if (streamId > maxStream) {
                    maxStream = streamId;
                }

                cuda.attentionLoop.call(streamId, qCU, l_key_cacheCU.pointer(), attCU,
                        attentionIndex, keyBase, kv_dim, queryIndex, pos, head_size);
            }
            for (int streamId = 1; streamId <= maxStream; streamId++) {
                cuda.synchronizeStream(streamId);
            }

            for (h = 0; h < p.n_heads; h++) {
                int attentionIndex = h * p.seq_len;
                // softmax the scores to get attention weights, from 0..pos inclusively
//                softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                // zzz redundant ?
//                cuda.memZeroFloat.call(0, tmp1CU, 0, 1);
                cuda.findMax.call(0, tmp1CU, attCU, attentionIndex, pos + 1);

                // exp and sum
                // zzz redundant ?
//                cuda.memZeroFloat.call(0, tmp2CU, 0, 1);
                cuda.expAndSum.call(0, tmp2CU, attCU, tmp1CU, attentionIndex, pos + 1);

                // normalize
                cuda.normalize.call(0, attCU, tmp2CU, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;
                cuda.memZeroFloat.call(0, xbCU, xbIndex, head_size);

                int valueBase = loff + (h / kv_mul) * head_size;
                cuda.accumWeightedValue.call(0, xbCU, attCU, l_value_cacheCU, pos, xbIndex,
                        valueBase, head_size, kv_dim, attentionIndex);
            }

            // final matmul to get the output of the attention
            cuda.matMul.callI8(0, xb2CU, xbCU, l_woCU, l * dim * dim, dim, dim);
            // residual connection back into x
            cuda.accum.call(0, xCU, xb2CU, dim);

            // ffn rmsnorm
//            rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            cuda.sumOfSquares.call(0, tmp1CU, xCU, dim);
            cuda.weightNormalizeAndScale.callI8(0, xbCU, xCU, l_rms_ffn_weightCU, l * dim, tmp1CU, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            cuda.matMul.callI8(0, hbCU, xbCU, l_w1CU, l * dim * hidden_dim, dim, hidden_dim);
            cuda.matMul.callI8(0, hb2CU, xbCU, l_w3CU, l * dim * hidden_dim, dim, hidden_dim);

            cuda.silu.call(0, hbCU, hb2CU, hidden_dim);

            // final matmul to get the output of the ffn
            cuda.matMul.callI8(0, xbCU, hbCU, l_w2CU, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            cuda.accum.call(0, xCU, xbCU, dim);

        } // layers

        // final rmsnorm
//        rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        cuda.sumOfSquares.call(0, tmp1CU, xCU, dim);
        cuda.weightNormalizeAndScale.callI8(0, xCU, xCU, rms_final_weightCU, 0, tmp1CU, dim);

        // classifier into logits, and send in OUTPUT_STREAM
        cuda.matMul.callFP32(0, logitsCU, xCU, wclsCU, p.dim, p.vocab_size);
    }

    private static void log(int pos, String name, ContextCUDA cuda, Pointer pointer, int size) {
        float[] hostArray = new float[size];
        cuda.synchronizeDevice();
        cuda.copyFromDeviceToHost(0, pointer, hostArray.length, hostArray);
        cuda.synchronizeDevice();
        log(pos, name, hostArray);
    }

    private static void log(int pos, String name, float[] hostArray) {
        System.out.println("START LOG pos " + pos + " " + name + "------------------------------------------------------------------");
        for (int i = 0; i < hostArray.length; i++) {
            System.out.print(String.format("%4d", i) + "\t" + String.format("%.3f", hostArray[i]) + "\t");
            if (i % 10 == 9) {
                System.out.println();
            }
        }
        if (hostArray.length % 10 != 9) {
            System.out.println();
        }
        System.out.println("END LOG pos " + pos + " " + name + "------------------------------------------------------------------");
    }

    private static void summarize(int pos, String name, ContextCUDA cuda, Pointer pointer, int size) {
        float[] hostArray = new float[size];
        cuda.synchronizeDevice();
        cuda.copyFromDeviceToHost(0, pointer, hostArray.length, hostArray);
        cuda.synchronizeDevice();
        summarize(pos, name, hostArray);
    }

    private static void summarize(int pos, String name, float[] a) {
        String crc32 = crc32(a);
        System.out.println("\nSUMMARY pos " + pos + " " + name + " " + crc32);
    }

// ----------------------------------------------------------------------------
// utilities

    private static long time() {
        return System.currentTimeMillis();
    }

    private static String crc32(float[] values) {
        CRC32 crc = new CRC32();

        // Use ByteBuffer to convert float to byte array
        ByteBuffer buffer = ByteBuffer.allocate(4 * values.length); // each float is 4 bytes
        for (float value : values) {
            buffer.putFloat(value);
        }

        byte[] byteRepresentation = buffer.array();
        crc.update(byteRepresentation);

        return Long.toHexString(crc.getValue());
    }

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

    public static void main(String[] args) {
        CommandLine commandLine = new CommandLine(args);

        Mode mode = commandLine.getMode();

        Long rngSeed = commandLine.getSeed();
        if (rngSeed == null) {
            rngSeed = time();
        }

        Config p = new Config();
        TransformerWeights w;

        Quant quant = new Quant(QUANT_GROUP_SIZE, QUANT_BITS);

        // read in the checkpoint file
        long startModelRead = time();
        LLogger.info("Start reading checkpoint " + commandLine.getCheckpoint());

        LayerAllocation layerAllocation;
        Context context;

        String binFile = MODELS_DIRECTORY + File.separator + commandLine.getCheckpoint();

        try (BinFileReader reader = new BinFileReader(binFile)) {
            // read in the config header
            p.dim = reader.nextInt(); // transformer dimension
            p.hidden_dim = reader.nextInt(); // for ffn layers
            p.n_layers = reader.nextInt(); // number of layers
            p.n_heads = reader.nextInt(); // number of query heads
            p.n_kv_heads = reader.nextInt(); // number of key/value heads (can be < query heads because of multiquery)
            p.vocab_size = reader.nextInt(); // vocabulary size, usually 256 (byte-level)
            p.seq_len = reader.nextInt(); // max sequence length

            // negative vocab size is hacky way of signaling unshared weights. bit yikes.
            boolean shared_weights = p.vocab_size > 0;
            p.vocab_size = abs(p.vocab_size);

            LLogger.info(p.toString());

            layerAllocation = new LayerAllocation(commandLine.getGpuMem(), p, mode, quant, shared_weights);

            context = new Context(layerAllocation);

            w = new TransformerWeights(context, reader, MODELS_DIRECTORY, p, quant, shared_weights);
        } catch (IOException e) {
            throw new RuntimeException("Initialization caused unexpected", e);
        }

        RunState s = new RunState(context, p);

        long endModelRead = time();

        LLogger.info("Read checkpoint in " + String.format("%.2f", (endModelRead - startModelRead) / 1000d) + " s");

        int steps = commandLine.getSteps();
        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > p.seq_len) {
            steps = p.seq_len;
        }

        Tokenizer tokenizer = new Tokenizer(
                MODELS_DIRECTORY + File.separator + commandLine.getTokenizer(), p.vocab_size);

        // process the prompt, if any
        int[] prompt_tokens = tokenizer.bpe_encode(commandLine.getPrompt());

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence
        float[] logits = mode != Mode.CUDA ? s.logits : new float[p.vocab_size];

        int lastDev = layerAllocation.deviceCount - 1;

        Sampler sampler = new Sampler(rngSeed, p.vocab_size);

        try {
            while (pos < steps) {
                // forward the transformer to get logits for the next token
                transformer(token, pos, mode, p, s, w, context);

                // if in cuda mode, copy logits from CUDA to CPU
                if (mode == Mode.CUDA) {
                    context.lastCuda().synchronizeStream(0);
                    context.lastCuda().copyFromDeviceToHost(0, s.logitsCU[lastDev], p.vocab_size, logits);
                    context.lastCuda().synchronizeStream(0);
                }

                // advance the state machine
                if (pos < prompt_tokens.length) {
                    // if we are still processing the input prompt, force the next prompt token
                    next = prompt_tokens[pos];
                } else {
                    // sample the next token (in this version, sampling is done on CPU)
                    if (commandLine.getTemperature() == 0.0f) {
                        // greedy argmax sampling: take the token with the highest probability
                        next = sampler.argmax(logits, p.vocab_size);
                    } else {
                        // apply the temperature to the logits
                        for (int q = 0; q < p.vocab_size; q++) {
                            logits[q] /= commandLine.getTemperature();
                        }
                        // apply softmax to the logits to get the probabilities for next token
//                    softmax(state.logits, 0, config.vocab_size);

                        // find max value (for numerical stability)
                        float[] max = {0f};
                        FindMax.call(max, logits, 0, p.vocab_size);

                        // exp and sum
                        float[] sum = {0f};
                        ExpAndSum.call(sum, logits, max, 0, p.vocab_size);

                        // normalize
                        Normalize.call(logits, sum, 0, p.vocab_size);

                        if (commandLine.getTopp() > 0.999) {
                            // we sample from this distribution to get the next token
                            next = sampler.sample(logits, p.vocab_size);
                        } else {
                            // top-p (nucleus) sampling, clamping the least likely tokens to zero
                            next = sampler.sample_topp(logits, p.vocab_size, commandLine.getTopp());
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
        } catch (Exception e) {
            System.out.println();
            LLogger.error("Unexpected", e);
        }

        Output.emit("\n");

        long end = time();

        // cleanup, free memory

        s.close();

        context.close();

        tokenizer.close();

        // report achieved tok/s
        if (pos > 1) {
            LLogger.debug("\nachieved tok/s: " + String.format("%.1f", (pos - 1) / (double) (end - start) * 1000));
        }
    }
}
