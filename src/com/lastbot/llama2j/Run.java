package com.lastbot.llama2j;

import com.lastbot.llama2j.kernel.*;
import jcuda.Pointer;

import java.io.File;
import java.io.IOException;
import java.util.stream.IntStream;

import static java.lang.Math.abs;

/*
Inference for Llama-2 Transformer model in pure Java and with optional CUDA on multiple GPU.

Adapted from and inspired by: :https://github.com/karpathy/llama2.c

See file upstream.txt for details on the commit that this version is consistent with.

Comments containing c-code are left to help to refer to the llama2.c implementation.

*/
public class Run {
    /**
     * Group size for quantization, set freely to any value, but
     * consider only 32, 64, 128, 256, 515 for the best performance.
     */
    public static final int QUANT_GROUP_SIZE = 256;
    public static final int QUANT_BITS = 8;

    /**
     * Directory relative to the current to load model snapshot from and to write
     * quant cache files to.
     */
    private static final String MODELS_DIRECTORY = "models";


    /**
     * Executes transformer in the desired mode and context
     * Note: in this version migration of transformer execution and run state propagation from CUDA to CPU is
     * not implemented. When desired, we need to change this function to call transformerCPU after transformerCUDA()
     * in case the run state has not reached final layer.
     */
    private static void transformer(int token, int pos, Mode mode, Config p, RunState s,
                                    TransformerWeights w, Context context) {
        switch (mode) {
            case CPU -> transformerCPU(token, pos, p, s, w);
            case CUDA -> transformerCUDA(token, pos, p, s, w, context);
            case TEST -> transformerTest(token, pos, p, s, w, context);
        }
    }

    /**
     * Implements CPU only transformer.
     *
     * @param token current token
     * @param pos   position in the sequence
     * @param p     config
     * @param s     run state
     * @param w     weights
     */
    private static void transformerCPU(int token, int pos, Config p, RunState s, TransformerWeights w) {
        // a few convenience variables
        final int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery

        final int hidden_dim = p.hidden_dim;
        final int head_size = dim / p.n_heads;

        // copy the token embedding into x
        System.arraycopy(w.token_embedding_table, token * dim, s.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // attention rmsnorm
            // rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);
            RootMeanSquare.call(s.tmp1, s.x, dim);

            WeightNormalizeAndScale.callI8(s.xb, s.x, w.l_rms_att_weight, l * dim, s.tmp1, dim);

            // qkv matmuls for this position
            MatMul.callI8(s.q, s.xb, w.l_wq, l * dim * dim, dim, dim);
            MatMul.callI8(s.k, s.xb, w.l_wk, l * dim * kv_dim, dim, kv_dim);
            MatMul.callI8(s.v, s.xb, w.l_wv, l * dim * kv_dim, dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            ApplyRope.call(s.q, s.k, pos, dim, kv_dim);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

            // float* key_cache_row = s->key_cache + loff + pos * kv_dim;
            // memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            System.arraycopy(s.k, 0, s.l_key_cache, loff + pos * kv_dim, kv_dim);

            // float* value_cache_row = s->value_cache + loff + pos * kv_dim;
            // memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
            System.arraycopy(s.v, 0, s.l_value_cache, loff + pos * kv_dim, kv_dim);

            // multihead attention. iterate over all heads in parallel
            IntStream.range(0, p.n_heads).parallel().forEach(h -> {
                // get the query vector for this head
                // float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;

                int keyBase = loff + (h / kv_mul) * head_size;

                int maxIndex = h;

                AttentionLoop.call(s.q, s.l_key_cache, s.att, s.tmp1, maxIndex, attentionIndex, keyBase,
                        kv_dim, queryIndex, pos, head_size);

                // softmax the scores to get attention weights, from 0..pos inclusively
                // softmax(s.att, attentionIndex, pos + 1);

                // find max value (for numerical stability)
                // FindMax.call(s.tmp1, maxIndex, s.att, attentionIndex, pos + 1);

                // exp and sum
                ExpSumNormalize.call(s.att, s.tmp1, maxIndex, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;

                int valueBase = loff + (h / kv_mul) * head_size;
                AccumWeightedValue.call(s.xb, s.att, s.l_value_cache, pos, xbIndex,
                        valueBase, head_size, kv_dim, attentionIndex);
            });

            // final matmul to get the output of the attention
            MatMul.callI8(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            Accum.call(s.x, s.xb2, dim);

            // ffn rmsnorm
            // rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            // MemZeroFloat.call(s.tmp1, 0, 1);
            RootMeanSquare.call(s.tmp1, s.x, dim);
            WeightNormalizeAndScale.callI8(s.xb, s.x, w.l_rms_ffn_weight, l * dim, s.tmp1, dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            MatMul.callI8(s.hb, s.xb, w.l_w1, l * dim * hidden_dim, dim, hidden_dim);
            MatMul.callI8(s.hb2, s.xb, w.l_w3, l * dim * hidden_dim, dim, hidden_dim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            // elementwise multiply with w3(x)
            Silu.call(s.hb, s.hb2, hidden_dim);

            // final matmul to get the output of the ffn
            MatMul.callI8(s.xb, s.hb, w.l_w2, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            Accum.call(s.x, s.xb, dim);
        } // layers

        // final rmsnorm
        // rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        // MemZeroFloat.call(s.tmp1, 0, 1);
        RootMeanSquare.call(s.tmp1, s.x, dim);
        WeightNormalizeAndScale.callI8(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        MatMul.callFP32(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
    }

    /**
     * Implements CPU and CUDA transformer. Logic runs on CPU only, but each kernel is run both on
     * CPU and CUDA and the results are compared and an exception is thrown if the results deviate
     * more than a kernel specific threshold.
     * <p>
     * This method is 100x slower, due to excessive copying between CPU and GPU devices. It is
     * intended only for validation of equivalent results between CPU and CUDA implementations.
     *
     * @param token   current token
     * @param pos     position in the sequence
     * @param p       config
     * @param s       run state
     * @param w       weights
     * @param context execution context
     */
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
            // rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);
            cuda.rootMeanSquare.test(s.tmp1, s.x, dim);

            cuda.weightNormalizeAndScale.testI8(s.xb, s.x, w.l_rms_att_weight, l * dim, s.tmp1, dim);

            // qkv matmuls for this position
            cuda.matMul.testI8(s.q, s.xb, w.l_wq, l * dim * dim, dim, dim);
            cuda.matMul.testI8(s.k, s.xb, w.l_wk, l * dim * kv_dim, dim, kv_dim);
            cuda.matMul.testI8(s.v, s.xb, w.l_wv, l * dim * kv_dim, dim, kv_dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            cuda.applyRope.test(s.q, s.k, pos, dim, kv_dim, head_size);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

            // float* key_cache_row = s->key_cache + loff + pos * kv_dim;
            // memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
            System.arraycopy(s.k, 0, s.l_key_cache, loff + pos * kv_dim, kv_dim);

            // float* value_cache_row = s->value_cache + loff + pos * kv_dim;
            // memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
            System.arraycopy(s.v, 0, s.l_value_cache, loff + pos * kv_dim, kv_dim);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
                // float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;

                int keyBase = loff + (h / kv_mul) * head_size;

                int maxIndex = h;

                cuda.attentionLoop.test(s.q, s.l_key_cache, s.att, s.tmp1, maxIndex, attentionIndex, keyBase,
                        kv_dim, queryIndex, pos, head_size);

                // softmax the scores to get attention weights, from 0..pos inclusively
                // softmax(s.att, attentionIndex, pos + 1);

                // exp and sum
                // redundant
                // cuda.memZeroFloat.test(s.tmp2, 0, 1);
                cuda.expSumNormalize.test(s.att, s.tmp1, maxIndex, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;

                int valueBase = loff + (h / kv_mul) * head_size;
                cuda.accumWeightedValue.test(s.xb, s.att, s.l_value_cache, pos, xbIndex,
                        valueBase, head_size, kv_dim, attentionIndex);
            }

            // final matmul to get the output of the attention
            cuda.matMul.testI8(s.xb2, s.xb, w.l_wo, l * dim * dim, dim, dim);
            // residual connection back into x
            cuda.accum.test(s.x, s.xb2, dim);

            // ffn rmsnorm
            // rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            cuda.rootMeanSquare.test(s.tmp1, s.x, dim);
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
        // rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        cuda.rootMeanSquare.test(s.tmp1, s.x, dim);
        cuda.weightNormalizeAndScale.testI8(s.x, s.x, w.rms_final_weight, 0, s.tmp1, dim);

        // classifier into logits
        cuda.matMul.testFP32(s.logits, s.x, w.wcls, 0, p.dim, p.vocab_size);
    }

    /**
     * Implements CUDA only transformer for one or multiple GPU devices. Layers are allocated across GPUs
     * and run state is propagated to the next GPU. This allows for models to use total of available VRAM.
     *
     * @param token current token
     * @param pos   position in the sequence
     * @param p     config
     * @param s     run state
     * @param w     weights
     */
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
        Pointer wclsCU = w.wclsCU[dev];

        // copy the token embedding into x
        cuda.copyFloatsFromDeviceToDevice(0, token_embedding_tableCU, (long) token * dim,
                xCU, 0, dim);

        // forward all the layers
        for (int l = 0; l < p.n_layers; l++) {
            // hand RunState over to the next device
            if (l > context.layerAllocation.lastLayer[dev]) {
                dev++;
                ContextCUDA newCuda = context.cudas[dev];

                // copy state from the current device to the new device
                // do not copy layer specific state that remains in the current device
                // both source and destination use 0 streamId

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
                wclsCU = w.wclsCU[dev];

                newCuda.synchronizeStream(0);
                cuda = newCuda;
            }

            // attention rmsnorm
            // rmsnorm(s.xb, s.x, w.l_rms_att_weight, layer * dim, dim);

            cuda.rootMeanSquare.call(0, tmp1CU, xCU, dim);

            cuda.weightNormalizeAndScale.callI8(
                    0, xbCU, xCU, l_rms_att_weightCU, l * dim, tmp1CU, dim);

            cuda.synchronizeStream(0);

            // qkv matmuls for this position
            cuda.matMul.callI8(0, qCU, xbCU, l_wqCU, l * dim * dim, dim, dim);
            cuda.matMul.callI8(1, kCU, xbCU, l_wkCU, l * dim * kv_dim, dim, kv_dim);
            cuda.matMul.callI8(2, vCU, xbCU, l_wvCU, l * dim * kv_dim, dim, kv_dim);

            cuda.synchronizeStream(1);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            cuda.applyRope.call(0, qCU, kCU, pos, dim, kv_dim, head_size);

            cuda.synchronizeStream(2);

            // save key,value at this time step (pos) to our kv cache
            int loff = l * p.seq_len * kv_dim; // kv cache layer offset for convenience

            cuda.copyFloatsFromDeviceToDevice(1, kCU, 0,
                    l_key_cacheCU.withIndex(loff + pos * kv_dim), 0, kv_dim);

            cuda.copyFloatsFromDeviceToDevice(0, vCU, 0,
                    l_value_cacheCU.withIndex(loff + pos * kv_dim), 0, kv_dim);

            cuda.synchronizeStream(1);

            // multihead attention. iterate over all heads
            int h;
            for (h = 0; h < p.n_heads; h++) {
                // get the query vector for this head
                // float*q = s -> q + h * head_size;
                int queryIndex = h * head_size;
                // attention scores for this head
                int attentionIndex = h * p.seq_len;

                int keyBase = loff + (h / kv_mul) * head_size - Math.toIntExact(l_key_cacheCU.floatOffset());

                int streamId = h % ContextCUDA.STREAM_COUNT;
                int maxIndex = h;

                // tmp1CU collects max values per head, and is used in ExpSumNormalize below
                cuda.attentionLoop.call(streamId, qCU, l_key_cacheCU.pointer(), attCU, tmp1CU, maxIndex,
                        attentionIndex, keyBase, kv_dim, queryIndex, pos, head_size);

                // softmax the scores to get attention weights, from 0..pos inclusively
                // softmax(s.att, attentionIndex, pos + 1);

                // exp and sum
                cuda.expSumNormalize.call(streamId, attCU, tmp1CU, maxIndex, attentionIndex, pos + 1);

                // weighted sum of the values, store back into xb
                int xbIndex = h * head_size;

                int valueBase = loff + (h / kv_mul) * head_size;
                cuda.accumWeightedValue.call(streamId, xbCU, attCU, l_value_cacheCU, pos, xbIndex,
                        valueBase, head_size, kv_dim, attentionIndex);
            }

            cuda.synchronizeDevice();

            // final matmul to get the output of the attention
            cuda.matMul.callI8(0, xb2CU, xbCU, l_woCU, l * dim * dim, dim, dim);
            // residual connection back into x
            cuda.accum.call(0, xCU, xb2CU, dim);

            // ffn rmsnorm
            // rmsnorm(s.xb, s.x, w.l_rms_ffn_weight, layer * dim, dim);

            cuda.rootMeanSquare.call(0, tmp1CU, xCU, dim);
            cuda.weightNormalizeAndScale.callI8(0, xbCU, xCU, l_rms_ffn_weightCU, l * dim, tmp1CU, dim);

            cuda.synchronizeStream(0);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            cuda.matMul.callI8(1, hbCU, xbCU, l_w1CU, l * dim * hidden_dim, dim, hidden_dim);
            cuda.matMul.callI8(0, hb2CU, xbCU, l_w3CU, l * dim * hidden_dim, dim, hidden_dim);

            cuda.synchronizeStream(1);

            cuda.silu.call(0, hbCU, hb2CU, hidden_dim);

            // final matmul to get the output of the ffn
            cuda.matMul.callI8(0, xbCU, hbCU, l_w2CU, l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            cuda.accum.call(0, xCU, xbCU, dim);

        } // layers

        // final rmsnorm
        // rmsnorm(s.x, s.x, w.rms_final_weight, 0, dim);
        cuda.rootMeanSquare.call(0, tmp1CU, xCU, dim);
        cuda.weightNormalizeAndScale.callI8(0, xCU, xCU, rms_final_weightCU, 0, tmp1CU, dim);

        // classifier into logits, and send in OUTPUT_STREAM
        cuda.matMul.callFP32(0, logitsCU, xCU, wclsCU, p.dim, p.vocab_size);
    }

// ----------------------------------------------------------------------------
// utilities

    private static long time() {
        return System.currentTimeMillis();
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

            Silu.init();

            final int dim = p.dim;
            final int head_size = dim / p.n_heads;
            ApplyRope.init(p.dim, head_size, p.seq_len);

            LLogger.info(p.toString());

            layerAllocation = new LayerAllocation(commandLine.getGpuMem(), p, mode, quant, shared_weights);

            context = new Context(layerAllocation);

            w = new TransformerWeights(context, reader, p, quant, shared_weights);
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
                        // softmax(state.logits, 0, config.vocab_size);

                        // find max value (for numerical stability)
                        float[] max = {0f};
                        FindMax.call(max, 0, logits, 0, p.vocab_size);

                        // exp and sum
                        ExpSumNormalize.call(logits, max, 0, 0, p.vocab_size);

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
            LLogger.debug("\nachieved tok/s: " + String.format("%.2f", (pos - 1) / (double) (end - start) * 1000));
        }
    }
}
