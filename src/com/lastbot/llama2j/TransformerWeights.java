package com.lastbot.llama2j;

import jcuda.Pointer;

public class TransformerWeights {
    private final Context context;
    private final Config config;  // token embedding table

    float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float[] rms_att_weight; // (layer, dim) rmsnorm weights
    float[] rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float[] wq; // (layer, dim, dim)
    float[] wk; // (layer, dim, dim)
    float[] wv; // (layer, dim, dim)
    float[] wo; // (layer, dim, dim)
    // weights for ffn
    float[] w1; // (layer, hidden_dim, dim)
    float[] w2; // (layer, dim, hidden_dim)
    float[] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float[] rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float[] freq_cis_real; // (seq_len, dim/2)
    float[] freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float[] wcls;

    public long bytesStatic(boolean sharedWeights) {
        int head_size = config.dim / config.n_heads;
        return (long) Float.BYTES * (
                ((long) config.vocab_size * config.dim) +
                        ((long) config.dim) +
                        ((long) config.seq_len * head_size / 2) +
                        ((long) config.seq_len * head_size / 2) +
                        (sharedWeights ? 0L : (long) config.vocab_size * config.dim)
        );
    }

    public long bytesPerLayer() {
        return (long) Float.BYTES * (
                ((long) config.dim) +
                        ((long) config.dim * config.dim) +
                        ((long) config.dim * config.dim) +
                        ((long) config.dim * config.dim) +
                        ((long) config.dim * config.dim) +
                        ((long) config.dim) +
                        ((long) config.dim * config.hidden_dim) +
                        ((long) config.hidden_dim * config.dim) +
                        ((long) config.dim * config.hidden_dim)
        );
    }

    Pointer token_embedding_tableCU;    // (vocab_size, dim)
    Pointer rms_att_weightCU; // (layer, dim) rmsnorm weights
    Pointer rms_ffn_weightCU; // (layer, dim)
    Pointer wqCU; // (layer, dim, dim)
    Pointer wkCU; // (layer, dim, dim)
    Pointer wvCU; // (layer, dim, dim)
    Pointer woCU; // (layer, dim, dim)
    Pointer w1CU; // (layer, hidden_dim, dim)
    Pointer w2CU; // (layer, dim, hidden_dim)
    Pointer w3CU; // (layer, hidden_dim, dim)
    Pointer rms_final_weightCU; // (dim,)
    Pointer freq_cis_realCU; // (seq_len, dim/2)
    Pointer freq_cis_imagCU; // (seq_len, dim/2)
    Pointer wclsCU;

    public TransformerWeights(Context context, BinFileReader reader, Config config, boolean sharedWeights) {
        this.context = context;
        this.config = config;

        LLogger.info("Static bytes " + String.format("%,d", bytesStatic(sharedWeights)));
        LLogger.info("Per layer bytes " + String.format("%,d", bytesPerLayer()));

        // always read weights first to CPU memory
        long t0 = System.currentTimeMillis();
        token_embedding_table = reader.nextFloatArray(config.vocab_size * config.dim);
        rms_att_weight = reader.nextFloatArray(config.n_layers * config.dim);
        wq = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        wk = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        wv = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        wo = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        rms_ffn_weight = reader.nextFloatArray(config.n_layers * config.dim);
        w1 = reader.nextFloatArray(config.n_layers * config.dim * config.hidden_dim);
        w2 = reader.nextFloatArray(config.n_layers * config.hidden_dim * config.dim);
        w3 = reader.nextFloatArray(config.n_layers * config.dim * config.hidden_dim);
        rms_final_weight = reader.nextFloatArray(config.dim);
        int head_size = config.dim / config.n_heads;
        freq_cis_real = reader.nextFloatArray(config.seq_len * head_size / 2);
        freq_cis_imag = reader.nextFloatArray(config.seq_len * head_size / 2);

        wcls = sharedWeights ? token_embedding_table : reader.nextFloatArray(config.vocab_size * config.dim);
        long t1 = System.currentTimeMillis();
        LLogger.time("Create TransformerWeights CPU", t0, t1);

        // copy weights to CUDA
        if (context.target.CUDA()) {
            long t2 = System.currentTimeMillis();
            ContextCUDA cu = context.cuda;
            token_embedding_tableCU = cu.allocateAndCopyToDevice(token_embedding_table);
            rms_att_weightCU = cu.allocateAndCopyToDevice(rms_att_weight);
            wqCU = cu.allocateAndCopyToDevice(wq);
            wkCU = cu.allocateAndCopyToDevice(wk);
            wvCU = cu.allocateAndCopyToDevice(wv);
            woCU = cu.allocateAndCopyToDevice(wo);
            rms_ffn_weightCU = cu.allocateAndCopyToDevice(rms_ffn_weight);
            w1CU = cu.allocateAndCopyToDevice(w1);
            w2CU = cu.allocateAndCopyToDevice(w2);
            w3CU = cu.allocateAndCopyToDevice(w3);
            rms_final_weightCU = cu.allocateAndCopyToDevice(rms_final_weight);
            freq_cis_realCU = cu.allocateAndCopyToDevice(freq_cis_real);
            freq_cis_imagCU = cu.allocateAndCopyToDevice(freq_cis_imag);
            wclsCU = sharedWeights ? token_embedding_tableCU : cu.allocateAndCopyToDevice(wcls);
            long t3 = System.currentTimeMillis();
            LLogger.time("Create TransformerWeights CUDA", t2, t3);
        }
    }

}
