package com.lastbot.llama2j;

import jcuda.Pointer;

import static com.lastbot.llama2j.LayerMemoryUtil.allocateAndCopyLayers;

public class TransformerWeights {
    float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float[] l_rms_att_weight; // (layer, dim) rmsnorm weights
    float[] l_rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float[] l_wq; // (layer, dim, n_heads * head_size)
    float[] l_wk; // (layer, dim, n_kv_heads * head_size)
    float[] l_wv; // (layer, dim, n_kv_heads * head_size)
    float[] l_wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float[] l_w1; // (layer, hidden_dim, dim)
    float[] l_w2; // (layer, dim, hidden_dim)
    float[] l_w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float[] rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float[] freq_cis_real; // (seq_len, head_size/2)
    float[] freq_cis_imag; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    float[] wcls;

    public static long bytesStatic(Config config, boolean sharedWeights) {
        int head_size = config.dim / config.n_heads;
        return (long) Float.BYTES * (
                ((long) config.vocab_size * config.dim) +
                        ((long) config.dim) +
                        ((long) config.seq_len * head_size / 2) +
                        ((long) config.seq_len * head_size / 2) +
                        (sharedWeights ? 0L : (long) config.vocab_size * config.dim)
        );
    }

    public static long bytesPerLayer(Config config) {
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
    Pointer l_rms_att_weightCU; // (layer, dim) rmsnorm weights
    Pointer l_rms_ffn_weightCU; // (layer, dim)
    Pointer l_wqCU; // (layer, dim, dim)
    Pointer l_wkCU; // (layer, dim, dim)
    Pointer l_wvCU; // (layer, dim, dim)
    Pointer l_woCU; // (layer, dim, dim)
    Pointer l_w1CU; // (layer, hidden_dim, dim)
    Pointer l_w2CU; // (layer, dim, hidden_dim)
    Pointer l_w3CU; // (layer, hidden_dim, dim)
    Pointer rms_final_weightCU; // (dim,)
    Pointer freq_cis_realCU; // (seq_len, dim/2)
    Pointer freq_cis_imagCU; // (seq_len, dim/2)
    Pointer wclsCU;

    public TransformerWeights(Context context, BinFileReader reader, Config p, boolean sharedWeights) {
        // token embedding table
        int head_size = p.dim / p.n_heads;
        // always read weights first to CPU memory
        long t0 = System.currentTimeMillis();
        token_embedding_table = reader.nextFloatArray(p.vocab_size * p.dim);
        l_rms_att_weight = reader.nextFloatArray(p.n_layers * p.dim);
        l_wq = reader.nextFloatArray(p.n_layers * p.dim * (p.n_heads * head_size));
        l_wk = reader.nextFloatArray(p.n_layers * p.dim * (p.n_kv_heads * head_size)); // zzz
        l_wv = reader.nextFloatArray(p.n_layers * p.dim * (p.n_kv_heads * head_size)); // zzz
        l_wo = reader.nextFloatArray(p.n_layers * (p.n_heads * head_size) * p.dim);
        l_rms_ffn_weight = reader.nextFloatArray(p.n_layers * p.dim);
        l_w1 = reader.nextFloatArray(p.n_layers * p.dim * p.hidden_dim);
        l_w2 = reader.nextFloatArray(p.n_layers * p.hidden_dim * p.dim);
        l_w3 = reader.nextFloatArray(p.n_layers * p.dim * p.hidden_dim);
        rms_final_weight = reader.nextFloatArray(p.dim);
        freq_cis_real = reader.nextFloatArray(p.seq_len * head_size / 2);
        freq_cis_imag = reader.nextFloatArray(p.seq_len * head_size / 2);

        wcls = sharedWeights ? token_embedding_table : reader.nextFloatArray(p.vocab_size * p.dim);
        long t1 = System.currentTimeMillis();
        LLogger.time("Create TransformerWeights CPU", t0, t1);

        // copy weights to CUDA
        if (context.layerAllocation.deviceCount > 0) {
            long t2 = System.currentTimeMillis();
            for (int dev = 0; dev < context.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = context.cudas[dev];
                token_embedding_tableCU = cu.allocateAndCopyToDevice(token_embedding_table, true);
                int firstLayer = context.layerAllocation.firstLayer[dev];
                int lastLayer = context.layerAllocation.lastLayer[dev];

                l_rms_att_weightCU = allocateAndCopyLayers(cu, l_rms_att_weight, firstLayer, lastLayer, p.n_layers);
                l_wqCU = allocateAndCopyLayers(cu, l_wq, firstLayer, lastLayer, p.n_layers);
                l_wkCU = allocateAndCopyLayers(cu, l_wk, firstLayer, lastLayer, p.n_layers);
                l_wvCU = allocateAndCopyLayers(cu, l_wv, firstLayer, lastLayer, p.n_layers);
                l_woCU = allocateAndCopyLayers(cu, l_wo, firstLayer, lastLayer, p.n_layers);
                l_rms_ffn_weightCU = allocateAndCopyLayers(cu, l_rms_ffn_weight, firstLayer, lastLayer, p.n_layers);
                l_w1CU = allocateAndCopyLayers(cu, l_w1, firstLayer, lastLayer, p.n_layers);
                l_w2CU = allocateAndCopyLayers(cu, l_w2, firstLayer, lastLayer, p.n_layers);
                l_w3CU = allocateAndCopyLayers(cu, l_w3, firstLayer, lastLayer, p.n_layers);
                rms_final_weightCU = cu.allocateAndCopyToDevice(rms_final_weight, true);
                freq_cis_realCU = cu.allocateAndCopyToDevice(freq_cis_real, true);
                freq_cis_imagCU = cu.allocateAndCopyToDevice(freq_cis_imag, true);
                wclsCU = sharedWeights ? token_embedding_tableCU : cu.allocateAndCopyToDevice(wcls, true);
            }
            long t3 = System.currentTimeMillis();
            LLogger.time("Create TransformerWeights CUDA", t2, t3);
        }
    }

}
