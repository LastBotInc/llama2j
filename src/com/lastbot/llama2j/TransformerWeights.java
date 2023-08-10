package com.lastbot.llama2j;

import jcuda.Pointer;

public class TransformerWeights {
    private final Context context;
    private final Config config;  // token embedding table

    float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float[] l_rms_att_weight; // (layer, dim) rmsnorm weights
    float[] l_rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float[] l_wq; // (layer, dim, dim)
    float[] l_wk; // (layer, dim, dim)
    float[] l_wv; // (layer, dim, dim)
    float[] l_wo; // (layer, dim, dim)
    // weights for ffn
    float[] l_w1; // (layer, hidden_dim, dim)
    float[] l_w2; // (layer, dim, hidden_dim)
    float[] l_w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float[] rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float[] freq_cis_real; // (seq_len, dim/2)
    float[] freq_cis_imag; // (seq_len, dim/2)
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

    public TransformerWeights(Context context, BinFileReader reader, Config config, boolean sharedWeights) {
        this.context = context;
        this.config = config;

        // always read weights first to CPU memory
        long t0 = System.currentTimeMillis();
        token_embedding_table = reader.nextFloatArray(config.vocab_size * config.dim);
        l_rms_att_weight = reader.nextFloatArray(config.n_layers * config.dim);
        l_wq = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        l_wk = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        l_wv = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        l_wo = reader.nextFloatArray(config.n_layers * config.dim * config.dim);
        l_rms_ffn_weight = reader.nextFloatArray(config.n_layers * config.dim);
        l_w1 = reader.nextFloatArray(config.n_layers * config.dim * config.hidden_dim);
        l_w2 = reader.nextFloatArray(config.n_layers * config.hidden_dim * config.dim);
        l_w3 = reader.nextFloatArray(config.n_layers * config.dim * config.hidden_dim);
        rms_final_weight = reader.nextFloatArray(config.dim);
        int head_size = config.dim / config.n_heads;
        freq_cis_real = reader.nextFloatArray(config.seq_len * head_size / 2);
        freq_cis_imag = reader.nextFloatArray(config.seq_len * head_size / 2);

        wcls = sharedWeights ? token_embedding_table : reader.nextFloatArray(config.vocab_size * config.dim);
        long t1 = System.currentTimeMillis();
        LLogger.time("Create TransformerWeights CPU", t0, t1);

        // copy weights to CUDA
        if (context.layerAllocation.deviceCount > 0) {
            long t2 = System.currentTimeMillis();
            for (int dev = 0; dev < context.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = context.cudas[dev];
                token_embedding_tableCU = cu.allocateAndCopyToDevice(token_embedding_table);
                int firstLayer = context.layerAllocation.firstLayer[dev];
                int lastLayer = context.layerAllocation.lastLayer[dev];

                l_rms_att_weightCU = allocateAndCopyLayers(cu, l_rms_att_weight, firstLayer, lastLayer);
                l_wqCU = allocateAndCopyLayers(cu, l_wq, firstLayer, lastLayer);
                l_wkCU = allocateAndCopyLayers(cu, l_wk, firstLayer, lastLayer);
                l_wvCU = allocateAndCopyLayers(cu, l_wv, firstLayer, lastLayer);
                l_woCU = allocateAndCopyLayers(cu, l_wo, firstLayer, lastLayer);
                l_rms_ffn_weightCU = allocateAndCopyLayers(cu, l_rms_ffn_weight, firstLayer, lastLayer);
                l_w1CU = allocateAndCopyLayers(cu, l_w1, firstLayer, lastLayer);
                l_w2CU = allocateAndCopyLayers(cu, l_w2, firstLayer, lastLayer);
                l_w3CU = allocateAndCopyLayers(cu, l_w3, firstLayer, lastLayer);
//                l_rms_att_weightCU = cu.allocateAndCopyToDevice(l_rms_att_weight);
//                l_wqCU = cu.allocateAndCopyToDevice(l_wq);
//                l_wkCU = cu.allocateAndCopyToDevice(l_wk);
//                l_wvCU = cu.allocateAndCopyToDevice(l_wv);
//                l_woCU = cu.allocateAndCopyToDevice(l_wo);
//                l_rms_ffn_weightCU = cu.allocateAndCopyToDevice(l_rms_ffn_weight);
//                l_w1CU = cu.allocateAndCopyToDevice(l_w1);
//                l_w2CU = cu.allocateAndCopyToDevice(l_w2);
//                l_w3CU = cu.allocateAndCopyToDevice(l_w3);
                rms_final_weightCU = cu.allocateAndCopyToDevice(rms_final_weight);
                freq_cis_realCU = cu.allocateAndCopyToDevice(freq_cis_real);
                freq_cis_imagCU = cu.allocateAndCopyToDevice(freq_cis_imag);
                wclsCU = sharedWeights ? token_embedding_tableCU : cu.allocateAndCopyToDevice(wcls);
            }
            long t3 = System.currentTimeMillis();
            LLogger.time("Create TransformerWeights CUDA", t2, t3);
        }
    }

    private Pointer allocateAndCopyLayers(ContextCUDA cu, float[] cpuArray, int firstLayer, int lastLayer) {
        int floatOffset = layerFloatOffset(cpuArray, firstLayer);
        int floatSize = layerFloatSize(cpuArray, firstLayer, lastLayer);

        Pointer pointer = cu.allocateAndCopyToDeviceWithOffset(cpuArray, floatOffset, floatSize);
        return pointer;
    }

    private int layerFloatOffset(float[] cpuArray, int firstLayer) {
        int bytesPerLayer = cpuArray.length / config.n_layers;
        int offset = firstLayer * bytesPerLayer;
        return offset;
    }

    private int layerFloatSize(float[] cpuArray, int firstLayer, int lastLayer) {
        int bytesPerLayer = cpuArray.length / config.n_layers;
        int size = (lastLayer - firstLayer + 1) * bytesPerLayer;
        return size;
    }
}
