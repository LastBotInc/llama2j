package com.lastbot.llama2j;

import jcuda.Pointer;

import static com.lastbot.llama2j.ContextCUDA.STREAM_COUNT;
import static com.lastbot.llama2j.ContextCUDA.allocateAndCopyLayers;

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

    Pointer[] token_embedding_tableCU;    // (vocab_size, dim)
    SlicePointer[] l_rms_att_weightCU; // (layer, dim) rmsnorm weights
    SlicePointer[] l_rms_ffn_weightCU; // (layer, dim)
    SlicePointer[] l_wqCU; // (layer, dim, dim)
    SlicePointer[] l_wkCU; // (layer, dim, dim)
    SlicePointer[] l_wvCU; // (layer, dim, dim)
    SlicePointer[] l_woCU; // (layer, dim, dim)
    SlicePointer[] l_w1CU; // (layer, hidden_dim, dim)
    SlicePointer[] l_w2CU; // (layer, dim, hidden_dim)
    SlicePointer[] l_w3CU; // (layer, hidden_dim, dim)
    Pointer[] rms_final_weightCU; // (dim,)
    Pointer[] freq_cis_realCU; // (seq_len, dim/2)
    Pointer[] freq_cis_imagCU; // (seq_len, dim/2)
    Pointer[] wclsCU;

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

            int n = context.layerAllocation.deviceCount;

            token_embedding_tableCU = new Pointer[n];
            l_rms_att_weightCU = new SlicePointer[n];
            l_rms_ffn_weightCU = new SlicePointer[n];
            l_wqCU = new SlicePointer[n];
            l_wkCU = new SlicePointer[n];
            l_wvCU = new SlicePointer[n];
            l_woCU = new SlicePointer[n];
            l_w1CU = new SlicePointer[n];
            l_w2CU = new SlicePointer[n];
            l_w3CU = new SlicePointer[n];
            rms_final_weightCU = new Pointer[n];
            freq_cis_realCU = new Pointer[n];
            freq_cis_imagCU = new Pointer[n];
            wclsCU = new Pointer[n];

            for (int dev = 0; dev < context.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = context.cudas[dev];
                token_embedding_tableCU[dev] = cu.allocateAndCopyToDevice(0, token_embedding_table, true);
                int firstLayer = context.layerAllocation.firstLayer[dev];
                int lastLayer = context.layerAllocation.lastLayer[dev];

                l_rms_att_weightCU[dev] = allocateAndCopyLayers(1 % STREAM_COUNT, cu, l_rms_att_weight, firstLayer, lastLayer, p.n_layers);
                l_rms_ffn_weightCU[dev] = allocateAndCopyLayers(2 % STREAM_COUNT, cu, l_rms_ffn_weight, firstLayer, lastLayer, p.n_layers);
                l_wqCU[dev] = allocateAndCopyLayers(3 % STREAM_COUNT, cu, l_wq, firstLayer, lastLayer, p.n_layers);
                l_wkCU[dev] = allocateAndCopyLayers(4 % STREAM_COUNT, cu, l_wk, firstLayer, lastLayer, p.n_layers);
                l_wvCU[dev] = allocateAndCopyLayers(5 % STREAM_COUNT, cu, l_wv, firstLayer, lastLayer, p.n_layers);
                l_woCU[dev] = allocateAndCopyLayers(6 % STREAM_COUNT, cu, l_wo, firstLayer, lastLayer, p.n_layers);
                l_w1CU[dev] = allocateAndCopyLayers(7 % STREAM_COUNT, cu, l_w1, firstLayer, lastLayer, p.n_layers);
                l_w2CU[dev] = allocateAndCopyLayers(8 % STREAM_COUNT, cu, l_w2, firstLayer, lastLayer, p.n_layers);
                l_w3CU[dev] = allocateAndCopyLayers(9 % STREAM_COUNT, cu, l_w3, firstLayer, lastLayer, p.n_layers);

                rms_final_weightCU[dev] = cu.allocateAndCopyToDevice(10 % STREAM_COUNT, rms_final_weight, true);
                freq_cis_realCU[dev] = cu.allocateAndCopyToDevice(11 % STREAM_COUNT, freq_cis_real, true);
                freq_cis_imagCU[dev] = cu.allocateAndCopyToDevice(12 % STREAM_COUNT, freq_cis_imag, true);
                wclsCU[dev] = sharedWeights ? token_embedding_tableCU[dev] :
                        cu.allocateAndCopyToDevice(13 % STREAM_COUNT, wcls, true);
            }
            // as transfers are async, make sure all devices are sync
            for (int dev = 0; dev < context.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = context.cudas[dev];
                cu.synchronizeDevice();
            }
            long t3 = System.currentTimeMillis();
            LLogger.time("Create TransformerWeights CUDA", t2, t3);
        }
    }
}
