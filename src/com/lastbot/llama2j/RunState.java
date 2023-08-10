package com.lastbot.llama2j;

import jcuda.Pointer;

import java.io.Closeable;

public class RunState implements Closeable {
    private final Context c;
    private final Config config;
    // current wave of activations
    float[] x; // activation at current time stamp (dim,)
    float[] xb; // same, but inside a residual branch (dim,)
    float[] xb2; // an additional buffer just for convenience (dim,)
    float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float[] q; // query (dim,)
    float[] k; // key (dim,)
    float[] v; // value (dim,)
    float[] att; // buffer for scores/attention values (n_heads, seq_len)
    float[] logits; // output logits
    // kv cache
    float[] l_key_cache;   // (layer, seq_len, dim)
    float[] l_value_cache; // (layer, seq_len, dim)

    Pointer xCU;
    Pointer xbCU;
    Pointer xb2CU;
    Pointer hbCU;
    Pointer hb2CU;
    Pointer qCU;
    Pointer kCU;
    Pointer vCU;
    Pointer attCU;
    Pointer logitsCU;
    Pointer l_key_cacheCU;
    Pointer l_value_cacheCU;

    public static long bytesStatic(Config config) {
        return (long) Float.BYTES * (
                ((long) config.dim) +
                        ((long) config.dim) +
                        ((long) config.dim) +
                        ((long) config.hidden_dim) +
                        ((long) config.hidden_dim) +
                        ((long) config.dim) +
                        ((long) config.dim) +
                        ((long) config.dim) +
                        ((long) config.n_heads * config.seq_len) +
                        ((long) config.vocab_size)
        );
    }

    public static long bytesPerLayer(Config config) {
        return (long) Float.BYTES * (
                ((long) config.seq_len * config.dim) +
                        ((long) config.seq_len * config.dim)
        );
    }

    public RunState(Context context, Config config) {
        this.config = config;
        this.c = context;

        if (context.cpu != null) {
            long t0 = System.currentTimeMillis();
            x = c.cpu.allocateFloatArray(config.dim);
            xb = c.cpu.allocateFloatArray(config.dim);
            xb2 = c.cpu.allocateFloatArray(config.dim);
            hb = c.cpu.allocateFloatArray(config.hidden_dim);
            hb2 = c.cpu.allocateFloatArray(config.hidden_dim);
            q = c.cpu.allocateFloatArray(config.dim);
            k = c.cpu.allocateFloatArray(config.dim);
            v = c.cpu.allocateFloatArray(config.dim);
            att = c.cpu.allocateFloatArray((long) config.n_heads * config.seq_len);
            logits = c.cpu.allocateFloatArray(config.vocab_size);
            l_key_cache = c.cpu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            l_value_cache = c.cpu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            long t1 = System.currentTimeMillis();
            LLogger.time("Create RunState CPU", t0, t1);
        }

        if (context.layerAllocation.deviceCount > 0) {
            long t0 = System.currentTimeMillis();
            for (int dev = 0; dev < context.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = c.cudas[dev];
                int firstLayer = context.layerAllocation.firstLayer[dev];
                int lastLayer = context.layerAllocation.lastLayer[dev];

                xCU = cu.allocateFloatArray(config.dim);
                xbCU = cu.allocateFloatArray(config.dim);
                xb2CU = cu.allocateFloatArray(config.dim);
                hbCU = cu.allocateFloatArray(config.hidden_dim);
                hb2CU = cu.allocateFloatArray(config.hidden_dim);
                qCU = cu.allocateFloatArray(config.dim);
                kCU = cu.allocateFloatArray(config.dim);
                vCU = cu.allocateFloatArray(config.dim);
                attCU = cu.allocateFloatArray((long) config.n_heads * config.seq_len);
                logitsCU = cu.allocateFloatArray(config.vocab_size);

                int nLayers = lastLayer - firstLayer + 1;
                int lengthOfLayerData = nLayers * (config.seq_len * config.dim);

                l_key_cacheCU = cu.allocateFloatArray(lengthOfLayerData);
                l_value_cacheCU = cu.allocateFloatArray(lengthOfLayerData);

//                l_key_cacheCU = cu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
//                l_value_cacheCU = cu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            }
            long t1 = System.currentTimeMillis();
            LLogger.time("Create RunState CUDA", t0, t1);
        }
    }

    @Override
    public void close() {
        if (c.cpu != null) {
            c.cpu.close();
        }
        if (c.cudas != null) {
            for (int i = 0; i < c.cudas.length; i++) {
                c.cudas[i].close();
            }
        }
    }
}
