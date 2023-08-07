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
    float[] key_cache;   // (layer, seq_len, dim)
    float[] value_cache; // (layer, seq_len, dim)

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
    Pointer key_cacheCU;
    Pointer value_cacheCU;

    RunState(Context context, Config config) {
        this.config = config;
        this.c = context;

        if (context.target.CPU()) {
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
            key_cache = c.cpu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            value_cache = c.cpu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            long t1 = System.currentTimeMillis();
            LLogger.time("Create RunState CPU", t0, t1);
        }

        if (context.target.CUDA()) {
            long t0 = System.currentTimeMillis();
            xCU = c.cuda.allocateFloatArray(config.dim);
            xbCU = c.cuda.allocateFloatArray(config.dim);
            xb2CU = c.cuda.allocateFloatArray(config.dim);
            hbCU = c.cuda.allocateFloatArray(config.hidden_dim);
            hb2CU = c.cuda.allocateFloatArray(config.hidden_dim);
            qCU = c.cuda.allocateFloatArray(config.dim);
            kCU = c.cuda.allocateFloatArray(config.dim);
            vCU = c.cuda.allocateFloatArray(config.dim);
            attCU = c.cuda.allocateFloatArray((long) config.n_heads * config.seq_len);
            logitsCU = c.cuda.allocateFloatArray(config.vocab_size);
            key_cacheCU = c.cuda.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            value_cacheCU = c.cuda.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            long t1 = System.currentTimeMillis();
            LLogger.time("Create RunState CUDA", t0, t1);
        }
    }

    @Override
    public void close() {
        c.cpu.close();
        c.cuda.close();
    }
}
