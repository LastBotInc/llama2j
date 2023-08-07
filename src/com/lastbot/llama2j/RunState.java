package com.lastbot.llama2j;

import jcuda.Pointer;

import java.io.Closeable;

public class RunState implements Closeable {
    private final Target t;
    private final Config c;
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

    ContextCUDA cuda;
    ContextCPU cpu;

    RunState(Config c, Target t) {
        this.c = c;
        this.t = t;

        if (t.CPU()) {
            long t0 = System.currentTimeMillis();
            cpu = new ContextCPU("contextCPU0", 0, 20);
            x = cpu.allocateFloatArray(c.dim);
            xb = cpu.allocateFloatArray(c.dim);
            xb2 = cpu.allocateFloatArray(c.dim);
            hb = cpu.allocateFloatArray(c.hidden_dim);
            hb2 = cpu.allocateFloatArray(c.hidden_dim);
            q = cpu.allocateFloatArray(c.dim);
            k = cpu.allocateFloatArray(c.dim);
            v = cpu.allocateFloatArray(c.dim);
            att = cpu.allocateFloatArray((long) c.n_heads * c.seq_len);
            logits = cpu.allocateFloatArray(c.vocab_size);
            key_cache = cpu.allocateFloatArray((long) c.n_layers * c.seq_len * c.dim);
            value_cache = cpu.allocateFloatArray((long) c.n_layers * c.seq_len * c.dim);
            long t1 = System.currentTimeMillis();
            LLogger.time("Create RunState CPU", t0, t1);
        }

        if (t.CUDA()) {
            long t0 = System.currentTimeMillis();
            cuda = new ContextCUDA("contextCUDA0", 0, 20);
            xCU = cuda.allocateFloatArray(c.dim);
            xbCU = cuda.allocateFloatArray(c.dim);
            xb2CU = cuda.allocateFloatArray(c.dim);
            hbCU = cuda.allocateFloatArray(c.hidden_dim);
            hb2CU = cuda.allocateFloatArray(c.hidden_dim);
            qCU = cuda.allocateFloatArray(c.dim);
            kCU = cuda.allocateFloatArray(c.dim);
            vCU = cuda.allocateFloatArray(c.dim);
            attCU = cuda.allocateFloatArray((long) c.n_heads * c.seq_len);
            logitsCU = cuda.allocateFloatArray(c.vocab_size);
            key_cacheCU = cuda.allocateFloatArray((long) c.n_layers * c.seq_len * c.dim);
            value_cacheCU = cuda.allocateFloatArray((long) c.n_layers * c.seq_len * c.dim);
            long t1 = System.currentTimeMillis();
            LLogger.time("Create RunState CUDA", t0, t1);
        }
    }

    @Override
    public void close() {
        cpu.close();
        cuda.close();
    }
}
