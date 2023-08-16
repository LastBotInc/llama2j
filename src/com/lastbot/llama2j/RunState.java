package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.Sizeof;

import java.io.Closeable;

public class RunState implements Closeable {
    // struct used when sorting probabilities during top-p sampling, CPU only
    public static class ProbIndex implements Comparable<ProbIndex> {
        public float prob;
        public int index;

        @Override
        public int compareTo(ProbIndex o) {
            if (prob > o.prob) {
                return -1;
            } else if (prob < o.prob) {
                return 1;
            } else {
                return 0;
            }
        }
    }

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
    float[] tmp1;
    float[] tmp2;
    float[] tmp3;

    // kv cache
    float[] l_key_cache;   // (layer, seq_len, dim)
    float[] l_value_cache; // (layer, seq_len, dim)

    ProbIndex[] probIndex; // buffer used in top-p sampling, CPU only

    Pointer[] xCU;
    Pointer[] xbCU;
    Pointer[] xb2CU;
    Pointer[] hbCU;
    Pointer[] hb2CU;
    Pointer[] qCU;
    Pointer[] kCU;
    Pointer[] vCU;
    Pointer[] attCU;
    Pointer[] logitsCU;
    Pointer[] l_key_cacheCU;
    Pointer[] l_value_cacheCU;
    Pointer[] tmp1CU;
    Pointer[] tmp2CU;
    Pointer[] tmp3CU;

    public static long bytesStatic(Config p) {
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        return (long) Float.BYTES * (
                ((long) p.dim) +
                        ((long) p.dim) +
                        ((long) p.dim) +
                        ((long) p.hidden_dim) +
                        ((long) p.hidden_dim) +
                        ((long) p.dim) +
                        ((long) kv_dim) +
                        ((long) kv_dim) +
                        ((long) p.n_heads * p.seq_len) +
                        ((long) p.vocab_size) +
                        ((long) 1) +
                        ((long) 1) +
                        ((long) 1)
                // omit probIndex, as it is currently only CPU
        );
    }

    public static long bytesPerLayer(Config p) {
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        return (long) Float.BYTES * (
                ((long) p.seq_len * kv_dim) +
                        ((long) p.seq_len * kv_dim)
        );
    }

    public RunState(Context context, Config p) {
        this.config = p;
        this.c = context;

        long t0 = System.currentTimeMillis();
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        x = c.cpu.allocateFloatArray(p.dim);
        xb = c.cpu.allocateFloatArray(p.dim);
        xb2 = c.cpu.allocateFloatArray(p.dim);
        hb = c.cpu.allocateFloatArray(p.hidden_dim);
        hb2 = c.cpu.allocateFloatArray(p.hidden_dim);
        q = c.cpu.allocateFloatArray(p.dim);
        k = c.cpu.allocateFloatArray(kv_dim);
        v = c.cpu.allocateFloatArray(kv_dim);
        att = c.cpu.allocateFloatArray((long) p.n_heads * p.seq_len);
        logits = c.cpu.allocateFloatArray(p.vocab_size);
        tmp1 = c.cpu.allocateFloatArray(1);
        tmp2 = c.cpu.allocateFloatArray(1);
        tmp3 = c.cpu.allocateFloatArray(1);

        probIndex = new ProbIndex[p.vocab_size];
        for (int i = 0; i < p.vocab_size; i++) {
            probIndex[i] = new ProbIndex();
        }

        l_key_cache = c.cpu.allocateFloatArray((long) p.n_layers * p.seq_len * kv_dim);
        l_value_cache = c.cpu.allocateFloatArray((long) p.n_layers * p.seq_len * kv_dim);
        long t1 = System.currentTimeMillis();
        LLogger.time("Create RunState CPU", t0, t1);

        if (context.layerAllocation.deviceCount > 0) {
            long t2 = System.currentTimeMillis();
            int n = context.layerAllocation.deviceCount;

            xCU = new Pointer[n];
            xbCU = new Pointer[n];
            xb2CU = new Pointer[n];
            hbCU = new Pointer[n];
            hb2CU = new Pointer[n];
            qCU = new Pointer[n];
            kCU = new Pointer[n];
            vCU = new Pointer[n];
            attCU = new Pointer[n];
            logitsCU = new Pointer[n];
            l_key_cacheCU = new Pointer[n];
            l_value_cacheCU = new Pointer[n];
            tmp1CU = new Pointer[n];
            tmp2CU = new Pointer[n];
            tmp3CU = new Pointer[n];

            for (int dev = 0; dev < context.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = c.cudas[dev];
                int firstLayer = context.layerAllocation.firstLayer[dev];
                int lastLayer = context.layerAllocation.lastLayer[dev];

                xCU[dev] = cu.allocateFloatArray(p.dim, true);
                xbCU[dev] = cu.allocateFloatArray(p.dim, true);
                xb2CU[dev] = cu.allocateFloatArray(p.dim, true);
                hbCU[dev] = cu.allocateFloatArray(p.hidden_dim, true);
                hb2CU[dev] = cu.allocateFloatArray(p.hidden_dim, true);
                qCU[dev] = cu.allocateFloatArray(p.dim, true);
                kCU[dev] = cu.allocateFloatArray(kv_dim, true);
                vCU[dev] = cu.allocateFloatArray(kv_dim, true);
                attCU[dev] = cu.allocateFloatArray((long) p.n_heads * p.seq_len, true);
                logitsCU[dev] = cu.allocateFloatArray(p.vocab_size, true);

                int nLayers = lastLayer - firstLayer + 1;
                int lengthOfLayerData = nLayers * (p.seq_len * kv_dim);

                l_key_cacheCU[dev] = cu.allocateFloatArray(lengthOfLayerData, true);
                l_value_cacheCU[dev] = cu.allocateFloatArray(lengthOfLayerData, true);

                tmp1CU[dev] = cu.allocateFloatArray(1, true);
                tmp2CU[dev] = cu.allocateFloatArray(1, true);
                tmp3CU[dev] = cu.allocateFloatArray(1, true);

//                l_key_cacheCU = cu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
//                l_value_cacheCU = cu.allocateFloatArray((long) config.n_layers * config.seq_len * config.dim);
            }
            long t3 = System.currentTimeMillis();
            LLogger.time("Create RunState CUDA", t2, t3);
        }
    }

    @Override
    public void close() {
        // rely on context.close
    }
}
