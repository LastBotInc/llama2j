package com.lastbot.llama2j;

import java.io.Closeable;

/**
 * CPU context which provides memory allocation. Used simple heap array allocation,
 * and checks the memory size against the set limit.
 */
public class ContextCPU implements Closeable {

    public ContextCPU() {
    }

    public float[] allocateFloatArray(long elements) {
        if (elements <= Limits.ARRAY_MAX_SIZE) {
            int size = Math.toIntExact(elements);
            return new float[size];
        }

        double ratio = (double) elements / (double) Limits.ARRAY_MAX_SIZE;
        String s = "Tried to allocate " + String.format("%,d", elements) +
                " elements, which is " + String.format("%,.1f", ratio) +
                " times more than supported";
        LLogger.error(s);
        System.exit(1);
        return null;
    }

    @Override
    public void close() {
    }
}
