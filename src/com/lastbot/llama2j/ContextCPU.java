package com.lastbot.llama2j;

import java.io.Closeable;

public class ContextCPU implements Closeable {
    private final String name;

    public ContextCPU(String name) {
        this.name = name;
    }

    public float[] allocateFloatArray(long elements) {
        if (elements <= Limits.ARRAY_MAX_SIZE) {
            int size = (int) elements;
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

    public int[] allocateIntArray(long elements) {
        if (elements <= Limits.ARRAY_MAX_SIZE) {
            int size = (int) elements;
            return new int[size];
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
