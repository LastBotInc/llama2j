package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;

public class ContextCPU implements Closeable {
    private final String name;
    private final int deviceId;
    private final int maxMemoryInGigaBytes;

    public ContextCPU(String name, int deviceId, int maxMemoryInGigaBytes) {
        this.name = name;
        this.deviceId = deviceId;
        this.maxMemoryInGigaBytes = maxMemoryInGigaBytes;
    }

    public float[] allocateFloatArray(long elements) {
        if (elements <= Limits.FLOAT_ARRAY_MAX_SIZE) {
            int size = (int) elements;
            return new float[size];
        }

        double ratio = (double) elements / (double) Limits.FLOAT_ARRAY_MAX_SIZE;
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
