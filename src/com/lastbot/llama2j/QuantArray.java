package com.lastbot.llama2j;

import java.io.DataOutputStream;
import java.io.IOException;

public class QuantArray {
    private final Quant quant;
    private final byte[] data;
    private final long floatOffset;

    public QuantArray(Quant quant, byte[] data, long floatOffset) {
        this.data = data;
        this.floatOffset = floatOffset;
        this.quant = quant;
    }

    public int byteOffsetByFloatIndex(int floatIndex) {
        int adjustedFloatIndex = Math.toIntExact(floatIndex - floatOffset);
        if (adjustedFloatIndex < 0) {
            throw new RuntimeException("adjustedFloatIndex < 0");
        }
        int byteOffset = quant.byteOffsetByFloatIndex(adjustedFloatIndex);
        return byteOffset;
    }

    public long getFloatOffset() {
        return floatOffset;
    }

    public Quant getQuant() {
        return quant;
    }

    public byte[] getByteArray() {
        return data;
    }

    public void write(DataOutputStream dos) throws IOException {
        dos.writeLong(floatOffset);
        dos.writeInt(data.length);
        dos.write(data);
    }
}
