package com.lastbot.llama2j;

import jcuda.Pointer;

public class QuantPointer {
    private final Quant quant;
    private final Pointer pointer;
    private final long floatOffset;

    public QuantPointer(Quant quant, Pointer pointer, long floatOffset) {
        this.pointer = pointer;
        this.floatOffset = floatOffset;
        this.quant = quant;
    }

    public Pointer pointerOfFloatIndex(int floatIndex) {
        int adjustedFloatIndex =  (int) (floatIndex - floatOffset);
        if (adjustedFloatIndex < 0) {
            throw new RuntimeException("adjustedFloatIndex < 0");
        }
        int quantIndex = quant.byteOffsetByFloatIndex(adjustedFloatIndex);
        return pointer.withByteOffset(quantIndex);
    }

    public Quant getQuant() {
        return quant;
    }

    public Pointer getPointer() {
        return pointer;
    }
}
