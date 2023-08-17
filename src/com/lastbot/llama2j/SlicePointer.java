package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.Sizeof;

public record SlicePointer(Pointer pointer, long floatOffset, long byteOffset, long byteSize) {
    public Pointer withIndex(int index) {
        return withByteOffset((long) index * Sizeof.FLOAT);
    }

    public Pointer withByteOffset(long additionalOffset) {
        long offset = additionalOffset - byteOffset;
        if (offset < 0) {
            throw new RuntimeException("SlicePointer.withByteOffset negative offset = " + offset);
        } else if (offset > byteSize) {
            throw new RuntimeException("SlicePointer.withByteOffset out of array offset = " + offset +
                    ", byteSize = " + byteSize);
        }
        return this.pointer.withByteOffset(offset);
    }
}
