package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Class used to manage arrays on CUDA devices, where the data representing different layers
 * is divided into different GPU devices. This is a convenience class that ensures that
 * memory references take into account.
 *
 * This is based on activations in FP32.
 *
 * @param pointer       CUDA device specific pointer to the array memory
 * @param floatOffset   offset how many floats addressing needs to be adjusted for this Slice
 * @param byteOffset    offset how many bytes addressing needs to be adjusted for this Slice
 * @param byteSize      total size of this slice in bytes
 */
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
