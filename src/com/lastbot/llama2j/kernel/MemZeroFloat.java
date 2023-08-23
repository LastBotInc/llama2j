package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;

import java.util.Arrays;

public class MemZeroFloat extends Kernel {
    public static final int BLOCK_SIZE = 64;

    private final ContextCUDA cuda;

    public MemZeroFloat(ContextCUDA cuda) {
        super(cuda, "memSetFloat");
        this.cuda = cuda;
    }

    public static void call(float[] a, int index, int size) {
        for (int i = 0; i < size; i++) {
            a[index + i] = 0f;
        }
    }

    public void test(float[] a, int index, int size) {
        float[] copyOfA = Arrays.copyOf(a, a.length);
        Pointer pa = cuda.allocateAndCopyToDevice(TEST_STREAM, a, false);
        cuda.synchronizeStream(TEST_STREAM);
        call(TEST_STREAM, pa, index, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pa, a.length, a);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pa);

        call(copyOfA, index, size);

        compareWithThreshold("MemSetFloat.call", a, copyOfA, 1e-5f);
    }

    public void call(int streamId, Pointer a, int index, int size) {
        cuda.memZero(streamId, a, index, size);
    }
}
