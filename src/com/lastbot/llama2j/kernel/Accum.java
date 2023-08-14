package com.lastbot.llama2j.kernel;

public class Accum {


    private static void accum(float[] a, float[] b, int size) {
        for (int i = 0; i < size; i++) {
            a[i] += b[i];
        }
    }

}
