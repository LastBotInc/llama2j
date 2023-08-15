package com.lastbot.llama2j;

import jcuda.Pointer;

public class LayerMemoryUtil {
    public static Pointer allocateAndCopyLayers(ContextCUDA cu, float[] cpuArray, int firstLayer, int lastLayer,
                                                 int nLayers) {
        int floatOffset = layerFloatOffset(cpuArray, firstLayer, nLayers);
        int floatSize = layerFloatSize(cpuArray, firstLayer, lastLayer, nLayers);

        Pointer pointer = cu.allocateAndCopyToDeviceWithOffset(cpuArray, floatOffset, floatSize, true);
        return pointer;
    }

    private static int layerFloatOffset(float[] cpuArray, int firstLayer, int nLayers) {
        return layerFloatOffset(cpuArray.length, firstLayer, nLayers);
    }

    private static int layerFloatSize(float[] cpuArray, int firstLayer, int lastLayer, int nLayers) {
        return layerFloatSize(cpuArray.length, firstLayer, lastLayer, nLayers);
    }

    private static int layerFloatOffset(int length, int firstLayer, int nLayers) {
        int bytesPerLayer = length / nLayers;
        int offset = firstLayer * bytesPerLayer;
        return offset;
    }

    private static int layerFloatSize(int length, int firstLayer, int lastLayer, int nLayers) {
        int bytesPerLayer = length / nLayers;
        int size = (lastLayer - firstLayer + 1) * bytesPerLayer;
        return size;
    }
}
