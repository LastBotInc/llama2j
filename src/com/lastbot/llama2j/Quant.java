package com.lastbot.llama2j;

import it.unimi.dsi.util.XoRoShiRo128PlusRandom;

import java.nio.ByteBuffer;
import java.util.Random;

public record Quant(int groupSize, int bits) {
    private static final int TEST_ITERATIONS = 100_000;
    private static final double TEST_THRESHOLD_RELATIVE = 0.01;

    private int originalBytesPerGroup() {
        return groupSize * Float.BYTES;
    }

    public int encodedBytesPerGroup() {
        return ((groupSize * bits) / Byte.SIZE) + 8;
    }

    public int numberOfGroupsByFloatSize(int floatSize) {
        int numberOfGroups = (int) Math.ceil((double) floatSize / groupSize);
        return numberOfGroups;
    }

    public int numberOfBytesByFloatSize(int floatSize) {
        int numberOfGroups = numberOfGroupsByFloatSize(floatSize);
        int bytes = numberOfGroups * encodedBytesPerGroup();
        return bytes;
    }

    public int groupIndexByFloatIndex(int floatIndex) {
        int groupIndex = floatIndex / groupSize; // round down
        return groupIndex;
    }

    public int byteOffsetByFloatIndex(int floatIndex) {
        int groupIndex = groupIndexByFloatIndex(floatIndex);
        int byteOffset = groupIndex * encodedBytesPerGroup();
        return byteOffset;
    }

    public double compression() {
        return (double) encodedBytesPerGroup() / originalBytesPerGroup();
    }

    public interface DecodeProcessor {
        void process(int floatIndex, float value);
    }

    public static void main(String[] args) {
        int SIZE = 100;
        float[] d = new float[SIZE];
        for (int i = 0; i < SIZE; i++) {
            d[i] = i;
        }
        Quant quant = new Quant(Run.QUANT_CHUNK_SIZE, Run.QUANT_BITS);
        quant.testEncode(d);
    }

    public void decode(byte[] encoded, int floatIndex, int size, DecodeProcessor processor) {
        int groupSize = groupSize();

        int startGroupIndex = groupIndexByFloatIndex(floatIndex);
        int endGroupIndex = groupIndexByFloatIndex(floatIndex + size - 1);

        float min;
        float max;
        float range;
        int groupBase;
        int groupPayloadBase;
        int jj;
        for (int group = startGroupIndex; group <= endGroupIndex; group++) {
            groupBase = group * encodedBytesPerGroup();
            groupPayloadBase = groupBase + 8;
            min = bytesToFloat(encoded, groupBase);
            max = bytesToFloat(encoded, groupBase + 4);
            range = max - min;

            int startFloatIndex = group * groupSize;
            for (int j = 0; j < groupSize; j++) {
                jj = startFloatIndex + j;
                if (jj >= floatIndex && jj < floatIndex + size) {
                    int byteValue = encoded[groupPayloadBase + j] & 0xff;
                    float value = byteValue / 255f * range + min;
                    processor.process(jj, value);
                }
            }
        }
    }

    private static float bytesToFloat(byte[] bytes, int index) {
        int asInt = (bytes[index + 3] & 0xFF)
                | ((bytes[index + 2] & 0xFF) << 8)
                | ((bytes[index + 1] & 0xFF) << 16)
                | ((bytes[index] & 0xFF) << 24);
        return Float.intBitsToFloat(asInt);
    }

    public ByteBuffer testEncode(float[] input) {
        int floatSize = input.length;
        ByteBuffer byteBuffer = encode(input);
        int encodedSize = byteBuffer.remaining();
        byte[] encoded = new byte[encodedSize];
        byteBuffer.get(encoded);
        byteBuffer.rewind();

        Random random = new XoRoShiRo128PlusRandom(101);
        int startIndex;
        int size;
        int[] expectedIndex = new int[1];

        for (int k = 0; k < TEST_ITERATIONS; k++) {
            startIndex = random.nextInt(floatSize);
            size = random.nextInt(floatSize - startIndex);
            if (random.nextFloat() < 0.1D) {
                size = floatSize - startIndex;
            }
            expectedIndex[0] = startIndex;
            decode(encoded, startIndex, size,
                    (floatIndex, value) -> {
                        if (floatIndex != expectedIndex[0]) {
                            LLogger.error("floatIndex != expectedIndex[0]");
                        }
                        expectedIndex[0]++;
                        float original = input[floatIndex];
                        float diff = Math.abs(input[floatIndex] - value);
                        double relativeDiff = diff / Math.abs(original);
                        if (relativeDiff > TEST_THRESHOLD_RELATIVE) {
                            LLogger.error("floatIndex " + floatIndex + " relativeDiff " + relativeDiff);
                        }
                    });
        }
        return byteBuffer;
    }

    public ByteBuffer encode(float[] input) {
        int length = input.length;
        int nChunks = (int) Math.ceil((double) length / groupSize);
        int byteSize = nChunks * encodedBytesPerGroup();
        ByteBuffer byteBuffer = ByteBuffer.allocate(byteSize);

        // in chunks of quantSize, store first min and max as floats, and then
        // groupSize bytes encoded as n-bit integers representing linear values between min and max
        // this version encodes only to 8-bit, but the same structure can be extended to any other bit quantity
        for (int chunk = 0; chunk < nChunks; chunk++) {
            int i = chunk * groupSize;
            float min = input[i];
            float max = min;
            for (int j = 1; j < groupSize; j++) {
                if (i + j < length) {
                    if (input[i + j] < min) {
                        min = input[i + j];
                    }
                    if (input[i + j] > max) {
                        max = input[i + j];
                    }
                }
            }
            float range = max - min;
            byteBuffer.putFloat(min);
            byteBuffer.putFloat(max);
            for (int j = 0; j < groupSize; j++) {
                if (i + j < length) {
                    int value = Math.round((input[i + j] - min) / range * 255);
                    byteBuffer.put((byte) value);
                } else {
                    byteBuffer.put((byte) 0); // filler
                }
            }
        }
        byteBuffer.flip();
        return byteBuffer;
    }

    public String extension() {
        return "_" + groupSize + "_" + bits + ".quant";
    }

    @Override
    public String toString() {
        return "Quant{" +
                "groupSize=" + groupSize +
                ", bits=" + bits +
                ", compression=" + String.format("%.2f", compression() * 100.0) + "%" +
                '}';
    }
}
