package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class FastBinFileReader implements Closeable {
    private final String filePath;

    private long readSize = 0;

    public FastBinFileReader(String filePath) throws IOException {
        this.filePath = filePath;
    }

    private MappedByteBuffer read(long size) {
        try (RandomAccessFile file = new RandomAccessFile(filePath, "r");
             FileChannel channel = file.getChannel()) {
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, readSize, size);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            readSize += size;
            return buffer;
        } catch (IOException e) {
            LLogger.error("Failed to read " + String.format("%,d", size) + " bytes from " + filePath);
            return null;
        }
    }

    private static final int MAX_CHUNK_SIZE = (Integer.MAX_VALUE / 4) - 3;

    public void nextFloatArray(int count, FloatArrayProcessor processor) {
        new Thread(() -> {
            float[] data = new float[count];
            long remaining = count;
            int index = 0;

            while (remaining > 0) {
                long size = Math.min(remaining, MAX_CHUNK_SIZE);
                MappedByteBuffer buffer = read(size * 4L);
                if (buffer != null) {
                    for (int i = 0; i < size; i++) {
                        data[index] = buffer.getFloat();
                        index++;
                    }
                    remaining -= size;
                } else {
                    processor.process(null);
                    return;
                }
            }
            processor.process(data);
        }).start();
    }

    private void error() {
        throw new RuntimeException("Failed to read " + filePath);
    }

    public float nextFloat() {
        MappedByteBuffer buffer = read(4L);
        if (buffer == null) {
            error();
        }
        return buffer.getFloat();
    }

    public int nextInt() {
        MappedByteBuffer buffer = read(4L);
        if (buffer == null) {
            error();
        }
        return buffer.getInt();
    }

    public String nextString(int length) {
        MappedByteBuffer buffer = read(length);
        if (buffer == null) {
            error();
        }
        char[] buf = new char[length];
        for (int i = 0; i < length; i++) {
            buf[i] = (char) buffer.get();
        }
        return new String(buf, 0, length);
    }

    @Override
    public void close() {
        // nothing
    }
}
