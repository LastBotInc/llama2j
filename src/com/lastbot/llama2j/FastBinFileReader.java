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

    public void nextFloatArray(int count, FloatArrayProcessor processor) {
        new Thread(() -> {
            MappedByteBuffer buffer = read(count * 4L);
            if (buffer != null) {
                float[] data = new float[count];
                for (int i = 0; i < count; i++) {
                    data[i] = buffer.getFloat();
                }
                processor.process(data);
            } else {
                processor.process(null);
            }

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

    public float nextInt() {
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
