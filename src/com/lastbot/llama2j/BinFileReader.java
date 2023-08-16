package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;

public class BinFileReader implements Closeable {
    private static final int ALIGNED_BY_4_BYTES_MAX_BUFFER_SIZE = (Integer.MAX_VALUE / 4) * 4;

    private MappedByteBuffer[] buffers;
    private int currentBuffer = 0;

    public BinFileReader(String filePath) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(filePath, "r");
             FileChannel channel = file.getChannel()) {

            long readSize = 0;
            long fileSize = file.length();

            int numberOfBuffers = (int) (fileSize / ALIGNED_BY_4_BYTES_MAX_BUFFER_SIZE + 1);
            buffers = new MappedByteBuffer[numberOfBuffers];

            for (int i = 0; i < numberOfBuffers; i++) {
                long remaining = fileSize - readSize;
                long chunkSize = Math.min(remaining, ALIGNED_BY_4_BYTES_MAX_BUFFER_SIZE);

                buffers[i] = channel.map(FileChannel.MapMode.READ_ONLY, readSize, chunkSize);
                buffers[i].order(ByteOrder.LITTLE_ENDIAN);
                readSize += chunkSize;
            }
        } catch (IOException e) {
            String s = "Failed to read bin file" + filePath;
            LLogger.error(s, e);
            throw e;
        }
    }

    public float[] nextFloatArray(int count) {
        int remaining = count;
        int index = 0;
        float[] data = new float[count];
        while (remaining > 0) {
            MappedByteBuffer buffer = buffers[currentBuffer];
            int size = Math.min(remaining, buffer.remaining() / 4);
            for (int i = 0; i < size; i++) {
                data[index++] = buffer.getFloat();
            }
            remaining -= size;
            if (buffer.remaining() == 0) {
                currentBuffer++;
            }
        }
        return data;
    }

    private void rollover(long length) {
        if (buffers[currentBuffer].remaining() < length) {
            currentBuffer++;
            if (currentBuffer >= buffers.length) {
                String m = "Tried to read " + String.format("%,d", length) + ", but all buffers exhausted";
                LLogger.error(m);
                throw new RuntimeException(m);
            }
            int remaining = buffers[currentBuffer].remaining();
            if (remaining < length) {
                String m = "Tried to read " + String.format("%,d", length) + "," +
                        "but only " + String.format("%,d", remaining) + " is remaining";
                LLogger.error(m);
                throw new RuntimeException(m);
            }
        }
    }

    public float nextFloat() {
        rollover(4);
        return buffers[currentBuffer].getFloat();
    }

    public int nextInt() {
        rollover(4);
        return buffers[currentBuffer].getInt();
    }

    public String nextString(int length, Charset charset) {
        rollover(length);
        byte[] buf = new byte[length];
        for (int i = 0; i < length; i++) {
            buf[i] = buffers[currentBuffer].get();
        }
        String s = new String(buf, charset);
        return s;
    }

    @Override
    public void close() {
        buffers = null;
    }
}
