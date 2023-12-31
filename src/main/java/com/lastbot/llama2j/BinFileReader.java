package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Generic reader for binary files of binary files using fast memory mapping,
 * with convenience functions for arrays of base types
 */
public class BinFileReader implements Closeable {
    private final String filePath;
    private long readSize = 0;
    private final FileChannel channel;

    /**
     * Byte order of data files
     */
    private final ByteOrder BYTE_ORDER = ByteOrder.LITTLE_ENDIAN;
    
    @SuppressWarnings("resource")
    public BinFileReader(String filePath) throws IOException {
        this.filePath = filePath;
        RandomAccessFile file = new RandomAccessFile(filePath, "r");
        this.channel = file.getChannel();
    }

    public String getDirectory() {
        Path path = Paths.get(filePath);
        Path parentDir = path.getParent();
        return (parentDir != null) ? parentDir.toString() : null;
    }

    public String getBaseName() {
        Path path = Paths.get(filePath);
        String fileName = path.getFileName().toString();
        int dotIndex = fileName.lastIndexOf('.');
        if (dotIndex > 0) {
            return fileName.substring(0, dotIndex);
        }
        return fileName;
    }

    public List<MappedByteBuffer> nextByteBufferByFloatCount(int floatCount) throws IOException {
        long bytes = (long) floatCount * Float.BYTES;
        return nextByteBuffer(bytes);
    }

    public List<MappedByteBuffer> nextByteBufferByIntCount(int intCount) throws IOException {
        long bytes = (long) intCount * Integer.BYTES;
        return nextByteBuffer(bytes);
    }

    private List<MappedByteBuffer> nextByteBuffer(long bytes) throws IOException {

        List<MappedByteBuffer> buffers = new ArrayList<>();

        long remaining = bytes;
        while (remaining > 0) {
            int size = Math.toIntExact(Math.min(Limits.ARRAY_MAX_SIZE, remaining));
            try {
                MappedByteBuffer byteBuffer = channel.map(FileChannel.MapMode.READ_ONLY, readSize, size);
                byteBuffer.order(BYTE_ORDER);
                buffers.add(byteBuffer);
                readSize += size;
                remaining -= size;
            } catch (IllegalArgumentException e) {
                LLogger.error("channel.map", e);
            }
        }
        return buffers;
    }

    public void skipFloats(int floatSize) {
        readSize += (long) floatSize * (long) Float.BYTES;
    }

    public ByteBuffer nextByteBufferByByteCount(int bytes) throws IOException {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bytes);
        MappedByteBuffer mappedByteBuffer = channel.map(FileChannel.MapMode.READ_ONLY, readSize, bytes);
        mappedByteBuffer.order(BYTE_ORDER);
        byteBuffer.order(BYTE_ORDER);
        byteBuffer.put(mappedByteBuffer);
        byteBuffer.flip();
        return byteBuffer;
    }

    public float[] nextFloatArray(int floatCount) throws IOException {
        List<MappedByteBuffer> buffers = nextByteBufferByFloatCount(floatCount);

        long totalBytes = 0;
        for (MappedByteBuffer buffer : buffers) {
            totalBytes += buffer.remaining();
        }

        if (totalBytes / Float.BYTES > Limits.ARRAY_MAX_SIZE) {
            throw new RuntimeException("totalBytes > Limits.BYTE_ARRAY_MAX_SIZE");
        }

        if (totalBytes % Float.BYTES != 0) {
            throw new RuntimeException("totalBytes % Float.BYTES != 0");
        }
        int totalFloats = Math.toIntExact(totalBytes / Float.BYTES);

        if (totalFloats != floatCount) {
            throw new RuntimeException("totalFloats != floatCount");
        }

        float[] floatArray = new float[floatCount];

        int arrayPos = 0;
        for (ByteBuffer byteBuffer : buffers) {
            FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
            int floatSize = floatBuffer.remaining();
            floatBuffer.get(floatArray, arrayPos, floatSize);
            arrayPos += floatSize;
        }
        return floatArray;
    }

    public int[] nextIntArray(int intCount) throws IOException {
        List<MappedByteBuffer> buffers = nextByteBufferByIntCount(intCount);

        long totalBytes = 0;
        for (MappedByteBuffer buffer : buffers) {
            totalBytes += buffer.remaining();
        }

        if (totalBytes / Integer.BYTES > Limits.ARRAY_MAX_SIZE) {
            throw new RuntimeException("totalBytes > Limits.BYTE_ARRAY_MAX_SIZE");
        }

        if (totalBytes % Integer.BYTES != 0) {
            throw new RuntimeException("totalBytes % Int.BYTES != 0");
        }
        int totalInts = Math.toIntExact(totalBytes / Integer.BYTES);

        if (totalInts != intCount) {
            throw new RuntimeException("totalInts != intCount");
        }

        int[] intArray = new int[totalInts];

        int arrayPos = 0;
        for (ByteBuffer byteBuffer : buffers) {
            IntBuffer intBuffer = byteBuffer.asIntBuffer();
            int intSize = intBuffer.remaining();
            intBuffer.get(intArray, arrayPos, intSize);
            arrayPos += intSize;
        }
        return intArray;
    }

    public float nextFloat() throws IOException {
        float[] floatArray = nextFloatArray(1);
        return floatArray[0];
    }

    public int nextInt() throws IOException {
        int[] intArray = nextIntArray(1);
        return intArray[0];
    }

    public String nextString(int length, Charset charset) throws IOException {
        List<MappedByteBuffer> buffers = nextByteBuffer(length);
        if (buffers.size() != 1) {
            throw new RuntimeException("nextString(): buffers.size() != 1");
        }
        MappedByteBuffer byteBuffer = buffers.get(0);
        String s = charset.decode(byteBuffer).toString();
        return s;
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
