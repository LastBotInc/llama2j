package com.lastbot.llama2j;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

/**
 * Generic writer of binary files using fast memory mapping
 */
public class BinFileWriter {
    private final String filePath;

    @SuppressWarnings("resource")
    public BinFileWriter(String filePath) {
        this.filePath = filePath;
    }

    public void write(ByteBuffer byteBuffer) throws IOException {
        try (RandomAccessFile file = new RandomAccessFile(filePath, "rw");
             FileChannel channel = file.getChannel()) {
            int bytes = Integer.BYTES + byteBuffer.remaining();
            long writtenSize = 0;
            ByteBuffer mappedBuffer = channel.map(FileChannel.MapMode.READ_WRITE, writtenSize, bytes);
            mappedBuffer.order(ByteOrder.LITTLE_ENDIAN);

            // Write the length of the buffer to the mapped byte buffer
            mappedBuffer.putInt(byteBuffer.remaining());

            // Write the data of the buffer to the mapped byte buffer
            mappedBuffer.put(byteBuffer);
        }
    }
}
