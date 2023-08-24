package com.lastbot.llama2j;

import jcuda.Pointer;

import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CountDownLatch;

import static com.lastbot.llama2j.TransformerWeights.QuantAllocationPolicy.*;

public class TransformerWeights {
    public enum QuantAllocationPolicy {
        SPLIT_TO_LAYERS,
        FIRST_DEVICE_ONLY,
        LAST_DEVICE_ONLY,
        ALL_DEVICES
    }

    private final Quant quant;

    float[] token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    QuantArray l_rms_att_weight; // (layer, dim) rmsnorm weights
    QuantArray l_rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantArray l_wq; // (layer, dim, n_heads * head_size)
    QuantArray l_wk; // (layer, dim, n_kv_heads * head_size)
    QuantArray l_wv; // (layer, dim, n_kv_heads * head_size)
    QuantArray l_wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantArray l_w1; // (layer, hidden_dim, dim)
    QuantArray l_w2; // (layer, dim, hidden_dim)
    QuantArray l_w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    QuantArray rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float[] wcls;

    public static void writeArray(DataOutputStream dos, float[] d) throws IOException {
        dos.writeInt(d.length);
        for (float v : d) {
            dos.writeFloat(v);
        }
    }

    public TransformerWeights(Context c, BinFileReader reader, Config p, Quant quant, boolean sharedWeights)
            throws IOException {
        this.quant = quant;
        // token embedding table
        int head_size = p.dim / p.n_heads;
        // always read weights first to CPU memory
        long t0 = System.currentTimeMillis();
        token_embedding_table = reader.nextFloatArray(p.vocab_size * p.dim);

        CountDownLatch latch = new CountDownLatch(10);

        readAsQuant("l_rms_att_weight", reader, c, p.n_layers * p.dim, SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_rms_att_weightCU = cudaPointers;
                    l_rms_att_weight = cpuArray;
                });
        readAsQuant("l_wq", reader, c, p.n_layers * p.dim * (p.n_heads * head_size), SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_wqCU = cudaPointers;
                    l_wq = cpuArray;
                });

        readAsQuant("l_wk", reader, c, p.n_layers * p.dim * (p.n_kv_heads * head_size), SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_wkCU = cudaPointers;
                    l_wk = cpuArray;
                });

        readAsQuant("l_wv", reader, c, p.n_layers * p.dim * (p.n_kv_heads * head_size), SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_wvCU = cudaPointers;
                    l_wv = cpuArray;
                });

        readAsQuant("l_wo", reader, c, p.n_layers * (p.n_heads * head_size) * p.dim, SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_woCU = cudaPointers;
                    l_wo = cpuArray;
                });

        readAsQuant("l_rms_ffn_weight", reader, c, p.n_layers * p.dim, SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_rms_ffn_weightCU = cudaPointers;
                    l_rms_ffn_weight = cpuArray;
                });

        readAsQuant("l_w1", reader, c, p.n_layers * p.dim * p.hidden_dim, SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_w1CU = cudaPointers;
                    l_w1 = cpuArray;
                });

        readAsQuant("l_w2", reader, c, p.n_layers * p.hidden_dim * p.dim, SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_w2CU = cudaPointers;
                    l_w2 = cpuArray;
                });

        readAsQuant("l_w3", reader, c, p.n_layers * p.dim * p.hidden_dim, SPLIT_TO_LAYERS, latch,
                (cudaPointers, cpuArray) -> {
                    l_w3CU = cudaPointers;
                    l_w3 = cpuArray;
                });

        readAsQuant("rms_final_weight", reader, c, p.dim, LAST_DEVICE_ONLY, latch,
                (cudaPointers, cpuArray) -> {
                    rms_final_weightCU = cudaPointers;
                    rms_final_weight = cpuArray;
                });

        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        reader.skipFloats(p.seq_len * head_size / 2); // freq_cis_real
        reader.skipFloats(p.seq_len * head_size / 2); // freq_cis_imag

        wcls = sharedWeights ? token_embedding_table : reader.nextFloatArray(p.vocab_size * p.dim);

        // copy non quant (FP32) weights to CUDA
        if (c.layerAllocation.deviceCount > 0) {
            int n = c.layerAllocation.deviceCount;
            token_embedding_tableCU = new Pointer[n];
            wclsCU = new Pointer[n];

            for (int dev = 0; dev < c.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = c.cudas[dev];
                token_embedding_tableCU[dev] = cu.allocateAndCopyToDevice(0, token_embedding_table, true);
                wclsCU[dev] = sharedWeights ? token_embedding_tableCU[dev] :
                        cu.allocateAndCopyToDevice(0, wcls, true);
            }
            // make sure all devices are sync
            for (int dev = 0; dev < c.layerAllocation.deviceCount; dev++) {
                ContextCUDA cu = c.cudas[dev];
                cu.synchronizeDevice();
            }
            long t1 = System.currentTimeMillis();
            LLogger.time("Created TransformerWeights", t0, t1);
        }
    }

    public static long bytesStatic(Config config, boolean sharedWeights) {
        int head_size = config.dim / config.n_heads;
        return (long) Float.BYTES * (
                ((long) config.vocab_size * config.dim) +
                        ((long) config.dim) +
                        ((long) config.seq_len * head_size / 2) +
                        ((long) config.seq_len * head_size / 2) +
                        (sharedWeights ? 0L : (long) config.vocab_size * config.dim)
        );
    }

    public static long bytesPerLayer(Config config, Quant quant) {
        // average estimate
        // todo zzz implement accurate calculation based on the dimensions of each array
        double c = quant.compression();
        return (long) Float.BYTES * (
                ((long) config.dim) +
                        ((long) (config.dim * config.dim * c)) +
                        ((long) (config.dim * config.dim * c)) +
                        ((long) (config.dim * config.dim * c)) +
                        ((long) (config.dim * config.dim * c)) +
                        ((long) (config.dim * c)) +
                        ((long) (config.dim * config.hidden_dim * c)) +
                        ((long) (config.hidden_dim * config.dim * c)) +
                        ((long) (config.dim * config.hidden_dim * c))
        );
    }

    Pointer[] token_embedding_tableCU;    // (vocab_size, dim)
    QuantPointer[] l_rms_att_weightCU; // (layer, dim) rmsnorm weights
    QuantPointer[] l_rms_ffn_weightCU; // (layer, dim)
    QuantPointer[] l_wqCU; // (layer, dim, dim)
    QuantPointer[] l_wkCU; // (layer, dim, dim)
    QuantPointer[] l_wvCU; // (layer, dim, dim)
    QuantPointer[] l_woCU; // (layer, dim, dim)
    QuantPointer[] l_w1CU; // (layer, hidden_dim, dim)
    QuantPointer[] l_w2CU; // (layer, dim, hidden_dim)
    QuantPointer[] l_w3CU; // (layer, hidden_dim, dim)
    QuantPointer[] rms_final_weightCU; // (dim,)
    Pointer[] wclsCU;

    private interface QuantReadProcessor {
        void process(QuantPointer[] cudaPointers, QuantArray cpuArray);
    }

    private void readAsQuant(String name, BinFileReader reader, Context c, int floatSize, QuantAllocationPolicy policy,
                             CountDownLatch latch, QuantReadProcessor processor)
            throws IOException {
        String quantCache = reader.getDirectory() + File.separator + reader.getBaseName() + "_" + name + quant.extension();
        Path quantCachePath = Paths.get(quantCache);

        float[] data;
        if (!Files.exists(quantCachePath)) {
            data = reader.nextFloatArray(floatSize);
        } else {
            data = null;
            reader.skipFloats(floatSize);
        }
        Thread.ofVirtual().start(() -> {
            try {
                ByteBuffer byteBuffer;
                if (data != null) {
                    byteBuffer = quant.encode(data);
                    LLogger.info("Writing quant cache file " + quantCache);
                    BinFileWriter writer = new BinFileWriter(quantCache);
                    try {
                        writer.write(byteBuffer);
                        byteBuffer.rewind();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                } else {
                    LLogger.info("Reading quant cache file " + quantCache);
                    try (BinFileReader cacheReader = new BinFileReader(quantCache)) {
                        int size = cacheReader.nextInt();
                        byteBuffer = cacheReader.nextByteBufferByByteCount(size);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }
                LLogger.info("Reading " + name);

                int totalByteSize = byteBuffer.remaining();
                int expectedTotalByteSize = quant.numberOfBytesByFloatSize(floatSize);
                if (totalByteSize != expectedTotalByteSize) {
                    throw new RuntimeException("totalByteSize != quant.numberOfBytesByFloatSize(floatSize)");
                }
                int totalLayers = c.layerAllocation.nLayers;

                QuantPointer[] quantPointers;
                QuantArray quantArray;

                if (policy == SPLIT_TO_LAYERS) {
                    if (c.layerAllocation.deviceCount > 0) {
                        int n = c.layerAllocation.deviceCount;
                        quantPointers = new QuantPointer[n];
                        for (int dev = 0; dev < n; dev++) {
                            ContextCUDA cu = c.cudas[dev];

                            int firstLayer = c.layerAllocation.firstLayer[dev];
                            int lastLayer = c.layerAllocation.lastLayer[dev];
                            int devLayers = lastLayer - firstLayer + 1;
                            int devFloatOffset = (int) ((long) floatSize * firstLayer / totalLayers);
                            int devFloatSize = (int) ((long) floatSize * devLayers / totalLayers);

                            quantPointers[dev] = cu.allocateQuantAndCopyToDevice(0, quant, byteBuffer,
                                    devFloatOffset, devFloatSize, true);
                        }
                    } else {
                        quantPointers = null;
                    }
                    quantArray = null;
                    if (c.layerAllocation.hasCPULayers()) {
                        int firstLayer = c.layerAllocation.firstCPULayer;
                        int lastLayer = c.layerAllocation.lastCPULayer;
                        int cpuLayers = lastLayer - firstLayer + 1;
                        int cpuFloatOffset = (int) ((long) floatSize * firstLayer / totalLayers);
                        int cpuByteOffset = (int) ((long) totalByteSize * firstLayer / totalLayers);
                        int cpuByteSize = (int) ((long) totalByteSize * cpuLayers / totalLayers);

                        byte[] cpuData = new byte[cpuByteSize];
                        byteBuffer.rewind();
                        int remaining = byteBuffer.remaining();
                        if (remaining != cpuByteSize) {
                            throw new RuntimeException("remaining != cpuByteSize");
                        }
                        byteBuffer.get(cpuData, cpuByteOffset, cpuByteSize);
                        quantArray = new QuantArray(quant, cpuData, cpuFloatOffset);
                    }
                } else if (policy == LAST_DEVICE_ONLY) {
                    if (c.layerAllocation.deviceCount > 0) {
                        int n = c.layerAllocation.deviceCount;
                        quantPointers = new QuantPointer[n];
                        int dev = n - 1;
                        ContextCUDA cu = c.cudas[dev];

                        quantPointers[dev] = cu.allocateQuantAndCopyToDevice(0, quant, byteBuffer,
                                0, floatSize, true);
                    } else {
                        quantPointers = null;
                    }
                    quantArray = null;
                    if (c.layerAllocation.hasCPULayers()) {
                        int firstLayer = c.layerAllocation.firstCPULayer;
                        int lastLayer = c.layerAllocation.lastCPULayer;
                        if (lastLayer == totalLayers - 1) {
                            int cpuLayers = lastLayer - firstLayer + 1;
                            int cpuFloatOffset = (int) ((long) floatSize * firstLayer / totalLayers);
                            int cpuByteOffset = (int) ((long) totalByteSize * firstLayer / totalLayers);
                            int cpuByteSize = (int) ((long) totalByteSize * cpuLayers / totalLayers);

                            byte[] cpuData = new byte[cpuByteSize];
                            byteBuffer.rewind();
                            byteBuffer.get(cpuData, cpuByteOffset, cpuByteSize);
                            quantArray = new QuantArray(quant, cpuData, cpuFloatOffset);
                        }
                    }
                } else if (policy == FIRST_DEVICE_ONLY) {
                    if (c.layerAllocation.deviceCount > 0) {
                        int n = c.layerAllocation.deviceCount;
                        quantPointers = new QuantPointer[n];
                        int dev = 0;
                        ContextCUDA cu = c.cudas[dev];

                        quantPointers[dev] = cu.allocateQuantAndCopyToDevice(0, quant, byteBuffer,
                                0, floatSize, true);
                    } else {
                        quantPointers = null;
                    }
                    quantArray = null;
                    if (c.layerAllocation.hasCPULayers()) {
                        int firstLayer = c.layerAllocation.firstCPULayer;
                        int lastLayer = c.layerAllocation.lastCPULayer;
                        if (firstLayer == 0) {
                            int cpuLayers = lastLayer - firstLayer + 1;
                            int cpuFloatOffset = (int) ((long) floatSize * firstLayer / totalLayers);
                            int cpuByteOffset = (int) ((long) totalByteSize * firstLayer / totalLayers);
                            int cpuByteSize = (int) ((long) totalByteSize * cpuLayers / totalLayers);

                            byte[] cpuData = new byte[cpuByteSize];
                            byteBuffer.rewind();
                            byteBuffer.get(cpuData, cpuByteOffset, cpuByteSize);
                            quantArray = new QuantArray(quant, cpuData, cpuFloatOffset);
                        }
                    }
                } else if (policy == ALL_DEVICES) {
                    if (c.layerAllocation.deviceCount > 0) {
                        int n = c.layerAllocation.deviceCount;
                        quantPointers = new QuantPointer[n];
                        for (int dev = 0; dev < n; dev++) {
                            ContextCUDA cu = c.cudas[dev];

                            quantPointers[dev] = cu.allocateQuantAndCopyToDevice(0, quant, byteBuffer,
                                    0, floatSize, true);
                        }
                    } else {
                        quantPointers = null;
                    }
                    quantArray = null;
                    if (c.layerAllocation.hasCPULayers()) {
                        byte[] cpuData = new byte[totalByteSize];
                        byteBuffer.rewind();
                        byteBuffer.get(cpuData, 0, totalByteSize);
                        quantArray = new QuantArray(quant, cpuData, 0);
                    }
                } else {
                    throw new RuntimeException("Policy " + policy + " not implemented");
                }
                processor.process(quantPointers, quantArray);
            } finally {
                latch.countDown();
            }
        });
    }

}
