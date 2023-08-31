package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;

/**
 * Execution context with ContextCPU to manage CPU memory allocations, and optional number of
 * ContextCUDA, which each provides memory, transfer, kernel, and some convenience functions
 * specific to one CUDA device.
 */
public class Context implements Closeable {
    final LayerAllocation layerAllocation;
    final ContextCUDA[] cudas;
    final ContextCPU cpu;

    public Context(LayerAllocation layerAllocation) throws IOException {
        this.cpu = new ContextCPU();
        this.cudas = new ContextCUDA[layerAllocation.deviceCount];
        this.layerAllocation = layerAllocation;
        for (int dev = 0; dev < layerAllocation.deviceCount; dev++) {
            this.cudas[dev] = new ContextCUDA("contextCUDA" + dev, dev);
        }
    }

    public ContextCUDA lastCuda() {
        return cudas[cudas.length - 1];
    }

    @Override
    public void close() {
        cpu.close();
        for (ContextCUDA cuda : cudas) {
            cuda.close();
        }
    }
}
