package com.lastbot.llama2j;

import java.io.Closeable;
import java.io.IOException;

public class Context implements Closeable {
    final LayerAllocation layerAllocation;
    final ContextCUDA[] cudas;
    final ContextCPU cpu;

    public Context(LayerAllocation layerAllocation) throws IOException {
        this.cpu = layerAllocation.hasCPULayers() ? new ContextCPU("contextCPU0") : null;
        this.cudas = new ContextCUDA[layerAllocation.deviceCount];
        this.layerAllocation = layerAllocation;
        for (int dev = 0; dev < layerAllocation.deviceCount; dev++) {
            this.cudas[dev] = new ContextCUDA("contextCUDA" + dev, dev);
        }
    }

    @Override
    public void close() {
        if (cpu != null) {
            cpu.close();
        }
        if (cudas != null) {
            for (int i = 0; i < cudas.length; i++) {
                cudas[i].close();
            }
        }
    }
}
