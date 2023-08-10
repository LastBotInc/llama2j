package com.lastbot.llama2j;

public class Context {
    final LayerAllocation layerAllocation;
    final ContextCUDA[] cudas;
    final ContextCPU cpu;

    public Context(LayerAllocation layerAllocation) {
        this.cpu = layerAllocation.hasCPULayers() ? new ContextCPU("contextCPU0") : null;
        this.cudas = new ContextCUDA[layerAllocation.deviceCount];
        this.layerAllocation = layerAllocation;
        for (int dev = 0; dev < layerAllocation.deviceCount; dev++) {
            this.cudas[dev] = new ContextCUDA("contextCUDA0", dev);
        }
    }
}
