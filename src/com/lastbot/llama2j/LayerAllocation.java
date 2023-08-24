package com.lastbot.llama2j;

public class LayerAllocation {
    public final int nLayers;
    public final int deviceCount;
    public final long staticBytes;
    public final long bytesPerLayer;
    public final int[] firstLayer;
    public final int[] lastLayer;
    public final int firstCPULayer;
    public final int lastCPULayer;

    public boolean hasCPULayers() {
        return firstCPULayer >= 0;
    }

    public LayerAllocation(long[] gpuMem, Config p, Mode mode, Quant quant, boolean sharedWeights) {
        this.nLayers = p.n_layers;
        long weightBytesStatic = TransformerWeights.bytesStatic(p, sharedWeights);
        long weightBytesPerLayer = TransformerWeights.bytesPerLayer(p, quant);
        long stateStatic = RunState.bytesStatic(p);
        long statePerLayer = RunState.bytesPerLayer(p);

        LLogger.info("--------- Model Size ---------");

        LLogger.info("TransformerWeights: Static bytes " + String.format("%,d", weightBytesStatic));
        LLogger.info("TransformerWeights: Per layer bytes " + String.format("%,d", weightBytesPerLayer));

        LLogger.info("RunState: Static bytes " + String.format("%,d", stateStatic));
        LLogger.info("RunState: Per layer bytes " + String.format("%,d", statePerLayer));

        this.staticBytes = weightBytesStatic + stateStatic;
        this.bytesPerLayer = weightBytesPerLayer + statePerLayer;

        LLogger.info("One Device: Static bytes " + String.format("%,d", staticBytes));
        LLogger.info("One Device: Layer bytes " + String.format("%,d", p.n_layers * bytesPerLayer));

        if (mode == Mode.CPU) {
            this.deviceCount = 0;
            this.firstCPULayer = 0;
            this.lastCPULayer = p.n_layers  - 1;
            this.firstLayer = null;
            this.lastLayer = null;
        } else {
            this.deviceCount = gpuMem.length;

            long staticSize = deviceCount * staticBytes;
            long layerSize = p.n_layers * bytesPerLayer;

            LLogger.info(deviceCount + " Devices: Static bytes " + String.format("%,d", staticSize));
            LLogger.info(deviceCount + " Devices: layer bytes " + String.format("%,d", layerSize));

            this.firstLayer = new int[deviceCount];
            this.lastLayer = new int[deviceCount];

            LLogger.info("--------- Capacity ---------");

            long[] layerBytes = new long[deviceCount];

            long layerCapacity = 0L;
            for (int dev = 0; dev < deviceCount; dev++) {
                layerBytes[dev] = gpuMem[dev] - staticBytes;
                layerCapacity += layerBytes[dev];
            }

            LLogger.info(deviceCount + " Devices: layer capacity " + String.format("%,d", layerCapacity));

            double layerUtilization = (double) layerSize / layerCapacity;
            double LayerUtilizationPercentage = 100D * layerUtilization;

            LLogger.info(deviceCount + " Devices: layer capacity utilization " +
                    String.format("%.2f", LayerUtilizationPercentage) + "%");

            int nextLayer = 0;
            for (int dev = 0; dev < deviceCount; dev++) {
                int layersPerDevice = Math.toIntExact(Math.round(layerBytes[dev] * layerUtilization / bytesPerLayer));
                // adjust lower as needed
                while (staticBytes + layersPerDevice * bytesPerLayer > gpuMem[dev]) {
                    layersPerDevice--;
                }

                this.firstLayer[dev] = nextLayer;
                this.lastLayer[dev] = Math.min(nextLayer + layersPerDevice - 1, p.n_layers - 1);
                nextLayer += layersPerDevice;
                layerCapacity += layerBytes[dev];
            }
            if (mode == Mode.TEST) {
                firstCPULayer = 0;
                lastCPULayer = p.n_layers - 1;
            } else // if (mode == Mode.CUDA)
            {
                // this version does not allow for roll over CUDA layers to CPU
                // todo enable GPU roll over to CPU layers in CUDA mode
                firstCPULayer = -1;
                lastCPULayer = -1;
//                if (nextLayer < p.n_layers) {
//                    firstCPULayer = nextLayer;
//                    lastCPULayer = p.n_layers - 1;
//                }
            }
        }
        LLogger.info("--------- Allocation ---------");

        LLogger.info("Total layers " + nLayers);

        if (deviceCount > 0) {
            for (int dev = 0; dev < deviceCount; dev++) {
                int nLayers = lastLayer[dev] - firstLayer[dev] + 1;
                long usagePerDevice = staticBytes + nLayers * bytesPerLayer;
                double utilizationPercentagePerDevice = 100D * usagePerDevice / gpuMem[dev];
                LLogger.info("Device " + dev + ": Layers " + firstLayer[dev] + " - " + lastLayer[dev] +
                        " (" + nLayers +
                        " layers) using " + String.format("%,d", usagePerDevice) +
                        " of " + String.format("%,d", gpuMem[dev]) + " bytes" +
                        " at " + String.format("%.1f", utilizationPercentagePerDevice) + "% utilization");
            }
        }
        if (firstCPULayer >= 0) {
            int nLayers = lastCPULayer - firstCPULayer + 1;
            long usagePerDevice = staticBytes + nLayers * bytesPerLayer;
            Runtime runtime = Runtime.getRuntime();
            long allocatedMemory = runtime.totalMemory() - runtime.freeMemory();
            long presumableFreeMemory = runtime.maxMemory() - allocatedMemory;
            double utilizationPercentage = 100D * usagePerDevice / presumableFreeMemory;
            LLogger.info("CPU: Layers " + firstCPULayer + " - " + lastCPULayer +
                    " (" + nLayers +
                    " layers) using " + String.format("%,d", usagePerDevice) +
                    " of " + String.format("%,d", presumableFreeMemory) + " bytes" +
                    " at " + String.format("%.1f", utilizationPercentage) + "% utilization");
        }
    }
}
