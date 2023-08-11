package com.lastbot.llama2j;

public class LayerAllocation {
    public int deviceCount;
    public long staticBytes;
    public long bytesPerLayer;
    public int[] firstLayer;
    public int[] lastLayer;
    public int firstCPULayer = -1;
    public int lastCPULayer = -1;

    public boolean hasCPULayers() {
        return firstCPULayer >= 0;
    }

    public boolean hasGPULayers() {
        return deviceCount > 0;
    }

    public LayerAllocation(long[] gpuMem, Config config, Target target, boolean sharedWeights) {
        long weightStatic = TransformerWeights.bytesStatic(config, sharedWeights);
        long weightPerLayer = TransformerWeights.bytesPerLayer(config);
        long stateStatic = RunState.bytesStatic(config);
        long statePerLayer = RunState.bytesPerLayer(config);

        LLogger.info("--------- Model Size ---------");

        LLogger.info("TransformerWeights: Static bytes " + String.format("%,d", weightStatic));
        LLogger.info("TransformerWeights: Per layer bytes " + String.format("%,d", weightPerLayer));

        LLogger.info("RunState: Static bytes " + String.format("%,d", stateStatic));
        LLogger.info("RunState: Per layer bytes " + String.format("%,d", statePerLayer));

        this.staticBytes = weightStatic + stateStatic;
        this.bytesPerLayer = weightPerLayer + statePerLayer;

        LLogger.info("One Device: Static bytes " + String.format("%,d", staticBytes));
        LLogger.info("One Device: Layer bytes " + String.format("%,d", config.n_layers * bytesPerLayer));

        if (!target.CUDA() || gpuMem == null) {
            this.deviceCount = 0;
            this.firstCPULayer = 0;
            this.lastCPULayer = config.n_layers;
        } else {
            this.deviceCount = gpuMem.length;

            long staticSize = deviceCount * staticBytes;
            long layerSize = config.n_layers * bytesPerLayer;

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

            LLogger.info(deviceCount + " Devices: layer capacity utilization " + String.format("%.2f", layerUtilization));

            int nextLayer = 0;
            for (int dev = 0; dev < deviceCount; dev++) {
                int layersPerDevice = (int) Math.round(layerBytes[dev] * layerUtilization / bytesPerLayer);
                // adjust lower as needed
                while (staticBytes + layersPerDevice * bytesPerLayer > gpuMem[dev]) {
                    layersPerDevice--;
                }

                this.firstLayer[dev] = nextLayer;
                this.lastLayer[dev] = Math.min(nextLayer + layersPerDevice - 1, config.n_layers - 1);
                nextLayer += layersPerDevice;
                layerCapacity += layerBytes[dev];
            }
            if (nextLayer < config.n_layers) {
                firstCPULayer = nextLayer;
                lastCPULayer = config.n_layers - 1;
            }
            // zzz for testing only override
            firstCPULayer = 0;
            lastCPULayer = config.n_layers - 1;
        }
        LLogger.info("--------- Allocation ---------");

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
            Runtime runtime           = Runtime.getRuntime();
            long allocatedMemory      = runtime.totalMemory() - runtime.freeMemory();
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
