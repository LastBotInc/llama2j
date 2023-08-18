package com.lastbot.llama2j;

public enum Mode {
    /**
     * Run on CPU only
     */
    CPU,

    /**
     * Divides the model across CUDA devices that are configured in the command line.
     * Automatically uses CPU if the model does not fit in the configured CUDA device memory.
     */
    CUDA,

    /**
     * Runs the model in CPU, for any kernel function runs both CPU and CUDA, and compares
     * the results. This mode is extremely (such as 100x) slow as it moves all data for any kernel
     * from CPU to CUDA and results back synchronously. The mode is intended only for
     * development as it helps to validate that CPU and CUDA kernels perform within
     * desired accuracy threshold.
     */
    TEST
}
