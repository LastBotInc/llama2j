package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;

import java.io.File;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

public class Kernel {
    private final CUfunction function = new CUfunction();
    private final ContextCUDA cuda;

    public Kernel(ContextCUDA cuda, String cubinFileName, String functionName) {
        this.cuda = cuda;
        cuda.setDevice();

        // Load the cubin file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, cubinFileName);

        // Obtain a function pointer to the kernel function.
        cuModuleGetFunction(function, module, functionName);
    }

    public void call_2_1(Pointer p1, Pointer p2, int n, int kernelStreamId) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(p1),
                Pointer.to(p2),
                Pointer.to(new int[]{n})
        );

        // Set up the kernel launch parameters.
        int blockSizeX = 256;
        int gridSizeX = (int) Math.ceil((double) n / blockSizeX);

        // Launch the kernel function.
        cuLaunchKernel(function,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, new CUstream(cuda.getKernelStream(kernelStreamId)),  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }

    private static final String CUDA_DIR = "/usr/local/cuda";
    private static final String NVCC_PATH = CUDA_DIR + File.separator + "bin" + File.separator + "nvcc";
    private static final String CUDA_ARCHITECTURE = "compute_89";
    private static final String CUDA_CODE = "sm_89";

    public static String prepareCubinFile(String cuFileName) {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) endIndex = cuFileName.length() - 1;
        String cubinFileName = cuFileName.substring(0, endIndex + 1) + "cubin";

        ProcessBuilder processBuilder = new ProcessBuilder(
                NVCC_PATH,
                "-arch=" + CUDA_ARCHITECTURE,
                "-code=" + CUDA_CODE,
                "-cubin",
                cuFileName,
                "-o",
                cubinFileName
        );
        processBuilder.inheritIO();  // To display the output in console
        int exitVal = -1;
        Process process = null;
        try {
            process = processBuilder.start();
            exitVal = process.waitFor();
            if (exitVal != 0) {
                LLogger.error("When compiling " + cuFileName + " into " + cubinFileName + " got exit value " + exitVal);
                return null;
            }
            return cubinFileName;
        } catch (IOException | InterruptedException e) {
            LLogger.error("When compiling " + cuFileName + " into " + cubinFileName, e);
            return null;
        }
    }
}
