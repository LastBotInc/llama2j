package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import com.lastbot.llama2j.LLogger;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.runtime.JCuda.cudaGetErrorString;
import static jcuda.runtime.cudaError.cudaSuccess;

public abstract class Kernel {
    protected final boolean SYNC_KERNEL_CALLS = false;

    protected final ContextCUDA cuda;

    protected final String name;

    private static final String KERNEL_DIRECTORY = "src/cuda";

    private static final String GENERATED_CUDA_SOURCE_PREFIX = "gen_";
    private static final String CUDA_SOURCE_EXTENSION = ".cu";
    private static final String CUBIN_EXTENSION = ".cubin";

    private static final String CUDA_DIR = "/usr/local/cuda-12.0";
    private static final String NVCC_PATH = CUDA_DIR + File.separator + "bin" + File.separator + "nvcc";
    private static final String CUDA_ARCHITECTURE = "compute_89";
    private static final String CUDA_CODE = "sm_89";

    protected final int TEST_STREAM = ContextCUDA.TEST_STREAM;

    protected static CUfunction defaultFunction;

    protected Kernel(ContextCUDA cuda, String name) {
        this.cuda = cuda;
        this.name = name;
    }

    protected Kernel(ContextCUDA cuda, String name, String fileName, String functionName) throws IOException {
        this.cuda = cuda;
        this.name = name;
        defaultFunction = loadFromFile(fileName, functionName);
    }

    protected CUfunction loadFromCode(String code, String functionName) {
        String sourceFileName = GENERATED_CUDA_SOURCE_PREFIX + name + "_" + functionName + CUDA_SOURCE_EXTENSION;
        String sourceFilePath = KERNEL_DIRECTORY + File.separator + sourceFileName;
        try (PrintWriter writer = new PrintWriter(sourceFilePath)) {
            writer.println(code);
        } catch (IOException e) {
            LLogger.error("Cannot write kernel source file " + sourceFilePath);
            throw new RuntimeException(e);
        }
        try {
            return loadFromFile(sourceFileName, functionName);
        } catch (IOException e) {
            LLogger.error("Cannot load kernel file " + sourceFilePath);
            throw new RuntimeException(e);
        }
    }

    private CUfunction loadFromFile(String fileName, String functionName) throws IOException {
        String cubinFileName;
        if (fileName.endsWith(CUDA_SOURCE_EXTENSION)) {
            cuda.setDevice();
            String cuFilename = KERNEL_DIRECTORY + File.separator + fileName;
            int endIndex = cuFilename.lastIndexOf('.');
            if (endIndex == -1) endIndex = cuFilename.length() - 1;
            cubinFileName = cuFilename.substring(0, endIndex + 1) + "cubin";

            boolean needCompile = false;
            Path cuPath = Paths.get(cuFilename);
            Path cubinPath = Paths.get(cubinFileName);
            if (!Files.exists(cubinPath)) {
                needCompile = true;
            } else {
                FileTime cuTime = Files.getLastModifiedTime(cuPath);
                FileTime cubinTime = Files.getLastModifiedTime(cubinPath);
                if (cuTime.compareTo(cubinTime) > 0) {
                    needCompile = true;
                }
            }
            if (needCompile) {
                Kernel.prepareCubinFile(cuFilename, cubinFileName);
            }
        } else {
            if (fileName.endsWith(CUBIN_EXTENSION)) {
                cubinFileName = fileName;
            } else {
                throw new RuntimeException("Unknown kernel file type " + fileName);
            }
        }

        // Load the cubin file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, cubinFileName);

        CUfunction function = new CUfunction();

        // Obtain a function pointer to the kernel function.
        cuModuleGetFunction(function, module, functionName);

        return function;
    }

    private static void prepareCubinFile(String cuFileName, String cubinFileName) throws IOException {

        ProcessBuilder processBuilder = new ProcessBuilder(
                NVCC_PATH,
                "-arch=" + CUDA_ARCHITECTURE,
                "-code=" + CUDA_CODE,
                "--use_fast_math",
                "-cubin",
                cuFileName,
                "-o",
                cubinFileName
        );
        processBuilder.inheritIO();  // To display the output in console
        try {
            Process process = processBuilder.start();
            int exitVal = process.waitFor();
            if (exitVal != 0) {
                LLogger.error("When compiling " + cuFileName + " into " + cubinFileName + " got exit value " + exitVal);
                throw new RuntimeException();
            }
        } catch (InterruptedException e) {
            LLogger.error("When compiling " + cuFileName + " into " + cubinFileName, e);
            throw new RuntimeException();
        }
    }

    protected void compareWithThreshold(String function, float[] a, float[] b, float threshold) {
        if (a.length != b.length) {
            throw new RuntimeException("Compare " + function + " different lengths");
        }
        int length = a.length;
        int errors = 0;
        float maxDiff = 0f;
        for (int i = 0; i < length; i++) {
            float diff = Math.abs(a[i] - b[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
            if (diff > threshold) {
                LLogger.error("Compare " + function + "[" + i + "]: " + a[i] + " != " + b[i]);
                errors++;
            }
        }
        if (errors > 0) {
            LLogger.error("Compare " + function + " total of " + String.format("%,d", errors) + " out of " +
                    String.format("%,d", length) + " maxDiff " + maxDiff);
        }
    }

    protected boolean isError(int result) {
        if (result != cudaSuccess) {
            String errorMessage = cudaGetErrorString(result);
            LLogger.error("CUDA error on kernel " + name + " with code " + result + "\n" +
                    errorMessage);
            return true;
        }
        return false;
    }

    protected static int findNextPowerOf2(int n) {
        // Subtract 1 to handle the case where n is already a power of 2
        n = n - 1;

        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;

        return n + 1;
    }

    protected static float bytesToFloat(byte[] bytes, int index) {
        int asInt = (bytes[index + 3] & 0xFF)
                | ((bytes[index + 2] & 0xFF) << 8)
                | ((bytes[index + 1] & 0xFF) << 16)
                | ((bytes[index] & 0xFF) << 24);
        return Float.intBitsToFloat(asInt);
    }
}
