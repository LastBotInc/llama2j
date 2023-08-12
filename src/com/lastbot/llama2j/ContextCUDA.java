package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaError.cudaSuccess;
import static jcuda.runtime.cudaMemcpyKind.*;

public class ContextCUDA implements Closeable {
    static {
        // Initialize the JCuda driver API
        JCuda.setExceptionsEnabled(true);
    }

    private final String name;
    private final int deviceId; // device id
    private final List<Pointer> memoryPointerList = new ArrayList<>();
    private final cudaStream_t transferStream;
    private final cudaStream_t kernelStream;

    public ContextCUDA(String name, int deviceId) {
        this.name = name;
        this.deviceId = deviceId;

        setDevice();

        this.transferStream = new cudaStream_t();
        if (isError(cudaStreamCreate(transferStream))) {
            throw new RuntimeException();
        }
        this.kernelStream = new cudaStream_t();
        if (isError(cudaStreamCreate(kernelStream))) {
            throw new RuntimeException();
        }
    }

    private void setDevice() {
        if (isError(cudaSetDevice(deviceId))) {
            throw new RuntimeException();
        }
    }

    public Pointer allocateAndCopyToDevice(float[] hostArray) {
        Pointer targetDeviceArray = allocateFloatArray(hostArray.length);

        long byteSize = (long) hostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(hostArray), byteSize, cudaMemcpyHostToDevice, transferStream))) {
            return null;
        }
        return targetDeviceArray;
    }

    public Pointer allocateAndCopyToDeviceWithOffset(float[] hostArray, int offset, int size) {
        Pointer targetDeviceArray = allocateFloatArray(size);

        Pointer hostArrayOffset = Pointer.to(hostArray).withByteOffset((long) offset * Float.BYTES);

        long byteSize = (long) (size) * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, hostArrayOffset, byteSize, cudaMemcpyHostToDevice, transferStream))) {
            return null;
        }
        return targetDeviceArray;
    }

    public Pointer allocateFloatArray(long elements) {
        if (elements <= Limits.FLOAT_ARRAY_MAX_SIZE) {
            long byteSize = elements * Sizeof.FLOAT;

            setDevice();

            // Create device array
            Pointer newDeviceArray = new Pointer();
            if (isError(cudaMalloc(newDeviceArray, byteSize))) {
                return null;
            }
            memoryPointerList.add(newDeviceArray);
            return newDeviceArray;
        } else {
            return null;
        }
    }

    public boolean copyFromHostToDevice(float[] sourceHostArray, Pointer targetDeviceArray) {
        long byteSize = (long) sourceHostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        return !isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(sourceHostArray),
                byteSize, cudaMemcpyHostToDevice, transferStream));
    }

    public boolean copyFromDeviceToHost(Pointer sourceDeviceArray, float[] targetHostArray) {
        long byteSize = (long) targetHostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        return !isError(cudaMemcpyAsync(Pointer.to(targetHostArray),
                sourceDeviceArray, byteSize, cudaMemcpyDeviceToHost, transferStream));
    }

    public boolean copyFromDeviceToDevice(Pointer sourceDeviceArray, Pointer targetDeviceArray, long byteSize) {
        setDevice();

        // Asynchronous copy from device to host
        return !isError(cudaMemcpyAsync(sourceDeviceArray, targetDeviceArray, byteSize, cudaMemcpyDeviceToDevice, transferStream));
    }

    public boolean copyFromDeviceToAnotherDevice(Pointer sourceDeviceArray, Pointer targetDeviceArray,
                                                 ContextCUDA targetContext, float[] hostArray) {
        setDevice();

        if (copyFromDeviceToHost(sourceDeviceArray, hostArray)) {
            synchronize(transferStream);
            return targetContext.copyFromHostToDevice(hostArray, targetDeviceArray);
        }
        return false;
    }

    public boolean synchronizeTransfer() {
        return synchronize(transferStream);
    }

    private boolean synchronizeKernel() {
        return synchronize(kernelStream);
    }

    private boolean synchronize(cudaStream_t stream) {
        setDevice();
        return !isError(cudaStreamSynchronize(stream));
    }

    private boolean isError(int result) {
        if (result != cudaSuccess) {
            String errorMessage = cudaGetErrorString(result);
            LLogger.error("CUDA error on context " + name + " with code " + result + "\n" +
                    errorMessage);
            return true;
        }
        return false;
    }

    //    private static final int TEST_SIZE = FLOAT_ARRAY_MAX_SIZE;
//    private static final int TEST_SIZE = 576_512; // estimated for LLama 2, 7B
    private static final int TEST_SIZE = 2 * 576_512; // estimated for LLama 2, 70B; dim = 4096, seq_len = 4096

    private static void test() {
        float[] d1 = new float[TEST_SIZE];
        float[] temp = new float[TEST_SIZE];
        float[] d2 = new float[TEST_SIZE];
        for (int i = 0; i < TEST_SIZE; i++) {
            d1[i] = i / 1000f;
        }
        long t1, t1b, t2, t3, t4, t5, t6, t7;

        try (ContextCUDA context0 = new ContextCUDA("context0", 0);
             ContextCUDA context1 = new ContextCUDA("context1", 1)) {

            t1 = System.currentTimeMillis();
            Pointer d1Pointer = context0.allocateFloatArray(TEST_SIZE);
            t1b = System.currentTimeMillis();
            context0.copyFromHostToDevice(d1, d1Pointer);

            t2 = System.currentTimeMillis();
            Pointer d2Pointer = context1.allocateFloatArray(TEST_SIZE);
            t3 = System.currentTimeMillis();

            if (d1Pointer != null) {
                if (d2Pointer != null) {
                    if (context0.synchronizeTransfer()) {
                        t4 = System.currentTimeMillis();
                        if (context0.copyFromDeviceToAnotherDevice(d1Pointer, d2Pointer, context1, temp)) {
                            t5 = System.currentTimeMillis();
                            if (context1.synchronizeTransfer()) {
                                t6 = System.currentTimeMillis();
                                if (context1.copyFromDeviceToHost(d2Pointer, d2)) {
                                    t7 = System.currentTimeMillis();
                                    int errorCount = 0;
                                    for (int i = 0; i < TEST_SIZE; i++) {
                                        if (d1[i] != d2[i]) {
                                            LLogger.error(i + "d1 " + d1[i] + ", d2 " + d2[i]);
                                            errorCount++;
                                        }
                                    }
                                    double gigabytes = ((double) TEST_SIZE * Float.BYTES) / 1024 / 1024 / 1024;

                                    LLogger.info(String.format("%,.0f", gigabytes) + " GB");

                                    LLogger.time("allocateFloatArray", t1, t1b);
                                    LLogger.time("allocateAndCopyToDevice", t1b, t2);
                                    LLogger.info("host to device " + String.format("%,.1f",
                                            (gigabytes / ((t2 - t1b) / 1000D))) + " GB per second");

                                    LLogger.time("allocateFloatArray", t2, t3);
                                    LLogger.time("context0.synchronizeTransfer", t3, t4);
                                    LLogger.time("context0.copyFromDeviceToAnotherDevice", t4, t5);
                                    LLogger.info("device to device " + String.format("%,.1f",
                                            (gigabytes / ((t5 - t4) / 1000D))) + " GB per second");

                                    LLogger.time("context1.synchronizeTransfer()", t5, t6);
                                    LLogger.time("context1.copyFromDeviceToHost", t6, t7);

                                    if (errorCount == 0) {
                                        LLogger.info("Success");
                                    } else {
                                        LLogger.info("Failure with " + String.format("%,d", errorCount) + " errors");
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
        LLogger.info("Failure, processing terminated with an error");
    }

    public static void main(String[] args) {
        test();
        test();
    }

    @Override
    public void close() {
        LLogger.debug("Closing context " + name);
        setDevice();

        for (Pointer pointer : memoryPointerList) {
            cudaFree(pointer);
        }

        // Destroy the CUDA streams
        cudaStreamDestroy(transferStream);
        cudaStreamDestroy(kernelStream);
    }

    public class JCudaVectorAddition {
        public static void main(String args[]) {
            // Initialize the driver and create a context for the first device.
            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);

            // Load the ptx file.
            CUmodule module = new CUmodule();
            String ptxFileName = preparePtxFile("vectorAddition.cu");
            cuModuleLoad(module, ptxFileName);

            // Obtain a function pointer to the kernel function.
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, "add");

            // Allocate and set the device input data.
            int n = 50000;
            float hostInputA[] = new float[n];
            float hostInputB[] = new float[n];
            for (int i = 0; i < n; i++) {
                hostInputA[i] = (float) i;
                hostInputB[i] = (float) i;
            }
            CUdeviceptr deviceInputA = new CUdeviceptr();
            cuMemAlloc(deviceInputA, n * Sizeof.FLOAT);
            cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA), n * Sizeof.FLOAT);

            CUdeviceptr deviceInputB = new CUdeviceptr();
            cuMemAlloc(deviceInputB, n * Sizeof.FLOAT);
            cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB), n * Sizeof.FLOAT);

            // Allocate device output memory.
            CUdeviceptr deviceOutput = new CUdeviceptr();
            cuMemAlloc(deviceOutput, n * Sizeof.FLOAT);

            // Create and set up the kernel parameters.
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[]{n}),
                    Pointer.to(deviceInputA),
                    Pointer.to(deviceInputB),
                    Pointer.to(deviceOutput)
            );

            // Set up the kernel launch parameters.
            int blockSizeX = 256;
            int gridSizeX = (int) Math.ceil((double) n / blockSizeX);
            CUstream stream = new CUstream();
            cuStreamCreate(stream, 0);

            // Launch the kernel function.
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, stream,             // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            // Copy the device output to the host.
            float hostOutput[] = new float[n];
            cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, n * Sizeof.FLOAT);

            // Cleanup.
            cuMemFree(deviceInputA);
            cuMemFree(deviceInputB);
            cuMemFree(deviceOutput);
            cuCtxDestroy(context);

            // Print out the result.
            for (int i = 0; i < n; i++) {
                System.out.println(hostOutput[i]);
            }
        }

        private static String preparePtxFile(String cuFileName) {
            int endIndex = cuFileName.lastIndexOf('.');
            if (endIndex == -1) endIndex = cuFileName.length() - 1;
            String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";

            String modelString = "-m" + System.getProperty("sun.arch.data.model");
            String cmd = "nvcc " + modelString + " -ptx " + cuFileName + " -o " + ptxFileName;

            System.out.println("Executing:\n" + cmd);
            try {
                Process process = Runtime.getRuntime().exec(cmd);
//                String errorMessage =
//                        new String(JCudaDriver.toByteArray(process.getErrorStream()));
//                String outputMessage =
//                        new String(JCudaDriver.toByteArray(process.getInputStream()));
                int exitValue = 0;
                try {
                    exitValue = process.waitFor();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException(
                            "Interrupted while waiting for nvcc output", e);
                }

                if (exitValue != 0) {
                    System.out.println("nvcc process exitValue " + exitValue);
//                    System.out.println("errorMessage:\n"+errorMessage);
//                    System.out.println("outputMessage:\n"+outputMessage);
//                    throw new IOException(
//                            "Could not create .ptx file: "+errorMessage);
                }
            } catch (IOException e) {
                throw new RuntimeException(
                        "Could not create .ptx file", e);
            }

            return ptxFileName;
        }
    }
}
