package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;

import static com.lastbot.llama2j.Limits.FLOAT_ARRAY_MAX_SIZE;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaError.cudaSuccess;
import static jcuda.runtime.cudaMemcpyKind.*;

public class CuContext implements Closeable {

    static {
        // Initialize the JCuda driver API
        JCuda.setExceptionsEnabled(true);
    }

    private final String name;
    private final int deviceId; // device id
    private final int gpuMem; // max memory usage in GB
    private final List<Pointer> memoryPointerList = new ArrayList<>();
    private final cudaStream_t transferStream;
    private final cudaStream_t kernelStream;

    public CuContext(String name, int deviceId, int gpuMem) {
        this.name = name;
        this.deviceId = deviceId;
        this.gpuMem = gpuMem;

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

    public Pointer allocateFloatArray(long floatSize) {
        long byteSize = floatSize * Sizeof.FLOAT;

        setDevice();

        // Create device array
        Pointer newDeviceArray = new Pointer();
        if (isError(cudaMalloc(newDeviceArray, byteSize))) {
            return null;
        }
        memoryPointerList.add(newDeviceArray);
        return newDeviceArray;
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
                                                 CuContext targetContext, float[] hostArray) {
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

    private static final int TEST_SIZE = FLOAT_ARRAY_MAX_SIZE;

    private static void test() {
        float[] d1 = new float[TEST_SIZE];
        float[] temp = new float[TEST_SIZE];
        float[] d2 = new float[TEST_SIZE];
        for (int i = 0; i < TEST_SIZE; i++) {
            d1[i] = i / 1000f;
        }
        long t1, t1b, t2, t3, t4, t5, t6, t7;

        try (CuContext context0 = new CuContext("context0", 0, 10);
             CuContext context1 = new CuContext("context1", 1, 10)) {

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
}
