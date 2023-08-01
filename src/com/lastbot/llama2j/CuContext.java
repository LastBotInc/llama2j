package com.lastbot.llama2j;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;

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
    private final cudaStream_t copyStream;
    private final cudaStream_t kernelStream;

    public CuContext(String name, int deviceId, int gpuMem) {
        this.name = name;
        this.deviceId = deviceId;
        this.gpuMem = gpuMem;

        setDevice();

        this.copyStream = new cudaStream_t();
        if (isError(cudaStreamCreate(copyStream))) {
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
        long byteSize = (long) hostArray.length * Sizeof.FLOAT;

        setDevice();

        // Create device array
        Pointer targetDeviceArray = new Pointer();
        if (isError(cudaMalloc(targetDeviceArray, byteSize))) {
            return null;
        }
        memoryPointerList.add(targetDeviceArray);

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(hostArray), byteSize, cudaMemcpyHostToDevice, copyStream))) {
            return null;
        }
        return targetDeviceArray;
    }

    public boolean copyFromHostToDevice(float[] sourceHostArray, Pointer targetDeviceArray) {
        long byteSize = (long) sourceHostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        return !isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(sourceHostArray),
                byteSize, cudaMemcpyHostToDevice, copyStream));
    }

    public boolean copyFromDeviceToHost(Pointer sourceDeviceArray, float[] targetHostArray) {
        long byteSize = (long) targetHostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        return !isError(cudaMemcpyAsync(Pointer.to(targetHostArray),
                sourceDeviceArray, byteSize, cudaMemcpyDeviceToHost, copyStream));
    }

    public boolean copyFromDeviceToDevice(Pointer sourceDeviceArray, Pointer targetDeviceArray, long byteSize) {
        setDevice();

        // Asynchronous copy from device to host
        return !isError(cudaMemcpyAsync(sourceDeviceArray, targetDeviceArray, byteSize, cudaMemcpyDeviceToDevice, copyStream));
    }

    public boolean copyFromDeviceToAnotherDevice(Pointer sourceDeviceArray, Pointer targetDeviceArray,
                                                 CuContext targetContext, float[] hostArray) {
        setDevice();

        if (copyFromDeviceToHost(sourceDeviceArray, hostArray)) {
            synchronize(copyStream);
            return targetContext.copyFromHostToDevice(hostArray, targetDeviceArray);
        }
        return false;
    }

    public boolean synchronize(cudaStream_t stream) {
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

    public void test() {
        float[] d1 = new float[]{1f, 2f, 3f};
        Pointer d1Pointer = allocateAndCopyToDevice(d1);
        if (synchronize(copyStream)) {
            LLogger.info("Synchronized");
        }
    }

    public static void main(String[] args) {
        try (CuContext context = new CuContext("testContext", 0, 10)) {
            context.test();
        }
    }

    @Override
    public void close() {
        setDevice();

        for (Pointer pointer : memoryPointerList) {
            cudaFree(pointer);
        }

        // Destroy the CUDA streams
        cudaStreamDestroy(copyStream);
        cudaStreamDestroy(kernelStream);
    }
}
