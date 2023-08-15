package com.lastbot.llama2j;

import com.lastbot.llama2j.kernel.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUstream;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaError.cudaSuccess;
import static jcuda.runtime.cudaMemcpyKind.*;

public class ContextCUDA implements Closeable {
    private static final int KERNEL_STREAM_COUNT = 3;

    // optimized kernels for transformer
    public final Accum accum;
    public final ExpAndSum expAndSum;
    public final FindMax findMax;
    public final SumOfSquares sumOfSquares;
    public final MatMul matMul;
    public final Normalize normalize;
    public final WeightNormalizeAndScale weightNormalizeAndScale;

    static {
        // Initialize the JCuda driver API
        JCuda.setExceptionsEnabled(true);
    }

    private final String name;
    private final int deviceId; // device id
    private final List<Pointer> memoryPointerList = new ArrayList<>();
    private final cudaStream_t transferStream;
    private final CUstream CUTransferStream;
    private final cudaStream_t[] kernelStreams = new cudaStream_t[KERNEL_STREAM_COUNT];
    private final CUstream[] CUKernelStreams = new CUstream[KERNEL_STREAM_COUNT];

    public ContextCUDA(String name, int deviceId) throws IOException {
        this.name = name;
        this.deviceId = deviceId;

        setDevice();

        this.transferStream = new cudaStream_t();
        if (isError(cudaStreamCreate(transferStream))) {
            throw new RuntimeException();
        }
        this.CUTransferStream = new CUstream(transferStream);

        for (int k = 0; k < KERNEL_STREAM_COUNT; k++) {
            this.kernelStreams[k] = new cudaStream_t();
            if (isError(cudaStreamCreate(kernelStreams[k]))) {
                throw new RuntimeException();
            }
            this.CUKernelStreams[k] = new CUstream(kernelStreams[k]);
        }
        this.accum = new Accum(this);
        this.expAndSum = new ExpAndSum(this);
        this.findMax = new FindMax(this);
        this.sumOfSquares = new SumOfSquares(this);
        this.matMul = new MatMul(this);
        this.normalize = new Normalize(this);
        this.weightNormalizeAndScale = new WeightNormalizeAndScale(this);
    }

    public void setDevice() {
        if (isError(cudaSetDevice(deviceId))) {
            throw new RuntimeException();
        }
    }

    public Pointer allocateAndCopyToDevice(float[] hostArray, boolean autoFree) {
        Pointer targetDeviceArray = allocateFloatArray(hostArray.length, autoFree);

        long byteSize = (long) hostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(hostArray), byteSize,
                cudaMemcpyHostToDevice, transferStream))) {
            return null;
        }
        return targetDeviceArray;
    }

    public Pointer allocateAndCopyToDeviceWithOffset(float[] hostArray, int offset, int size, boolean autoFree) {
        Pointer targetDeviceArray = allocateFloatArray(size, autoFree);

        Pointer hostArrayOffset = Pointer.to(hostArray).withByteOffset((long) offset * Float.BYTES);

        long byteSize = (long) (size) * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, hostArrayOffset, byteSize,
                cudaMemcpyHostToDevice, transferStream))) {
            return null;
        }
        return targetDeviceArray;
    }

    public Pointer allocateFloatArray(long elements, boolean autoFree) {
        if (elements <= Limits.FLOAT_ARRAY_MAX_SIZE) {
            long byteSize = elements * Sizeof.FLOAT;

            setDevice();

            // Create device array
            Pointer newDeviceArray = new Pointer();
            if (isError(cudaMalloc(newDeviceArray, byteSize))) {
                return null;
            }
            if (autoFree) {
                memoryPointerList.add(newDeviceArray);
            }
            return newDeviceArray;
        } else {
            return null;
        }
    }

    public Pointer allocate(long byteSize) {
        setDevice();

        // Create device array
        Pointer newDeviceArray = new Pointer();
        if (isError(cudaMalloc(newDeviceArray, byteSize))) {
            return null;
        }
        return newDeviceArray;
    }

    public void free(Pointer pointer) {
        isError(cudaFree(pointer));
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

    public boolean synchronizeKernel(int k) {
        return !isError(cudaStreamSynchronize(kernelStreams[k]));
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
            Pointer d1Pointer = context0.allocateFloatArray(TEST_SIZE, true);
            t1b = System.currentTimeMillis();
            context0.copyFromHostToDevice(d1, d1Pointer);

            t2 = System.currentTimeMillis();
            Pointer d2Pointer = context1.allocateFloatArray(TEST_SIZE, true);
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
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        LLogger.info("Failure, processing terminated with an error");
    }

    public static void main(String[] args) {
        test();
        test();
    }

    public cudaStream_t getTransferStream() {
        return transferStream;
    }

    public CUstream getCUTransferStream() {
        return CUTransferStream;
    }

    public cudaStream_t getKernelStream(int k) {
        return kernelStreams[k];
    }

    public CUstream getCUKernelStream(int k) {
        return CUKernelStreams[k];
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
        for (cudaStream_t kernelStream : kernelStreams) {
            cudaStreamDestroy(kernelStream);
        }
    }
}
