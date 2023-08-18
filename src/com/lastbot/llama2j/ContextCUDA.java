package com.lastbot.llama2j;

import com.lastbot.llama2j.kernel.*;
import jcuda.CudaException;
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
    public static final int STREAM_COUNT = 16;
    public static final int TEST_STREAM = STREAM_COUNT - 1;

    // optimized kernels for transformer
    public final Accum accum;
    public final AccumWeightedValue accumWeightedValue;
    public final ApplyRope applyRope;
    public final Attention attention;
    public final ExpAndSum expAndSum;
    public final FindMax findMax;
    public final SumOfSquares sumOfSquares;
    public final MatMul matMul;
    public final MemZeroFloat memZeroFloat;
    public final Normalize normalize;
    public final Silu silu;
    public final WeightNormalizeAndScale weightNormalizeAndScale;

    static {
        // Initialize the JCuda driver API
        JCuda.setExceptionsEnabled(true);
    }

    private final String name;
    private final int deviceId; // device id
    private final List<Pointer> memoryPointerList = new ArrayList<>();
    private final cudaStream_t[] streams = new cudaStream_t[STREAM_COUNT];
    private final CUstream[] CUKernelStreams = new CUstream[STREAM_COUNT];

    public ContextCUDA(String name, int deviceId) throws IOException {
        this.name = name;
        this.deviceId = deviceId;

        setDevice();

        for (int k = 0; k < STREAM_COUNT; k++) {
            this.streams[k] = new cudaStream_t();
            if (isError(cudaStreamCreate(streams[k]))) {
                throw new RuntimeException();
            }
            this.CUKernelStreams[k] = new CUstream(streams[k]);
        }

        this.accum = new Accum(this);
        this.accumWeightedValue = new AccumWeightedValue(this);
        this.applyRope = new ApplyRope(this);
        this.attention = new Attention(this);
        this.expAndSum = new ExpAndSum(this);
        this.findMax = new FindMax(this);
        this.sumOfSquares = new SumOfSquares(this);
        this.matMul = new MatMul(this);
        this.memZeroFloat = new MemZeroFloat(this);
        this.normalize = new Normalize(this);
        this.silu = new Silu(this);
        this.weightNormalizeAndScale = new WeightNormalizeAndScale(this);
    }

    public void setDevice() {
        if (isError(cudaSetDevice(deviceId))) {
            throw new RuntimeException();
        }
    }

    public Pointer allocateAndCopyToDevice(int streamId, float[] hostArray, boolean autoFree) {
        Pointer targetDeviceArray = allocateFloatArray(hostArray.length, autoFree);

        long byteSize = (long) hostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(hostArray), byteSize,
                cudaMemcpyHostToDevice, streams[streamId]))) {
            return null;
        }
        return targetDeviceArray;
    }


    public SlicePointer allocateSliceAndCopyToDevice(int streamId, float[] hostArray, int floatOffset,
                                                     int size, boolean autoFree) {
        Pointer targetDeviceArray = allocateFloatArray(size, autoFree);

        long byteOffset = (long) floatOffset * Float.BYTES;

        Pointer hostArrayWithOffset = Pointer.to(hostArray).withByteOffset(byteOffset);

        long byteSize = (long) (size) * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, hostArrayWithOffset, byteSize,
                cudaMemcpyHostToDevice, streams[streamId]))) {
            return null;
        }
        SlicePointer slicePointer = new SlicePointer(targetDeviceArray, floatOffset, byteOffset, byteSize);
        return slicePointer;
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

    public static SlicePointer allocateAndCopyLayers(int streamId, ContextCUDA cu, float[] cpuArray,
                                                     int firstLayer, int lastLayer, int nLayers) {
        int floatOffset = layerFloatOffset(cpuArray, firstLayer, nLayers);
        int floatSize = layerFloatSize(cpuArray, firstLayer, lastLayer, nLayers);

        SlicePointer slicePointer =
                cu.allocateSliceAndCopyToDevice(streamId, cpuArray, floatOffset, floatSize, true);
        return slicePointer;
    }

    private static int layerFloatOffset(float[] cpuArray, int firstLayer, int nLayers) {
        return layerFloatOffset(cpuArray.length, firstLayer, nLayers);
    }

    private static int layerFloatSize(float[] cpuArray, int firstLayer, int lastLayer, int nLayers) {
        return layerFloatSize(cpuArray.length, firstLayer, lastLayer, nLayers);
    }

    private static int layerFloatOffset(int length, int firstLayer, int nLayers) {
        int bytesPerLayer = length / nLayers;
        int offset = firstLayer * bytesPerLayer;
        return offset;
    }

    private static int layerFloatSize(int length, int firstLayer, int lastLayer, int nLayers) {
        int bytesPerLayer = length / nLayers;
        int size = (lastLayer - firstLayer + 1) * bytesPerLayer;
        return size;
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

    public void memZero(int streamId, Pointer a, int index, int size) {
        Pointer aWithOffset = index == 0 ? a : a.withByteOffset((long) index * Sizeof.FLOAT);
        // sets 4 bytes to integer 0, this works because 0.0f is also all bits zero
        cudaMemsetAsync(aWithOffset, 0, (long) size * Sizeof.FLOAT, streams[streamId]);
    }

    public void copyFromHostToDevice(int streamId, float[] sourceHostArray, Pointer targetDeviceArray) {
        long byteSize = (long) sourceHostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(sourceHostArray),
                byteSize, cudaMemcpyHostToDevice, streams[streamId]));
    }

    public void copyFromDeviceToHost(int streamId, Pointer sourceDeviceArray, float[] targetHostArray) {
        long byteSize = (long) targetHostArray.length * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        isError(cudaMemcpyAsync(Pointer.to(targetHostArray),
                sourceDeviceArray, byteSize, cudaMemcpyDeviceToHost, streams[streamId]));
    }

    public void copyFloatsFromDeviceToDevice(int streamId, Pointer sourceDeviceArray, long sourceIndex,
                                             Pointer targetDeviceArray, long targetIndex,
                                             long floatSize) {
        copyBytesFromDeviceToDevice(streamId, sourceDeviceArray, sourceIndex * Sizeof.FLOAT,
                targetDeviceArray, targetIndex * Sizeof.FLOAT, floatSize * Sizeof.FLOAT);
    }

    private void copyBytesFromDeviceToDevice(int streamId, Pointer sourceDeviceArray, long sourceOffset,
                                            Pointer targetDeviceArray, long targetOffset,
                                            long byteSize) {
        setDevice();

        Pointer source = sourceOffset == 0 ? sourceDeviceArray : sourceDeviceArray.withByteOffset(sourceOffset);
        Pointer target = targetOffset == 0 ? targetDeviceArray : targetDeviceArray.withByteOffset(targetOffset);

        // Asynchronous copy from device to device
        isError(cudaMemcpyAsync(target, source, byteSize,
                cudaMemcpyDeviceToDevice, streams[streamId]));
    }

    public void copyBytesFromDeviceToDevice(int streamId, Pointer sourceDeviceArray, Pointer targetDeviceArray,
                                            long byteSize) {
        setDevice();

        // Asynchronous copy from device to host
        isError(cudaMemcpyAsync(targetDeviceArray, sourceDeviceArray, byteSize,
                cudaMemcpyDeviceToDevice, streams[streamId]));
    }

    public void copyFromDeviceToAnotherDevice(int sourceStreamId, Pointer sourceDeviceArray, Pointer targetDeviceArray,
                                              ContextCUDA targetContext, int targetStreamId, float[] hostArray) {
        setDevice();

        copyFromDeviceToHost(sourceStreamId, sourceDeviceArray, hostArray);
        synchronizeStream(sourceStreamId);
        targetContext.copyFromHostToDevice(targetStreamId, hostArray, targetDeviceArray);
    }

    public void synchronizeAllStreams() {
        setDevice();
        for (int streamId = 0; streamId < STREAM_COUNT; streamId++) {
            synchronizeStream(streamId);
        }
    }

    public void synchronizeStream(int streamId) {
        setDevice();
        try {
            if (isError(cudaStreamSynchronize(streams[streamId]))) {
                throw new RuntimeException("synchronizeStream " + streamId + " failed");
            }
        } catch (CudaException e) {
            throw new RuntimeException("synchronizeStream CudaException", e);
        }
    }

    private boolean isError(int result) {
        if (result != cudaSuccess) {
            String errorMessage = cudaGetErrorString(result);
            String msg = "CUDA error on context " + name + " with code " + result + "\n" +
                    errorMessage;
            throw new RuntimeException(msg);
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
            context0.copyFromHostToDevice(0, d1, d1Pointer);

            t2 = System.currentTimeMillis();
            Pointer d2Pointer = context1.allocateFloatArray(TEST_SIZE, true);
            t3 = System.currentTimeMillis();

            if (d1Pointer != null) {
                if (d2Pointer != null) {
                    context0.synchronizeStream(0);
                    t4 = System.currentTimeMillis();
                    context0.copyFromDeviceToAnotherDevice(0, d1Pointer, d2Pointer,
                            context1, 1, temp);
                    t5 = System.currentTimeMillis();
                    context1.synchronizeStream(1);
                    t6 = System.currentTimeMillis();
                    context1.copyFromDeviceToHost(1, d2Pointer, d2);
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
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        LLogger.info("Failure, processing terminated with an error");
    }

    public static void main(String[] args) {
        test();
        test();
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
        for (cudaStream_t kernelStream : streams) {
            cudaStreamDestroy(kernelStream);
        }
    }
}
