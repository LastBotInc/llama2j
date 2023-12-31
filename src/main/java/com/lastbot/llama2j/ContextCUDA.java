package com.lastbot.llama2j;

import com.lastbot.llama2j.kernel.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUstream;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaError.cudaSuccess;
import static jcuda.runtime.cudaMemcpyKind.*;

/**
 * Execution context for a single CUDA device. Provides utilities such as memory, transfer, kernel,
 * and some convenience functions.
 */
public class ContextCUDA implements Closeable {
    public static final int STREAM_COUNT;
    public static final int TEST_STREAM;

    // optimized kernels for transformer
    public final Accum accum;
    public final AccumWeightedValue accumWeightedValue;
    public final ApplyRope applyRope;
    public final AttentionLoop attentionLoop;
    public final ExpSumNormalize expSumNormalize;
    public final FindMax findMax;
    public final RootMeanSquare rootMeanSquare;
    public final MatMul matMul;
    public final Normalize normalize;
    public final Silu silu;
    public final WeightNormalizeAndScale weightNormalizeAndScale;

    static {
        // Initialize the JCuda driver API
        JCuda.setExceptionsEnabled(true);

        String s = System.getenv("CUDA_DEVICE_MAX_CONNECTIONS");
        int value;
        int DEFAULT = 8;
        if (s != null) {
            try {
                value = Integer.parseInt(s.strip());
                if (value > 4 && value < 32) {
                    LLogger.info("Using CUDA_DEVICE_MAX_CONNECTIONS " + value);
                } else {
                    LLogger.info("Invalid CUDA_DEVICE_MAX_CONNECTIONS " + value + ", using default " + DEFAULT);
                    value = DEFAULT;
                }
            } catch (NumberFormatException e) {
                LLogger.info("Invalid CUDA_DEVICE_MAX_CONNECTIONS " + s + ", using default " + DEFAULT);
                value = DEFAULT;
            }
        } else {
            LLogger.info("Using default CUDA_DEVICE_MAX_CONNECTIONS " + DEFAULT);
            value = DEFAULT;
        }
        STREAM_COUNT = value;
        TEST_STREAM = STREAM_COUNT - 1;
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
        this.attentionLoop = new AttentionLoop(this);
        this.expSumNormalize = new ExpSumNormalize(this);
        this.findMax = new FindMax(this);
        this.rootMeanSquare = new RootMeanSquare(this);
        this.matMul = new MatMul(this);
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

    public Pointer allocateAndCopyToDevice(int streamId, byte[] hostArray, boolean autoFree) {
        Pointer targetDeviceArray = allocateByteArray(hostArray.length, autoFree);

        long byteSize = hostArray.length;

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(hostArray), byteSize,
                cudaMemcpyHostToDevice, streams[streamId]))) {
            return null;
        }
        return targetDeviceArray;
    }

    public QuantPointer allocateQuantAndCopyToDevice(int streamId, Quant quant,
                                                     ByteBuffer byteBuffer, int floatOffset,
                                                     int floatSize, boolean autoFree) {
        int byteSize = quant.numberOfBytesByFloatSize(floatSize);

        Pointer targetDeviceArray = allocateByteArray(byteSize, autoFree);

        int byteOffset = quant.byteOffsetByFloatIndex(floatOffset);

        Pointer hostArrayWithOffset = Pointer.to(byteBuffer).withByteOffset(byteOffset);

        setDevice();

        // Asynchronous copy from host to device
        if (isError(cudaMemcpyAsync(targetDeviceArray, hostArrayWithOffset, byteSize,
                cudaMemcpyHostToDevice, streams[streamId]))) {
            return null;
        }
        QuantPointer quantPointer = new QuantPointer(quant, targetDeviceArray, floatOffset);
        return quantPointer;
    }

    public Pointer allocateFloatArray(long elements, boolean autoFree) {
        if (elements <= Limits.ARRAY_MAX_SIZE) {
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

    public Pointer allocateByteArray(long byteSize, boolean autoFree) {
        if (byteSize <= Limits.ARRAY_MAX_SIZE) {

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

    public void memZero(int streamId, Pointer a, int index, int size) {
        Pointer aWithOffset = index == 0 ? a : a.withByteOffset((long) index * Sizeof.FLOAT);
        // sets 4 bytes to integer 0, this works because 0.0f is also all bits zero
        cudaMemsetAsync(aWithOffset, 0, (long) size * Sizeof.FLOAT, streams[streamId]);
    }

    public void copyFromHostToDevice(int streamId, float[] sourceHostArray, int floatSize, Pointer targetDeviceArray) {
        long byteSize = (long) floatSize * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        isError(cudaMemcpyAsync(targetDeviceArray, Pointer.to(sourceHostArray),
                byteSize, cudaMemcpyHostToDevice, streams[streamId]));
    }

    public void copyFromDeviceToHost(int streamId, Pointer sourceDeviceArray, int floatSize, float[] targetHostArray) {
        long byteSize = (long) floatSize * Sizeof.FLOAT;

        setDevice();

        // Asynchronous copy from device to host
        isError(cudaMemcpyAsync(Pointer.to(targetHostArray),
                sourceDeviceArray, byteSize, cudaMemcpyDeviceToHost, streams[streamId]));
    }

    public void copyFloatsFromDeviceToDevice(int streamId, QuantPointer sourceDeviceArray, long sourceIndex,
                                             Pointer targetDeviceArray, long targetIndex,
                                             long floatSize) {
        Pointer sourcePointer = sourceDeviceArray.pointer();
        int sourceOffset = Math.toIntExact(sourceIndex - sourceDeviceArray.floatOffset()) * Sizeof.FLOAT;

        copyBytesFromDeviceToDevice(streamId, sourcePointer, sourceOffset,
                targetDeviceArray, targetIndex * Sizeof.FLOAT, floatSize * Sizeof.FLOAT);
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

    public void copyFromDeviceToAnotherDevice(int sourceStreamId, Pointer sourceDeviceArray,
                                              Pointer targetDeviceArray, ContextCUDA targetContext,
                                              int targetStreamId, int floatSize, float[] hostArray) {
        if (floatSize > hostArray.length) {
            throw new RuntimeException("floatSize > hostArray.length");
        }
        setDevice();

        copyFromDeviceToHost(sourceStreamId, sourceDeviceArray, floatSize, hostArray);
        synchronizeStream(sourceStreamId);
        targetContext.copyFromHostToDevice(targetStreamId, hostArray, floatSize, targetDeviceArray);
    }

    public void synchronizeDevice() {
        setDevice();
        if (isError(cudaDeviceSynchronize())) {
            throw new RuntimeException("synchronizeDevice failed");
        }
    }

    public void synchronizeStream(int streamId) {
        setDevice();
        if (isError(cudaStreamSynchronize(streams[streamId]))) {
            throw new RuntimeException("synchronizeStream " + streamId + " failed");
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
