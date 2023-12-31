package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class MatMul extends Kernel {
    public static final int BLOCK_SIZE = 4;
    public static final int TARGET_THREAD_COUNT = 4 * Runtime.getRuntime().availableProcessors();

    private final CUfunction kernelFP32;
    private final CUfunction kernelI8;

    public MatMul(ContextCUDA cuda) {
        super(cuda, "matMul");
        cuda.setDevice();
        kernelFP32 = createFP32();
        kernelI8 = createI8();
    }

    public static void callFP32(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        int threadCount = TARGET_THREAD_COUNT;
        while (d % threadCount != 0) {
            threadCount /= 2;
        }

        int sizePerThread = d / threadCount;
        CountDownLatch latch = new CountDownLatch(threadCount);
        for (int threadId = 0; threadId < threadCount; threadId++) {
            // W (d,n) @ x (n,) -> xout (d,)
            final int end = Math.min(d, (threadId + 1) * sizePerThread);
            int finalThreadId = threadId;
            Thread.ofVirtual().start(() -> {
                float val;
                int weightPos;
                for (int i = finalThreadId * sizePerThread; i < end; i++) {
                    val = 0.0f;
                    weightPos = weightIndex + i * n;
                    for (int j = 0; j < n; j++) {
                        val += w[weightPos + j] * x[j];
                    }
                    xout[i] = val;
                }
                latch.countDown();
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            LLogger.error("callFP32 was interrupted");
        }
    }

    public static void callFP32simple(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        int i;
        float val;
        int weightPos;
        for (i = 0; i < d; i++) {
            weightPos = weightIndex + i * n;
            val = 0.0f;
            for (int jj = 0; jj < n; jj++) {
                val += w[weightPos + jj] * x[jj];
            }
            xout[i] = val;
        }
    }

    public static void callI8(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
        Quant q = w.quant();
        int groupSize = q.groupSize();
        if (weightIndex % groupSize == 0 && n % groupSize == 0) {
            callI8GroupAligns(xout, x, w, weightIndex, n, d);
        } else {
            callI8GroupDoesNotAlign(xout, x, w, weightIndex, n, d);
        }
    }

    public static void callI8Single(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
        Quant q = w.quant();
        int groupSize = q.groupSize();
        if (weightIndex % groupSize == 0 && n % groupSize == 0) {
            callI8GroupAligns(xout, x, w, weightIndex, n, d);
        } else {
            callI8GroupDoesNotAlign(xout, x, w, weightIndex, n, d);
        }
    }

    public static void callI8GroupAligns(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        Quant q = w.quant();

        byte[] encoded = w.data();

        int bytesPerGroup = q.encodedBytesPerGroup();

        int groupSize = q.groupSize();

        IntStream.range(0, d).parallel().forEach(i -> {
            // W (d,n) @ x (n,) -> xout (d,)
            int weightPos;

            int startGroupIndex;
            int endGroupIndex;

            float min;
            float max;
            float rangeMultiplier;
            int groupPayloadBase;

            int index;
            float val;

            int group;
            int j;
            int limit;

            weightPos = weightIndex + i * n;
            index = 0;
            val = 0f;

            startGroupIndex = weightPos / groupSize; // round down
            endGroupIndex = (weightPos + n - 1) / groupSize;

            groupPayloadBase = startGroupIndex * bytesPerGroup + 8;
            for (group = startGroupIndex; group <= endGroupIndex; group++) {
                min = bytesToFloat(encoded, groupPayloadBase - 8);
                max = bytesToFloat(encoded, groupPayloadBase - 4);
                rangeMultiplier = (max - min) / 255f;

                limit = Math.min(weightPos + n - group * groupSize, groupSize);
                for (j = 0; j < limit; j++) {
                    val += ((encoded[groupPayloadBase + j] & 0xff) * rangeMultiplier + min) * x[index++];
                }
                groupPayloadBase += bytesPerGroup;
            }

            if (index != n) {
                throw new RuntimeException("index != n");
            }
            xout[i] = val;
        });
    }

    public static void callI8GroupDoesNotAlign(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        Quant q = w.quant();

        byte[] encoded = w.data();

        int bytesPerGroup = q.encodedBytesPerGroup();

        int groupSize = q.groupSize();

        IntStream.range(0, d).parallel().forEach(i -> {
            // W (d,n) @ x (n,) -> xout (d,)
            int weightPos;

            int startGroupIndex;
            int endGroupIndex;

            float min;
            float max;
            float rangeMultiplier;
            int groupPayloadBase;

            int index;
            float val;

            int group;
            int j;

            int jj;
            int startFloatIndex;

            weightPos = weightIndex + i * n;
            index = 0;
            val = 0f;

            startGroupIndex = weightPos / groupSize; // round down
            endGroupIndex = (weightPos + n - 1) / groupSize;

            groupPayloadBase = startGroupIndex * bytesPerGroup + 8;
            for (group = startGroupIndex; group <= endGroupIndex; group++) {
                min = bytesToFloat(encoded, groupPayloadBase - 8);
                max = bytesToFloat(encoded, groupPayloadBase - 4);
                rangeMultiplier = (max - min) / 255f;

                startFloatIndex = group * groupSize;
                for (j = 0; j < groupSize; j++) {
                    jj = startFloatIndex + j;
                    if (jj >= weightPos && jj < weightPos + n) {
                        val += ((encoded[groupPayloadBase + j] & 0xff) * rangeMultiplier + min) * x[index++];
                    }
                }
                groupPayloadBase += bytesPerGroup;
            }

            if (index != n) {
                throw new RuntimeException("index != n");
            }
            xout[i] = val;
        });
    }

    public static void callI8Simple(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        Quant q = w.quant();

        byte[] encoded = w.data();

        int weightPos;
        for (int i = 0; i < d; i++) {
            weightPos = weightIndex + i * n;
            int[] index = new int[1];
            float[] val = new float[1];
            q.decode(encoded, weightPos, n,
                    (value) -> val[0] += value * x[index[0]++]);

            if (index[0] != n) {
                throw new RuntimeException("index[0] != n");
            }
            xout[i] = val[0];
        }

    }

    public void testFP32(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        float[] copyOfXout = Arrays.copyOf(xout, xout.length);
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer pXout = cuda.allocateAndCopyToDevice(TEST_STREAM, xout, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        Pointer pw = cuda.allocateAndCopyToDevice(TEST_STREAM, w, false);
        cuda.synchronizeStream(TEST_STREAM);
        callFP32(TEST_STREAM, pXout, px, pw, weightIndex, n, d);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pXout, xout.length, xout);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pXout);
        cuda.free(px);
        cuda.free(pw);

        callFP32(copyOfXout, copyOfx, w, weightIndex, n, d);

        compareWithThreshold("MatMul.call xout ",
                xout, copyOfXout, 1e-4f);
    }

    public void testI8(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
        float[] copyOfXout = Arrays.copyOf(xout, xout.length);
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer pXout = cuda.allocateAndCopyToDevice(TEST_STREAM, xout, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        Quant q = w.quant();
        Pointer pw = cuda.allocateAndCopyToDevice(TEST_STREAM, w.data(), false);
        cuda.synchronizeStream(TEST_STREAM);
        callI8(TEST_STREAM, pXout, px, pw, q, weightIndex, n, d);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pXout, xout.length, xout);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pXout);
        cuda.free(px);
        cuda.free(pw);

        callI8(copyOfXout, copyOfx, w, weightIndex, n, d);

        compareWithThreshold("MatMul.call xout ",
                xout, copyOfXout, 1e-3f);
    }

    public void callFP32(int streamId, Pointer xout, Pointer x, SlicePointer w, int weightIndex, int n, int d) {
        Pointer wIndexed = w.withIndex(weightIndex);
        callFP32(streamId, xout, x, wIndexed, n, d);
    }

    public void callFP32(int streamId, Pointer xout, Pointer x, Pointer w, int weightIndex, int n, int d) {
        Pointer wIndexed = w.withByteOffset((long) weightIndex * Sizeof.FLOAT);
        callFP32(streamId, xout, x, wIndexed, n, d);
    }

    public void callFP32(int streamId, Pointer xout, Pointer x, Pointer w, int n, int d) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int blockSizeX = BLOCK_SIZE;
        int gridSizeX = (int) Math.ceil((double) d / blockSizeX);

//        __global__ void matMul(float* xout, float* x, float* w, int n, int d) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(xout),
                Pointer.to(x),
                Pointer.to(w),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{d})
        );

        isError(cuLaunchKernel(kernelFP32,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
    }

    public void callI8(int streamId, Pointer xout, Pointer x, QuantPointer w, int weightIndex, int n, int d) {
        Pointer encoded = w.pointer();
        Quant q = w.quant();
        if ((long) weightIndex - w.floatOffset() < 0) {
            throw new RuntimeException("(long)weightIndex - w.getFloatOffset() < 0)");
        }
        if ((long) weightIndex - w.floatOffset() > Integer.MAX_VALUE) {
            throw new RuntimeException("(long)weightIndex - w.getFloatOffset() > Integer.MAX_VALUE");
        }
        int adjustedWeightIndex = Math.toIntExact(weightIndex - w.floatOffset());
        callI8(streamId, xout, x, encoded, q, adjustedWeightIndex, n, d);
    }

    public void callI8(int streamId, Pointer xout, Pointer x, Pointer encoded, Quant q, int weightIndex, int n, int d) {
        CUstream stream = cuda.getCUKernelStream(streamId);

        int bytesPerGroup = q.encodedBytesPerGroup();
        int groupSize = q.groupSize();

        int blockSizeX = BLOCK_SIZE;
        int gridSizeX = (int) Math.ceil((double) d / blockSizeX);

//        __global__ void matMul(float* xout, float* x, float* encoded, int weightIndex,
//        int groupSize, int bytesPerGroup, int n, int d ) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(xout),
                Pointer.to(x),
                Pointer.to(encoded),
                Pointer.to(new int[]{weightIndex}),
                Pointer.to(new int[]{groupSize}),
                Pointer.to(new int[]{bytesPerGroup}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{d})
        );

        isError(cuLaunchKernel(kernelI8,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
    }

    private CUfunction createI8() {
        String code =
                """
                            struct four_bytes {
                                unsigned char b0;
                                unsigned char b1;
                                unsigned char b2;
                                unsigned char b3;
                            };

                            union __align__(4) float_int_union {
                                four_bytes bytes;
                                int intVal;
                            };

                        extern "C"
                          __global__  void matMulI8(float* xout, float* x, unsigned char* encoded, int weightIndex,
                                                   int groupSize, int bytesPerGroup, int n, int d ) {

                                int i = blockIdx.x * blockDim.x + threadIdx.x;

                                if (i < d) {
                                    int weightPos;
                            
                                    int startGroupIndex;
                                    int endGroupIndex;
                            
                                    float min;
                                    float max;
                                    float rangeMultiplier;
                                    int groupBase;
                                    unsigned char* groupPayload;
                                    int j;
                                    
                                    float_int_union converter;

                                    int start;
                                    int end;
                                    
                                    unsigned char* groupEncoded;
                            
                                    int index = 0;
                                    float val;

                                    weightPos = weightIndex + i * n;
                                    val = 0.0f;

                                    startGroupIndex = weightPos / groupSize; // round down
                                    endGroupIndex = (weightPos + n - 1) / groupSize;

                                    for (int group = startGroupIndex; group <= endGroupIndex; group++) {
                                        groupBase = group * bytesPerGroup;
                                        groupEncoded = encoded + groupBase;
                                        groupPayload = groupEncoded + 8;

                                        converter.bytes.b0 = groupEncoded[3];
                                        converter.bytes.b1 = groupEncoded[2];
                                        converter.bytes.b2 = groupEncoded[1];
                                        converter.bytes.b3 = groupEncoded[0];

                                        min = __int_as_float(converter.intVal);

                                        converter.bytes.b0 = groupEncoded[7];
                                        converter.bytes.b1 = groupEncoded[6];
                                        converter.bytes.b2 = groupEncoded[5];
                                        converter.bytes.b3 = groupEncoded[4];

                                        max = __int_as_float(converter.intVal);

                                        rangeMultiplier = (max - min) / 255.0f;

                                        start = group * groupSize < weightPos ? weightPos - group * groupSize: 0;
                                        end = groupSize <= weightPos + n - (group * groupSize) ?
                                                groupSize : weightPos + n - (group * groupSize);
                                        #pragma unroll 32
                                        for (j = start; j < end; j++) {
                                                val += (groupPayload[j] * rangeMultiplier + min) * x[index++];
                                        }
                                    }
                                    xout[i] = val;
                                }
                            }
                        """;
        return loadFromCode(code, "matMulI8");
    }

    private CUfunction createFP32() {
        String code =
                """
                            extern "C"
                            __global__ void matMulFP32(float* xout, float* x, float* w, int n, int d) {
                                int i = blockIdx.x * blockDim.x + threadIdx.x;

                                if (i < d) {
                                    float* weightPos = w + i * n;
                                    float val = 0.0f;
                                    for (int j = 0; j < n; j++) {
                                        val += weightPos[j] * x[j];
                                    }
                                    xout[i] = val;
                                }
                            }
                        """;
        return loadFromCode(code, "matMulFP32");
    }
}