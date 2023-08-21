package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class MatMul extends Kernel {
    private final CUfunction kernelFP32;
    private final CUfunction kernelI8;

    public static final int THREAD_COUNT = 32;

    public MatMul(ContextCUDA cuda) {
        super(cuda, "matMul");
        cuda.setDevice();
        kernelFP32 = createFP32();
        kernelI8 = createI8();
    }

    public static void callFP32(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        int sizePerThread = d / THREAD_COUNT;
        CountDownLatch latch = new CountDownLatch(THREAD_COUNT);
        for (int threadId = 0; threadId < THREAD_COUNT; threadId++) {
            // W (d,n) @ x (n,) -> xout (d,)
            final int end = Math.min(d, (threadId + 1) * sizePerThread);
            int finalThreadId = threadId;
            Thread.ofVirtual().start(() -> {
                try {
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
                } finally {
                    latch.countDown();
                }
            });
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            LLogger.error("fastMatmul was interrupted");
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
        // W (d,n) @ x (n,) -> xout (d,)
        Quant q = w.getQuant();

        byte[] encoded = w.getByteArray();

        int weightPos;
        int groupSize = q.groupSize();
        for (int i = 0; i < d; i++) {
            weightPos = weightIndex + i * n;
            float min;
            float max;
            float range;
            int groupBase;
            int groupPayloadBase;
            int startGroupIndex = q.groupIndexByFloatIndex(weightPos);
            int endGroupIndex = q.groupIndexByFloatIndex(weightPos + n - 1);

            float val = 0.0f;

            int index;
            int count = 0;

            for (int group = startGroupIndex; group <= endGroupIndex; group++) {
                groupBase = group * q.encodedBytesPerGroup();
//                LLogger.debug("groupBase " + groupBase + " encoded.length " + encoded.length);
                groupPayloadBase = groupBase + 8;
                min = bytesToFloat(encoded, groupBase);
                max = bytesToFloat(encoded, groupBase + 4);
                range = max - min;

                int startFloatIndex = group * groupSize;
                for (int j = 0; j < groupSize; j++) {
                    index = startFloatIndex + j;
                    if (index >= weightPos && index < weightPos + n) {
//                        LLogger.debug("group " + group + " index " + index);
                        int byteValue = encoded[groupPayloadBase + j] & 0xff;
                        float value = byteValue / 255f * range + min;
                        val += value * x[count];
                        count++;
                    }
                }
            }

            if (count != n) {
                throw new RuntimeException("count != n");
            }
            xout[i] = val;
        }
    }

//    public static void callI8(float[] xout, float[] x, QuantArray w, int weightIndex, int n, int d) {
//        // W (d,n) @ x (n,) -> xout (d,)
//        Quant q = w.getQuant();
//        int numberOfGroupsPerI = q.numberOfGroupsByFloatSize(n);
//
//        int i;
//        int j;
//        float min;
//        float max;
//        float range;
//        float val;
//        int jj; // effective j (0..n) over chunks
//        byte[] data = w.getByteArray();
//        int groupSize = q.groupSize();
//        int nGroupBytes = q.encodedBytesPerGroup();
//        int groupPayloadBase;
//
//        for (i = 0; i < d; i++) {
//            val = 0.0f;
//            jj = 0;
//            for (int group = 0; group < numberOfGroupsPerI; group++) {
//                int groupBase = w.byteOffsetByFloatIndex(weightIndex + i * n) + group * nGroupBytes;
//                min = bytesToFloat(data, groupBase);
//                max = bytesToFloat(data, groupBase + 4);
//                if (min > max) {
//                    throw new RuntimeException("min > max");
//                }
//                range = max - min;
//                groupPayloadBase = groupBase + 8;
//                for (j = 0; j < groupSize; j++) {
//                    // hardcoded 8 bits quant in this version
//                    // todo implement other quant sizes (based on quant.bits)
//                    if (jj < n) {
//                        int byteValue = data[groupPayloadBase + j] & 0xff;
//                        float weight = byteValue / 255f * range + min;
//                        val += weight * x[jj];
//                        jj++;
//                    } else {
//                        break;
//                    }
//                }
//            }
//            if (Float.isNaN(val)) {
//                throw new RuntimeException("Float.isNaN(val)");
//            }
//            xout[i] = val;
//        }
//    }

    public void testFP32(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        float[] copyOfXout = Arrays.copyOf(xout, xout.length);
        float[] copyOfx = Arrays.copyOf(x, x.length);
        Pointer pXout = cuda.allocateAndCopyToDevice(TEST_STREAM, xout, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        Pointer pw = cuda.allocateAndCopyToDevice(TEST_STREAM, w, false);
        cuda.synchronizeStream(TEST_STREAM);
        callFP32(TEST_STREAM, pXout, px, pw, weightIndex, n, d);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pXout, xout);
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
        Pointer pw = cuda.allocateAndCopyToDevice(TEST_STREAM, w.getByteArray(), false);
        cuda.synchronizeStream(TEST_STREAM);
        callFP32(TEST_STREAM, pXout, px, pw, weightIndex, n, d);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pXout, xout);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pXout);
        cuda.free(px);
        cuda.free(pw);

        callI8(copyOfXout, copyOfx, w, weightIndex, n, d);

        compareWithThreshold("MatMul.call xout ",
                xout, copyOfXout, 1e-4f);
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
        int blockSizeX = Math.min(findNextPowerOf2(d), MAX_THREADS_PER_BLOCK);
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

    // continue here
    //   make similar changes to WeightNormalizeAndScale
    //   update calls at transformer to I8 where quants in use
    //   test

    public void callI8(int streamId, Pointer xout, Pointer x, QuantPointer w, int weightIndex, int n, int d) {
        CUstream stream = cuda.getCUKernelStream(streamId);
        int quantSize = w.getQuant().groupSize();
        if (d % quantSize != 0) {
            throw new RuntimeException("d % quantSize != 0");
        }
        if (weightIndex % quantSize != 0) {
            throw new RuntimeException("weightIndex % quantSize != 0");
        }

        int nQuant = d / quantSize;
        int blockSizeX = Math.min(findNextPowerOf2(nQuant), MAX_THREADS_PER_BLOCK);
        int gridSizeX = (int) Math.ceil((double) nQuant / blockSizeX);

//        __global__ void matMul(float* xout, float* x, float* w, int weightIndex,
//        int n, int d, int quantSize) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(xout),
                Pointer.to(x),
                Pointer.to(w.pointerOfFloatIndex(weightIndex)),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{d}),
                Pointer.to(new int[]{quantSize})
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

    private CUfunction createI8() {
        String code =
                """
                            extern "C"
                            __global__ void matMul(float* xout, float* x, float* w, int quantIndex,
                                                   int n, int d, int quantSize) {
                                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                                int quantIndex = quantIndex + quantSize * tid;
                                float min = w[quantIndex * quantSize];
                                float max = w[quantIndex * quantSize + 1];
                                float range = max - min;
                                unsigned char* data = (((unsigned char*) (w + quantIndex * quantSize)) + 8;
                                float* weightPos = w + (quantIndex * quantSize) + i * n;
                                
                                for (int k = 0; k < quantSize; k++)
                                {
                                    int i = tid * quantSize + k;
                                    float value;
                                    if (i < d) {
                                        unsigned char value = data[i];
                                        value = value * range / 255f + min;
                                        
                                        float val = 0.0f;
                                        for (int j = 0; j < n; j++) {
                                            val += weightPos[j] * x[j];
                                        }
                                        xout[i] = val;
                                    }
                                }
                            }
                        """;
        return loadFromCode(code, "matMul");
    }

    private CUfunction createFP32() {
        String code =
                """
                            extern "C"
                            __global__ void matMul(float* xout, float* x, float* w, int n, int d) {
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
        return loadFromCode(code, "matMul");
    }
}