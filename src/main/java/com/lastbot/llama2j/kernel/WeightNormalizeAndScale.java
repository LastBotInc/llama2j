package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class WeightNormalizeAndScale extends Kernel {
    public static final int BLOCK_SIZE = 64;

    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public WeightNormalizeAndScale(ContextCUDA cuda) {
        super(cuda, "weightNormalizeAndScale");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void callFP32(float[] out, float[] x, float[] weight, int weightIndex,
                                float[] sumOfSquares, int size) {
        float ss = sumOfSquares[0];
        for (int jj = 0; jj < size; jj++) {
            out[jj] = weight[weightIndex + jj] * (ss * x[jj]);
        }
    }

    public static void callI8(float[] out, float[] x, QuantArray w, int weightIndex,
                              float[] sumOfSquares, int size) {
        float ss = sumOfSquares[0];
        // W (d,n) @ x (n,) -> xout (d,)
        Quant q = w.quant();

        byte[] encoded = w.data();

        int groupSize = q.groupSize();
        int encodedBytesPerGroup = q.encodedBytesPerGroup();
        int startGroupIndex = q.groupIndexByFloatIndex(weightIndex);
        int endGroupIndex = q.groupIndexByFloatIndex(weightIndex + size - 1);
        int sizeGroupIndex = endGroupIndex - startGroupIndex + 1;

        float min;
        float max;
        float range;
        int groupBase;
        int groupPayloadBase;

        int index;
        int group;

        for (int i = 0; i < sizeGroupIndex; i++) {
            group = startGroupIndex + i;
            groupBase = group * encodedBytesPerGroup;
            groupPayloadBase = groupBase + 8;
            min = bytesToFloat(encoded, groupBase);
            max = bytesToFloat(encoded, groupBase + 4);
            range = max - min;

            int startFloatIndex = group * groupSize;
            for (int j = 0; j < groupSize; j++) {
                index = startFloatIndex + j;
                if (index >= weightIndex + size) {
                    break;
                }
                if (index >= weightIndex) {
                    int offset = index - weightIndex;
                    int byteValue = encoded[groupPayloadBase + j] & 0xff;
                    float value = byteValue / 255f * range + min;
                    out[offset] = value * (ss * x[offset]);
                }
            }
        }
    }

    public void testI8(float[] out, float[] x, QuantArray w, int weightIndex,
                       float[] sumOfSquares, int size) {
        float[] copyOfOut = Arrays.copyOf(out, out.length);
        float[] copyOfx = Arrays.copyOf(x, x.length); // x can point to out!
        Pointer pOut = cuda.allocateAndCopyToDevice(TEST_STREAM, out, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        QuantPointer pWeight = new QuantPointer(w.quant(),
                cuda.allocateAndCopyToDevice(TEST_STREAM, w.data(), false), 0);
        Pointer pSumOfSquares = cuda.allocateAndCopyToDevice(TEST_STREAM, sumOfSquares, false);
        cuda.synchronizeStream(TEST_STREAM);
        callI8(TEST_STREAM, pOut, px, pWeight, weightIndex, pSumOfSquares, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pOut, out.length, out);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pOut);
        cuda.free(px);
        cuda.free(pWeight.pointer());
        cuda.free(pSumOfSquares);

        callI8(copyOfOut, copyOfx, w, weightIndex, sumOfSquares, size);

        compareWithThreshold("WeightNormalizeAndScale.call", out, copyOfOut, 1e-5f);
    }

    public void callFP32(int streamId, Pointer out, Pointer x, SlicePointer weight, int weightIndex,
                         Pointer sumOfSquares, int size) {
        Pointer weightWithIndex = weight.withIndex(weightIndex);
        callFP32(streamId, out, x, weightWithIndex, sumOfSquares, size);
    }

    public void callFP32(int streamId, Pointer out, Pointer x, Pointer weight, int weightIndex,
                         Pointer sumOfSquares, int size) {
        Pointer weightWithIndex = weight.withByteOffset((long) weightIndex * Sizeof.FLOAT);
        callFP32(streamId, out, x, weightWithIndex, sumOfSquares, size);
    }

    public void callFP32(int streamId, Pointer out, Pointer x, Pointer weight, Pointer sumOfSquares, int size) {
//        __global__ void weightNormalizeAndScale(float *out, const float *x, const float *weight,
//                    const float* rootMeanSquare, const int size)
        Pointer kernelParameters = Pointer.to(
                Pointer.to(out),
                Pointer.to(x),
                Pointer.to(weight),
                Pointer.to(sumOfSquares),
                Pointer.to(new int[]{size})
        );

        // Set up the kernel launch parameters.
        int blockSizeX = Math.min(findNextPowerOf2(size), BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) size / blockSizeX);

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, cuda.getCUKernelStream(streamId),  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
    }

    public void callI8(int streamId, Pointer out, Pointer x, QuantPointer w, int weightIndex,
                       Pointer sumOfSquares, int size) {
        // W (d,n) @ x (n,) -> xout (d,)
        Quant q = w.quant();

        if ((long)weightIndex - w.floatOffset() < 0) {
            throw new RuntimeException("(long)weightIndex - w.getFloatOffset() < 0)");
        }
        if ((long)weightIndex - w.floatOffset() > Integer.MAX_VALUE) {
            throw new RuntimeException("(long)weightIndex - w.getFloatOffset() > Integer.MAX_VALUE");
        }
        int adjustedWeightIndex = Math.toIntExact(weightIndex - w.floatOffset());

        int groupSize = q.groupSize();
        int startGroupIndex = q.groupIndexByFloatIndex(adjustedWeightIndex);
        int endGroupIndex = q.groupIndexByFloatIndex(adjustedWeightIndex + size - 1);
        int sizeGroupIndex = endGroupIndex - startGroupIndex + 1;
        int encodedBytesPerGroup = q.encodedBytesPerGroup();

//        __global__ void weightNormalizeAndScale(float *out, const float *x, const float *encoded,
//                            const float* rootMeanSquare, const int weightIndex, const int startGroupIndex,
//                            const int encodedBytesPerGroup, const int groupSize, const int size)
        Pointer kernelParameters = Pointer.to(
                Pointer.to(out),
                Pointer.to(x),
                Pointer.to(w.pointer()),
                Pointer.to(sumOfSquares),
                Pointer.to(new int[]{adjustedWeightIndex}),
                Pointer.to(new int[]{startGroupIndex}),
                Pointer.to(new int[]{encodedBytesPerGroup}),
                Pointer.to(new int[]{groupSize}),
                Pointer.to(new int[]{sizeGroupIndex}),
                Pointer.to(new int[]{size})
        );

        // Set up the kernel launch parameters.
        int blockSizeX = Math.min(findNextPowerOf2(sizeGroupIndex), BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) sizeGroupIndex / blockSizeX);

        isError(cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, cuda.getCUKernelStream(streamId),  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        ));
        if (SYNC_KERNEL_CALLS) {
            cuda.synchronizeStream(streamId);
        }
    }

    private CUfunction create() {
        String code =
                """
                            extern "C"
                            __global__ void weightNormalizeAndScaleI8(float *out, const float *x, unsigned char* encoded,
                            const float* sumOfSquares, const int weightIndex, const int startGroupIndex,
                            const int encodedBytesPerGroup, const int groupSize, const int sizeGroupIndex, const int size)
                            {
                                int i = blockIdx.x * blockDim.x + threadIdx.x;

                                if (i < sizeGroupIndex) {
                                    int group = startGroupIndex + i;
                                    int groupBase = group * encodedBytesPerGroup;
                                    int groupPayloadBase = groupBase + 8;

                                    float min = __int_as_float((encoded[groupBase + 3] & 0xFF)
                                                    | ((encoded[groupBase + 2] & 0xFF) << 8)
                                                    | ((encoded[groupBase + 1] & 0xFF) << 16)
                                                    | ((encoded[groupBase] & 0xFF) << 24));

                                    float max = __int_as_float((encoded[groupBase + 7] & 0xFF)
                                                    | ((encoded[groupBase + 6] & 0xFF) << 8)
                                                    | ((encoded[groupBase + 5] & 0xFF) << 16)
                                                    | ((encoded[groupBase + 4] & 0xFF) << 24));

                                    float range = max - min;

                                    int index;
    
                                    float ss = sumOfSquares[0];
    
                                    int startFloatIndex = group * groupSize;
                                    for (int j = 0; j < groupSize; j++) {
                                        index = startFloatIndex + j;
                                        if (index >= weightIndex + size) {
                                            break;
                                        }
                                        if (index >= weightIndex) {
                                            int offset = index - weightIndex;
                                            float byteValue = static_cast<float>(encoded[groupPayloadBase + j]);
                                            float value = (byteValue * range / 255.0f) + min;
                                            out[offset] = value * (ss * x[offset]);
                                        }
                                    }
                                }
                            }
                        """;
        return loadFromCode(code, "weightNormalizeAndScaleI8");
    }
}
