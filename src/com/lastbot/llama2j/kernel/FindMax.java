package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class FindMax extends Kernel {
    public static final int BLOCK_SIZE = 64;
    private static final int SIMPLE_KERNEL_THRESHOLD = 32;

    private final ContextCUDA cuda;

    private final CUfunction simpleKernel;
    private final CUfunction findMaxKernel;

    public FindMax(ContextCUDA cuda) {
        super(cuda, "findMax");
        this.cuda = cuda;
        simpleKernel = createSimple();
        findMaxKernel = createFindMax();
    }

    public static void call(float[] max, int maxIndex, float[] x, int index, int size) {
        float max_val = x[index]; // index + 0
        for (int i = 1; i < size; i++) {
            if (x[index + i] > max_val) {
                max_val = x[index + i];
            }
        }
        max[maxIndex] = max_val;
    }

    public void test(float[] max, int maxIndex, float[] x, int index, int size) {
        float[] copyOfMax = Arrays.copyOf(max, max.length);
        Pointer pMax = cuda.allocateAndCopyToDevice(TEST_STREAM, max, false);
        Pointer px = cuda.allocateAndCopyToDevice(TEST_STREAM, x, false);
        call(TEST_STREAM, pMax, maxIndex, px, index, size);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.copyFromDeviceToHost(TEST_STREAM, pMax, max.length, max);
        cuda.synchronizeStream(TEST_STREAM);
        cuda.free(pMax);
        cuda.free(px);

        call(copyOfMax, maxIndex, x, index, size);

        compareWithThreshold("FindMax.call max (size " + size + ")",
                max, copyOfMax, 1e-2f);
    }

    public void call(int streamId, Pointer max, int maxIndex, Pointer x, int index, int size) {
        CUstream stream = cuda.getCUKernelStream(streamId);

        if (size == 1) {
            cuda.copyFloatsFromDeviceToDevice(streamId, x, index, max, 0, 1);
        } else if (size < SIMPLE_KERNEL_THRESHOLD) {
//            _global__ void simpleMax(float* input, int inputIndex, int inputSize,
//            float* max, int maxIndex) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(x),
                    Pointer.to(new int[]{index}),
                    Pointer.to(new int[]{size}),
                    Pointer.to(max),
                    Pointer.to(new int[]{maxIndex})
            );
            isError(cuLaunchKernel(simpleKernel,
                    1, 1, 1,          // Grid dimension
                    1, 1, 1,      // Block dimension
                    0, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            if (SYNC_KERNEL_CALLS) {
                cuda.synchronizeStream(streamId);
            }
        } else { // parallel
//            __global__ void findMax(float* d_input, int inputIndex, int inputSize,
//            float* max, int maxIndex) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(x),
                    Pointer.to(new int[]{index}),
                    Pointer.to(new int[]{size}),
                    Pointer.to(max),
                    Pointer.to(new int[]{maxIndex})
            );
            isError(cuLaunchKernel(findMaxKernel,
                    (size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1,          // Grid dimension
                    BLOCK_SIZE, 1, 1,      // Block dimension
                    (BLOCK_SIZE * Sizeof.FLOAT) + 1, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            ));
            if (SYNC_KERNEL_CALLS) {
                cuda.synchronizeStream(streamId);
            }
        }
    }

    private CUfunction createSimple() {
        String code =
                """
                            #include <cfloat>
                            
                            extern "C"
                            __global__ void simpleMax(float* d_input, int inputIndex, int inputSize,
                                                      float* max, int maxIndex) {
                                float myMax = d_input[inputIndex];
                                for (int i = inputIndex + 1; i < inputIndex + inputSize; i++) {
                                    myMax = fmaxf(myMax, d_input[i]);
                                }
                                max[maxIndex] = myMax;
                            }

                        """;
        return loadFromCode(code, "simpleMax");
    }

    private CUfunction createFindMax() {
        String code =
                """
                         #include <cfloat>

                         #define BLOCK_SIZE <BLOCK_SIZE>

                        __device__ __forceinline__ void atomicMaxf(float* address, float val) {
                              int* address_as_int = (int*)address;
                              int old = *address_as_int, assumed;
                              while (val > __int_as_float(old)) {
                                  assumed = old;
                                  old = atomicCAS(address_as_int, assumed, __float_as_int(val));
                                  if (old == assumed)
                                      break;
                              }
                          }
                         
                         extern "C"
                        __global__ void findMax(float* d_input, int inputIndex, int inputSize,
                                                      float* max, int maxIndex) {
                                 extern __shared__ float sdata[];
                             
                                 int tid = threadIdx.x;
                                 int i = inputIndex + blockIdx.x * blockDim.x + threadIdx.x;
                             
                                 if (tid == 0) {
                                     max[maxIndex] = -FLT_MAX;
                                 }
                                 __syncthreads();

                                 float myMax = (i >= inputIndex && i < (inputIndex + inputSize)) ? d_input[i] : -FLT_MAX;
                             
                                 sdata[tid] = myMax;
                                 __syncthreads();
                             
                                 // Standard reduction in shared memory
                                 for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                                     if (tid < s) {
                                         sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
                                     }
                                     __syncthreads();
                                 }
                             
                                 // Only thread 0 writes result for this block to global mem
                                 if (tid == 0) {
                                     atomicMaxf(max + maxIndex, sdata[0]);
                                 }
                        }
                                  """;
        code = code.replaceAll("<BLOCK_SIZE>", Integer.toString(BLOCK_SIZE));
        return loadFromCode(code, "findMax");
    }
}
