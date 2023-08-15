package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class ApplyRope extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction kernel;

    public ApplyRope(ContextCUDA cuda) {
        super(cuda, "applyRope");
        this.cuda = cuda;
        this.kernel = create();
    }

    public static void call(float[] q, float[] k, float[] freq_cis_real, float[] freq_cis_imag,
                             int dim, int head_size, int freq_cis_imag_row) {
        // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
        for (int i = 0; i < dim; i += 2) {
            float q0 = q[i];
            float q1 = q[i + 1];
            float k0 = k[i];
            float k1 = k[i + 1];
            int freq_cis_imag_index = freq_cis_imag_row + (i % head_size) / 2;
            float fcr = freq_cis_real[freq_cis_imag_index];
            float fci = freq_cis_imag[freq_cis_imag_index];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            k[i] = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
        }
    }

    public void test(float[] q, float[] k, float[] freq_cis_real, float[] freq_cis_imag,
                     int dim, int head_size, int freq_cis_imag_row) {
        float[] copyOfQ = Arrays.copyOf(q, q.length);
        float[] copyOfK = Arrays.copyOf(k, k.length);
        Pointer pq = cuda.allocateAndCopyToDevice(q, false);
        Pointer pk = cuda.allocateAndCopyToDevice(k, false);
        Pointer pFreq_cis_real = cuda.allocateAndCopyToDevice(freq_cis_real, false);
        Pointer pFreq_cis_imag = cuda.allocateAndCopyToDevice(freq_cis_imag, false);
        cuda.synchronizeTransfer();
        call(0, pq, pk, pFreq_cis_real, pFreq_cis_imag, dim, head_size, freq_cis_imag_row);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pk, k);
        cuda.copyFromDeviceToHost(pq, q);
        cuda.free(pq);
        cuda.free(pk);
        cuda.free(pFreq_cis_real);
        cuda.free(pFreq_cis_imag);

        call(copyOfQ, copyOfK, freq_cis_real, freq_cis_imag, dim, head_size, freq_cis_imag_row);

        cuda.synchronizeTransfer();

        compareWithThreshold("ApplyRope.call q", q, copyOfQ, 1e-5f);
        compareWithThreshold("ApplyRope.call k", k, copyOfK, 1e-5f);
    }

    public void call(int kernelStreamId, Pointer q, Pointer k, Pointer freq_cis_real, Pointer freq_cis_imag,
                     int dim, int head_size, int freq_cis_imag_row) {
//        __global__ void applyRope(    float *q, float *k,
//                                      const float *freq_cis_real, const float *freq_cis_imag,
//                                      int dim, int head_size, int freq_cis_imag_row)
        Pointer kernelParameters = Pointer.to(
                Pointer.to(q),
                Pointer.to(k),
                Pointer.to(freq_cis_real),
                Pointer.to(freq_cis_imag),
                Pointer.to(new int[]{dim}),
                Pointer.to(new int[]{head_size}),
                Pointer.to(new int[]{freq_cis_imag_row})
        );

        int halfDim = dim / 2;

        int blockSizeX = Math.min(findNextPowerOf2(halfDim), 1024);
        int gridSizeX = (int) Math.ceil((double) halfDim / blockSizeX);

        cuLaunchKernel(kernel,
                gridSizeX, 1, 1,          // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, cuda.getCUKernelStream(kernelStreamId),  // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
    }

    private CUfunction create() {
        String code =
                """
                                extern "C"
                                __global__ void applyRope(float *q, float *k,
                                const float *freq_cis_real, const float *freq_cis_imag,
                                int dim, int head_size, int freq_cis_imag_row)
                                {
                                    // process elements in steps of 2
                                    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
                                 
                                    // Ensure we don't go out of bounds
                                    if (i < dim) {
                                        float q0 = q[i];
                                        float q1 = q[i + 1];
                                        float k0 = k[i];
                                        float k1 = k[i + 1];

                                        int freq_cis_imag_index = freq_cis_imag_row + (i % head_size) / 2;
                                        float fcr = freq_cis_real[freq_cis_imag_index];
                                        float fci = freq_cis_imag[freq_cis_imag_index];

                                        q[i] = q0 * fcr - q1 * fci;
                                        q[i + 1] = q0 * fci + q1 * fcr;

                                        k[i] = k0 * fcr - k1 * fci;
                                        k[i + 1] = k0 * fci + k1 * fcr;
                                    }
                                }
                        """;
        return loadFromCode(code, "applyRope");
    }
}