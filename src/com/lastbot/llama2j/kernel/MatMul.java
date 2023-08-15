package com.lastbot.llama2j.kernel;

import com.lastbot.llama2j.ContextCUDA;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;

import java.util.Arrays;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public class MatMul extends Kernel {
    private final ContextCUDA cuda;
    private final CUfunction tileKernel;
    private final CUfunction reductionKernel;

    private static final int TILE_WIDTH = 32;
    private static final int BLOCK_SIZE = 256;

    public MatMul(ContextCUDA cuda) {
        super(cuda, "matMul");
        this.cuda = cuda;
        this.tileKernel = createTile();
        this.reductionKernel = createReduction();
    }

    public static void call(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        // W (d,n) @ x (n,) -> xout (d,)
        int i;
        float val;
        for (i = 0; i < d; i++) {
            val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[weightIndex + i * n + j] * x[j];
            }
            xout[i] = val;
        }
    }

    public void test(float[] xout, float[] x, float[] w, int weightIndex, int n, int d) {
        float[] copyOfXout = Arrays.copyOf(xout, xout.length);
        Pointer pXout = cuda.allocateAndCopyToDevice(xout, false);
        Pointer px = cuda.allocateAndCopyToDevice(x, false);
        Pointer pw = cuda.allocateAndCopyToDevice(w, false);
        cuda.synchronizeTransfer();
        call(0, pXout, px, pw, weightIndex, n, d);
        cuda.synchronizeKernel(0);
        cuda.copyFromDeviceToHost(pXout, xout);
        cuda.free(pXout);
        cuda.free(px);
        cuda.free(pw);

        call(copyOfXout, x, w, weightIndex, n, d);

        compareWithThreshold("MatMul.call xout ",
                xout, copyOfXout, 1e-2f);
    }

    private void call(int kernelStreamId, Pointer xout, Pointer x, Pointer w, int weightIndex, int n, int d) {
        CUstream stream = cuda.getCUKernelStream(kernelStreamId);
        int threadsPerBlock = BLOCK_SIZE;
        int blocksPerGrid = (int) Math.ceil((double) d / threadsPerBlock);
        {
            // __shared__ float s_x[TILE_WIDTH];
            // __shared__ float s_partialSum[BLOCK_SIZE];
            int sharedMemory = (TILE_WIDTH * TILE_WIDTH + 1) * Float.BYTES;

            int blockSizeX = TILE_WIDTH;
            int blockSizeY = TILE_WIDTH;
            int gridSizeX = (n - 1) / TILE_WIDTH + 1;
            int gridSizeY = (n - 1) / TILE_WIDTH + 1;

//            __global__ void tileKernel(float *d_xout, const float *d_w, const float *d_x, int weightIndex, int n, int d) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(xout),
                    Pointer.to(x),
                    Pointer.to(w),
                    Pointer.to(new int[]{weightIndex}),
                    Pointer.to(new int[]{n}),
                    Pointer.to(new int[]{d})
            );
            cuLaunchKernel(tileKernel,
                    gridSizeX, gridSizeY, 1,          // Grid dimension
                    blockSizeX, blockSizeY, 1,      // Block dimension
                    sharedMemory, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        }
        cuda.synchronizeKernel(kernelStreamId);
        // reduction
        {
            int blockSizeX = d; // ???
            int gridSizeX = n; // ???

//            __global__ void reduceKernel(float *d_xout, int n) {
            Pointer kernelParameters = Pointer.to(
                    Pointer.to(xout),
                    Pointer.to(new int[]{n})
            );
            cuLaunchKernel(reductionKernel,
                    gridSizeX, 1, 1,          // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, stream,  // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
        }
    }

    private CUfunction createTile() {
        String code =
                """
                    #include <stdio.h>

                    #define TILE_WIDTH __TILE_WIDTH__

                    __global__ void tileKernel(float *d_xout, const float *d_w, const float *d_x, int weightIndex, int n, int d) {
                        __shared__ float sh_w[TILE_WIDTH][TILE_WIDTH];
                        __shared__ float sh_x[TILE_WIDTH];

                        int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
                        int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

                        float val = 0.0f;

                        for (int m = 0; m < n / TILE_WIDTH; ++m) {
                            if (col < n && row < d) {
                                sh_w[threadIdx.y][threadIdx.x] = d_w[weightIndex + row * n + m * TILE_WIDTH + threadIdx.x];
                            } else {
                                sh_w[threadIdx.y][threadIdx.x] = 0.0f;
                            }

                            if (col < n) {
                                sh_x[threadIdx.x] = d_x[m * TILE_WIDTH + threadIdx.x];
                            } else {
                                sh_x[threadIdx.x] = 0.0f;
                            }

                            __syncthreads();

                            for (int k = 0; k < TILE_WIDTH; ++k) {
                                val += sh_w[threadIdx.y][k] * sh_x[k];
                            }

                            __syncthreads();
                        }

                        if (row < d) {
                            d_xout[row * gridDim.x + blockIdx.x] = val;
                        }
                    }
                """;
        code = code.replaceAll("__TILE_WIDTH__", Integer.toString(TILE_WIDTH));

        return loadFromCode(code, "matMulTile");
    }

    private CUfunction createReduction() {
        String code =
                """
                    extern "C"
                    __global__ void reduceKernel(float *d_xout, int n) {
                         int idx = blockIdx.x * blockDim.x + threadIdx.x;
                         if (idx < n) {
                             for (int stride = 1; stride < gridDim.x; stride *= 2) {
                                 if (threadIdx.x % (2 * stride) == 0 && (idx + stride) < n) {
                                     d_xout[idx] += d_xout[idx + stride];
                                 }
                                 __syncthreads();
                             }
                         }
                         if (threadIdx.x == 0) {
                             d_xout[blockIdx.x] = d_xout[idx];
                         }
                    }
                """;
        return loadFromCode(code, "reduction");
    }
}
