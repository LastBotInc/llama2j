//     private static void accum(float[] a, float[] b, int size) {
//         for (int i = 0; i < size; i++) {
//             a[i] += b[i];
//         }
//     }
extern "C"
__global__ void accum(float *a, float *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}
