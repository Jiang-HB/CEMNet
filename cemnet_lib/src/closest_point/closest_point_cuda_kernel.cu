#include "../cuda_utils.h"
#include "closest_point_cuda_kernel.h"
#include <cmath>

// input: srcs(B, 3, n) tgt(3, n)
// output: out(b, n)
//__global__ void closest_point_cuda_kernel(int b, int c, int n, int *closest_idxs, const float *srcs, const float *tgt)
//{
//    int n_block = gridDim.x;
//    int n_thread = blockDim.x;
//    int idx_block = blockIdx.x;
//    int idx_thread = threadIdx.x;
//
//    if (idx_block < n && idx_thread < b)
//    {
//        float min_val = 1000.0;
//        int idx = -1;
//        for (int k = 0; k < n; k += 1)
//        {
//            float val = 0.0;
//            for (int i = 0; i < c; i += 1)
//            {
//                val += pow(srcs[idx_thread * c * n + i * n + idx_block] - tgt[i * n + k], 2.0);
//            }
//            if (val < min_val)
//            {
//                min_val = val;
//                idx = k;
//            }
//        }
//        closest_idxs[idx_thread * n + idx_block] = idx;
//    }
//}

//void closest_point_cuda_launcher(int b, int c, int n, int *closest_idxs, const float *srcs, const float *tgt)
//{
//    dim3 grid(n, 1, 1);
//    dim3 block(b, 1, 1);
//    closest_point_cuda_kernel<<<grid, block, 0>>>(b, c, n, closest_idxs, srcs, tgt);
//}


// input: srcs(B, 3, n) tgt(3, n)
// output: out(b, n)
__global__ void closest_point_cuda_kernel(int b, int c, int n, int *closest_idxs, const float *srcs, const float *tgt)
{
    int idx_block = blockIdx.x;
    int idx_thread = threadIdx.x;

    if (idx_block < b && idx_thread < n)
    {
        float min_val = 1000.0;
        int idx = -1;
        for (int k = 0; k < n; k += 1)
        {
            float val = 0.0;
            for (int i = 0; i < c; i += 1)
            {
                val += pow(srcs[idx_block * c * n + i * n + idx_thread] - tgt[i * n + k], 2.0);
            }
            if (val < min_val)
            {
                min_val = val;
                idx = k;
            }
        }
        closest_idxs[idx_block * n + idx_thread] = idx;
    }
}

void closest_point_cuda_launcher(int b, int c, int n, int *closest_idxs, const float *srcs, const float *tgt)
{
    dim3 grid(b, 1, 1);
    dim3 block(n, 1, 1);
    closest_point_cuda_kernel<<<grid, block, 0>>>(b, c, n, closest_idxs, srcs, tgt);
}