#include "../cuda_utils.h"
#include "distances_cuda_kernel.h"
#include <cmath>

// input: srcs(b, 3, n) tgt(3, n) distances(b)
// output: None
__global__ void iou_distance_cuda_kernel(int b, int c, int n, float r, const float *srcs, const float *tgt, float * distances)
{
    int idx_block = blockIdx.x;   // b_idx
    int idx_thread = threadIdx.x; // n_idx

    if (idx_block < b && idx_thread < n)
    {
        float min_distance1, min_distance2 = 0.0, 0.0;
        for (int i = 0; i < n; i += 1)
        {
            float distance1, distance2 = 0.0, 0.0;
            for (int j = 0; j < c; j += 1)
            {
                distance1 += pow(srcs[idx_block * c * n + j * n + idx_thread] - tgt[j * n + i], 2.0);
                distance2 += pow(srcs[idx_block * c * n + j * n + i] - tgt[j * n + idx_thread], 2.0);
            }
            if (distance1 < min_distance1)
            {
                min_distance1 = distance1;
            }
            if (distance2 < min_distance2)
            {
                min_distance2 = distance2;
            }
        }
        if (min_distance1 <= r)
        {
            distances[idx_block] += (1.0 - min_distance1 / r);
        }
        if (min_distance2 <= r)
        {
            distances[idx_block] += (1.0 - min_distance2 / r);
        }
    }
}

void iou_distance_cuda_launcher(int b, int c, int n, float r, const float *srcs, const float *tgt, float * distances)
{
    dim3 grid(b, 1, 1);
    dim3 block(n, 1, 1);
    iou_distance_cuda_kernel<<<grid, block, 0>>>(b, c, n, r, srcs, tgt, distances);
}

// input: srcs(b, 3, n) tgt(3, n) distances(b)
// output: None
__global__ void cd_distance_cuda_kernel(int b, int c, int n, const float *srcs, const float *tgt, float * distances)
{
    int idx_block = blockIdx.x;   // b_idx
    int idx_thread = threadIdx.x; // n_idx

    if (idx_block < b && idx_thread < n)
    {
        float min_distance1, min_distance2 = 0.0, 0.0;
        for (int i = 0; i < n; i += 1)
        {
            float distance1, distance2 = 0.0, 0.0;
            for (int j = 0; j < c; j += 1)
            {
                distance1 += pow(srcs[idx_block * c * n + j * n + idx_thread] - tgt[j * n + i], 2.0);
                distance2 += pow(srcs[idx_block * c * n + j * n + i] - tgt[j * n + idx_thread], 2.0);
            }
            if (distance1 < min_distance1)
            {
                min_distance1 = distance1;
            }
            if (distance2 < min_distance2)
            {
                min_distance2 = distance2;
            }
        }
        distances[idx_block] += (min_distance1 + min_distance2);
    }
}

void cd_distance_cuda_launcher(int b, int c, int n, const float *srcs, const float *tgt, float * distances)
{
    dim3 grid(b, 1, 1);
    dim3 block(n, 1, 1);
    cd_distance_cuda_kernel<<<grid, block, 0>>>(b, c, n, srcs, tgt, distances);
}