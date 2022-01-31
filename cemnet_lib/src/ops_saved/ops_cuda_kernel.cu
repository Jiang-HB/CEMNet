#include "../cuda_utils.h"
#include "ops_cuda_kernel.h"
#include "math.h"

// input: srcs(B, 3, n) tgt(3, n)
// output: out(b, n)
__global__ void closest_point_cuda_kernel(int b, int c, int n, float *closest_points, const float *srcs, const float *tgt)
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
                val += (srcs[idx_block * c * n + i * n + idx_thread] - tgt[i * n + k]) * (srcs[idx_block * c * n + i * n + idx_thread] - tgt[i * n + k]);
            }
            if (val < min_val)
            {
                min_val = val;
                idx = k;
            }
        }
        closest_points[idx_block * n * c + idx_thread] = tgt[idx];
        closest_points[idx_block * n * c + n + idx_thread] = tgt[n + idx];
        closest_points[idx_block * n * c + 2 * n + idx_thread] = tgt[2 * n + idx];
    }
}

void closest_point_cuda_launcher(int b, int c, int n, float *closest_points, const float *srcs, const float *tgt)
{
    dim3 grid(b, 1, 1);
    dim3 block(n, 1, 1);
    closest_point_cuda_kernel<<<grid, block, 0>>>(b, c, n, closest_points, srcs, tgt);
}

// input: srcs(b, 3, n) tgt(3, n) distances(b)
// output: None
__global__ void iou_distance_cuda_kernel(int b, int c, int n, float r, const float *srcs, const float *tgt, float * distances)
{
    int idx_block = blockIdx.x;   // b_idx
    int idx_thread = threadIdx.x; // n_idx
    float rr = r * r;
    if (idx_block < b && idx_thread < n)
    {
        float min_distance1 = 1000.0;
        float min_distance2 = 1000.0;
        for (int i = 0; i < n; i += 1)
        {
            float distance1 = 0.0;
            float distance2 = 0.0;
            for (int j = 0; j < c; j += 1)
            {
                distance1 += (srcs[idx_block * c * n + j * n + idx_thread] - tgt[j * n + i]) * (srcs[idx_block * c * n + j * n + idx_thread] - tgt[j * n + i]);
                distance2 += (srcs[idx_block * c * n + j * n + i] - tgt[j * n + idx_thread]) * (srcs[idx_block * c * n + j * n + i] - tgt[j * n + idx_thread]);
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
        if (min_distance1 <= rr)
        {
            distances[idx_block * n * 2 + idx_thread * 2] = (1.0 - sqrt((float)(min_distance1)) / r);
        }
        if (min_distance2 <= rr)
        {
            distances[idx_block * n * 2 + idx_thread * 2 + 1] = (1.0 - sqrt((float)(min_distance2)) / r);
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
        float min_distance1 = 1000.0;
        float min_distance2 = 1000.0;
        for (int i = 0; i < n; i += 1)
        {
            float distance1 = 0.0;
            float distance2 = 0.0;
            for (int j = 0; j < c; j += 1)
            {
                distance1 += (srcs[idx_block * c * n + j * n + idx_thread] - tgt[j * n + i]) * (srcs[idx_block * c * n + j * n + idx_thread] - tgt[j * n + i]);
                distance2 += (srcs[idx_block * c * n + j * n + i] - tgt[j * n + idx_thread]) * (srcs[idx_block * c * n + j * n + i] - tgt[j * n + idx_thread]);
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
        distances[idx_block * n * 2 + idx_thread * 2] = min_distance1;
        distances[idx_block * n * 2 + idx_thread * 2 + 1] = min_distance2;
    }
}

void cd_distance_cuda_launcher(int b, int c, int n, const float *srcs, const float *tgt, float * distances)
{
    dim3 grid(b, 1, 1);
    dim3 block(n, 1, 1);
    cd_distance_cuda_kernel<<<grid, block, 0>>>(b, c, n, srcs, tgt, distances);
}