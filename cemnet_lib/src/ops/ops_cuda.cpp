#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "ops_cuda_kernel.h"

extern THCState *state;

void closest_point_cuda(at::Tensor srcs_tensor, at::Tensor tgt_tensor, at::Tensor closest_points_tensor){
    int b = srcs_tensor.size(0);
    int c = srcs_tensor.size(1);
    int n = srcs_tensor.size(2);
    const float *srcs = srcs_tensor.data<float>();
    const float *tgt = tgt_tensor.data<float>();
    float *closest_points = closest_points_tensor.data<float>();
    closest_point_cuda_launcher(b, c, n, closest_points, srcs, tgt);
}

void cd_distance_cuda(at::Tensor srcs_tensor, at::Tensor tgt_tensor, at::Tensor distances_tensor)
{
    int b = srcs_tensor.size(0);
    int c = srcs_tensor.size(1);
    int n = srcs_tensor.size(2);
    const float *srcs = srcs_tensor.data<float>();
    const float *tgts = tgt_tensor.data<float>();
    float *distances = distances_tensor.data<float>();
    cd_distance_cuda_launcher(b, c, n, srcs, tgts, distances);
}

void iou_distance_cuda(at::Tensor srcs_tensor, at::Tensor tgt_tensor, at::Tensor distances_tensor, float r, at::Tensor min_idxs_tensor)
{
    int b = srcs_tensor.size(0);
    int c = srcs_tensor.size(1);
    int n = srcs_tensor.size(2);
    const float *srcs = srcs_tensor.data<float>();
    const float *tgt = tgt_tensor.data<float>();
    float *distances = distances_tensor.data<float>();
    int *min_idxs = min_idxs_tensor.data<int>();
    iou_distance_cuda_launcher(b, c, n, r, srcs, tgt, distances, min_idxs);
}

void cycle_distance_cuda(at::Tensor srcs_tensor, at::Tensor tgt_tensor, at::Tensor distances_tensor, int N, at::Tensor min_idxs_tensor, at::Tensor n_distances_tensor, at::Tensor n_idxs_tensor)
{
    int b = srcs_tensor.size(0);
    int c = srcs_tensor.size(1);
    int n = srcs_tensor.size(2);
    const float *srcs = srcs_tensor.data<float>();
    const float *tgt = tgt_tensor.data<float>();
    float *distances = distances_tensor.data<float>();
    int *min_idxs = min_idxs_tensor.data<int>();
    float *n_distances = n_distances_tensor.data<float>();
    int *n_idxs = n_idxs_tensor.data<int>();
    cycle_distance_cuda_launcher(b, c, n, N, srcs, tgt, distances, min_idxs, n_distances, n_idxs);
}