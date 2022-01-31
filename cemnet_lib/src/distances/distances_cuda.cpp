#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "distances_cuda_kernel.h"

extern THCState *state;

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

void iou_distance_cuda(at::Tensor srcs_tensor, at::Tensor tgt_tensor, at::Tensor distances_tensor, float r)
{
    int b = srcs_tensor.size(0);
    int c = srcs_tensor.size(1);
    int n = srcs_tensor.size(2);
    const float *srcs = srcs_tensor.data<float>();
    const float *tgt = tgt_tensor.data<float>();
    float *distances = distances_tensor.data<float>();
    iou_distance_cuda_launcher(b, c, n, r, srcs, tgt, distances);
}