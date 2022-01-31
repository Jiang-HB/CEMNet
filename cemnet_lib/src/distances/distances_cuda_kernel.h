#ifndef _SAMPLING_CUDA_KERNEL
#define _SAMPLING_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void cd_distance_cuda(at::Tensor srcs, at::Tensor tgt, at::Tensor distances);
void iou_distance_cuda(at::Tensor srcs, at::Tensor tgt, at::Tensor distances, float r);

#ifdef __cplusplus
extern "C" {
#endif

void cd_distance_cuda_launcher(int b, int c, int n, const float *srcs, const float *tgt, float * distances);
void iou_distance_cuda_launcher(int b, int c, int n, float r, const float *srcs, const float *tgt, float * distances);

#ifdef __cplusplus
}
#endif
#endif