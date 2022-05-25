#ifndef _SAMPLING_CUDA_KERNEL
#define _SAMPLING_CUDA_KERNEL
#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void closest_point_cuda(at::Tensor srcs, at::Tensor tgt, at::Tensor closest_points);
void cd_distance_cuda(at::Tensor srcs, at::Tensor tgt, at::Tensor distances);
void mc_distance_cuda(at::Tensor srcs, at::Tensor tgt, at::Tensor distances, float r, at::Tensor min_idxs);
void cycle_distance_cuda(at::Tensor srcs, at::Tensor tgt, at::Tensor distances, int N, at::Tensor min_idxs, at::Tensor n_distances, at::Tensor n_idxs);

#ifdef __cplusplus
extern "C" {
#endif

void closest_point_cuda_launcher(int b, int c, int n, float *closest_points, const float *srcs, const float *tgt);
void cd_distance_cuda_launcher(int b, int c, int n, const float *srcs, const float *tgt, float * distances);
void mc_distance_cuda_launcher(int b, int c, int n, float r, const float *srcs, const float *tgt, float * distances, int *min_idxs);
void cycle_distance_cuda_launcher(int b, int c, int n, int N, const float *srcs, const float *tgt, float * distances, int *min_idxs, float * n_distances, int * n_idxs);

#ifdef __cplusplus
}
#endif
#endif