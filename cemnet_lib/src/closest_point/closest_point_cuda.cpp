#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include "closest_point_cuda_kernel.h"

extern THCState *state;

void closest_point_cuda(at::Tensor srcs_tensor, at::Tensor tgt_tensor, at::Tensor closest_idxs_tensor){
    int b = srcs_tensor.size(0);
    int c = srcs_tensor.size(1);
    int n = srcs_tensor.size(2);
    const float *srcs = srcs_tensor.data<float>();
    const float *tgt = tgt_tensor.data<float>();
    int *closest_idxs = closest_idxs_tensor.data<int>();
    closest_point_cuda_launcher(b, c, n, closest_idxs, srcs, tgt);
}




//#include <torch/serialize/tensor.h>
//#include <vector>
//#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>
//#include "sampling_cuda_kernel.h"
//
//extern THCState *state;
//
//void gathering_forward_cuda(int b, int c, int n, int m, at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor)
//{
//    const float *points = points_tensor.data<float>();
//    const int *idx = idx_tensor.data<int>();
//    float *out = out_tensor.data<float>();
//    gathering_forward_cuda_launcher(b, c, n, m, points, idx, out);
//}
//
//void gathering_backward_cuda(int b, int c, int n, int m, at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor)
//{
//
//    const float *grad_out = grad_out_tensor.data<float>();
//    const int *idx = idx_tensor.data<int>();
//    float *grad_points = grad_points_tensor.data<float>();
//    gathering_backward_cuda_launcher(b, c, n, m, grad_out, idx, grad_points);
//}
//
//void furthestsampling_cuda(int b, int n, int m, at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor)
//{
//    const float *points = points_tensor.data<float>();
//    float *temp = temp_tensor.data<float>();
//    int *idx = idx_tensor.data<int>();
//    furthestsampling_cuda_launcher(b, n, m, points, temp, idx);
//}