#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "ops/ops_cuda_kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("closest_point_cuda", &closest_point_cuda, "closest_point_cuda"); // name in python, cpp function address, docs
    m.def("cd_distance_cuda", &cd_distance_cuda, "cd_distance_cuda");
//    m.def("iou_distance_cuda", &iou_distance_cuda, "iou_distance_cuda");
    m.def("mc_distance_cuda", &mc_distance_cuda, "mc_distance_cuda");
//    m.def("cycle_distance_cuda", &cycle_distance_cuda, "cycle_distance_cuda");
}