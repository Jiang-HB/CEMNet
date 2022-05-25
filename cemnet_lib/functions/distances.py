import torch, cemnet_lib_cuda
from torch.autograd import Function

# closest point
class ClosestPoint(Function):
    @staticmethod
    def forward(ctx, srcs, tgt):
        """
        input: srcs: (B, 3, N), tgt: (3, N)
        return: closest_points: (B, 3, N)
        """
        closest_points = torch.zeros_like(srcs).to(srcs.device)
        cemnet_lib_cuda.closest_point_cuda(srcs, tgt, closest_points)
        return closest_points

    @staticmethod
    def backward(srcs=None, tgt=None, closest_idxs=None):
        return None, None

closest_point = ClosestPoint.apply

# Chamfer distance
class CD(Function):
    @staticmethod
    def forward(ctx, srcs, tgt):
        """
        srcs: (B, 3, N)
        tgt: (3, N)
        return: distances: (B)
        """
        distances = torch.zeros(len(srcs), srcs.size(2), 2).to(srcs.device)
        cemnet_lib_cuda.cd_distance_cuda(srcs, tgt, distances)
        distances = distances.sum(2).mean(1)
        return distances

    @staticmethod
    def backward(srcs=None, tgt=None, distances=None, r=None):
        return None, None

cd_distance = CD.apply

# Geman-McClure estimator based distance
class GM(Function):
    @staticmethod
    def forward(ctx, srcs, tgt, mu):
        """
        srcs: (B, 3, N)
        tgt: (3, N)
        return: distances: (B)
        """
        distances = torch.zeros(len(srcs), srcs.size(2), 2).to(srcs.device)
        cemnet_lib_cuda.cd_distance_cuda(srcs, tgt, distances) # (B, N, 2)
        distances = ((distances * mu) / (distances + mu)).sum(2).mean(1)
        return distances

    @staticmethod
    def backward(srcs=None, tgt=None, distances=None, r=None):
        return None, None

gm_distance = GM.apply

# Maximum consensus based distance
class MC(Function):
    @staticmethod
    def forward(ctx, srcs, tgt, epsilon):
        """
        srcs: (B, 3, N)
        tgt: (3, N)
        epsilon: float
        return: distances: (B)
        """
        distances = torch.zeros(len(srcs), srcs.size(2), 2).to(srcs.device)
        min_idxs = torch.zeros(len(srcs), srcs.size(2), 2).type(torch.int).to(srcs.device)
        cemnet_lib_cuda.mc_distance_cuda(srcs, tgt, distances, epsilon, min_idxs)
        distances = 2.0 - distances.sum(2).mean(1)
        return distances

    @staticmethod
    def backward(srcs=None, tgt=None, distances=None, r=None, is_min_idxs=None):
        return None, None

mc_distance = MC.apply