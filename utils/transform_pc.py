import torch, numpy as np
from utils.euler2mat import euler2mat_torch, euler2mat_np

def transform_pc_torch(pcs, Rs, ts):
    """
    :param pcs: point clouds, (B, 3, N),
    :param Rs: rotation matrix, (B, 3, 3)
    :param ts: translation vector, (B, 3)
    :return: transformed pcs, (B, 3, N)
    """
    return torch.matmul(Rs, pcs) + ts.unsqueeze(2)

def transform_pc_np(pcs, Rs, ts):
    """
    :param pcs: point clouds, (B, 3, N),
    :param Rs: rotation matrix, (B, 3, 3)
    :param ts: translation vector, (B, 3)
    :return: transformed pcs, (B, 3, N)
    """
    return np.matmul(Rs, pcs) + np.expand_dims(ts, axis=2)

def transform_pc_action_pytorch(pcs, actions):
    """
    :param pcs: point clouds, (B, 3, N),
    :param actions: rotation matrix, (B, 6)
    """
    Rs = euler2mat_torch(actions[:, :3], seq="zyx")
    ts = actions[:, 3:]
    return torch.matmul(Rs, pcs) + ts.unsqueeze(2)

def transform_pc_action_np(pcs, actions):
    """
    :param pcs: point clouds, (B, 3, N),
    :param actions: rotation matrix, (B, 6)
    """
    Rs = euler2mat_np(actions[:, :3], seq="zyx")
    ts = actions[:, 3:]
    return np.matmul(Rs, pcs) + np.expand_dims(ts, axis=2)
