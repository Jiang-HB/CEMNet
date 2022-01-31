import numpy as np, torch
from scipy.spatial.transform import Rotation

def mat2euler_np(mats, seq='zyx', is_degrees=True):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=is_degrees))
    return np.asarray(eulers, dtype='float32')

def mat2euler_torch(mats, seq='zyx', is_degrees=True):
    mats_np = mats.detach().cpu().numpy()
    eulers = []
    for i in range(mats_np.shape[0]):
        r = Rotation.from_dcm(mats_np[i])
        eulers.append(r.as_euler(seq, degrees=is_degrees))
    return torch.FloatTensor(np.asarray(eulers, dtype='float32')).to(mats.device)

def mat2euler(rot_mat, seq='xyz'):
    """
    convert rotation matrix to euler angle
    :param rot_mat: rotation matrix rx*ry*rz [B, 3, 3]
    :param seq: seq is xyz(rotate along z first) or zyx
    :return: three angles, x, y, z
    """
    r11 = rot_mat[:, 0, 0]
    r12 = rot_mat[:, 0, 1]
    r13 = rot_mat[:, 0, 2]
    r21 = rot_mat[:, 1, 0]
    r22 = rot_mat[:, 1, 1]
    r23 = rot_mat[:, 1, 2]
    r31 = rot_mat[:, 2, 0]
    r32 = rot_mat[:, 2, 1]
    r33 = rot_mat[:, 2, 2]
    if seq == 'xyz':
        z = torch.atan2(-r12, r11)
        y = torch.asin(r13)
        x = torch.atan2(-r23, r33)
    else:
        y = torch.asin(-r31)
        x = torch.atan2(r32, r33)
        z = torch.atan2(r21, r11)
    return torch.stack((z, y, x), dim=1)