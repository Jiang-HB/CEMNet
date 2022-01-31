import numpy as np, torch

def euler2mat_np(rs, seq="zyx"):
    assert seq == "zyx", "Invalid euler seq."
    rs_z, rs_y, rs_x = rs[:, [0]], rs[:, [1]], rs[:, [2]]
    sinx, siny, sinz = np.sin(rs_x), np.sin(rs_y), np.sin(rs_z)
    cosx, cosy, cosz = np.cos(rs_x), np.cos(rs_y), np.cos(rs_z)
    R = np.concatenate([cosy * cosz, - cosy * sinz, siny,
                            sinx * siny * cosz + cosx * sinz, - sinx * siny * sinz + cosx * cosz, - sinx * cosy,
                            - cosx * siny * cosz + sinx * sinz, cosx * siny * sinz + sinx * cosz, cosx * cosy],
                           1).reshape([-1, 3, 3])
    return R

def euler2mat_torch(rs, seq="zyx"):
    assert seq == "zyx", "Invalid euler seq."
    rs_z, rs_y, rs_x = rs[:, [0]], rs[:, [1]], rs[:, [2]]
    sinx, siny, sinz = torch.sin(rs_x), torch.sin(rs_y), torch.sin(rs_z)
    cosx, cosy, cosz = torch.cos(rs_x), torch.cos(rs_y), torch.cos(rs_z)
    R = torch.cat([cosy * cosz, - cosy * sinz, siny,
                            sinx * siny * cosz + cosx * sinz, - sinx * siny * sinz + cosx * cosz, - sinx * cosy,
                            - cosx * siny * cosz + sinx * sinz, cosx * siny * sinz + sinx * cosz, cosx * cosy],
                           1).view([-1, 3, 3])
    return R

