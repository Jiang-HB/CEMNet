import torch, pdb
from utils.transform_pc import transform_pc_torch
from torch_batch_svd import svd
from cemnet_lib.functions import closest_point

def one_step(srcs, tgt, Rs, ts):

    xs_mean = srcs.mean(2, keepdim=True)  # [B, 3, 1]
    xs_centered = srcs - xs_mean  # [B, 3, N]
    ys = closest_point(transform_pc_torch(srcs, Rs, ts), tgt)

    ys_mean = ys.mean(2, keepdim=True)
    ys_centered = ys - ys_mean


    H = torch.matmul(xs_centered, ys_centered.transpose(2, 1).contiguous())
    u, _, v = svd(H)
    Rs = torch.matmul(v, u.transpose(2, 1)).contiguous()
    r_det = torch.det(Rs)
    # Rs[:, 2, 2] = r_det
    diag = torch.eye(3, 3).unsqueeze(0).repeat(len(Rs), 1, 1).to(srcs.device)
    diag[:, 2, 2] = r_det
    Rs = torch.matmul(torch.matmul(v, diag), u.transpose(2, 1)).contiguous()

    # idxs = torch.where(r_det < 0)[0]
    # Rs[idxs, :, 2] *= -1
    # Rs[idxs, 2, :] *= -1
    ts = torch.matmul(- Rs, xs_mean) + ys_mean

    return Rs, ts.squeeze(-1)

def batch_icp(opts, srcs, tgt, Rs=None, ts=None, is_path=False):

    # srcs(b, c, n) tgt(c, n)
    if Rs is None:
        Rs = torch.eye(3).unsqueeze(0).repeat(len(srcs), 1, 1).to(srcs.device)
    if ts is None:
        ts = torch.zeros(len(srcs), 3).to(srcs.device)
    paths = [[Rs, ts]]
    for i in range(3):
        Rs, ts = one_step(srcs, tgt, Rs, ts)
        if is_path:
            paths.append([Rs, ts])
    if is_path:
        return Rs, ts, paths
    else:
        return Rs, ts