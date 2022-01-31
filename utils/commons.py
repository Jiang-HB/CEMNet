import pickle, numpy as np, matplotlib, pdb, torch, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.mat2euler import mat2euler_torch
from utils.euler2mat import euler2mat_torch

def load_data(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def save_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file)
    file.close()

def chunker_list(seq, size):
    return [seq[pos: pos + size] for pos in range(0, len(seq), size)]

def chunker_num(num, size):
    return [list(range(num))[pos: pos + size] for pos in range(0, num, size)]

def stack_action_seqs(action_seqs):
    """
    :param action_seqs: (H, B, D)
    :return: actions (B, D)
    """
    return action_seqs.sum(0)

def line_plot(xs, ys, title, path):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def cal_errors_np(rs_pred, ts_pred, rs_lb, ts_lb, is_degree=False):
    """
    :param rs_pred: (B, 3)
    :param ts_pred: (B, 3)
    :param rs_lb: (B, 3)
    :param ts_lb: (B, 3)
    :param is_degree: bool
    :return: dict key: val (B, )
    """
    if not is_degree:
        rs_pred = np.degrees(rs_pred)
        rs_lb = np.degrees(rs_lb)
    rs_mse = np.mean((rs_pred - rs_lb) ** 2, 1) # (B, )
    ts_mse = np.mean((ts_pred - ts_lb) ** 2, 1)
    rs_rmse = np.sqrt(rs_mse)
    ts_rmse = np.sqrt(ts_mse)
    rs_mae = np.mean(np.abs(rs_pred - rs_lb), 1)
    ts_mae = np.mean(np.abs(ts_pred - ts_lb), 1)

    return {"rs_mse": rs_mse, "ts_mse": ts_mse,
            "rs_rmse": rs_rmse, "ts_rmse": ts_rmse,
            "rs_mae": rs_mae, "ts_mae": ts_mae}

def plot_pc(pcs, save_path):
    # pcs = [[[nm, pc], [nm, pc]], [[nm, pc]]] pc (3, N)
    N = len(pcs)
    n_col = 3
    n_row = np.ceil(N / n_col)
    plt.figure(figsize=(n_col * 4, n_row * 4))
    colors = [(244/255, 17/255, 10/255), (44/255, 175/255, 53/255), (18/255, 72/255, 148/255), (246/255, 130/255, 11/255)]
    for i, _pcs in enumerate(pcs):
        ax = plt.subplot(n_row, n_col, i + 1, projection='3d')
        for j, (lb, pc) in enumerate(_pcs):
            ax.scatter(pc[0], pc[1], pc[2], color=colors[j], marker='.', label=lb)
            ax.legend(fontsize=12, frameon=True)
    plt.savefig(save_path)
    # plt.close()

def stack_transforms(transforms1, transforms2):
    """
    :param transforms1: (B, 6), tensor
    :return: transforms2 (B, 6), tensor
    """
    rs1, ts1 = transforms1[:, :3], transforms1[:, 3:] # (B, 3)
    rs2, ts2 = transforms2[:, :3], transforms2[:, 3:] # (B, 3)
    Rs1 = euler2mat_torch(rs1) # (B, 3, 3)
    Rs2 = euler2mat_torch(rs2) # (B, 3, 3)
    Rs = torch.matmul(Rs2, Rs1) # (B, 3, 3)
    ts = (torch.matmul(Rs2, ts1.unsqueeze(2)) + ts2.unsqueeze(2)).squeeze(2) # (B, 3)
    rs = mat2euler_torch(Rs, is_degrees=False) # (B, 3)
    return torch.cat([rs, ts], 1) # (B, 6)

def stack_transforms_seq(transforms):
    """
    :param transforms: (L, B, 6), tensor
    """
    L = len(transforms)
    if L == 1:
        actions = transforms[0, :, :]
    else:
        for l in range(L - 1):
            if l == 0:
                actions = stack_transforms(transforms[l, :, :], transforms[l + 1, :, :])
            else:
                actions = stack_transforms(actions, transforms[l + 1, :, :])

    return actions # (B, 6)
