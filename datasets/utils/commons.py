import pickle, matplotlib.pyplot as plt, numpy as np

def load_data(path):
    file = open(path, "rb")
    data = pickle.load(file)
    file.close()
    return data

def save_data(path, data):
    file = open(path, "wb")
    pickle.dump(data, file)
    file.close()

def line_plot(xs, ys, title, path):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def stack_action_seqs(action_seqs):
    """
    :param action_seqs: (H, B, D)
    :return: actions (B, D)
    """
    return action_seqs.sum(0)

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
