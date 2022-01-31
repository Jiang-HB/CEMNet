from tqdm import tqdm
import torch, numpy as np, os

def test(opts, model, test_loader, epoch):
    cal_score = lambda x: np.mean(np.concatenate(x, 0)).item()
    with torch.no_grad():
        rs_mse, ts_mse, rs_mae, ts_mae = [], [], [], []
        rs_prior_mse, ts_prior_mse, rs_prior_mae, ts_prior_mae = [], [], [], []
        for srcs, tgts, rs_lb, ts_lb in tqdm(test_loader):
            srcs, tgts = srcs.cuda(), tgts.cuda()
            results = model(srcs, tgts)
            # rs_pred, ts_pred, rs_prior, ts_prior = results["rs"], results["ts"], results["rs_init"], results["ts_init"]
            rs_pred, ts_pred, rs_prior, ts_prior = results["r"], results["t"], results["r_init"], results["t_init"]
            rs_mse.append(np.mean((np.degrees(rs_pred.cpu().numpy()) - np.degrees(rs_lb.numpy())) ** 2, 1))
            rs_mae.append(np.mean(np.abs(np.degrees(rs_pred.cpu().numpy()) - np.degrees(rs_lb.numpy())), 1))
            ts_mse.append(np.mean((ts_pred.cpu().numpy() - ts_lb.numpy()) ** 2, 1))
            ts_mae.append(np.mean(np.abs(ts_pred.cpu().numpy() - ts_lb.numpy()), 1))

            rs_prior_mse.append(np.mean((np.degrees(rs_prior.cpu().numpy()) - np.degrees(rs_lb.numpy())) ** 2, 1))
            rs_prior_mae.append(np.mean(np.abs(np.degrees(rs_prior.cpu().numpy()) - np.degrees(rs_lb.numpy())), 1))
            ts_prior_mse.append(np.mean((ts_prior.cpu().numpy() - ts_lb.numpy()) ** 2, 1))
            ts_prior_mae.append(np.mean(np.abs(ts_prior.cpu().numpy() - ts_lb.numpy()), 1))

        r_mse = cal_score(rs_mse)
        t_mse = cal_score(ts_mse)
        r_mae = cal_score(rs_mae)
        t_mae = cal_score(ts_mae)
        r_prior_mse = cal_score(rs_prior_mse)
        t_prior_mse = cal_score(ts_prior_mse)
        r_prior_mae = cal_score(rs_prior_mae)
        t_prior_mae = cal_score(ts_prior_mae)

        if not opts.is_debug:
            torch.save(model.state_dict(), os.path.join(opts.results_dir, 'model_epoch%d.pth' % (epoch)))

        print("[%d] Test: r_mse: %.8f, t_mse: %.8f, r_mae: %.8f, t_mae: %.8f" % (
                                                epoch, r_mse, t_mse, r_mae, t_mae))
        print("[%d] Prior test: r_mse: %.8f, t_mse: %.8f, r_mae: %.8f, t_mae: %.8f" % (
                                                epoch, r_prior_mse, t_prior_mse, r_prior_mae, t_prior_mae))