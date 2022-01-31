import numpy as np, torch, os, pdb
from utils.options import opts
from utils.transform_pc import transform_pc_torch
from utils.euler2mat import euler2mat_torch
from utils.recorder import Recorder
from utils.test import test
from utils.losses import CDLoss, GMLoss
from datasets.get_dataset import get_dataset
from cems.guided_cem import GuidedCEM
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

np.random.seed(opts.seed)
torch.manual_seed(opts.seed)
torch.cuda.manual_seed_all(opts.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def init_opts(opts):
    opts.is_debug = True
    opts.is_train = True
    opts.loss_type = [["GM", 0.01], ["CD", -1.0]][0]
    opts.cem.metric_type = [["MC", 0.1], ["CD"], ["GM", 0.01]][0]
    opts.cem.n_candidates = [1000, {"minibatch": 1000}]
    opts.cem.n_iters = 10
    opts.cem.is_fused_reward = [False, -1.0, 0]
    opts.results_dir = "./results/%s_n%d_unseen%d_noise%d_seed%s_v0" % (
        opts.db_nm, opts.db.n_sub_points, opts.db.unseen, opts.db.gaussian_noise, opts.seed)
    if not opts.is_debug:
        os.makedirs(opts.results_dir, exist_ok=True)
    return opts

def main(opts):
    ## initial setting
    opts = init_opts(opts)
    opts.recorder = Recorder(opts)
    model = GuidedCEM(opts).to(opts.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[35, 100, 150], gamma=0.7)
    if opts.loss_type[0] == "CD":
        loss_func = CDLoss(opts)
    elif opts.loss_type[0] == "GM":
        loss_func = GMLoss(opts)
    train_loader, _ = get_dataset(opts, db_nm=opts.db_nm, partition="train", is_normal=False,  batch_size=opts.batch_size, shuffle=True, drop_last=False)
    test_loader, _ = get_dataset(opts, db_nm=opts.db_nm, partition="test", is_normal=False,  batch_size=opts.batch_size, shuffle=False, drop_last=False)

    ## training
    print(opts.results_dir)
    for epoch in range(opts.n_epochs):
        scheduler.step()
        ## train
        losses = []
        for srcs, tgts, rs_lb, ts_lb in tqdm(train_loader):
            srcs, tgts, rs_lb, ts_lb = [x.to(opts.device) for x in [srcs, tgts, rs_lb, ts_lb]]
            results = model(srcs, tgts)
            rs_pred, ts_pred, rs_prior, ts_prior = results["r"], results["t"], results["r_init"], results["t_init"]
            transform_srcs_pred = transform_pc_torch(srcs, euler2mat_torch(rs_pred), ts_pred)
            transform_srcs_prior = transform_pc_torch(srcs, euler2mat_torch(rs_prior), ts_prior)
            loss = loss_func(transform_srcs_pred, tgts) + loss_func(transform_srcs_prior, tgts)
            if torch.isnan(loss):
                print("None, skip")
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Epoch[%d], losses: %.9f, %s." % (epoch, np.mean(losses), opts.results_dir))

        ## test
        test(opts, model, test_loader, epoch)

if __name__ == '__main__':
    main(opts)






