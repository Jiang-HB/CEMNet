import numpy as np, torch, time
from datasets.get_dataset import get_dataset
from utils.options import opts
from utils.recorder import Recorder
from utils.attr_dict import AttrDict
from cems.guided_cem import GuidedCEM
from tqdm import tqdm

np.random.seed(opts.seed)
torch.manual_seed(opts.seed)
torch.cuda.manual_seed_all(opts.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

opts.db_nm = "scene7"
opts.db = AttrDict(
    modelnet40 = AttrDict(
        path = "/test/datasets/registration/modelnet40/modelnet40_normal_n2048.pth",
        cls_idx = -1,  # None_class: -1, airplane: 0, car: 7, chair: 8, table: 33, lamp: 19
        is_neg_angle = False,
        unseen = False,
        gaussian_noise = False,
        n_points = 1024,
        n_sub_points = 768,
        factor = 4
    ),
    scene7 = AttrDict(
        path = "/test/datasets/registration/7scene/7scene_normal_n2048.pth",
        cls_idx = -1,
        is_neg_angle = False,
        unseen = False,
        gaussian_noise = False,
        n_points = 1024,
        n_sub_points = 768,
        factor = 4
    ),
    icl_nuim = AttrDict(
        path = "/test/datasets/registration/icl_nuim/icl_nuim_normal_n2048.pth",
        cls_idx = -1,
        is_neg_angle = False,
        unseen = False,
        gaussian_noise = False,
        n_points = 1024,
        n_sub_points = 768,
        factor = 4
    )
)
opts.db = opts.db[opts.db_nm]

def init_opts(opts):
    opts.is_train = False
    opts.cem.metric_type = [["MC", 0.1], ["CD"], ["GM", 0.01]][0]
    opts.cem.n_candidates = [1000, {"minibatch": 1000}]
    opts.cem.is_fused_reward = [True, 0.5, 3]
    opts.exploration_weight = 0.5
    opts.cem.n_iters = 10
    opts.is_debug = True
    opts.cem.init_sigma = AttrDict(
        modelnet40 = [1.0, 0.5],
        scene7 = [1.0, 0.5],
        icl_nuim = [1.0, 0.5]
    )

    # 1. ModelNet40 - Unseen Object
    # opts.model_path = "./results/modelnet40_n768_unseen0_noise0_seed123/model.pth"

    # 2. ModelNet40 - Unseen Catergory
    # opts.model_path = "./results/modelnet40_n768_unseen1_noise0_seed123/model.pth"

    # 3. ModelNet40 - Noise
    # opts.model_path = "./results/modelnet40_n768_unseen0_noise1_seed123/model.pth"

    # 4. 7Scene
    opts.model_path = "./results/scene7_n768_unseen0_noise0_seed123/model.pth"

    # 5. ICL-NUIM
    # opts.model_path = "./results/icl_nuim_n768_unseen0_noise0_seed123/model.pth"

    return opts

def test(opts, model, test_loader):
    rcd_times, n_cnt = 0, 0
    with torch.no_grad():
        r_mses, t_mses, r_maes, t_maes = [], [], [], []
        for srcs, tgts, rs_lb, ts_lb in tqdm(test_loader):
            srcs, tgts = srcs.cuda(), tgts.cuda()
            t1 = time.time()
            results = model(srcs, tgts)
            rcd_times += time.time() - t1
            n_cnt += len(srcs)
            rs_pred, ts_pred = results["r"], results["t"]
            r_mses.append(np.mean((np.degrees(rs_pred.cpu().numpy()) - np.degrees(rs_lb.numpy())) ** 2, 1))
            r_maes.append(np.mean(np.abs(np.degrees(rs_pred.cpu().numpy()) - np.degrees(rs_lb.numpy())), 1))
            t_mses.append(np.mean((ts_pred.cpu().numpy() - ts_lb.numpy()) ** 2, 1))
            t_maes.append(np.mean(np.abs(ts_pred.cpu().numpy() - ts_lb.numpy()), 1))

            r_mse = np.mean(np.concatenate(r_mses, 0)).item()
            t_mse = np.mean(np.concatenate(t_mses, 0)).item()
            r_mae = np.mean(np.concatenate(r_maes, 0)).item()
            t_mae = np.mean(np.concatenate(t_maes, 0)).item()

            print("--- Test: r_mse: %.8f, t_mse: %.8f, r_rmse: %.8f, t_rmse: %.8f, r_mae: %.8f, t_mae: %.8f, time: %.8f ---" % (
                                                    r_mse, t_mse, np.sqrt(r_mse), np.sqrt(t_mse), r_mae, t_mae, rcd_times / n_cnt))

def model_test(opts):
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ## initial setting
    opts.recorder = Recorder(opts)
    model = GuidedCEM(opts).to(opts.device).load_model(opts.model_path)

    test_loader, db1 = get_dataset(opts, db_nm=opts.db_nm, partition="test", is_normal=False,  batch_size=opts.batch_size, shuffle=False, drop_last=False, cls_idx=opts.db.cls_idx)
    ## testing
    test(opts, model, test_loader)

if __name__ == '__main__':
    opts = init_opts(opts)
    model_test(opts)