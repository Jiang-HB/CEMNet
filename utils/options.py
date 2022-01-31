from utils.attr_dict import AttrDict
import numpy as np, torch

opts = AttrDict()
## general setting
opts.db_nm = "scene7" # "modelnet40", "scene7", "icl_nuim
opts.is_debug = False
opts.device=torch.device("cuda")
opts.seed = 123
opts.batch_size = 35
opts.minibatch_size = 35
opts.n_epochs = 30

## dataset
opts.db = AttrDict(
    modelnet40 = AttrDict(
        path = "/test/datasets/registration/modelnet40/modelnet40_normal_n2048.pth",
        cls_idx = -1,  # None_class: -1, airplane: 0, car: 7, chair: 8, table: 33, lamp: 19
        is_neg_angle = False,
        unseen = False,
        gaussian_noise = True,
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

## cem module
opts.cem = AttrDict(
    metric_type = ["iou", {"epsilon": 0.01}],  # "iou", "cd"
    n_candidates = [1000, {"minibatch": 1000}],
    n_elites = 25,
    n_iters = 10,
    r_range = [-np.pi, np.pi],
    t_range = AttrDict(
        modelnet40 = [-1.0, 1.0],
        scene7 = [-1.0, 1.0],
        icl_nuim = [-1.0, 1.0]
    ),
    init_sigma = AttrDict(
        modelnet40 = [1.0, 0.5],
        scene7 = [1.0, 0.5],
        icl_nuim = [1.0, 0.5]
    ),
    planning_horizon = 1,
    is_icp_modification = [True, 0.5, 3]
)

# network setting
opts.pointer = "identity"  # or "transformer", "identity"
opts.head = "svd"  # "mlp", "svd
opts.eval = False
opts.emb_nn = "dgcnn"
opts.emb_dims = 512
opts.ff_dim = 1024
opts.n_blocks = 1
opts.n_heads = 4
opts.dropout = 0.
