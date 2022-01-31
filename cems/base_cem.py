import torch, cemnet_lib
from utils.euler2mat import euler2mat_torch
from utils.commons import stack_transforms_seq
from utils.transform_pc import transform_pc_torch
from utils.batch_icp import batch_icp

class BaseCEM(torch.nn.Module):
    def __init__(self, opts):
        super(BaseCEM, self).__init__()
        self.opts = opts
        self.n_iters = opts.cem.n_iters
        self.n_candidates = opts.cem.n_candidates[0]
        self.planning_horizon = opts.cem.planning_horizon
        self.n_elites = opts.cem.n_elites
        self.r_size, self.t_size, self.action_size = 3, 3, 6
        self.min_r, self.max_r = opts.cem.r_range
        self.min_t, self.max_t = opts.cem.t_range[opts.db_nm]
        self.r_init_sigma, self.t_init_sigma = opts.cem.init_sigma[opts.db_nm]
        self.recorder = opts.recorder
        self.device = opts.device

    def init_distrib(self):
        # mus
        mus_r = torch.zeros([self.planning_horizon, self.batch_size, 1, self.r_size]) + (self.min_r + self.max_r) / 2. # (H, B, 1, 3)
        mus_t = torch.zeros([self.planning_horizon, self.batch_size, 1, self.t_size]) + (self.min_t + self.max_t) / 2. # (H, B, 1, 3)
        self.mus = torch.cat([mus_r, mus_t], 3).to(self.device) # (H, B, 1, 6)
        init_r_t = stack_transforms_seq(self.mus.squeeze(2)) # (B, 6)
        self.r_init = init_r_t[:, :3] # (B, 3)
        self.t_init = init_r_t[:, 3:] # (B, 3)
        # sigmas
        self.sigmas = torch.ones_like(self.mus).to(self.device) # (H, B, 1, 6)
        self.sigmas[:, :, :, :3] *= self.r_init_sigma # (H, B, 1, 3)
        self.sigmas[:, :, :, 3:] *= self.t_init_sigma # (H, B, 1, 3)
        self.mus_init = self.mus
        self.sigmas_init = self.sigmas

    def sample_candidates(self):
        self.candidates = self.mus + self.sigmas * torch.randn(self.planning_horizon, self.batch_size, self.n_candidates, self.action_size).to(self.device)  # (H, B, C, 6)

    def perform_candidates(self):
        stack_candidates = stack_transforms_seq(self.candidates) # (B * C, 6)
        Rs = euler2mat_torch(stack_candidates[:, :self.r_size], seq="zyx")  # (B * C, 3, 3)
        ts = stack_candidates[:, self.r_size:]  # (B * C, 3)
        transformed_srcs_repeat = transform_pc_torch(self.srcs_repeat, Rs, ts).reshape(self.batch_size, self.n_candidates, 3, -1)  # (B, C, 3, N)
        return transformed_srcs_repeat

    def distance(self, srcs, tgt):
        if self.opts.cem.metric_type[0] == "MC":
            return cemnet_lib.mc_distance(srcs, tgt, self.opts.cem.metric_type[1])
        if self.opts.cem.metric_type[0] == "CD":
            return cemnet_lib.cd_distance(srcs, tgt)
        if self.opts.cem.metric_type[0] == "GM":
            return cemnet_lib.gm_distance(srcs, tgt, self.opts.cem.metric_type[1]) * 100

    def evaluate_candidates(self):
        transformed_srcs_repeat = self.perform_candidates() # (B, C, 3, N)
        ## current reward
        self.alignment_errors = torch.zeros(self.batch_size, self.n_candidates).to(self.device)  # (B, C)
        for k, (transformed_src_repeat, tgt) in enumerate(zip(transformed_srcs_repeat, self.tgts)):
            self.alignment_errors[k] = self.distance(transformed_src_repeat, tgt).detach()
        ## future potential
        if self.opts.cem.is_fused_reward[0] and self.k < self.opts.cem.is_fused_reward[2]:
            alpha = self.opts.cem.is_fused_reward[1]
            for k, (transformed_src_repeat, tgt) in enumerate(zip(transformed_srcs_repeat, self.tgts)):
                Rs_icp, ts_icp = batch_icp(self.opts, transformed_src_repeat, tgt) # (C, 3, 3), (C, 3)
                transformed_src_repeat_icp = transform_pc_torch(transformed_src_repeat, Rs_icp, ts_icp) # (C, 3, N)
                potential = self.distance(transformed_src_repeat_icp, tgt).detach()
                self.alignment_errors[k] = alpha * self.alignment_errors[k] + (1- alpha) * potential
        self.alignment_errors = self.alignment_errors.detach()
        return self.alignment_errors

    def update_distrib(self):
        self.mus = self.elites.mean(dim=2, keepdim=True) # mus: [H, B, 1, 6]
        self.sigmas = self.elites.std(dim=2, unbiased=False, keepdim=True) # sigmas: [H, B, 1, 6]

    def elite_selection(self):
        self.candidates = self.candidates.reshape(self.planning_horizon, self.batch_size * self.n_candidates, self.action_size)  # (H, B * C, 6)
        self.alignment_errors = self.evaluate_candidates() # (B, C)
        self.elite_errors, self.elite_idxs = self.alignment_errors.topk(self.n_elites, dim=1, largest=False, sorted=False)  # (B, K)
        self.elite_idxs += self.n_candidates * torch.arange(0, self.batch_size, dtype=torch.int64).reshape(self.batch_size, 1).to(self.device)  # topk: (B, K)
        self.elite_idxs = self.elite_idxs.view(-1)  # (B * K, )
        self.elites = self.candidates[:, self.elite_idxs].reshape(self.planning_horizon, self.batch_size, self.n_elites, self.action_size)  # (H, B, K, 6)
        return self.elites

    def forward(self, srcs, tgts):
        """
        :param ids:  ids
        :param srcs: (B, 3, N), torch.FloatTensor
        :param tgts: (B, 3, N), torch.FloatTensor
        :return: None
        """
        self.batch_size, self.n_points = srcs.size(0), srcs.size(2)
        self.srcs, self.tgts = srcs.to(self.device), tgts.to(self.device) # (B, 3, N)
        self.srcs_repeat = self.srcs.unsqueeze(1).repeat(1, self.n_candidates, 1, 1).reshape(self.batch_size * self.n_candidates,
                                                                                             3, self.n_points) # (B * C, 3, N)
        self.init_distrib() # (H, B, 1, 6)
        self.k = 0
        for iter_idx in range(self.n_iters):

            # 1. sample candidates
            self.sample_candidates()

            # 2. elite selection
            self.elite_selection()

            # 3. update distribution
            self.update_distrib()

            self.k += 1

        actions = stack_transforms_seq(self.mus.squeeze(2)) # (B, 6)
        return {"r": actions[:, :3], "t": actions[:, 3:], "r_init": self.r_init, "t_init": self.t_init}