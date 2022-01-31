from cems.base_cem import BaseCEM
from modules.dcp_net import DCPNet
from modules.sparsemax import Sparsemax
from utils.commons import stack_transforms_seq
import pdb, torch

class VanillaCEM(BaseCEM):

    def __init__(self, opts):
        super(VanillaCEM, self).__init__(opts)
        self.opts = opts

class GuidedCEM(BaseCEM):

    def __init__(self, opts):
        super(GuidedCEM, self).__init__(opts)
        self.coarse_policy = DCPNet(opts)
        self.top_k = Sparsemax(dim=1)
        self.opts = opts

    def add_exploration_noise(self, mus, sigmas):
        mus_ = torch.zeros_like(mus).to(self.device)
        sigmas_ = torch.ones_like(mus_).to(self.device) # (B, 6)
        sigmas_[:, :3] *= self.r_init_sigma
        sigmas_[:, 3:] *= self.t_init_sigma
        r = self.opts.exploration_weight
        mus = (1 - r) * mus + r * mus_
        sigmas = (1 - r) * sigmas + r * sigmas_
        return mus, sigmas

    def init_distrib(self):
        mus, sigmas = self.coarse_policy(self.srcs, self.tgts, is_sigma=True) # (H, B, 6)
        if not self.opts.is_train:
            mus, sigmas = self.add_exploration_noise(mus, sigmas)
        self.mus, self.sigmas = mus.unsqueeze(2), sigmas.unsqueeze(2) # (H, B, 1, 6)
        self.mus_init, self.sigmas_init = self.mus, self.sigmas
        init_r_t = stack_transforms_seq(self.mus.squeeze(2)).clone() # (B, 6)
        self.r_init = init_r_t[:, :3] # (B, 3)
        self.t_init = init_r_t[:, 3:] # (B, 3)

    def elite_selection(self):
        super().elite_selection()
        self.sparsemax_probs = self.top_k(-self.alignment_errors).reshape(1, self.batch_size, self.n_candidates, 1)  # (1, B, C, 1)
        self.elites = self.candidates.reshape(self.planning_horizon, self.batch_size, self.n_candidates, self.action_size) * self.sparsemax_probs  # (H, B, C, 6)

    def update_distrib(self):
        self.mus = self.elites.sum(dim=2, keepdim=True)  # (H, B, 1, 6)
        self.sigmas = (((self.candidates.reshape(self.planning_horizon, self.batch_size, self.n_candidates, self.action_size) -
                         self.mus) ** 2) * self.sparsemax_probs).sum(dim=2, keepdim=True).sqrt() # (H, B, 1, 6)

    def load_model(self,  model_path):
        self.load_state_dict(torch.load(model_path))
        return self