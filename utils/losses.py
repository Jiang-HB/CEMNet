import torch, torch.nn as nn, pdb

class CDLoss(nn.Module):
    def __init__(self, opts):
        super(CDLoss, self).__init__()
        self.device = opts.device

    def forward(self, srcs, tgts):
        P = self.pairwise_distance(srcs, tgts)
        return torch.min(P, 1)[0].mean() + torch.min(P, 2)[0].mean()

    def pairwise_distance(self, srcs, tgts):
        srcs, tgts = srcs.transpose(2, 1), tgts.transpose(2, 1)
        batch_size, n_points_src, _ = srcs.size()
        _, n_points_tgt, _ = tgts.size()
        srcs_dist = torch.bmm(srcs, srcs.transpose(2, 1)) # (B, n_points_src, n_points_src)
        tgts_dist = torch.bmm(tgts, tgts.transpose(2, 1)) # (B, n_points_tgt, n_points_tgt)
        srcs_tgts_dist = torch.bmm(srcs, tgts.transpose(2, 1)) # (B, n_points_src, n_points_tgt)
        diag_ind_srcs = torch.arange(0, n_points_src).long().to(self.device)
        diag_ind_tgts = torch.arange(0, n_points_tgt).long().to(self.device)
        rx = srcs_dist[:, diag_ind_srcs, diag_ind_srcs].unsqueeze(1).expand_as(srcs_tgts_dist.transpose(2, 1)) # (B, n_points_tgt, n_points_src)
        ry = tgts_dist[:, diag_ind_tgts, diag_ind_tgts].unsqueeze(1).expand_as(srcs_tgts_dist) # (B, n_points_src, n_points_tgt)
        P = (rx.transpose(2, 1) + ry - 2 * srcs_tgts_dist)
        return P

class GMLoss(nn.Module):
    def __init__(self, opts):
        super(GMLoss, self).__init__()
        self.device = opts.device
        self.opts = opts

    def forward(self, srcs, tgts):
        mu = self.opts.loss_type[1]
        srcs, tgts = srcs.transpose(2, 1), tgts.transpose(2, 1)
        P = torch.norm(srcs[:, :, None, :] - tgts[:, None, :, :], dim=-1, p=2).pow(2.0)
        distances = torch.cat([torch.min(P, 1)[0].unsqueeze(-1), torch.min(P, 2)[0].unsqueeze(-1)], -1)
        losses = ((mu * distances) / (distances + mu)).sum(2).mean(1).mean()
        return losses

    def pairwise_distance(self, srcs, tgts):
        batch_size, n_points_src, _ = srcs.size()
        _, n_points_tgt, _ = tgts.size()
        srcs_dist = torch.bmm(srcs, srcs.transpose(2, 1)) # (B, n_points_src, n_points_src)
        tgts_dist = torch.bmm(tgts, tgts.transpose(2, 1)) # (B, n_points_tgt, n_points_tgt)
        srcs_tgts_dist = torch.bmm(srcs, tgts.transpose(2, 1)) # (B, n_points_src, n_points_tgt)
        diag_ind_srcs = torch.arange(0, n_points_src).long().to(self.device)
        diag_ind_tgts = torch.arange(0, n_points_tgt).long().to(self.device)
        rx = srcs_dist[:, diag_ind_srcs, diag_ind_srcs].unsqueeze(1).expand_as(srcs_tgts_dist.transpose(2, 1)) # (B, n_points_tgt, n_points_src)
        ry = tgts_dist[:, diag_ind_tgts, diag_ind_tgts].unsqueeze(1).expand_as(srcs_tgts_dist) # (B, n_points_src, n_points_tgt)
        P = (rx.transpose(2, 1) + ry - 2 * srcs_tgts_dist)
        return P