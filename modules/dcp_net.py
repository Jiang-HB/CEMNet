import torch.nn as nn, torch
from modules.dgcnn import DGCNN
from utils.mat2euler import mat2euler

def pairwise_distance_batch(x,y):
    xx = torch.sum(torch.mul(x,x), 1, keepdim = True)#[b,1,n]
    yy = torch.sum(torch.mul(y,y),1, keepdim = True) #[b,1,n]
    inner = -2*torch.matmul(x.transpose(2,1),y) #[b,n,n]
    pair_distance = xx.transpose(2,1) + inner + yy #[b,n,n]
    device = torch.device('cuda')
    zeros_matrix = torch.zeros_like(pair_distance,device = device)
    pair_distance_square = torch.where(pair_distance > 0.0,pair_distance,zeros_matrix)
    error_mask = torch.le(pair_distance_square,0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float()*1e-16)
    pair_distances = torch.mul(pair_distances,(1.0-error_mask.float()))
    return pair_distances

class DCPNet(nn.Module):
    def __init__(self, opts):
        super(DCPNet, self).__init__()
        self.emb_nn = DGCNN(emb_dims = opts.emb_dims)
        self.emb_dims = opts.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

        self.opts = opts
        self.planning_horizon = self.opts.cem.planning_horizon
        self.nn = nn.Sequential(nn.Linear(self.emb_dims * 2, self.emb_dims // 4),
                                nn.BatchNorm1d(self.emb_dims // 4),
                                nn.LeakyReLU(),
                                nn.Linear(self.emb_dims // 4, self.emb_dims // 8),
                                nn.BatchNorm1d(self.emb_dims // 8),
                                nn.LeakyReLU(),
                                # nn.Linear(self.emb_dims // 4, self.emb_dims // 8),
                                # nn.BatchNorm1d(self.emb_dims // 8),
                                # nn.LeakyReLU(),
                                nn.Linear(self.emb_dims // 8, 6))

    def forward(self, srcs, tgts, is_sigma=False):
        batch_size = len(srcs)
        srcs_emb = self.emb_nn(srcs)  # 3, 512, 1024
        tgts_emb = self.emb_nn(tgts)
        scores = -pairwise_distance_batch(srcs_emb, tgts_emb)
        scores = torch.softmax(scores, dim=2)
        src_corr = torch.matmul(tgts, scores.transpose(2, 1).contiguous())
        src_centered = srcs - srcs.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        U, S, V = [], [], []
        R = []

        for i in range(srcs.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)
            U.append(u)
            S.append(s)
            V.append(v)
        R = torch.stack(R, dim=0)
        t = (torch.matmul(-R, srcs.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)).reshape(batch_size, -1)
        r = mat2euler(R)
        if is_sigma:
            sg = torch.cat([srcs_emb, tgts_emb], 1)
            sigmas = self.nn(sg.max(dim=-1)[0])
            sigmas[:, :3] = torch.nn.Sigmoid()(sigmas[:, :3]) * 1.0
            sigmas[:, 3:] = torch.nn.Sigmoid()(sigmas[:, 3:]) * 1.0
            return torch.cat([r, t], 1).unsqueeze(0), sigmas.unsqueeze(0)
        else:
            return {"r": r, "t": t}