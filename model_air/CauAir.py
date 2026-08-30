"""CauAir baseline adapted for BrainAI KnowAir loaders.

Reference: PoorOtterBob/CauAir, src/models/cauair.py.
The original model consumes pollutant history and covariates; this wrapper keeps
that design while matching BrainAI's model_air forward(pm25_hist, feature) API.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, p: float = -1.0, eps: float = 1e-8, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-0.5)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, expand_ratio: int = 4, dropout: float = 0.3):
        super().__init__()
        hidden = expand_ratio * dim_in
        self.W1 = nn.Linear(dim_in, hidden)
        self.W2 = nn.Linear(dim_in, hidden)
        self.W3 = nn.Linear(hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class CachAttention(nn.Module):
    def __init__(self, dim: int, dim_attn: int, rank: int, head: int = 4):
        super().__init__()
        if rank <= 0:
            raise ValueError('rank must be positive for CauAir CachAttention.')
        if head <= 0:
            raise ValueError('head must be positive for CauAir CachAttention.')
        if dim_attn % head != 0:
            raise ValueError('dim_attn must be divisible by head.')
        self.head_dim = dim_attn // head
        self.query = nn.Linear(dim, dim_attn)
        self.key = nn.Parameter(torch.randn((rank, head, self.head_dim)))
        self.value = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.full((head, 1), math.log(9.0)))
        self.beta = nn.Parameter(torch.full((head, 1), math.log(0.01)))
        self.rank = rank
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, f = x.shape
        q = self.query(x).reshape(b, n, self.head, self.head_dim)
        attn = torch.einsum('bnhd,rhd->bnhr', q, self.key) / (self.head_dim ** 0.5)
        v0 = self.value(x).reshape(b, n, self.head, self.head_dim)
        v = torch.einsum('bnhr,bnhd->brhd', F.softmax(attn, dim=-1), v0)
        v = torch.einsum('bnhr,brhd->bnhd', F.softmax(attn, dim=-3), v)
        v = torch.sigmoid(self.alpha) * v0 + torch.sigmoid(self.beta) * v
        return v.reshape(b, n, f)


class CachLormer(nn.Module):
    def __init__(self, dim: int, head: int = 4, rank: int = 10):
        super().__init__()
        self.MHA = CachAttention(dim, dim, rank, head)
        self.FFN = SwiGLU_FFN(dim, dim)
        self.norm1 = RMSNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(math.log(9.0)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2.0) + 1.0)))
        self.gamma = nn.Parameter(torch.tensor(math.log(9.0)))
        self.delta = nn.Parameter(torch.tensor(math.log(math.sqrt(2.0) + 1.0)))

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.norm1(torch.sigmoid(self.gamma) * x + torch.sigmoid(self.delta) * z)
        return torch.sigmoid(self.alpha) * self.FFN(x) + torch.sigmoid(self.beta) * self.MHA(x)


class CauAir(nn.Module):
    """CauAir model with BrainAI KnowAir baseline API.

    Args mirror the project baselines. ``in_dim`` is the concatenated dimension
    used by existing BrainAI models: PM2.5 history channel plus per-domain
    covariates. In ``forward`` we receive PM2.5 history separately, so the
    covariate dimension is ``feature.shape[-1]``.
    """

    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        in_dim: int,
        city_num: int,
        batch_size: int = 32,
        device=None,
        out_dim: int = 1,
        dim: int = 128,
        rank: int = 10,
        head: int = 2,
        cov_dim=None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.in_dim = int(in_dim)
        self.city_num = int(city_num)
        self.batch_size = int(batch_size)
        self.device = device
        self.out_dim = int(out_dim)
        self.dim = int(dim)
        self.rank = int(rank)
        self.head = int(head)
        self.cov_dim = int(cov_dim) if cov_dim is not None else max(1, int(in_dim) - 1)

        self.encoder1 = SwiGLU_FFN(self.hist_len, self.dim, dropout=dropout)
        self.encoder11 = SwiGLU_FFN(self.hist_len, self.dim, dropout=dropout)
        self.encoder2 = SwiGLU_FFN(self.hist_len * self.cov_dim, self.dim, dropout=dropout)
        self.encoder3 = SwiGLU_FFN(self.pred_len * self.cov_dim, self.dim, dropout=dropout)
        self.decoder = SwiGLU_FFN(self.dim, self.out_dim * self.pred_len, dropout=dropout)

        self.position1 = nn.Parameter(torch.zeros((self.city_num, self.dim)))
        self.position2 = nn.Parameter(torch.zeros((self.city_num, self.dim)))
        self.position3 = nn.Parameter(torch.zeros((self.city_num, self.dim)))
        self.position4 = nn.Parameter(torch.zeros((self.city_num, self.dim)))
        self.norm1 = RMSNorm(self.dim)
        self.norm2 = RMSNorm(self.dim)
        self.norm3 = RMSNorm(self.dim)
        self.norm4 = RMSNorm(self.dim)
        self.module1 = CachLormer(self.dim, self.head, self.rank)
        self.module2 = CachLormer(self.dim, self.head, self.rank)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def _pad_or_trim_feature(self, feature: torch.Tensor, target_len: int) -> torch.Tensor:
        if feature.size(1) == target_len:
            return feature
        if feature.size(1) > target_len:
            return feature[:, :target_len]
        pad = feature[:, -1:].repeat(1, target_len - feature.size(1), 1, 1)
        return torch.cat([feature, pad], dim=1)

    def forward(self, pm25_hist: torch.Tensor, feature: torch.Tensor):
        if pm25_hist.dim() != 4 or feature.dim() != 4:
            raise ValueError('CauAir expects pm25_hist and feature as [B,T,N,C].')
        cov_hist = self._pad_or_trim_feature(feature[:, :self.hist_len], self.hist_len)
        cov_future = self._pad_or_trim_feature(feature[:, self.hist_len:self.hist_len + self.pred_len], self.pred_len)

        x_pm25 = pm25_hist[..., 0].transpose(1, 2)
        z_hist = cov_hist.transpose(1, 2).reshape(-1, self.city_num, self.hist_len * self.cov_dim)
        z_future = cov_future.transpose(1, 2).reshape(-1, self.city_num, self.pred_len * self.cov_dim)

        z = self.encoder2(z_hist) + self.norm2(self.position2)
        label_z = self.encoder3(z_future) + self.norm3(self.position3)
        x1 = self.encoder1(x_pm25) + self.norm1(self.position1)
        x2 = self.encoder11(x_pm25) + self.norm4(self.position4)
        hidden = torch.sigmoid(self.alpha) * self.module1(x1, z) + torch.sigmoid(self.beta) * self.module2(x2, label_z)
        out = self.decoder(hidden).reshape(-1, self.city_num, self.pred_len, self.out_dim)
        prediction = out.permute(0, 2, 1, 3).contiguous()
        activate_freq = float((hidden > 0).float().mean().detach().cpu().item())
        return prediction, activate_freq
