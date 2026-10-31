"""Vector Neuron core layers (Deng et al., "Vector Neurons: A General
Framework for SO(3)-Equivariant Networks", ICCV 2021).

Faithful reimplementation of FlyingGiraffe/vnn's models/vn_layers.py and
models/utils/vn_dgcnn_util.py, trimmed to exactly what vn_pointnet.py and
vn_dgcnn.py here use.

Convention: a vector feature tensor has shape (B, C, 3, N) or, for edge
features, (B, C, 3, N, K) -- C vector channels, each a 3-vector, per point.
Rotating the input point cloud by R rotates every 3-vector by R
(equivariance); VNStdFeature converts to rotation-invariant scalars.
"""

import torch
import torch.nn as nn

EPS = 1e-6


# ── Linear / nonlinearity / norm ───────────────────────────────────────────────

class VNLinear(nn.Module):
    """Channel-mixing linear map; acts identically on each vector component,
    so it commutes with rotation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):                       # (B, C_in, 3, ...) -> (B, C_out, 3, ...)
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNLeakyReLU(nn.Module):
    """Vector LeakyReLU: learn a direction per channel, reflect the half-space
    component of vectors on the negative side of it."""

    def __init__(self, in_channels: int, share_nonlinearity: bool = False,
                 negative_slope: float = 0.2):
        super().__init__()
        out = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):                       # (B, C, 3, ...)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d))


class VNBatchNorm(nn.Module):
    """BatchNorm on vector norms only -- directions untouched, so the layer
    stays equivariant. dim=4 for (B, C, 3, N), dim=5 for (B, C, 3, N, K)."""

    def __init__(self, num_features: int, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features) if dim in (3, 4) else nn.BatchNorm2d(num_features)

    def forward(self, x):
        norm = torch.norm(x, dim=2) + EPS       # (B, C, N[, K])
        norm_bn = self.bn(norm)
        return x / norm.unsqueeze(2) * norm_bn.unsqueeze(2)


class VNLinearLeakyReLU(nn.Module):
    """The standard VN conv block, fused exactly as in the official repo:
    features p = BN(Wx), nonlinearity direction d = W_d x computed from the
    block INPUT (not from p)."""

    def __init__(self, in_channels: int, out_channels: int, dim: int = 5,
                 share_nonlinearity: bool = False, negative_slope: float = 0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        out_dir = 1 if share_nonlinearity else out_channels
        self.map_to_dir = nn.Linear(in_channels, out_dir, bias=False)

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        return self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))


def mean_pool(x, dim=-1):
    return x.mean(dim=dim)


# ── Invariant read-out ─────────────────────────────────────────────────────────

class VNStdFeature(nn.Module):
    """Learn a data-dependent equivariant frame z0 (B, 3, 3, N) and express the
    features in it: <x, z0> is rotation-INVARIANT. dim=4 input only."""

    def __init__(self, in_channels: int, dim: int = 4, negative_slope: float = 0.2):
        super().__init__()
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim,
                                     negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                     negative_slope=negative_slope)
        self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):                       # (B, C, 3, N)
        z0 = self.vn_lin(self.vn2(self.vn1(x)).transpose(1, -1)).transpose(1, -1)
        # (B, 3ch, 3vec, N) -> (B, 3vec, 3ch, N): the einsum must contract x's
        # vector axis with z0's VECTOR axis (sum_j R_jp R_jq = delta_pq) --
        # without this transpose the contraction hits z0's channel axis and the
        # output is NOT rotation-invariant (verified numerically, float64).
        z0 = z0.transpose(1, 2)
        x_std = torch.einsum('bijm,bjkm->bikm', x, z0)   # (B, C, 3, N), invariant
        return x_std, z0


# ── Graph features (vector-valued EdgeConv inputs) ─────────────────────────────

def knn(x, k: int):
    """x: (B, F, N) flattened features -> (B, N, k) neighbor indices."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1)
    return pairwise.topk(k=k, dim=-1)[1]


def _gather_neighbors(x, k: int):
    """Shared gather for the two graph-feature builders below.
    x: (B, C, 3, N) -> (center, neighbors), both (B, N, K, C, 3)."""
    B, C, _, N = x.shape
    x_flat = x.reshape(B, C * 3, N)
    idx = knn(x_flat, k)                                       # (B, N, k)
    idx_base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    x_t = x_flat.transpose(2, 1).contiguous().view(B * N, C * 3)
    nbrs = x_t[idx, :].view(B, N, k, C, 3)
    ctr = x_flat.transpose(2, 1).view(B, N, 1, C, 3).expand(B, N, k, C, 3)
    return ctr, nbrs


def get_graph_feature(x, k: int = 20):
    """Vector EdgeConv features cat(x_j - x_i, x_i): (B, C, 3, N) -> (B, 2C, 3, N, K)."""
    ctr, nbrs = _gather_neighbors(x, k)
    feat = torch.cat((nbrs - ctr, ctr), dim=3)                 # (B, N, K, 2C, 3)
    return feat.permute(0, 3, 4, 1, 2).contiguous()


def get_graph_feature_cross(x, k: int = 20):
    """Like get_graph_feature but with an extra cross-product channel,
    cat(x_j - x_i, x_i, x_j x x_i): (B, C, 3, N) -> (B, 3C, 3, N, K)."""
    ctr, nbrs = _gather_neighbors(x, k)
    cross = torch.cross(nbrs, ctr, dim=-1)
    feat = torch.cat((nbrs - ctr, ctr, cross), dim=3)          # (B, N, K, 3C, 3)
    return feat.permute(0, 3, 4, 1, 2).contiguous()
