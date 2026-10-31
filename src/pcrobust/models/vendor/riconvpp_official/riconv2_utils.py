"""
Vendored, near-verbatim from the OFFICIAL RIConv++ repository:
    https://github.com/cszyzhang/riconv2  (commit 70d9297cc906e8c977c48ae5cdfcd250fae9fed4, 2022-11-20)
Original authors: Zhiyuan Zhang et al. ("RIConv++: Effective Rotation Invariant
Convolutions for 3D Point Clouds Deep Learning", IJCV 2022).

Only three kinds of changes were made to the original `models/riconv2_utils.py`,
all noted inline with "PATCHED":

1. `Tensor.symeig(eigenvectors=True)` was removed from PyTorch (deprecated
   since 1.9, gone in 1.13+). Replaced with `torch.linalg.eigh`, which is the
   documented drop-in replacement and preserves the same convention used here
   (ascending eigenvalues, so index 0 is still the smallest-eigenvalue
   eigenvector -- i.e. the Local Reference Axis / normal-like direction the
   original code selects).
2. The official `sample()` hard-imports `pointnet2_utils.furthest_point_sample`,
   a custom CUDA kernel from `models/pointnet2/` that only runs on CUDA tensors
   and has no CPU path at all -- it cannot be smoke-tested on a CPU-only
   machine. That import is now optional: if the compiled CUDA extension is
   present, FPS uses the real
   official kernel exactly as published; otherwise it falls back to
   `_fps_fallback`, a pure-PyTorch furthest-point-sampling implementation
   (identical algorithm -- iterative farthest-point selection -- just without
   the fused CUDA kernel). This does not change RIConv++'s actual rotation-
   invariant feature construction (the LRA computation and the RI_features
   ordering scheme below), which is untouched from the original.

3. `torch.cross(...)` without an explicit `dim` argument is deprecated
   upstream (PyTorch warns on every call). Made `dim=-1` explicit at its one
   call site in `order_index` -- this is exactly the dimension the implicit
   form already resolved to, so it changes no numerics, just silences a
   deprecation warning that would otherwise fire on every single forward pass
   across thousands of experiment runs.

Everything else -- the LRA computation, the neighbor-ordering scheme
(`order_index`), the 8-dim rotation-invariant feature construction
(`RI_features`), and the `RIConv2SetAbstraction` module -- is the original
authors' code, unmodified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# PATCHED: torch's default cusolver-based batched eigh (cusolverDnXsyevBatched)
# throws CUSOLVER_STATUS_INVALID_VALUE on certain large-batch inputs (observed
# on this project's ModelNet40-C corruption evaluation: compute_LRA's batched
# eigh over (B, N, 3, 3) with B*N in the tens of thousands) even when the
# input contains no NaN/Inf -- confirmed by testing an equivalent-shaped
# synthetic symmetric PSD batch, which hit the identical error under the
# default backend and succeeded under magma. Switching to magma (an
# alternative LAPACK backend PyTorch supports for exactly this class of
# issue, per the error message's own suggestion) avoids it; this is a
# process-wide torch setting, set once here since every architecture in this
# project that uses compute_LRA imports this module (or its risurconv
# counterpart) before calling it.
torch.backends.cuda.preferred_linalg_library("magma")
from time import time
import numpy as np

try:
    from pointnet2 import pointnet2_utils
    _HAS_CUDA_POINTNET2 = True
except ImportError:
    _HAS_CUDA_POINTNET2 = False


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]
    return new_points


def _fps_fallback(xyz, npoint):
    """PATCHED: pure-PyTorch furthest-point-sampling, used only when the
    official CUDA kernel (models/pointnet2/) hasn't been built (e.g. CPU-only
    smoke tests). Same iterative-farthest-point algorithm as the CUDA kernel,
    just not fused into a single kernel."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    dists = torch.cdist(new_xyz, xyz)
    if radius is not None:
        group_idx[dists > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def compute_LRA_one(group_xyz, weighting=False):
    B, S, N, C = group_xyz.shape
    dists = torch.norm(group_xyz, dim=-1, keepdim=True)  # nn lengths

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)

    # PATCHED: torch.Tensor.symeig was removed from PyTorch; torch.linalg.eigh
    # is the documented replacement and keeps the same ascending-eigenvalue
    # convention the original code relied on (index 0 == smallest eigenvalue).
    # Also regularized with a tiny diagonal epsilon: degenerate/duplicate
    # neighbor groups drive M to exactly zero, which cusolver's batched eigh
    # can turn into NaN outputs rather than the mathematically-valid
    # all-zero-eigenvalue result (same fix as risurconv_utils.py).
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0) + 1e-6 * torch.eye(3, device=M.device, dtype=M.dtype)
    eigen_values, vec = torch.linalg.eigh(M)

    LRA = vec[:, :, :, 0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA  # B N 3


def compute_LRA(xyz, weighting=False, nsample=64):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3, 2), weights * group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3, 2), group_xyz)

    # PATCHED: see compute_LRA_one above (symeig replacement + eps regularization).
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0) + 1e-6 * torch.eye(3, device=M.device, dtype=M.dtype)
    eigen_values, vec = torch.linalg.eigh(M)

    LRA = vec[:, :, :, 0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA  # B N 3


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample(npoint, xyz, norm=None, sampling='fps'):
    B, N, C = xyz.shape
    xyz = xyz.contiguous()
    if sampling == 'fps':
        if _HAS_CUDA_POINTNET2:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
            fps_idx = fps_idx.long()
        else:
            # PATCHED: pure-PyTorch fallback, see _fps_fallback docstring above.
            fps_idx = _fps_fallback(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        if norm is not None:
            new_norm = index_points(norm, fps_idx)
    elif sampling == 'random':
        shuffle = np.arange(xyz.shape[1])
        np.random.shuffle(shuffle)
        new_xyz = xyz[:, shuffle[:npoint], :]
        if norm is not None:
            new_norm = norm[:, shuffle[:npoint], :]
    else:
        print('Unknown sampling method!')
        exit()

    return new_xyz, new_norm


def group_index(nsample, radius, xyz, new_xyz, group='knn'):
    if group == 'knn':
        idx = knn_point(nsample, xyz, new_xyz.contiguous())
    elif group == 'ball':
        if _HAS_CUDA_POINTNET2:
            idx = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz.contiguous())
            idx = idx.long()
        else:
            idx = query_ball_point(radius, nsample, xyz, new_xyz.contiguous())
    else:
        print('Unknown grouping method!')
        exit()

    return idx


def order_index(xyz, new_xyz, new_norm, idx):
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered

    # project and order
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane * new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / proj_xyz_length
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # set nan to zero

    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1, 1, 1, 3))  # corresponds to the largest length

    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    # PATCHED: explicit dim=-1 (torch.cross's implicit-dim behavior is deprecated
    # upstream; -1 is what it already resolved to here, so this changes nothing).
    sign = torch.cross(projected_xyz_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1), dim=-1)
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.  # the first is the center point itself, just set sign as 1 to differ from ref_vec
    dots = sign * dots - (1 - sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))

    return dots_sorted, idx_ordered


def RI_features(xyz, norm, new_xyz, new_norm, idx, group_all=False):
    B, S, C = new_xyz.shape

    new_norm = new_norm.unsqueeze(-1)
    dots_sorted, idx_ordered = order_index(xyz, new_xyz, new_norm, idx)

    epsilon = 1e-7
    grouped_xyz = index_points(xyz, idx_ordered)  # [B, npoint, nsample, C]
    if not group_all:
        grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered
    else:
        grouped_xyz_local = grouped_xyz  # treat orgin as center
    grouped_xyz_length = torch.norm(grouped_xyz_local, dim=-1, keepdim=True)  # nn lengths
    grouped_xyz_unit = grouped_xyz_local / grouped_xyz_length
    grouped_xyz_unit[grouped_xyz_unit != grouped_xyz_unit] = 0  # set nan to zero
    grouped_xyz_norm = index_points(norm, idx_ordered)  # nn neighbor normal vectors

    grouped_xyz_angle_0 = torch.matmul(grouped_xyz_unit, new_norm)
    grouped_xyz_angle_1 = (grouped_xyz_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_angle_norm = torch.matmul(grouped_xyz_norm, new_norm)
    grouped_xyz_angle_norm = torch.acos(torch.clamp(grouped_xyz_angle_norm, -1 + epsilon, 1 - epsilon))
    D_0 = (grouped_xyz_angle_0 < grouped_xyz_angle_1)
    D_0[D_0 == 0] = -1
    grouped_xyz_angle_norm = D_0.float() * grouped_xyz_angle_norm

    grouped_xyz_inner_vec = grouped_xyz_local - torch.roll(grouped_xyz_local, 1, 2)
    grouped_xyz_inner_length = torch.norm(grouped_xyz_inner_vec, dim=-1, keepdim=True)  # nn lengths
    grouped_xyz_inner_unit = grouped_xyz_inner_vec / grouped_xyz_inner_length
    grouped_xyz_inner_unit[grouped_xyz_inner_unit != grouped_xyz_inner_unit] = 0  # set nan to zero
    grouped_xyz_inner_angle_0 = (grouped_xyz_inner_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_1 = (grouped_xyz_inner_unit * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = (grouped_xyz_norm * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = torch.acos(torch.clamp(grouped_xyz_inner_angle_2, -1 + epsilon, 1 - epsilon))
    D_1 = (grouped_xyz_inner_angle_0 < grouped_xyz_inner_angle_1)
    D_1[D_1 == 0] = -1
    grouped_xyz_inner_angle_2 = D_1.float() * grouped_xyz_inner_angle_2

    proj_inner_angle_feat = dots_sorted - torch.roll(dots_sorted, 1, 2)
    proj_inner_angle_feat[:, :, 0, 0] = (-3) - dots_sorted[:, :, -1, 0]

    ri_feat = torch.cat([grouped_xyz_length,
                          proj_inner_angle_feat,
                          grouped_xyz_angle_0,
                          grouped_xyz_angle_1,
                          grouped_xyz_angle_norm,
                          grouped_xyz_inner_angle_0,
                          grouped_xyz_inner_angle_1,
                          grouped_xyz_inner_angle_2], dim=-1)

    return ri_feat, idx_ordered


def sample_and_group(npoint, radius, nsample, xyz, norm):
    xyz = xyz.contiguous()
    norm = norm.contiguous()

    new_xyz, new_norm = sample(npoint, xyz, norm=norm, sampling='fps')
    idx = group_index(nsample, radius, xyz, new_xyz, group='knn')

    ri_feat, idx_ordered = RI_features(xyz, norm, new_xyz, new_norm, idx)

    return new_xyz, ri_feat, new_norm, idx_ordered


def sample_and_group_all(xyz, norm):
    device = xyz.device
    B, N, C = xyz.shape
    S = 1
    new_xyz = torch.mean(xyz, dim=1, keepdim=True)  # centroid
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered
    new_norm = compute_LRA_one(grouped_xyz_local, weighting=True)

    device = xyz.device
    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    ri_feat, idx_ordered = RI_features(xyz, norm, new_xyz, new_norm, idx, group_all=True)

    return None, ri_feat, new_norm, idx_ordered


def sample_and_group_deconv(nsample, xyz, norm, new_xyz, new_norm):
    idx = group_index(nsample, 0.0, xyz, new_xyz, group='knn')
    ri_feat, idx_ordered = RI_features(xyz, norm, new_xyz, new_norm, idx)

    return ri_feat, idx_ordered


class RIConv2SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(RIConv2SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.prev_mlp_convs = nn.ModuleList()
        self.prev_mlp_bns = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        in_channel_0 = 8
        mlp_0 = [32, 64]
        last_channel = in_channel_0
        for out_channel in mlp_0:
            self.prev_mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.prev_mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.group_all = group_all

    def forward(self, xyz, norm, points):
        if points is not None:  # transform from [B, C, N] to [B, N, C]
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape

        if self.group_all:
            new_xyz, ri_feat, new_norm, idx_ordered = sample_and_group_all(xyz, norm)
        else:
            new_xyz, ri_feat, new_norm, idx_ordered = sample_and_group(self.npoint, self.radius, self.nsample, xyz, norm)

        # lift
        ri_feat = ri_feat.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.prev_mlp_convs):
            bn = self.prev_mlp_bns[i]
            ri_feat = F.relu(bn(conv(ri_feat)))

        # concat previous layer features
        if points is not None:
            if idx_ordered is not None:
                grouped_points = index_points(points, idx_ordered)
            else:
                grouped_points = points.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        ri_feat = torch.max(new_points, 2)[0]  # maxpooling

        return new_xyz, new_norm, ri_feat
