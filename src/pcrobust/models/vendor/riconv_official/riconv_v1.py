"""
Ported (not vendored verbatim -- see below) from the OFFICIAL RIConv repository:
    https://github.com/hkust-vgd/riconv (commit 3a094e5f6e794a72c842f0fbec911839316648ea,
    2019-09-14 "Initial code release")
Original authors: Zhiyuan Zhang, Binh-Son Hua, David W. Rosen, Sai-Kit Yeung,
"Rotation Invariant Convolutions for 3D Point Clouds Deep Learning" (3DV 2019).
Source file: `RIConv.py`, function `RIConv()` / `get_model()`.

Why this is a *port*, not a vendored copy
-------------------------------------------
Every other architecture in this package (RIConv++, RISurConv, PointMLP,
PointTransformer) is vendored by copying the official PyTorch source with
minimal, documented patches. RIConv predates all of those and its official
implementation is TensorFlow 1.x with custom TF ops (`tf_ops/grouping`,
`tf_ops/sampling`) -- there is no PyTorch code to import. This module is a
line-by-line translation of the exact tensor operations in the official
`RIConv()` function (fetched directly from the repository, not reconstructed
from the paper's prose) into PyTorch, preserving every formula exactly:
variable names below mirror the original TF variable names so the
correspondence can be checked line by line against the official repository's
`RIConv.py`. Only framework-mechanical differences apply (TF's
`tf.gather_nd`-based KNN/FPS ops -> this project's existing pure-PyTorch
KNN/FPS helpers, same pattern already used for every other architecture's
CUDA-kernel fallback path; TF's `tf_util.conv2d` 1x1 convs -> equivalent
`nn.Linear`/`nn.Conv1d` layers).

The rotation-invariant feature construction (4 scalars per neighbor)
-----------------------------------------------------------------------
For a query/center point `c` with K neighbors `{x_i}`:
    nn_local        = x_i - c
    dist_local      = ||x_i - c||                                  (d0)
    mean_local      = mean_i(x_i)                        (mean of the K neighbors)
    nn_local_mean   = x_i - mean_local
    dist_local_mean = ||x_i - mean_local||                         (d1)
    vec             = mean_local - c        (reference direction: center -> local mean)
    cos1            = (x_i - c) . vec / (dist_local * |vec|)       (a0)
    cos2            = (x_i - mean_local) . vec / (dist_local_mean * |vec|)  (a1)
ri_feat = [d0, d1, a0, a1]. All four are invariant to any rotation applied to
the whole input cloud, by the same norm/inner-product argument used
throughout this package's other RI architectures (every vector involved
transforms by the same rotation matrix, and norms/inner-products between
co-rotated vectors are unchanged).

The D-bin directional ordering/max-pool (the part that replaces a fixed
neighbor ordering, since raw KNN neighbor order is not rotation-consistent)
------------------------------------------------------------------------------
A second, independent reference direction is computed per query point from
the *global* mean of the whole current point set (not just the K neighbors):
`vec_g = mean(all points in this layer) - c`. Every neighbor's local offset
is projected onto this direction (`proj_g`), the projection range
`[min, max]` over the K neighbors is split into D equal bins, and each
neighbor is assigned to whichever bin its projection falls into. Features
are max-pooled within each bin (empty bins are excluded via a -inf fill
before the max, matching the original code's `-99999999999.0` sentinel),
giving a fixed-size, order-independent (D, C) tensor per query point
regardless of which K points a rotation happened to bring into the
neighborhood or in what order. This is then flattened and passed through one
more linear layer to the stage's output channel width -- equivalent to the
original code's `conv2d` with a kernel exactly spanning the D dimension
(a "VALID" conv over the full spatial extent is mathematically a fully
connected layer over that extent, so `nn.Linear(D * C_in, C_out)` below is
exact, not an approximation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def _square_dist(src, dst):
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += (src ** 2).sum(-1, keepdim=True)
    dist += (dst ** 2).sum(-1).unsqueeze(1)
    return dist


def _index_points(points, idx):
    B = points.shape[0]
    vs = list(idx.shape); vs[1:] = [1] * (len(vs) - 1)
    rs = list(idx.shape); rs[0] = 1
    bidx = torch.arange(B, device=points.device).view(vs).repeat(rs)
    return points[bidx, idx, :]


def _fps(xyz, npoint):
    """Pure-PyTorch furthest point sampling (same algorithm/precedent as the
    fallback paths used throughout this package for CUDA-only official
    kernels -- RIConv's official repo's FPS is a custom TF op with no CPU/
    portable path at all, so there is nothing to make "optional" here; this
    is simply the implementation)."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def _knn(nsample, xyz, new_xyz):
    sq = _square_dist(new_xyz, xyz)
    _, idx = torch.topk(sq, nsample, dim=-1, largest=False, sorted=False)
    return idx


def _ri_features(xyz, new_xyz, group_idx):
    """xyz: (B,N,3) full layer input. new_xyz: (B,P,3) query/center points.
    group_idx: (B,P,K) neighbor indices into xyz.
    Returns ri_feat (B,P,K,4) and nn_local (B,P,K,3) -- the latter reused by
    the D-bin step below (matches the official code computing `nn_pts_local`
    once and reusing it)."""
    grouped = _index_points(xyz, group_idx)                    # nn_pts        (B,P,K,3)
    center = new_xyz.unsqueeze(2)                               # nn_pts_center (B,P,1,3)
    nn_local = grouped - center                                 # nn_pts_local  (B,P,K,3)
    dist_local = nn_local.norm(dim=-1, keepdim=True)             # dists_local   (B,P,K,1)

    mean_local = grouped.mean(dim=2, keepdim=True)               # mean_local    (B,P,1,3)
    nn_local_mean = grouped - mean_local                         # nn_pts_local_mean (B,P,K,3)
    dist_local_mean = nn_local_mean.norm(dim=-1, keepdim=True)   # dists_local_mean  (B,P,K,1)

    vec = mean_local - center                                    # vec (B,P,1,3)
    vec_dist = vec.norm(dim=-1, keepdim=True)
    vec_norm = vec / (vec_dist + _EPS)                            # nan->0 via +_EPS (official code uses tf.where(isnan,0,.))

    proj1 = (nn_local * vec_norm).sum(-1, keepdim=True)           # nn_pts_local_proj   (B,P,K,1)
    cos1 = proj1 / (dist_local + _EPS)                            # nn_pts_local_proj_dot

    proj2 = (nn_local_mean * vec_norm).sum(-1, keepdim=True)      # nn_pts_local_proj_2
    cos2 = proj2 / (dist_local_mean + _EPS)                       # nn_pts_local_proj_dot_2

    ri_feat = torch.cat([dist_local, dist_local_mean, cos1, cos2], dim=-1)  # (B,P,K,4) = d0 d1 a0 a1
    return ri_feat, nn_local


def _dbin_maxpool(xyz, new_xyz, nn_local, feat, D):
    """The D-bin directional ordering/max-pool. `feat`: (B,P,K,C) already-
    lifted per-neighbor features (this stage's `pts_X_0` in the original
    code). Returns (B,P,D,C)."""
    B, P, K, C = feat.shape
    center = new_xyz.unsqueeze(2)                                  # (B,P,1,3)
    mean_global = xyz.mean(dim=1, keepdim=True).unsqueeze(2)        # (B,1,1,3) -- global mean of this layer's full point set
    vec_g = mean_global - center                                    # (B,P,1,3)
    vec_g_dist = vec_g.norm(dim=-1, keepdim=True)
    vec_g_norm = vec_g / (vec_g_dist + _EPS)

    proj_g = (nn_local * vec_g_norm).sum(-1, keepdim=True)          # (B,P,K,1) -- raw projection length, unnormalized (matches original: no division here)

    proj_min = proj_g.min(dim=2, keepdim=True)[0]                    # (B,P,1,1)
    proj_max = proj_g.max(dim=2, keepdim=True)[0]                    # (B,P,1,1)
    seg = (proj_max - proj_min) / D

    bin_range = torch.arange(D, device=xyz.device, dtype=xyz.dtype).view(1, 1, 1, D)
    limit_bottom = bin_range * seg + proj_min                        # (B,P,1,D)
    limit_up = limit_bottom + seg                                    # (B,P,1,D)

    idx_up = proj_g <= limit_up          # (B,P,K,1) vs (B,P,1,D) -> (B,P,K,D)
    idx_bottom = proj_g >= limit_bottom
    bin_mask = (idx_up & idx_bottom)                                  # (B,P,K,D) bool -- true bin membership

    # DEVIATION from the official code: the official RIConv.py flags "empty"
    # slots by re-checking `nn_fts_rect == 0.0` *after* already multiplying
    # by the membership mask (see riconv/RIConv.py:99-101), which conflates
    # "not a member of this bin" with "member of this bin, but this channel's
    # post-ReLU activation legitimately is 0" -- the latter is extremely
    # common (ReLU) and, empirically here, was pushing large numbers of
    # legitimate zero features to the -1e11 sentinel, poisoning the
    # following BatchNorm layer and preventing training from converging at
    # all (loss stuck at ln(10), the ~chance-level starting point, for the
    # entire training run). `bin_mask` above already carries unambiguous
    # membership information *before* it's collapsed against feature values,
    # so we use it directly instead of re-deriving (ambiguously) "emptiness"
    # from the zeroed product.
    # SECOND deviation, found after retraining with the fix above: a bin can
    # be *genuinely* empty (D=4/D=2 stages only -- D=1 can't have an empty
    # bin, since its single bin spans the whole [proj_min, proj_max] range
    # by construction), and the official code's -99999999999.0 sentinel
    # magnitude is fine for the max() comparison itself but is numerically
    # radioactive once it reaches `self.out`'s Linear layer and the
    # following BatchNorm: BN's running_var accumulates via an EMA of
    # squared deviations, so a single training batch containing one
    # genuinely-empty bin permanently poisons running_var to ~1e18-1e21
    # (confirmed empirically: sa1/sa2's out_bn, which can see empty D=4/D=2
    # bins, ended up with running_var ~1e18-1e21 after retraining; sa3's
    # out_bn, D=1, stayed sane at ~0.2). Training itself looked fine because
    # BatchNorm uses live batch statistics in train() mode, not the poisoned
    # running stats -- the corruption only surfaces at eval()-mode inference
    # (every downstream script), where it collapsed classification to
    # flat chance-level accuracy. Since `feat` is always >=0 (post-ReLU),
    # any negative sentinel makes an empty slot strictly lose the max()
    # against a real slot wherever both exist for a channel -- there is no
    # need for astronomical magnitude, so a small, numerically tame sentinel
    # preserves the exact same masking logic without poisoning BatchNorm.
    feat_exp = feat.unsqueeze(3)                                      # (B,P,K,1,C)
    mask_exp = bin_mask.unsqueeze(-1)                                 # (B,P,K,D,1)
    neg_fill = torch.full((), -1.0, device=feat.device, dtype=feat.dtype)
    rect = torch.where(mask_exp, feat_exp, neg_fill)                  # (B,P,K,D,C)
    rect = rect.max(dim=2)[0]                                         # max over K -> (B,P,D,C)
    return rect


class RIConvLayer(nn.Module):
    """One RIConv set-abstraction stage: FPS-sample P query points, KNN-group
    K neighbors, lift the 4-dim RI feature to 64 channels (`with_local`
    branch, C_pts_fts=64 in the original code), concat previous-layer
    features gathered at the same neighbor indices, D-bin max-pool, then a
    linear layer spanning the full D extent to `out_channel`."""

    def __init__(self, npoint, nsample, D, in_channel, out_channel):
        super().__init__()
        self.npoint, self.nsample, self.D = npoint, nsample, D
        self.lift1 = nn.Linear(4, 32)
        self.lift_bn1 = nn.BatchNorm1d(32)
        self.lift2 = nn.Linear(32, 64)
        self.lift_bn2 = nn.BatchNorm1d(64)
        bin_in = D * (64 + in_channel)
        self.out = nn.Linear(bin_in, out_channel)
        self.out_bn = nn.BatchNorm1d(out_channel)

    def forward(self, xyz, points):
        """xyz: (B,N,3). points: (B,N,C_in) previous-layer features, or None
        for the first stage. Returns new_xyz (B,P,3), new_points (B,P,out_channel)."""
        B, N, _ = xyz.shape
        fps_idx = _fps(xyz, self.npoint)
        new_xyz = _index_points(xyz, fps_idx)                        # (B,P,3)
        group_idx = _knn(self.nsample, xyz, new_xyz)                  # (B,P,K)

        ri_feat, nn_local = _ri_features(xyz, new_xyz, group_idx)     # (B,P,K,4)
        P, K = ri_feat.shape[1], ri_feat.shape[2]

        lifted = F.relu(self.lift_bn1(self.lift1(ri_feat).reshape(-1, 32)).reshape(B, P, K, 32))
        lifted = F.relu(self.lift_bn2(self.lift2(lifted).reshape(-1, 64)).reshape(B, P, K, 64))

        if points is not None:
            grouped_prev = _index_points(points, group_idx)           # (B,P,K,C_in)
            feat = torch.cat([lifted, grouped_prev], dim=-1)
        else:
            feat = lifted

        binned = _dbin_maxpool(xyz, new_xyz, nn_local, feat, self.D)   # (B,P,D,C)
        flat = binned.reshape(B, P, -1)                                  # (B,P,D*C)
        out = F.relu(self.out_bn(self.out(flat).transpose(1, 2)).transpose(1, 2))  # (B,P,out_channel)
        return new_xyz, out
