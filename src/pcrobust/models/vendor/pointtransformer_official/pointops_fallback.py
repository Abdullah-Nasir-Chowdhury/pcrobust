"""
Pure-PyTorch fallback for the two `pointops` CUDA kernels the official Point
Transformer v1 classification/segmentation code (`point_transformer_seg.py`,
`point_transformer_cls.py`) calls: `knn_query_and_group` and
`farthest_point_sampling`. Not part of the official repo -- this module
exists because `libs/pointops` (https://github.com/Pointcept/
PointTransformerV2/tree/main/libs/pointops) is a custom CUDA extension with
no CPU path at all (every kernel asserts CUDA tensors), so it can't be
smoke-tested on a CPU-only machine. `point_transformer_seg.py` imports
this module instead of the real `pointops` package when the compiled
extension isn't present; when the compiled kernels are installed, they are
used and this file is never imported.

Both functions below match the real library's exact call signature and
semantics -- operating on the same flattened "offset" batching convention
(all point clouds in a batch concatenated into one (n_total, 3) tensor, with
`offset` marking each cloud's cumulative end index) -- verified against
`libs/pointops/functions/{query,sampling,utils}.py` in the official repo.
`grouping()` (turning neighbor indices into gathered feature tensors) needs
no fallback at all: the official library already implements it in plain
PyTorch indexing, not a CUDA kernel, so `grouping` below is copied verbatim
rather than reimplemented.
"""

import torch


def _square_distance(src, dst):
    return torch.cdist(src, dst) ** 2


def _offsets_to_slices(offset):
    offset = offset.tolist()
    starts = [0] + offset[:-1]
    return list(zip(starts, offset))


def grouping(idx, feat, xyz, new_xyz=None, with_xyz=False):
    """Verbatim from libs/pointops/functions/grouping.py -- already pure
    PyTorch in the official code, not a CUDA kernel."""
    if new_xyz is None:
        new_xyz = xyz
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    xyz = torch.cat([xyz, torch.zeros([1, 3], device=xyz.device, dtype=xyz.dtype)], dim=0)
    feat = torch.cat([feat, torch.zeros([1, c], device=feat.device, dtype=feat.dtype)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)

    if with_xyz:
        mask = torch.sign(idx + 1)
        grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) - new_xyz.unsqueeze(1)
        grouped_xyz = torch.einsum("n s c, n s -> n s c", grouped_xyz, mask)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat


def _knn_query(nsample, xyz, offset, new_xyz=None, new_offset=None):
    """Fallback for libs/pointops/functions/query.py::KNNQuery. Per-cloud
    brute-force distance + topk, restricted to each cloud's own [start,end)
    segment -- the offsets ensure different clouds in the same batch are
    never grouped together, matching the CUDA kernel's own per-batch-item
    restriction."""
    if new_xyz is None or new_offset is None:
        new_xyz = xyz
        new_offset = offset

    idx_parts, dist_parts = [], []
    for (s0, s1), (d0, d1) in zip(_offsets_to_slices(offset), _offsets_to_slices(new_offset)):
        seg_xyz = xyz[s0:s1].unsqueeze(0)      # (1, n_i, 3)
        seg_new = new_xyz[d0:d1].unsqueeze(0)  # (1, m_i, 3)
        n_i = seg_xyz.shape[1]
        k = min(nsample, n_i)
        sq = _square_distance(seg_new, seg_xyz)  # (1, m_i, n_i)
        d, idx = torch.topk(sq, k, dim=-1, largest=False, sorted=True)
        if k < nsample:  # pad by repeating the nearest neighbor (same convention pointops uses for undersized groups elsewhere)
            pad_n = nsample - k
            idx = torch.cat([idx, idx[..., -1:].expand(-1, -1, pad_n)], dim=-1)
            d = torch.cat([d, d[..., -1:].expand(-1, -1, pad_n)], dim=-1)
        idx_parts.append((idx.squeeze(0) + s0).to(torch.int64))
        dist_parts.append(d.squeeze(0))

    idx = torch.cat(idx_parts, dim=0)
    dist = torch.cat(dist_parts, dim=0)
    return idx, torch.sqrt(torch.clamp(dist, min=0))


def _fps_dense(xyz, npoint):
    """Iterative farthest-point sampling on a single dense (1, N, 3) cloud --
    same algorithm used elsewhere in this project's CUDA fallbacks
    (riconvpp/risurconv/pointmlp)."""
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


def farthest_point_sampling(xyz, offset, new_offset):
    """Fallback for libs/pointops/functions/sampling.py::FarthestPointSampling.
    Per-cloud FPS restricted to each cloud's own segment; returns absolute
    (global) indices into the flat `xyz`, matching the real kernel's
    contract (callers index the full flat tensor with this output directly)."""
    new_offset_list = new_offset.tolist()
    new_starts = [0] + new_offset_list[:-1]
    m_per_cloud = [b - a for a, b in zip(new_starts, new_offset_list)]

    idx_parts = []
    for (s0, s1), m in zip(_offsets_to_slices(offset), m_per_cloud):
        seg = xyz[s0:s1].unsqueeze(0)             # (1, n_i, 3)
        local_idx = _fps_dense(seg, m).squeeze(0)  # (m,)
        idx_parts.append((local_idx + s0).to(torch.int64))
    return torch.cat(idx_parts, dim=0)


def knn_query_and_group(feat, xyz, offset=None, new_xyz=None, new_offset=None,
                         idx=None, nsample=None, with_xyz=False):
    """Fallback for libs/pointops/functions/utils.py::knn_query_and_group."""
    if idx is None:
        assert nsample is not None
        idx, _ = _knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return grouping(idx, feat, xyz, new_xyz, with_xyz), idx
