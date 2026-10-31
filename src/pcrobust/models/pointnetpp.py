"""PointNet++ SSG backbone — matches yanx27/Pointnet_Pointnet2_pytorch checkpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Geometry helpers ───────────────────────────────────────────────────────────

def _square_dist(src, dst):
    """Pairwise squared Euclidean distances. src/dst: (B, N, 3)."""
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += (src ** 2).sum(-1, keepdim=True)
    dist += (dst ** 2).sum(-1).unsqueeze(1)
    return dist  # (B, N, M)


def _index(points, idx):
    """Gather rows of points by idx. points: (B,N,C), idx: (B,S) or (B,S,K)."""
    B    = points.shape[0]
    vs   = list(idx.shape); vs[1:] = [1] * (len(vs) - 1)
    rs   = list(idx.shape); rs[0]  = 1
    bidx = torch.arange(B, device=points.device).view(vs).repeat(rs)
    return points[bidx, idx, :]


def _fps(xyz, npoint):
    """Farthest-point sampling. xyz: (B,N,3) → (B,npoint) indices."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist      = torch.full((B, N), 1e10, device=xyz.device)
    farthest  = torch.zeros(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c    = xyz[torch.arange(B), farthest].unsqueeze(1)
        d    = ((xyz - c) ** 2).sum(-1)
        dist = torch.min(dist, d)
        farthest = dist.max(1)[1]
    return centroids


def _ball_query(radius, nsample, xyz, new_xyz):
    """Ball query: (B,S,nsample) neighbor indices."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    sq  = _square_dist(new_xyz, xyz)
    idx[sq > radius ** 2] = N
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    first = idx[:, :, 0:1].expand_as(idx)
    idx[idx == N] = first[idx == N]
    return idx


def _sample_and_group(npoint, radius, nsample, xyz, points):
    fps_idx  = _fps(xyz, npoint)
    new_xyz  = _index(xyz, fps_idx)
    idx      = _ball_query(radius, nsample, xyz, new_xyz)
    grouped  = _index(xyz, idx) - new_xyz.unsqueeze(2)     # relative coords
    if points is not None:
        new_points = torch.cat([grouped, _index(points, idx)], dim=-1)
    else:
        new_points = grouped
    return new_xyz, new_points


def _sample_and_group_all(xyz, points):
    B, N, _ = xyz.shape
    new_xyz   = torch.zeros(B, 1, 3, device=xyz.device)
    grouped   = xyz.unsqueeze(1)
    if points is not None:
        new_points = torch.cat([grouped, points.unsqueeze(1)], dim=-1)
    else:
        new_points = grouped
    return new_xyz, new_points


# ── Set Abstraction ────────────────────────────────────────────────────────────

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint    = npoint
        self.radius    = radius
        self.nsample   = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz, points):
        # xyz: (B, N, 3), points: (B, N, C) or None
        if self.group_all:
            new_xyz, new_pts = _sample_and_group_all(xyz, points)
        else:
            new_xyz, new_pts = _sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points)
        # new_pts: (B, npoint, nsample, C) → (B, C, nsample, npoint)
        new_pts = new_pts.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_pts = F.relu(bn(conv(new_pts)))
        new_pts = new_pts.max(dim=2)[0]              # (B, C_out, npoint)
        return new_xyz, new_pts.permute(0, 2, 1)    # (B, npoint, C_out)


# ── PointNet++ Embedder ────────────────────────────────────────────────────────

class PointNetPPEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        # SA layer names match the yanx27 pretrained checkpoint directly
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, 3,       [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.embed_head = nn.Linear(256, embed_dim)
        self.cls_head   = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):               # x: (B, 3, N)
        xyz = x.permute(0, 2, 1)        # (B, N, 3)
        xyz, pts = self.sa1(xyz, None)
        xyz, pts = self.sa2(xyz, pts)
        _, pts   = self.sa3(xyz, pts)
        f = pts.squeeze(1)              # (B, 1024)
        f = F.relu(self.bn1(self.fc1(f)))
        f = F.relu(self.bn2(self.fc2(f)))
        return f                        # (B, 256)

    def forward(self, x):
        f   = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


# ── Pretrained loading (yanx27, ModelNet40 SSG without normals) ────────────────

_HEADS = ("embed_head", "cls_head")


def load_pretrained_backbone(model: PointNetPPEmbedder, ckpt_path: str) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = raw.get("model_state_dict", raw.get("state_dict", raw))
    dst = model.state_dict()

    # Layer names match directly; only fc3 (40-class head) has no match in our model
    update = {k: v for k, v in src.items()
              if k in dst and dst[k].shape == v.shape
              and not k.startswith(_HEADS)}
    dst.update(update)
    model.load_state_dict(dst)
    total = sum(1 for k in dst if not k.startswith(_HEADS))
    print(f"[pretrained] {len(update)}/{total} backbone tensors loaded "
          f"from {Path(ckpt_path).name}")
