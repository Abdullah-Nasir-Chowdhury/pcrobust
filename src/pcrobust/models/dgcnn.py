"""DGCNN backbone — matches antao97/dgcnn.pytorch ModelNet40 checkpoint."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Graph helpers ──────────────────────────────────────────────────────────────

def _knn(x, k):
    """k-nearest neighbor indices via negative squared distance.
    x: (B, C, N) → (B, N, k) indices."""
    inner = -2 * torch.bmm(x.transpose(2, 1), x)        # (B, N, N)
    xx    = (x ** 2).sum(1, keepdim=True)                # (B, 1, N)
    neg_dist = -xx - inner - xx.transpose(2, 1)          # (B, N, N)
    return neg_dist.topk(k=k, dim=-1)[1]                 # (B, N, k)


def _get_graph_feature(x, k):
    """EdgeConv edge features: cat(x_j - x_i, x_i) for each of k neighbors.
    x: (B, C, N) → (B, 2C, N, k)."""
    B, C, N = x.shape
    idx     = _knn(x, k)                                  # (B, N, k)
    base    = torch.arange(B, device=x.device).view(B, 1, 1) * N
    idx_flat = (idx + base).reshape(-1)                   # (B*N*k,)
    x_t     = x.transpose(2, 1).reshape(B * N, C)        # (B*N, C)
    nbrs    = x_t[idx_flat].reshape(B, N, k, C).permute(0, 3, 1, 2)  # (B, C, N, k)
    x_i     = x.unsqueeze(3).expand_as(nbrs)             # (B, C, N, k)
    return torch.cat([nbrs - x_i, x_i], dim=1)           # (B, 2C, N, k)


def _edgeconv(conv, x, k):
    return conv(_get_graph_feature(x, k)).max(dim=-1)[0]


# ── DGCNN Embedder ─────────────────────────────────────────────────────────────

class DGCNNEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128, k: int = 20):
        super().__init__()
        self.k = k
        # Layer names and Sequential indices match antao97 checkpoint (after stripping module.)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6,   64,  1, bias=False), nn.BatchNorm2d(64),   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64,  1, bias=False), nn.BatchNorm2d(64),   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128),  nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256),  nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear1    = nn.Linear(2048, 512, bias=False)
        self.bn6        = nn.BatchNorm1d(512)
        self.linear2    = nn.Linear(512, 256)
        self.bn7        = nn.BatchNorm1d(256)
        self.embed_head = nn.Linear(256, embed_dim)
        self.cls_head   = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                                         # x: (B, 3, N)
        x1 = _edgeconv(self.conv1, x,  self.k)                   # (B, 64,  N)
        x2 = _edgeconv(self.conv2, x1, self.k)                   # (B, 64,  N)
        x3 = _edgeconv(self.conv3, x2, self.k)                   # (B, 128, N)
        x4 = _edgeconv(self.conv4, x3, self.k)                   # (B, 256, N)
        x  = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))      # (B, 1024, N)
        g  = torch.cat([x.max(-1)[0], x.mean(-1)], dim=1)        # (B, 2048)
        g  = F.leaky_relu(self.bn6(self.linear1(g)), 0.2)
        g  = F.dropout(g, p=0.5, training=self.training)
        g  = F.leaky_relu(self.bn7(self.linear2(g)), 0.2)
        g  = F.dropout(g, p=0.5, training=self.training)
        return g                                                   # (B, 256)

    def forward(self, x):
        f   = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


# ── Pretrained loading (antao97/dgcnn.pytorch, ModelNet40, k=20) ───────────────

_HEADS = ("embed_head", "cls_head")


def load_pretrained_backbone(model: DGCNNEmbedder, ckpt_path: str) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = raw.get("model_state_dict", raw.get("state_dict", raw))
    dst = model.state_dict()

    update = {}
    for k, v in src.items():
        key = k.removeprefix("module.")  # strip DataParallel wrapper
        if key in dst and dst[key].shape == v.shape and not key.startswith(_HEADS):
            update[key] = v
        # bn1..bn5 are stored both as conv*.1.* and as standalone bn*.* in antao97;
        # our model only has the conv*.1.* form, so standalone keys naturally miss dst.

    dst.update(update)
    model.load_state_dict(dst)
    total = sum(1 for k in dst if not k.startswith(_HEADS))
    print(f"[pretrained] {len(update)}/{total} backbone tensors loaded "
          f"from {Path(ckpt_path).name}")
