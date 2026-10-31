"""PointNet backbone — TNet + PointNetEmbedder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Architecture ───────────────────────────────────────────────────────────────

class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k    = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),     nn.BatchNorm1d(64),   nn.ReLU(),
            nn.Conv1d(64, 128, 1),   nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        B = x.size(0)
        x = self.conv(x).max(dim=2)[0]
        x = self.fc(x)
        I = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).expand(B, -1)
        return (x + I).view(B, self.k, self.k)


class PointNetEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.tnet3  = TNet(3)
        self.mlp1   = nn.Sequential(
            nn.Conv1d(3, 64, 1),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.tnet64 = TNet(64)
        self.mlp2   = nn.Sequential(
            nn.Conv1d(64, 128, 1),   nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.embed_head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):                   # x: (B, 3, N)
        T3  = self.tnet3(x)
        x   = torch.bmm(T3, x)
        x   = self.mlp1(x)
        T64 = self.tnet64(x)
        x   = torch.bmm(T64, x)
        x   = self.mlp2(x).max(dim=2)[0]
        emb = F.normalize(self.embed_head(x), dim=1)
        return self.cls_head(emb), emb, T64

    def get_embedding(self, x):
        return self.forward(x)[1]


# ── Pretrained loading (meder411/PointNet-PyTorch, ModelNet40) ─────────────────

_HEADS = ("embed_head", "cls_head")

_MEDER411_MAP = {
    "base.input_transformer.block1.0": "tnet3.conv.0",
    "base.input_transformer.block1.1": "tnet3.conv.1",
    "base.input_transformer.block2.0": "tnet3.conv.3",
    "base.input_transformer.block2.1": "tnet3.conv.4",
    "base.input_transformer.block3.0": "tnet3.conv.6",
    "base.input_transformer.block3.1": "tnet3.conv.7",
    "base.input_transformer.mlp.0":    "tnet3.fc.0",
    "base.input_transformer.mlp.1":    "tnet3.fc.1",
    "base.input_transformer.mlp.3":    "tnet3.fc.3",
    "base.input_transformer.mlp.4":    "tnet3.fc.4",
    "base.input_transformer.mlp.6":    "tnet3.fc.6",
    "base.mlp1.0": "mlp1.0", "base.mlp1.1": "mlp1.1",
    "base.mlp1.3": "mlp1.3", "base.mlp1.4": "mlp1.4",
    "base.embedding_transformer.block1.0": "tnet64.conv.0",
    "base.embedding_transformer.block1.1": "tnet64.conv.1",
    "base.embedding_transformer.block2.0": "tnet64.conv.3",
    "base.embedding_transformer.block2.1": "tnet64.conv.4",
    "base.embedding_transformer.block3.0": "tnet64.conv.6",
    "base.embedding_transformer.block3.1": "tnet64.conv.7",
    "base.embedding_transformer.mlp.0":    "tnet64.fc.0",
    "base.embedding_transformer.mlp.1":    "tnet64.fc.1",
    "base.embedding_transformer.mlp.3":    "tnet64.fc.3",
    "base.embedding_transformer.mlp.4":    "tnet64.fc.4",
    "base.embedding_transformer.mlp.6":    "tnet64.fc.6",
    "base.mlp2.3": "mlp2.0", "base.mlp2.4": "mlp2.1",
    "base.mlp2.6": "mlp2.3", "base.mlp2.7": "mlp2.4",
}


def load_pretrained_backbone(model: PointNetEmbedder, ckpt_path: str) -> None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = raw.get("state_dict", raw)
    dst = model.state_dict()

    update = {}
    for src_key, tensor in src.items():
        if src_key in dst and dst[src_key].shape == tensor.shape:
            update[src_key] = tensor
            continue
        for sp, op in _MEDER411_MAP.items():
            if src_key.startswith(sp + "."):
                ok = op + src_key[len(sp):]
                if ok in dst and dst[ok].shape == tensor.shape:
                    update[ok] = tensor
                break

    backbone = {k: v for k, v in update.items() if not k.startswith(_HEADS)}
    dst.update(backbone)
    model.load_state_dict(dst)
    total = sum(1 for k in dst if not k.startswith(_HEADS))
    print(f"[pretrained] {len(backbone)}/{total} backbone tensors loaded "
          f"from {Path(ckpt_path).name}")
