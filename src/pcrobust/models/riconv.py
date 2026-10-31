"""
RIConv backbone — adapter around a PyTorch port of the OFFICIAL implementation.

Ported from https://github.com/hkust-vgd/riconv (commit
3a094e5f6e794a72c842f0fbec911839316648ea, 2019-09-14) — Zhang et al.,
"Rotation Invariant Convolutions for 3D Point Clouds Deep Learning" (3DV 2019),
the predecessor to RIConv++ (`pcrobust/models/riconvpp.py`). The official code
is TensorFlow 1.x with custom TF ops and has no PyTorch release, so unlike
this package's other 4 vendored architectures, this one is a direct line-by-
line port of the official `RIConv()` function's tensor math to PyTorch rather
than an adapted copy of existing PyTorch source — see
`pcrobust/models/vendor/riconv_official/riconv_v1.py`'s docstring for the exact
correspondence and why this counts as "using the official code" despite no
code being literally reused.

Architecture: 3 RIConvLayer stages with the exact (K, D, P, C) hyperparameters
from the official repo's `train_val_cls.py` (ModelNet40 classification
config, already tuned for 1024-point inputs):
    stage 1: K=64 neighbors, D=4 bins, P=256 query points, C=128 channels
    stage 2: K=32,           D=2,      P=128,              C=256
    stage 3: K=16,           D=1,      P=64,               C=512
followed by the same two fully-connected layers the official `fc_params`
specifies (512, dropout 0) then (256, dropout 0.5).

One deliberate deviation: the official classifier applies a *per-point*
classification head to all 64 remaining points and averages the resulting
logits (`tf.reduce_mean(pred, axis=-2)`) to get one prediction. This adapter
instead mean-pools the 256-dim *features* across those 64 points first, then
applies this project's own embedding + classification head — mathematically
similar in spirit (both aggregate over the same 64 points) but produces the
single embedding vector this project's `Embedder` contract requires, which
per-point-logit-averaging does not.

No pretrained checkpoint is available for this architecture.
"""

import torch.nn as nn
import torch.nn.functional as F

from .vendor.riconv_official.riconv_v1 import RIConvLayer


class RIConvEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.sa1 = RIConvLayer(npoint=256, nsample=64, D=4, in_channel=0, out_channel=128)
        self.sa2 = RIConvLayer(npoint=128, nsample=32, D=2, in_channel=128, out_channel=256)
        self.sa3 = RIConvLayer(npoint=64, nsample=16, D=1, in_channel=256, out_channel=512)

        self.fc1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.embed_head = nn.Linear(256, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                       # x: (B, 3, N)
        xyz = x.permute(0, 2, 1).contiguous()    # (B, N, 3)
        xyz, pts = self.sa1(xyz, None)
        xyz, pts = self.sa2(xyz, pts)
        _, pts = self.sa3(xyz, pts)              # (B, 64, 512)

        B, P, C = pts.shape
        f = F.relu(self.bn1(self.fc1(pts).reshape(-1, 512)).reshape(B, P, 512))
        f = self.drop1(f)
        f = F.relu(self.bn2(self.fc2(f).reshape(-1, 256)).reshape(B, P, 256))
        f = self.drop2(f)

        f = f.mean(dim=1)                        # (B, 256) -- mean-pool over the 64 remaining points
        return f

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: RIConvEmbedder, ckpt_path: str) -> None:
    raise NotImplementedError(
        "No pretrained checkpoint is available for riconv -- train without --pretrained."
    )
