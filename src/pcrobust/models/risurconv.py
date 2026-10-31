"""
RISurConv backbone — adapter around the OFFICIAL implementation.

Vendored from https://github.com/cszyzhang/RISurConv (commit
27c48dd3656a7af14a3d35ef3c45cef3ae9bc356, 2024-08-13) — Zhang et al.,
"RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for
3D Point Cloud Classification and Segmentation" (ECCV 2024). The vendored
core lives in `pcrobust/models/vendor/risurconv_official/risurconv_utils.py`;
see that file's docstring for the two documented patches (optional CUDA
furthest-point-sampling with a pure-PyTorch fallback, and one deprecated
`torch.cross` call made explicit). The rotation-invariant surface-feature
construction itself — the same LRA machinery as RIConv++ (`pcrobust/models/
riconvpp.py`) plus RISurConv's own 14-dim "surface triangle" features
(`RISP_features`) and its attention-augmented set-abstraction layers
(`SA_Layer`/`SA_Layer_2d`) — is the original authors' code, unmodified.

Architecture and the initial-normal fallback
----------------------------------------------
`sc0..sc4` below mirror `models/risurconv_cls.py`'s `n=2` configuration
(the official repo's own setting for 1024-point inputs), plus the official
model's 3-layer TransformerEncoder refinement over the final pooled feature.
As with RIConv++, this project's datasets don't carry curated per-point
surface normals, so the initial per-point reference normal is seeded with
the vendored module's own `compute_LRA()` — the same fallback the official
classifier script documents in a comment but never actually implements for
its `normal_channel=False` path.

One deliberate deviation from the official model script: it calls
`nn.TransformerEncoder` with the library default `batch_first=False`, but
feeds it a `(B, 1, 512)` tensor without ever transposing to the `(seq, batch,
feature)` layout that default expects. Since the official model's sequence
length is always 1 (one pooled token per object, from the `group_all` stage),
that mismatch makes the *batch* dimension look like the sequence dimension --
i.e. self-attention would run across different objects in the same batch
rather than being a no-op over a length-1 sequence, which cannot be the
intended behavior for a per-object classifier (predictions would depend on
what else happens to be in the same batch). This adapter uses
`batch_first=True` instead, which keeps batch items independent and makes
the transformer layer the harmless per-object refinement it's clearly meant
to be. Everything upstream of this (the actual rotation-invariant feature
construction) is unaffected either way.

No pretrained checkpoint is available for this architecture.
"""

import torch.nn as nn
import torch.nn.functional as F

from .vendor.risurconv_official.risurconv_utils import RISurConvSetAbstraction, compute_LRA

_LRA_NSAMPLE = 32


class RISurConvEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()

        self.sc0 = RISurConvSetAbstraction(npoint=1024, radius=0.12, nsample=8, in_channel=0, out_channel=32, group_all=False)
        self.sc1 = RISurConvSetAbstraction(npoint=512, radius=0.16, nsample=16, in_channel=32, out_channel=64, group_all=False)
        self.sc2 = RISurConvSetAbstraction(npoint=256, radius=0.24, nsample=32, in_channel=64, out_channel=128, group_all=False)
        self.sc3 = RISurConvSetAbstraction(npoint=128, radius=0.48, nsample=64, in_channel=128, out_channel=256, group_all=False)
        self.sc4 = RISurConvSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, out_channel=512, group_all=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.05, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.embed_head = nn.Linear(128, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                      # x: (B, 3, N)
        xyz = x.permute(0, 2, 1).contiguous()   # (B, N, 3)
        norm = compute_LRA(xyz, weighting=True, nsample=_LRA_NSAMPLE)

        l0_xyz, l0_norm, l0_points = self.sc0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        _, _, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)

        # l4_points: (B, 512, 1) -- batch_first transformer over the length-1 sequence
        seq = l4_points.permute(0, 2, 1)        # (B, 1, 512)
        seq = self.transformer_encoder(seq)
        f = seq.reshape(x.shape[0], 512)

        f = F.relu(self.bn1(self.fc1(f)))
        f = F.relu(self.bn2(self.fc2(f)))       # (B, 128)
        return f

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: RISurConvEmbedder, ckpt_path: str) -> None:
    raise NotImplementedError(
        "No pretrained checkpoint is available for risurconv -- train without --pretrained."
    )
