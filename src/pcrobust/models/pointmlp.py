"""
PointMLP backbone — adapter around the OFFICIAL implementation.

Vendored from https://github.com/ma-xu/pointMLP-pytorch (commit
1ab7f04e88d2aeb8fd16921a2b2fc4fba950b6df, 2025-02-02) — Ma et al.,
"Rethinking Network Design and Local Geometry in Point Cloud: A Simple
Residual MLP Framework" (ICLR 2022). The vendored core lives in
`pcrobust/models/vendor/pointmlp_official/pointmlp.py`; see that file's
docstring for the one documented patch (optional CUDA furthest-point-
sampling with a pure-PyTorch fallback). The actual architecture -- the
geometric affine module in `LocalGrouper`, the residual point-MLP blocks --
is the original authors' code, unmodified, using the exact `pointMLP()`
factory hyperparameters (the official repo's standard, non-"Elite"
configuration, tuned for 1024-point inputs -- matching this project's
convention).

Unlike RIConv++/RISurConv, PointMLP is intentionally NOT rotation invariant
-- it consumes locally-normalized raw xyz coordinates directly. That's the
point of including it here: the pose/noise robustness studies are exactly
about contrasting architectures that are invariant by construction against
ones that aren't.

This adapter reuses the official backbone verbatim up to its final pooled
feature (1024-dim, via `Model.forward_features()` -- a vendored-file
addition that factors out the official `forward()` at the exact point it
hands off to the classifier, see that file's docstring), then attaches this
project's own embedding + classification head in place of the official
`Model.classifier`, matching the `Embedder`/`(logits, embedding, aux)`
contract the other backbones in this package use.

No pretrained checkpoint is available for this architecture.
"""

import torch.nn as nn
import torch.nn.functional as F

from .vendor.pointmlp_official.pointmlp import Model as _PointMLPBackbone

_BACKBONE_OUT_DIM = 1024  # embed_dim(64) * prod(dim_expansion=[2,2,2,2]) in the pointMLP() factory config


class PointMLPEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        # Exact hyperparameters of the official `pointMLP()` factory function.
        self.backbone = _PointMLPBackbone(
            points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
            activation="relu", bias=False, use_xyz=False, normalize="anchor",
            dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
            k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2],
        )
        # self.backbone.classifier is unused (never called) -- forward_features()
        # stops right before it -- so it just sits idle rather than being deleted,
        # keeping the vendored Model class untouched.

        self.fc1 = nn.Linear(_BACKBONE_OUT_DIM, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.embed_head = nn.Linear(256, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                       # x: (B, 3, N)
        f = self.backbone.forward_features(x)    # (B, 1024)
        f = F.relu(self.bn1(self.fc1(f)))
        f = F.relu(self.bn2(self.fc2(f)))         # (B, 256)
        return f

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: PointMLPEmbedder, ckpt_path: str) -> None:
    raise NotImplementedError(
        "No pretrained checkpoint is available for pointmlp -- train without --pretrained."
    )
