"""
PointTransformerV2 backbone — adapter around the OFFICIAL implementation.

Vendored from https://github.com/Pointcept/PointTransformerV2 (commit
5386c4d71f3d6c42c24a8105fce8750e9355dc54, 2023-06-05). The vendored code
lives in `pcrobust/models/vendor/pointtransformer_official/`; see that
directory's file docstrings for the documented patches (optional CUDA
kernels with a pure-PyTorch fallback for the offset-batched KNN/FPS ops,
and a backbone/classifier split so this project's own head can be attached).

Why this is Point Transformer *v1*'s classification model, not PTv2's own
architecture
--------------------------------------------------------------------------
PTv3's official code is scene-segmentation-only, so it cannot serve as a
classification backbone here. The
*same* substitution problem applies to PTv2 itself: the official
PTv2 model (`pcr/models/point_transformer2/point_transformer_v2m2_base.py`)
is an encoder-decoder U-Net over voxelized, variable-length *scenes*
(`configs/_base_/datasets/modelnet40.py` in that repo is explicitly marked
"Unmaintained, kept for reference" by the authors) -- not a fit for this
project's fixed-size, single-object embedding task, and not one the authors
themselves stand behind for classification.

The *same officially-maintained PointTransformerV2 repository* also ships
`pcr/models/point_transformer/point_transformer_cls.py` -- the Pointcept
team's own re-implementation of the original Point Transformer (v1, Zhao et
al. ICCV 2021), purpose-built for object classification, which this repo's
own docstring understatedly calls out as possibly "a bit different from the
original paper." This is what "PointTransformerV2" refers to in this
project: not the PTv2 architecture, but Point Transformer v1 as shipped by
the officially-maintained PTv2 repo -- genuinely official code, and correctly
shaped for the task, unlike the alternative of forcing PTv2's segmentation
U-Net through this project's classification pipeline. `PointTransformerCls38`
(the middle of the repo's 3 unwired depth options) is used below.

Data-format adapter
--------------------
The vendored model expects Pointcept's flattened "offset" batching
convention (`{"coord", "feat", "offset"}`, all clouds in a batch
concatenated into one tensor) rather than this project's fixed (B, 3, N)
tensor. Since every point cloud in a batch here has the same N, this is a
trivial reshape, done in `_to_offset_batch()` below.

No pretrained checkpoint is available for this architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendor.pointtransformer_official.point_transformer_cls import PointTransformerCls38

_BACKBONE_OUT_DIM = 512  # planes[4] in PointTransformerCls


def _to_offset_batch(x):
    """x: (B, 3, N) -> Pointcept's flattened {coord, feat, offset} format."""
    B, _, N = x.shape
    coord = x.permute(0, 2, 1).reshape(B * N, 3).contiguous()
    offset = torch.arange(1, B + 1, device=x.device, dtype=torch.int64) * N
    return {"coord": coord, "feat": coord, "offset": offset}


class PointTransformerEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        self.backbone = PointTransformerCls38(in_channels=3, num_classes=num_classes)
        # self.backbone.cls is unused (never called) -- forward_backbone() stops
        # right before it -- so it just sits idle rather than being deleted,
        # keeping the vendored class untouched.
        self.embed_head = nn.Linear(_BACKBONE_OUT_DIM, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                              # x: (B, 3, N)
        input_dict = _to_offset_batch(x)
        f = self.backbone.forward_backbone(input_dict)  # (B, 512)
        return f

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: PointTransformerEmbedder, ckpt_path: str) -> None:
    raise NotImplementedError(
        "No pretrained checkpoint is available for pointtransformer -- train without --pretrained."
    )
