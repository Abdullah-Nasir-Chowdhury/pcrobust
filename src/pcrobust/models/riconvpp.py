"""
RIConv++ backbone — adapter around the OFFICIAL implementation.

Vendored from https://github.com/cszyzhang/riconv2 (commit
70d9297cc906e8c977c48ae5cdfcd250fae9fed4, 2022-11-20) — Zhang et al.,
"RIConv++: Effective Rotation Invariant Convolutions for 3D Point Clouds Deep
Learning" (IJCV 2022). The vendored core lives in
`pcrobust/models/vendor/riconvpp_official/riconv2_utils.py`; see that file's
docstring for the exact (minimal, documented) patches applied — replacing the
removed `Tensor.symeig` API and making the CUDA furthest-point-sampling
kernel optional with a pure-PyTorch fallback for CPU-only smoke tests. The
rotation-invariant feature construction itself (Local Reference Axis via
PCA/eigendecomposition, the neighbor-ordering scheme, the 8-dim RI feature)
is the original authors' code, unmodified.

Why the official LRA computation is used verbatim
--------------------------------------------------
A simpler reimplementation might pick its per-group reference point by a
bare `argmax` over distance-to-center — cruder than even the "old,
unstable" reference vector the RIConv++ paper's own authors call out as a
problem in their motivation (Section 4.3/4.4 of the paper). The official
design instead computes a Local Reference Axis (LRA) per point: the
eigenvector of the *smallest* eigenvalue of a distance-weighted covariance
matrix over a local neighborhood (`compute_LRA` below) — averaged over many
neighbors rather than picking a single farthest point, which is exactly what
makes it robust to noise. This
file therefore uses the actual official LRA computation and RI-feature
pipeline instead of reimplementing the idea from scratch.

Supplying the initial per-point normal
----------------------------------------
The official classification model (`models/riconv2_cls.py`) expects either
curated surface normals as an extra input channel (`normal_channel=True`,
e.g. exact CAD-derived normals shipped with some ModelNet40 distributions) or
a documented-but-never-implemented fallback ("compute the LRA and use as
normal") when normals aren't available. This project's datasets (ModelNet40/
ShapeNet/ScanObjectNN/OmniObject3D point samples) don't carry curated
per-point normals, so this adapter fills that gap using the official repo's
own `compute_LRA()` — it's the same LRA machinery the rest of the pipeline
already uses, just applied once up front to seed the first set-abstraction
layer's `norm` argument, exactly as the original comment describes.

Architecture hyperparameters (npoint/radius/nsample per stage, and the
520->256->128 embedding head widths before this project's own embed/class
heads) mirror `models/riconv2_cls.py`'s `n=2` configuration, which is what
the official repo itself uses for 1024-point inputs — matching this
project's 1024-point/unit-sphere convention exactly.

There is no pretrained checkpoint available for this architecture (as
before) -- load_pretrained_backbone() is a stub that raises rather than
silently doing nothing.
"""

import torch.nn as nn
import torch.nn.functional as F

from .vendor.riconvpp_official.riconv2_utils import RIConv2SetAbstraction, compute_LRA

_LRA_NSAMPLE = 32  # local neighborhood size for the initial per-point LRA estimate


class RIConvPPEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128):
        super().__init__()
        in_channel = 64  # fixed width the official code lifts every raw 8-dim RI feature to (see RIConv2SetAbstraction.prev_mlp_convs)

        self.sa0 = RIConv2SetAbstraction(npoint=1024, radius=0.12, nsample=8,
                                          in_channel=0 + in_channel, mlp=[32], group_all=False)
        self.sa1 = RIConv2SetAbstraction(npoint=512, radius=0.16, nsample=16,
                                          in_channel=32 + in_channel, mlp=[64], group_all=False)
        self.sa2 = RIConv2SetAbstraction(npoint=256, radius=0.24, nsample=32,
                                          in_channel=64 + in_channel, mlp=[128], group_all=False)
        self.sa3 = RIConv2SetAbstraction(npoint=128, radius=0.48, nsample=64,
                                          in_channel=128 + in_channel, mlp=[256], group_all=False)
        self.sa4 = RIConv2SetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256 + in_channel, mlp=[512], group_all=True)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.embed_head = nn.Linear(128, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                      # x: (B, 3, N)
        xyz = x.permute(0, 2, 1).contiguous()   # (B, N, 3)
        norm = compute_LRA(xyz, weighting=True, nsample=_LRA_NSAMPLE)

        l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sa3(l2_xyz, l2_norm, l2_points)
        _, _, l4_points = self.sa4(l3_xyz, l3_norm, l3_points)

        f = l4_points.view(x.shape[0], 512)
        f = F.relu(self.bn1(self.fc1(f)))
        f = F.relu(self.bn2(self.fc2(f)))       # (B, 128)
        return f

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: RIConvPPEmbedder, ckpt_path: str) -> None:
    raise NotImplementedError(
        "No pretrained checkpoint is available for riconvpp -- train without --pretrained."
    )
