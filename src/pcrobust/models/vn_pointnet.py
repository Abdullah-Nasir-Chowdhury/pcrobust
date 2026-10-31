"""VN-PointNet backbone -- Vector Neuron version of PointNet (Deng et al.,
ICCV 2021), following FlyingGiraffe/vnn's models/vn_pointnet.py encoder.

Registry key:
"vn_pointnet". Same Embedder API as every other pcrobust.models backbone:
forward((B, 3, N)) -> (logits, L2-normalized 128-d embedding, aux).

Two documented deviations from the official encoder, both deliberate:
  - no VN feature-transform (fstn): the training loop applies an orthogonality
    regularizer to any non-None aux, which is meaningless for a VN vector
    frame, so aux is None here (like every non-PointNet backbone in this
    suite) and the fstn branch is dropped rather than silently
    mis-regularized. conv2's input width shrinks accordingly (64//3 instead
    of the fstn-concat 64//3*2).
  - mean pooling over the k neighbors (the official repo's recommended
    setting for classification) rather than VN max pooling.

Architecturally SO(3)-equivariant up to the k-NN graph (rotation-preserving)
and float precision; the embedding is rotation-invariant via VNStdFeature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vn_layers import (VNLinear, VNBatchNorm, VNLinearLeakyReLU, VNStdFeature,
                        get_graph_feature_cross, mean_pool)


class VNPointNetEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128, k: int = 20):
        super().__init__()
        self.k = k
        c64, c128, c1024 = 64 // 3, 128 // 3, 1024 // 3
        self.conv_pos = VNLinearLeakyReLU(3, c64, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(c64, c64, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(c64, c128, dim=4, negative_slope=0.0)
        self.conv3 = VNLinear(c128, c1024)
        self.bn3 = VNBatchNorm(c1024, dim=4)
        self.std_feature = VNStdFeature(c1024 * 2, dim=4, negative_slope=0.0)

        feat_dim = c1024 * 2 * 3                       # 341*2*3 = 2046
        # Head mirrors pointnet.py's embed_head/cls_head structure.
        self.embed_head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                              # x: (B, 3, N)
        B, _, N = x.shape
        x = x.unsqueeze(1)                             # (B, 1, 3, N)
        feat = get_graph_feature_cross(x, k=self.k)    # (B, 3, 3, N, k)
        x = mean_pool(self.conv_pos(feat))             # (B, 21, 3, N)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))                    # (B, 341, 3, N)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)                  # (B, 682, 3, N)
        x, _ = self.std_feature(x)                     # invariant
        x = x.reshape(B, -1, N)                        # (B, 2046, N)
        return x.max(dim=-1)[0]                        # (B, 2046)

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: VNPointNetEmbedder, ckpt_path: str) -> None:
    """No public pretrained checkpoint matches this embedder; VN baselines are
    trained from scratch like the RI family in this suite."""
    raise NotImplementedError(
        "vn_pointnet has no pretrained backbone -- train from scratch "
        f"(got --pretrained {ckpt_path})")
