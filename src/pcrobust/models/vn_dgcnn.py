"""VN-DGCNN backbone -- Vector Neuron version of DGCNN (Deng et al., ICCV
2021), following FlyingGiraffe/vnn's models/vn_dgcnn_cls.py.

Registry key:
"vn_dgcnn". Same Embedder API as every other pcrobust.models backbone:
forward((B, 3, N)) -> (logits, L2-normalized 128-d embedding, aux=None).

Mean pooling over the k neighbors (the official repo's recommended setting
for classification). The scalar head after VNStdFeature mirrors dgcnn.py's
linear1/bn6/linear2/bn7/embed_head/cls_head structure so the two DGCNNs
differ only in their equivariant vs. plain feature extraction.

Architecturally SO(3)-equivariant up to the k-NN graph (rotation-preserving)
and float precision; the embedding is rotation-invariant via VNStdFeature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vn_layers import (VNLinearLeakyReLU, VNStdFeature, get_graph_feature,
                        mean_pool)


class VNDGCNNEmbedder(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 128, k: int = 20):
        super().__init__()
        self.k = k
        c64, c128, c256, c1024 = 64 // 3, 128 // 3, 256 // 3, 1024 // 3
        self.conv1 = VNLinearLeakyReLU(2,        c64,  dim=5)
        self.conv2 = VNLinearLeakyReLU(c64 * 2,  c64,  dim=5)
        self.conv3 = VNLinearLeakyReLU(c64 * 2,  c128, dim=5)
        self.conv4 = VNLinearLeakyReLU(c128 * 2, c256, dim=5)
        self.conv5 = VNLinearLeakyReLU(c64 * 2 + c128 + c256, c1024, dim=4,
                                       share_nonlinearity=True)
        self.std_feature = VNStdFeature(c1024 * 2, dim=4)

        feat_dim = c1024 * 2 * 3 * 2                   # 341*2*3 * (max+avg) = 4092
        self.linear1    = nn.Linear(feat_dim, 512, bias=False)
        self.bn6        = nn.BatchNorm1d(512)
        self.linear2    = nn.Linear(512, 256)
        self.bn7        = nn.BatchNorm1d(256)
        self.embed_head = nn.Linear(256, embed_dim)
        self.cls_head   = nn.Linear(embed_dim, num_classes)

    def _encode(self, x):                              # x: (B, 3, N)
        B, _, N = x.shape
        x = x.unsqueeze(1)                             # (B, 1, 3, N)
        x1 = mean_pool(self.conv1(get_graph_feature(x,  self.k)))   # (B, 21, 3, N)
        x2 = mean_pool(self.conv2(get_graph_feature(x1, self.k)))   # (B, 21, 3, N)
        x3 = mean_pool(self.conv3(get_graph_feature(x2, self.k)))   # (B, 42, 3, N)
        x4 = mean_pool(self.conv4(get_graph_feature(x3, self.k)))   # (B, 85, 3, N)
        x = self.conv5(torch.cat((x1, x2, x3, x4), dim=1))          # (B, 341, 3, N)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)                  # (B, 682, 3, N)
        x, _ = self.std_feature(x)                     # invariant
        x = x.reshape(B, -1, N)                        # (B, 2046, N)
        g = torch.cat([x.max(-1)[0], x.mean(-1)], dim=1)            # (B, 4092)
        g = F.leaky_relu(self.bn6(self.linear1(g)), 0.2)
        g = F.dropout(g, p=0.5, training=self.training)
        g = F.leaky_relu(self.bn7(self.linear2(g)), 0.2)
        g = F.dropout(g, p=0.5, training=self.training)
        return g                                       # (B, 256)

    def forward(self, x):
        f = self._encode(x)
        emb = F.normalize(self.embed_head(f), dim=1)
        return self.cls_head(emb), emb, None

    def get_embedding(self, x):
        return self.forward(x)[1]


def load_pretrained_backbone(model: VNDGCNNEmbedder, ckpt_path: str) -> None:
    """No public pretrained checkpoint matches this embedder; VN baselines are
    trained from scratch like the RI family in this suite."""
    raise NotImplementedError(
        "vn_dgcnn has no pretrained backbone -- train from scratch "
        f"(got --pretrained {ckpt_path})")
