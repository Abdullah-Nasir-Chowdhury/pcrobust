"""
Vendored, near-verbatim from the OFFICIAL PointTransformerV2 repository:
    https://github.com/Pointcept/PointTransformerV2
    (commit 5386c4d71f3d6c42c24a8105fce8750e9355dc54, 2023-06-05)
Original authors: Xiaoyang Wu et al. Source file:
`pcr/models/point_transformer/point_transformer_cls.py` -- "Point Transformer
V1 for Object Classification" (their own docstring notes: "Might be a bit
different from the original paper").

This is the model actually used for "PointTransformerV2" in this project
(see `pcrobust/models/pointtransformer.py` for why): the repo's own PTv2
architecture (`pcr/models/point_transformer2/`) is a segmentation-only
encoder-decoder over voxelized, variable-length scenes, with its ModelNet40
classification config explicitly marked "Unmaintained" by the authors --
a poor and unsupported fit for this project's fixed-size single-object
embedding task. This file, from the *same* officially-maintained repository,
is the properly classification-shaped model the Pointcept authors ship
alongside PTv2 for benchmarking comparisons.

Only the registry decorator/import was removed (noted "PATCHED") --
`@MODELS.register_module(...)` and `from ..builder import MODELS` require
Pointcept's full model-registry infrastructure, which this project has no
use for outside that one repo. Removing it changes zero computation; the
classes are otherwise identical, plain `nn.Module`s.

`PointTransformerCls38` (blocks=[1,2,2,2,2]) is used by this project's
adapter -- the official repo defines 3 depths (26/38/50) but wires none of
them into an example config, so this is my own reasonable default (a
middle depth) rather than an official recommendation.
"""

import torch
import torch.nn as nn

from .point_transformer_seg import TransitionDown, Bottleneck


class PointTransformerCls(nn.Module):
    def __init__(self, block, blocks, in_channels=6, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes,
                                   stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes,
                                   stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes,
                                   stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes,
                                   stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes,
                                   stride=stride[4], nsample=nsample[4])  # N/256
        self.cls = nn.Sequential(nn.Linear(planes[4], 256),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(256, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(128, num_classes))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward_backbone(self, input_dict):
        """PATCHED addition (not in the official file): factors the official
        forward() into a backbone half (through the final per-object average
        pool, 512-dim) and the classifier half (self.cls), so the adapter
        (pcrobust/models/pointtransformer.py) can attach this project's own
        embedding head after the exact same backbone. Numerically identical
        to the official forward() up to where it calls self.cls."""
        p0 = input_dict["coord"]
        x0 = input_dict["feat"]
        o0 = input_dict["offset"].int()
        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x = []
        for i in range(o5.shape[0]):
            if i == 0:
                s_i, e_i, cnt = 0, o5[0], o5[0]
            else:
                s_i, e_i, cnt = o5[i - 1], o5[i], o5[i] - o5[i - 1]
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x.append(x_b)
        x = torch.cat(x, 0)
        return x

    def forward(self, input_dict):
        x = self.forward_backbone(input_dict)
        x = self.cls(x)
        return x


class PointTransformerCls26(PointTransformerCls):
    def __init__(self, **kwargs):
        super(PointTransformerCls26, self).__init__(Bottleneck, [1, 1, 1, 1, 1], **kwargs)


class PointTransformerCls38(PointTransformerCls):
    def __init__(self, **kwargs):
        super(PointTransformerCls38, self).__init__(Bottleneck, [1, 2, 2, 2, 2], **kwargs)


class PointTransformerCls50(PointTransformerCls):
    def __init__(self, **kwargs):
        super(PointTransformerCls50, self).__init__(Bottleneck, [1, 2, 3, 5, 2], **kwargs)
