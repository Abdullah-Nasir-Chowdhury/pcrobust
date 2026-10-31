"""Vendored verbatim from the official PointTransformerV2 repo
(pcr/models/point_transformer/utils.py)."""

import torch
import torch.nn as nn


class LayerNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
