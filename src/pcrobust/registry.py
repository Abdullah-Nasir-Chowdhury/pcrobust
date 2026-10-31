"""Model registry metadata: architecture keys, families, display names.

The ``family`` field is load-bearing for every auto-generated legend,
table, and report. The taxonomy is fixed:

- ``rotation-invariant``: RIConv, RIConv++, RISurConv — and only these
  three. Their features are constructed to be invariant to rotation.
- ``equivariant``: VN-PointNet, VN-DGCNN. Vector Neurons networks are
  rotation-equivariant, not invariant; the two terms are not
  interchangeable and must never be collapsed into one label.
- ``standard``: everything else.

Model *builders* (torch modules) live in :mod:`pcrobust.models`; this
module is import-light on purpose so that listing models never requires
torch.
"""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES = ("standard", "rotation-invariant", "equivariant")

# arch key -> (family, display name)
_MODEL_INFO: dict[str, tuple[str, str]] = {
    "pointnet":         ("standard",           "PointNet"),
    "pointnetpp":       ("standard",           "PointNet++"),
    "dgcnn":            ("standard",           "DGCNN"),
    "pointmlp":         ("standard",           "PointMLP"),
    "pointtransformer": ("standard",           "Point Transformer"),
    "riconv":           ("rotation-invariant", "RIConv"),
    "riconvpp":         ("rotation-invariant", "RIConv++"),
    "risurconv":        ("rotation-invariant", "RISurConv"),
    "vn_pointnet":      ("equivariant",        "VN-PointNet"),
    "vn_dgcnn":         ("equivariant",        "VN-DGCNN"),
}


@dataclass(frozen=True)
class ModelInfo:
    key: str
    family: str
    display_name: str


def list_models() -> list[str]:
    """Keys of every architecture in the benchmark, in paper order."""
    return list(_MODEL_INFO)


def model_info(key: str) -> ModelInfo:
    try:
        family, display = _MODEL_INFO[key]
    except KeyError:
        raise KeyError(
            f"unknown architecture {key!r}; known: {', '.join(_MODEL_INFO)}"
        ) from None
    return ModelInfo(key=key, family=family, display_name=display)


def family_of(key: str) -> str:
    return model_info(key).family
