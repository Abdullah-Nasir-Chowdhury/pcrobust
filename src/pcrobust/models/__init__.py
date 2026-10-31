"""Model builders: one Embedder class per architecture, plus the REGISTRY.

Every backbone follows the same Embedder API:
``forward((B, 3, N)) -> (logits, L2-normalized embedding, aux)``.

Family taxonomy (see :mod:`pcrobust.registry`): riconv/riconvpp/risurconv
are rotation-invariant; vn_pointnet/vn_dgcnn are rotation-*equivariant*
(Vector Neurons); everything else is standard.
"""

from .pointnet  import PointNetEmbedder,   load_pretrained_backbone as _pn_load
from .pointnetpp import PointNetPPEmbedder, load_pretrained_backbone as _pp_load
from .dgcnn      import DGCNNEmbedder,      load_pretrained_backbone as _dgcnn_load
from .riconv     import RIConvEmbedder,     load_pretrained_backbone as _riconv_load
from .riconvpp   import RIConvPPEmbedder,   load_pretrained_backbone as _riconvpp_load
from .risurconv  import RISurConvEmbedder,  load_pretrained_backbone as _risurconv_load
from .pointmlp   import PointMLPEmbedder,   load_pretrained_backbone as _pointmlp_load
from .pointtransformer import PointTransformerEmbedder, load_pretrained_backbone as _pt_load
# Vector Neuron versions of PointNet and DGCNN (Deng et al., ICCV 2021) --
# rotation-equivariant baselines (equivariant, not rotation-invariant).
from .vn_pointnet import VNPointNetEmbedder, load_pretrained_backbone as _vnpn_load
from .vn_dgcnn   import VNDGCNNEmbedder,   load_pretrained_backbone as _vndg_load

REGISTRY = {
    "pointnet":         (PointNetEmbedder,         _pn_load),
    "pointnetpp":       (PointNetPPEmbedder,        _pp_load),
    "dgcnn":            (DGCNNEmbedder,             _dgcnn_load),
    "riconv":           (RIConvEmbedder,            _riconv_load),
    "riconvpp":         (RIConvPPEmbedder,          _riconvpp_load),
    "risurconv":        (RISurConvEmbedder,         _risurconv_load),
    "pointmlp":         (PointMLPEmbedder,          _pointmlp_load),
    "pointtransformer": (PointTransformerEmbedder,  _pt_load),
    "vn_pointnet":      (VNPointNetEmbedder,        _vnpn_load),
    "vn_dgcnn":         (VNDGCNNEmbedder,           _vndg_load),
}


def get_model(arch: str, num_classes: int, embed_dim: int = 128):
    cls, _ = REGISTRY[arch]
    return cls(num_classes=num_classes, embed_dim=embed_dim)


def load_backbone(arch: str, model, ckpt_path: str):
    _, fn = REGISTRY[arch]
    fn(model, ckpt_path)


__all__ = [
    "REGISTRY",
    "get_model",
    "load_backbone",
    "PointNetEmbedder",
    "PointNetPPEmbedder",
    "DGCNNEmbedder",
    "RIConvEmbedder",
    "RIConvPPEmbedder",
    "RISurConvEmbedder",
    "PointMLPEmbedder",
    "PointTransformerEmbedder",
    "VNPointNetEmbedder",
    "VNDGCNNEmbedder",
]
