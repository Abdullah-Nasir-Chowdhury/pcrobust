"""pcrobust — statistical robustness benchmark for point cloud classification.

Companion package to "Do We Still Need Rotation-Invariant Architectures?
A Statistical Robustness Benchmark for Point Cloud Classification"
(WACV 2027).

Top-level API::

    import pcrobust
    pcrobust.list_models()                          # the 10 architectures
    model, classes = pcrobust.load_model("modelnet40/pointnet.pth")  # zoo checkpoint
    net = pcrobust.get_model("dgcnn", num_classes=10)                # untrained
    pts2 = pcrobust.corrupt(pts, "occlusion", severity=3)            # paper ladders

``import pcrobust`` stays light: torch and open3d load only when a model or
corruption function is first used.
"""

from pcrobust.registry import FAMILIES, ModelInfo, family_of, list_models, model_info

__version__ = "1.0.2"

from pcrobust._welcome import show_once as _show_welcome

_show_welcome(__version__)

_LAZY = {
    "load_model": ("pcrobust.models.zoo", "load_model"),
    "get_model": ("pcrobust.models", "get_model"),
    "corrupt": ("pcrobust.cli", "corrupt_points"),
}

__all__ = [
    "FAMILIES",
    "ModelInfo",
    "corrupt",
    "family_of",
    "get_model",
    "list_models",
    "load_model",
    "model_info",
    "__version__",
]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
