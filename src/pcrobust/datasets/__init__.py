"""Dataset preparation, download, and loading for the pcrobust benchmark.

One module per domain (``modelnet40``, ``shapenet``, ``scanobjectnn``,
``abo``, ``s3dis``, plus the ``modelnet40c`` corruption benchmark), each
folding that domain's download / preparation / object-pool pipeline.
Data lives under the cache root returned by :func:`pcrobust_home`
(``$PCROBUST_HOME``, default ``~/.cache/pcrobust``).
"""

from pcrobust.datasets._paths import (
    checkpoint_dir,
    checkpoint_file,
    domain_root,
    pcrobust_home,
    results_dir,
)
from pcrobust.datasets.gallery import MeshGalleryDataset

_DOMAIN_MODULES = (
    "modelnet40",
    "modelnet40c",
    "shapenet",
    "scanobjectnn",
    "abo",
    "s3dis",
)

__all__ = [
    "domain_module",
    "MeshGalleryDataset",
    "checkpoint_dir",
    "checkpoint_file",
    "domain_root",
    "pcrobust_home",
    "results_dir",
    *_DOMAIN_MODULES,
]


def __getattr__(name):
    # Domain modules are imported lazily so that `import pcrobust.datasets`
    # does not require every domain's optional dependencies (h5py, gdown, ...).
    if name in _DOMAIN_MODULES:
        import importlib

        return importlib.import_module(f"pcrobust.datasets.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def domain_module(domain: str):
    """The module for a benchmark domain (extract_object_pts, pools, prepare_dataset)."""
    if domain not in _DOMAIN_MODULES or domain == "modelnet40c":
        raise ValueError(f"unknown domain {domain!r}; known: "
                         f"{', '.join(d for d in _DOMAIN_MODULES if d != 'modelnet40c')}")
    import importlib

    return importlib.import_module(f"pcrobust.datasets.{domain}")
