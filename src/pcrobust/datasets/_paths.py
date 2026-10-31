"""Cache-root convention for pcrobust dataset files.

Every dataset domain keeps its raw downloads and prepared data under
``<PCROBUST_HOME>/<domain>/``, where ``PCROBUST_HOME`` defaults to
``~/.cache/pcrobust`` and can be overridden via the environment variable
of the same name. Below the per-domain root, the layout is:

    <domain>/_raw_download/   raw archives / fetched files
    <domain>/data/train/<class>/<class><i>.ply
    <domain>/data/test/<class>/<class>_test<i>.ply
    <domain>/data/gallery/<class><i>_gallery.ply
"""

import os
from pathlib import Path


def pcrobust_home() -> Path:
    """Root of the pcrobust data cache ($PCROBUST_HOME, default ~/.cache/pcrobust)."""
    return Path(os.environ.get("PCROBUST_HOME", os.path.expanduser("~/.cache/pcrobust")))


def domain_root(domain: str) -> Path:
    """Per-domain data root: ``<PCROBUST_HOME>/<domain>/``."""
    return pcrobust_home() / domain


# Checkpoints and results share the same cache root. A checkpoint is
# addressed by (domain, key), where key is the architecture key optionally
# followed by a training-recipe suffix (``pointnet``, ``pointnet_invreg``,
# ``pointnet_invreg_warmup``, ``pointnet_invreg_pose270`` ...). This is the
# layout pcrobust.models.zoo downloads into, so files fetched from the zoo
# and files copied in by hand are found the same way.

CHECKPOINT_REVISION = "wacv2027"


def checkpoint_dir(domain: str = "modelnet40",
                   revision: str = CHECKPOINT_REVISION) -> Path:
    """``<PCROBUST_HOME>/checkpoints/<revision>/<domain>/``."""
    return pcrobust_home() / "checkpoints" / revision / domain


def checkpoint_file(key: str, domain: str = "modelnet40",
                    revision: str = CHECKPOINT_REVISION) -> Path:
    """``<checkpoint_dir(domain)>/<key>.pth``."""
    return checkpoint_dir(domain, revision) / f"{key}.pth"


def results_dir(*parts: str) -> Path:
    """``<PCROBUST_HOME>/results/<parts...>/`` (not created here)."""
    return pcrobust_home().joinpath("results", *parts)
