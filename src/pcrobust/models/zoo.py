"""Checkpoint zoo: manifest-driven download of released model checkpoints.

Released checkpoints are hosted as GitHub Release assets and described by
``zoo_manifest.json`` (shipped next to this module). Each manifest entry
records the file's SHA256 and size; ``checkpoint_path()`` downloads a named
checkpoint on first use into the local cache
(``$PCROBUST_HOME/checkpoints/<revision>/``, defaulting to
``~/.cache/pcrobust``), verifies its SHA256, and returns the cached path on
every subsequent call.

The manifest ships with ``base_url: null`` and an empty ``files`` dict; it
is populated at release time once the assets are uploaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

DEFAULT_REVISION = "wacv2027"

_MANIFEST_PATH = Path(__file__).with_name("zoo_manifest.json")


def cache_root() -> Path:
    """Local cache directory (override with the PCROBUST_HOME env var)."""
    return Path(os.environ.get("PCROBUST_HOME",
                               os.path.expanduser("~/.cache/pcrobust")))


def _load_manifest() -> dict:
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_checkpoints(revision: str = DEFAULT_REVISION) -> list[str]:
    """Names of every checkpoint the manifest describes for ``revision``."""
    manifest = _load_manifest()
    if revision != manifest.get("revision"):
        raise ValueError(
            f"unknown revision {revision!r}; this package's manifest "
            f"describes revision {manifest.get('revision')!r}")
    return list(manifest.get("files") or {})


def checkpoint_path(name: str, revision: str = DEFAULT_REVISION) -> Path:
    """Return the local path of checkpoint ``name``, downloading if needed.

    Looks ``name`` up in ``zoo_manifest.json``, downloads
    ``<base_url>/<name>`` into ``$PCROBUST_HOME/checkpoints/<revision>/``,
    verifies its SHA256 against the manifest (raising ``RuntimeError`` on
    mismatch), and caches the verified file. A cached file whose hash still
    matches is returned without touching the network.
    """
    manifest = _load_manifest()
    if revision != manifest.get("revision"):
        raise ValueError(
            f"unknown revision {revision!r}; this package's manifest "
            f"describes revision {manifest.get('revision')!r}")

    files = manifest.get("files") or {}
    if name not in files:
        known = ", ".join(files) or "(manifest is empty until release)"
        raise KeyError(f"unknown checkpoint {name!r}; known: {known}")
    entry = files[name]

    dest = cache_root() / "checkpoints" / revision / name
    if dest.exists():
        if _sha256(dest) == entry["sha256"]:
            return dest
        dest.unlink()  # corrupt or stale cache entry: re-download

    base_url = manifest.get("base_url")
    if not base_url:
        raise RuntimeError(
            "zoo_manifest.json has no base_url yet -- it is populated at "
            "release time once the checkpoint assets are uploaded")

    url = base_url.rstrip("/") + "/" + name
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        digest = _sha256(tmp)
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"SHA256 mismatch for {name} downloaded from {url}: "
                f"expected {entry['sha256']}, got {digest}")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def load_model(name: str, pretrained: bool = True,
               revision: str = DEFAULT_REVISION, device=None):
    """Fetch the named released checkpoint and restore its model.

    Thin wrapper: ``checkpoint_path()`` for the download/cache/verify step,
    then :func:`pcrobust.models.loading.load_model` to rebuild the model.
    Returns ``(model, classes)`` exactly like the loading function.
    """
    if not pretrained:
        raise ValueError(
            "load_model() restores released checkpoints (pretrained=True); "
            "for an untrained model use pcrobust.models.get_model()")
    import torch

    from pcrobust.models.loading import load_model as _load

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = checkpoint_path(name, revision=revision)
    return _load(str(path), device)
