"""Checkpoint loading shared by every evaluation entry point.

``load_model()`` restores a trained Embedder from a checkpoint dict of the
form ``{state_dict, arch, classes, embed_dim, num_classes}`` (the format
every training run in this benchmark saves).

The checkpoint's ``arch`` field is either a base architecture key
(``pointnet``) or a base key plus a training-recipe suffix
(``pointnet_invreg``, ``pointnet_invreg_warmup``, ...). Recipe variants
reuse the base architecture class unchanged, so the suffix only names the
training run; :func:`base_arch` strips it.
"""

import torch
import torch.nn as nn

from pcrobust.models import REGISTRY, get_model


def base_arch(key: str) -> str:
    """Map a checkpoint key to its architecture: ``pointnetpp_invreg`` -> ``pointnetpp``."""
    if key in REGISTRY:
        return key
    # Longest match first, so pointnetpp_* is not read as pointnet + "pp_*"
    # and vn_pointnet_* is not read as pointnet.
    for arch in sorted(REGISTRY, key=len, reverse=True):
        if key.startswith(arch + "_"):
            return arch
    raise KeyError(f"no architecture matches checkpoint key {key!r}; "
                   f"known: {', '.join(REGISTRY)}")


def load_model(ckpt_path: str,
               device: torch.device) -> tuple[nn.Module, list[str]]:
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch  = ckpt.get("arch", "pointnet")   # backward-compatible default
    model = get_model(base_arch(arch), num_classes=ckpt["num_classes"],
                      embed_dim=ckpt["embed_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[load]  arch={arch}")
    return model, ckpt["classes"]
