"""Evaluate trained ModelNet40 checkpoints on ModelNet40-C (Sun et al., 2022),
filtered to the benchmark's 10-class subset (see
:mod:`pcrobust.datasets.modelnet40c` for the filter, relabelling, and axis
fix).

Covers both training conditions the paper compares, by checkpoint key:

  - baseline checkpoints (``pointnet``), trained with classification loss only
  - invariance-trained checkpoints (``pointnet_invreg``): the SAME
    architecture class retrained with cls_loss + lambda_inv *
    (inv_rotation + inv_noise + inv_occlusion); ``<arch>_invreg_warmup``
    adds a linear lambda_inv warmup

Baseline and invreg checkpoints for one architecture carry an identical
classes list and num_classes/embed_dim (10/128), so one label mapping
serves both.

Output: <PCROBUST_HOME>/results/modelnet40c/raw_<key>.csv (or --out_dir),
one row per (corruption, severity, instance).

Usage:
    python -m pcrobust.evaluate.modelnet40c --arch pointnet
    python -m pcrobust.evaluate.modelnet40c --arch pointnet_invreg
    python -m pcrobust.evaluate.modelnet40c --arch all --device cuda
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

from pcrobust.datasets import checkpoint_dir, results_dir
from pcrobust.datasets.modelnet40c import (
    _CLEAN_HDF5_DIR,
    CORRUPTIONS,
    SEVERITIES,
    STD40_TO_OURS,
    load_and_filter,
)
from pcrobust.models.loading import load_model

BASELINE_ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp",
                   "risurconv", "pointmlp", "pointtransformer",
                   "vn_pointnet", "vn_dgcnn"]
INVREG_ARCHS = [f"{a}_invreg" for a in BASELINE_ARCHS]
WARMUP_ARCHS = [f"{a}_invreg_warmup" for a in BASELINE_ARCHS]
EXTRA_INVREG_ARCHS = ["pointnetpp_invreg_modelnet40b", "riconvpp_invreg_modelnet40b"]
ALL_ARCHS = BASELINE_ARCHS + INVREG_ARCHS + WARMUP_ARCHS + EXTRA_INVREG_ARCHS

_CKPT_DIR = None   # set by main(); None means the zoo cache for modelnet40
_RESULTS = None


def evaluate(model, data: np.ndarray, labels: np.ndarray, device) -> np.ndarray:
    """Feeds each (corruption, severity) file's NATIVE point count straight to
    the model, no resampling to a fixed N. This matters: the official
    ModelNet40-C release does not ship uniform-N corruptions -- lidar is 768
    pts, cutout/density shrink from ~965 down to 649 by severity 5,
    upsampling grows from 1228 up to 2048 -- and the official repo's own
    dataloader.py (ModelNet40C.__getitem__: `self.data[item]#[:self.num_points]`,
    truncation explicitly commented out) deliberately feeds that native count
    through unmodified. An earlier version of this script instead resampled
    every corruption to exactly 1024 points (this project's own training
    convention, inherited from evaluate_corruption.py) -- silently forcing
    every input through a data assumption the official benchmark doesn't
    make, and never actually necessary: empirically verified (2026-08-08,
    on GPU) that all 8 of this project's architectures forward-pass cleanly
    at every N actually present in the release (649 through 2048) with no
    resampling at all. All instances within a single (corruption, severity)
    file share one fixed N already (it's one array), so batches stack
    directly -- no per-instance sampling call needed anywhere in this
    function."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), 32):
            batch = data[i:i + 32]
            x = torch.from_numpy(batch.astype(np.float32)).transpose(1, 2).to(device)
            logits, _, _ = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def run_one(arch: str, device: torch.device, out_tag: str = "", class_map: str = "modelnet40"):
    ckpt = (_CKPT_DIR or checkpoint_dir("modelnet40")) / f"{arch}.pth"
    if not ckpt.exists():
        print(f"[eval] SKIP {arch}: checkpoint not found at {ckpt}")
        return
    print(f"[eval] arch: {arch}  device: {device}  ckpt: {ckpt}")
    model, ckpt_classes = load_model(str(ckpt), device)
    model.eval()

    mapping = STD40_TO_OURS
    if class_map == "auto":
        # Rebuild the standard-40-index -> checkpoint-label mapping from the
        # checkpoint's own classes list against the official shape_names.txt
        # order (for checkpoints trained on a different 10-class subset).
        names = (_CLEAN_HDF5_DIR / "shape_names.txt").read_text().split()
        missing = [c for c in ckpt_classes if c not in names]
        if missing:
            sys.exit(f"[eval] --class_map auto: checkpoint classes {missing} "
                     f"not in shape_names.txt")
        mapping = {names.index(c): i for i, c in enumerate(ckpt_classes)}
        print(f"[eval] auto class map ({len(mapping)} classes): {mapping}")

    rows = []
    print("[eval] clean baseline (none) ...")
    data, labels = load_and_filter("none", None, mapping)
    preds = evaluate(model, data, labels, device)
    acc = (preds == labels).mean()
    print(f"    clean acc = {acc:.4f}  (n={len(labels)})")
    for i, (t, p) in enumerate(zip(labels, preds)):
        rows.append({"corruption": "none", "severity": 0, "instance_idx": i,
                     "true_label": int(t), "pred_label": int(p), "correct": int(t == p)})

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            data, labels = load_and_filter(corruption, severity, mapping)
            preds = evaluate(model, data, labels, device)
            acc = (preds == labels).mean()
            print(f"    {corruption:<16s} severity={severity}  acc={acc:.4f}  (n={len(labels)})")
            for i, (t, p) in enumerate(zip(labels, preds)):
                rows.append({"corruption": corruption, "severity": severity, "instance_idx": i,
                             "true_label": int(t), "pred_label": int(p), "correct": int(t == p)})

    _RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS / f"raw_{arch}{out_tag}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["corruption", "severity", "instance_idx",
                                           "true_label", "pred_label", "correct"])
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] saved {out_path}  ({len(rows)} rows)\n")


def main(argv=None):
    global _CKPT_DIR, _RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=ALL_ARCHS + ["all"])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--ckpt_root", default=None,
                        help="Directory holding <key>.pth for every checkpoint "
                             "(default: the zoo cache location for modelnet40).")
    parser.add_argument("--out_dir", default=None,
                        help="Default: <PCROBUST_HOME>/results/modelnet40c.")
    parser.add_argument("--out_tag", default="",
                        help="Appended to the output CSV filename (raw_<arch><out_tag>.csv) "
                             "so replicate runs never overwrite the published raw_<arch>.csv.")
    parser.add_argument("--class_map", default="modelnet40", choices=["modelnet40", "auto"],
                        help="'auto' rebuilds the standard-40-index mapping from the "
                             "checkpoint's own classes list.")
    args = parser.parse_args(argv)

    torch.manual_seed(42)
    _CKPT_DIR = Path(args.ckpt_root) if args.ckpt_root else None
    _RESULTS = Path(args.out_dir) if args.out_dir else results_dir("modelnet40c")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    archs = ALL_ARCHS if args.arch == "all" else [args.arch]
    for arch in archs:
        run_one(arch, device, out_tag=args.out_tag, class_map=args.class_map)


if __name__ == "__main__":
    main()
