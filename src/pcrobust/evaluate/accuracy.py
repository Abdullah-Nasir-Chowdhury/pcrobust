"""
Held-out classification-accuracy testing under clean/pose/noise/occlusion
conditions, for any architecture in pcrobust.models.REGISTRY. This is the
"test_results.csv"-shaped deliverable from the original project plan
(Phase 8: "domain, architecture, test_accuracy, ...") that was never
actually produced -- test.py's CLI eval was removed at some point and
never replaced -- expanded per explicit user request to also cover
degraded conditions, not just clean: this measures whether the model
still predicts the right CLASS under the same perturbation severities
(ANGLES/SIGMAS/LEVELS) the pose/noise/occlusion self-similarity studies
use, a directly-interpretable companion metric to those studies' "does
the embedding still look similar to itself" question.

Uses data/test/ (the held-out split -- 300 instances/10 classes for
ModelNet40) rather than the self-similarity studies' confirmatory pool
(740 instances): this is genuine held-out test data, the correct choice
for a classification-accuracy number, and keeps this experiment lighter
given occlusion's per-instance Hidden-Point-Removal cost.

Per instance: 5-replicate softmax-averaging (same K_REPLICATES=5
convention as every other study here) then argmax -- not a single noisy
sample's decision.

Usage:
  python -m pcrobust.evaluate.accuracy --arch pointnetpp --domain modelnet40

Runs ALL conditions in one invocation: clean (once), pose (ANGLES=[20,40,
60,80,120]), noise (SIGMAS=[0.005,0.01,0.02,0.03,0.05]), occlusion
(LEVELS=[0.3,0.6,1.2,2.5,6.0]).

Output: <PCROBUST_HOME>/results/classification_accuracy/<domain>/
        accuracy_raw_<arch>.csv    (condition, level, identity, scan,
                                     true_class, pred_class, correct)
        accuracy_summary_<arch>.csv (condition, level, n, accuracy)
"""

import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

_rng = np.random.default_rng(42)

from pcrobust.corruptions.occlusion import LEVELS, occlude, random_direction
from pcrobust.datasets import checkpoint_file, domain_module, results_dir
from pcrobust.models.loading import load_model
from pcrobust.utils import _sample

ANGLES = [20, 40, 60, 80, 120]
SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05]

K_REPLICATES = 5
N_POINTS = 1024


def rotate_yaw(pts: np.ndarray, deg: float) -> np.ndarray:
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ R.T


def add_noise(pts: np.ndarray, sigma: float) -> np.ndarray:
    return pts + np.random.randn(*pts.shape).astype(np.float32) * sigma


def predict(model, pts, device, sample_fn, classes, k=K_REPLICATES):
    probs = []
    for _ in range(k):
        p = sample_fn(pts.copy(), N_POINTS)
        x = torch.from_numpy(p).unsqueeze(0).transpose(1, 2).to(device)
        with torch.no_grad():
            logits, _, _ = model(x)
        probs.append(F.softmax(logits, dim=1))
    mean_prob = torch.stack(probs).mean(0)
    pred_idx = mean_prob.argmax(1).item()
    return classes[pred_idx]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True,
                        choices=["pointnet", "pointnetpp", "dgcnn", "riconv",
                                 "riconvpp", "risurconv", "pointmlp", "pointtransformer",
                                 "vn_pointnet", "vn_dgcnn"])
    parser.add_argument("--domain", default="modelnet40")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint path (default: the zoo cache location for --arch/--domain).")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: <PCROBUST_HOME>/results/classification_accuracy/<domain>).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    global _rng
    np.random.seed(42)
    torch.manual_seed(42)
    _rng = np.random.default_rng(42)

    dom = domain_module(args.domain)
    extract_object_pts = dom.extract_object_pts
    ckpt = Path(args.ckpt) if args.ckpt else checkpoint_file(args.arch, args.domain)
    test_dir = dom.TEST_DIR

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[classification_accuracy] arch={args.arch} domain={args.domain} "
          f"device={device}", flush=True)

    model, classes = load_model(str(ckpt), device)

    files = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.glob("*.ply")):
            files.append((class_dir.name, f))
    print(f"[classification_accuracy] N = {len(files)} test instances", flush=True)

    out_root = Path(args.out_dir) if args.out_dir else results_dir("classification_accuracy", args.domain)
    out_root.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    for i, (true_class, path) in enumerate(files, 1):
        try:
            canonical_pts = extract_object_pts(str(path))
        except Exception as e:
            print(f"[classification_accuracy] EXTRACTION FAILED for {path.name}: "
                  f"{e} -- SKIPPING", flush=True)
            continue

        pred = predict(model, canonical_pts, device, _sample, classes)
        raw_rows.append({"condition": "clean", "level": "", "identity": true_class,
                          "scan": path.stem, "true_class": true_class, "pred_class": pred,
                          "correct": int(pred == true_class)})

        for angle in ANGLES:
            rotated = rotate_yaw(canonical_pts, angle)
            pred = predict(model, rotated, device, _sample, classes)
            raw_rows.append({"condition": "pose", "level": angle, "identity": true_class,
                              "scan": path.stem, "true_class": true_class, "pred_class": pred,
                              "correct": int(pred == true_class)})

        for sigma in SIGMAS:
            noisy = add_noise(canonical_pts, sigma)
            pred = predict(model, noisy, device, _sample, classes)
            raw_rows.append({"condition": "noise", "level": sigma, "identity": true_class,
                              "scan": path.stem, "true_class": true_class, "pred_class": pred,
                              "correct": int(pred == true_class)})

        direction = random_direction(_rng)
        for dist_mult in LEVELS:
            occluded = occlude(canonical_pts, dist_mult, direction)
            pred = predict(model, occluded, device, _sample, classes)
            raw_rows.append({"condition": "occlusion", "level": dist_mult, "identity": true_class,
                              "scan": path.stem, "true_class": true_class, "pred_class": pred,
                              "correct": int(pred == true_class)})

        print(f"[classification_accuracy] ({i}/{len(files)}) {true_class}/{path.name} done",
              flush=True)

    if not raw_rows:
        sys.exit("[classification_accuracy] No instances processed -- aborting.")

    raw_path = out_root / f"accuracy_raw_{args.arch}.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "level", "identity", "scan",
                                          "true_class", "pred_class", "correct"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"[classification_accuracy] Saved {raw_path}", flush=True)

    groups = defaultdict(list)
    for r in raw_rows:
        groups[(r["condition"], r["level"])].append(r["correct"])
    summary_rows = []
    for (cond, level), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        summary_rows.append({"condition": cond, "level": level, "n": len(vals),
                              "accuracy": round(sum(vals) / len(vals), 6)})
    summary_path = out_root / f"accuracy_summary_{args.arch}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "level", "n", "accuracy"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[classification_accuracy] Saved {summary_path}", flush=True)
    for r in summary_rows:
        print(f"    {r['condition']:10s} level={str(r['level']):6s}  "
              f"acc={r['accuracy']:.4f}  n={r['n']}", flush=True)


if __name__ == "__main__":
    main()
