"""
Fine-grained rotation sweep for Sec. 7 of the paper
("How Much Rotation Does the Loss Buy?"). For each of the 8 baseline
architectures and their invariance-trained (invreg) counterparts
(16 ModelNet40 models total), rotates every one of the 300
held-out test instances in 10-degree steps through the full 360 degrees,
about four axes -- yaw, tilt, roll (identical rotation matrices to
pcrobust.evaluate.robustness's ROTATE_FN) and a fourth 'so3' axis: one random unit
axis per instance (same uniform-sphere sampling as
pcrobust.corruptions.rotation.random_rotation), swept through the same
angle range via the identical Rodrigues formula that function uses,
standing in for the paper's "free SO(3) rotation" curve.

Per (instance, axis, angle) records BOTH classification accuracy and
embedding drift angle theta = arccos(cos_sim(emb(rotated), emb(canonical)))
from a single forward pass (every architecture here returns a uniform
(logits, embedding, extra) tuple -- see e.g. archs/pointnet.py's
PointNetEmbedder.forward), rather than calling predict()/embed_replicates()
separately the way classification_accuracy.py/eval_robustness.py do.

Deliberately ONE deterministic pass per cell, not the 5-replicate
softmax/embedding-averaging protocol those two scripts use: this is a
descriptive sweep feeding one figure and one crossover-angle number, no
hypothesis test or CI is claimed for it in the paper, and 36 angles x 4
axes x 16 models is already roughly two orders of magnitude more forward
passes than a single-axis 5-angle confirmatory run.

Uses the same 300-instance ModelNet40 test split as
pcrobust.evaluate.accuracy, so results are directly comparable to the
paper's already-quoted "known anchors" (e.g. yaw-120 accuracy figures).

Usage:
  python -m pcrobust.evaluate.rotation_sweep
  python -m pcrobust.evaluate.rotation_sweep --out_dir /tmp/pilot --archs pointnet --limit_instances 20 --angles 0 60 120 180
"""
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from pcrobust.datasets import checkpoint_file, results_dir
from pcrobust.datasets.modelnet40 import TEST_DIR, extract_object_pts
from pcrobust.models.loading import load_model
from pcrobust.utils import _sample

ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp",
         "risurconv", "pointmlp", "pointtransformer"]
ANGLES = list(range(0, 360, 10))          # 36 steps, matches Sec. 7's "10-degree steps"
N_POINTS = 1024



def rotate_yaw(pts, deg):
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ R.T


def rotate_tilt(pts, deg):
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return pts @ R.T


def rotate_roll(pts, deg):
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return pts @ R.T


def rotate_so3(pts, deg, axis):
    """Rodrigues' rotation formula about a fixed unit `axis`, matching
    pcrobust.corruptions.rotation.random_rotation's own K-matrix
    construction exactly (just parameterized by a specific angle/axis
    instead of drawing both randomly each call)."""
    t = deg * np.pi / 180.0
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    # np.sin/np.cos on a Python float return a numpy float64 SCALAR, which
    # (unlike a plain Python float) promotes a float32 array to float64 under
    # NEP 50 -- force back to float32 explicitly rather than let every
    # downstream forward pass silently upcast to double.
    R = (np.eye(3, dtype=np.float32) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)).astype(np.float32)
    return pts @ R.T


def random_axis(rng):
    """Uniform-sphere unit axis, same construction as
    pcrobust.corruptions.rotation.random_rotation."""
    theta = rng.uniform(0, 2 * np.pi)
    phi = np.arccos(2 * rng.uniform(0, 1) - 1)
    return np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
                     dtype=np.float32)


def forward_one(model, pts, device):
    p = _sample(pts.copy(), N_POINTS)
    x = torch.from_numpy(p).unsqueeze(0).transpose(1, 2).to(device)
    with torch.no_grad():
        logits, emb, _ = model(x)
    return logits, emb


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None,
                        help="Default: <PCROBUST_HOME>/results/rotation_sweep.")
    parser.add_argument("--models_dir", default=None,
                        help="Directory holding <key>.pth (default: the zoo cache "
                             "location for modelnet40).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--archs", nargs="*", default=ARCHS)
    parser.add_argument("--angles", nargs="*", type=int, default=None)
    parser.add_argument("--limit_instances", type=int, default=None,
                         help="Debug/timing only -- truncates the 300-instance test set.")
    parser.add_argument("--conditions", nargs="*", default=["baseline", "invreg"])
    args = parser.parse_args(argv)
    angles = args.angles if args.angles is not None else ANGLES
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[rotation_sweep] device={device} archs={args.archs} n_angles={len(angles)}", flush=True)

    test_dir = TEST_DIR
    by_class = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        by_class.append([(class_dir.name, f) for f in sorted(class_dir.glob("*.ply"))])
    if args.limit_instances:
        # Class-balanced subsample -- files are grouped by class (30/class),
        # so a naive files[:N] prefix would grab only the first 1-2 classes
        # and silently produce a single-class accuracy number. Take an even
        # slice per class instead (round-robin) so every class is represented.
        n_classes = len(by_class)
        per_class = max(1, args.limit_instances // n_classes)
        files = []
        for cls_files in by_class:
            files.extend(cls_files[:per_class])
        files = files[:args.limit_instances] if len(files) > args.limit_instances else files
    else:
        files = [f for cls_files in by_class for f in cls_files]
    print(f"[rotation_sweep] N = {len(files)} test instances "
          f"({len(set(c for c, _ in files))} classes)", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else results_dir("rotation_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "rotation_sweep_raw.csv"
    fieldnames = ["arch", "condition", "axis", "angle_deg", "identity", "scan",
                  "true_class", "pred_class", "correct", "cos_sim", "drift_deg"]
    raw_f = open(raw_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(raw_f, fieldnames=fieldnames)
    writer.writeheader()

    t_start = time.time()
    for arch in args.archs:
        for condition in args.conditions:
            key = arch if condition == "baseline" else f"{arch}_invreg" if condition == "invreg" \
                else f"{arch}_invreg_{condition}"
            ckpt_path = (Path(args.models_dir) / f"{key}.pth" if args.models_dir
                         else checkpoint_file(key, "modelnet40"))
            t_model = time.time()
            model, classes = load_model(str(ckpt_path), device)
            model.eval()

            # Re-seeded per (arch, condition) so every model/condition sees the
            # SAME per-instance so3 axes -- curves stay comparable across the
            # 16-model grid rather than each drawing its own random geometry.
            axis_rng = np.random.default_rng(42)

            n_rows = 0
            for identity, path in files:
                canonical_pts = extract_object_pts(str(path))
                so3_axis = random_axis(axis_rng)

                canonical_logits, canonical_emb = forward_one(model, canonical_pts, device)
                canonical_emb = F.normalize(canonical_emb, dim=1)

                for axis_name, rotate_fn in [
                    ("yaw", rotate_yaw), ("tilt", rotate_tilt), ("roll", rotate_roll),
                    ("so3", lambda p, d: rotate_so3(p, d, so3_axis)),
                ]:
                    for angle in angles:
                        rotated = rotate_fn(canonical_pts, angle)
                        logits, emb = forward_one(model, rotated, device)
                        emb = F.normalize(emb, dim=1)
                        pred = classes[logits.argmax(1).item()]
                        cos_sim = float(np.clip(F.cosine_similarity(emb, canonical_emb).item(), -1.0, 1.0))
                        drift_deg = float(np.degrees(np.arccos(cos_sim)))
                        writer.writerow({
                            "arch": arch, "condition": condition, "axis": axis_name,
                            "angle_deg": angle, "identity": identity, "scan": path.stem,
                            "true_class": identity, "pred_class": pred,
                            "correct": int(pred == identity),
                            "cos_sim": round(cos_sim, 6), "drift_deg": round(drift_deg, 4),
                        })
                        n_rows += 1
            raw_f.flush()
            elapsed = time.time() - t_model
            print(f"[rotation_sweep] {arch}/{condition}: {n_rows} rows in "
                  f"{elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    raw_f.close()
    total_elapsed = time.time() - t_start
    print(f"[rotation_sweep] ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min). "
          f"Saved {raw_path}", flush=True)

    # -- Summary aggregation --
    groups = defaultdict(lambda: {"n": 0, "correct": 0, "drifts": []})
    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["arch"], row["condition"], row["axis"], int(row["angle_deg"]))
            g = groups[key]
            g["n"] += 1
            g["correct"] += int(row["correct"])
            g["drifts"].append(float(row["drift_deg"]))

    summary_path = out_dir / "rotation_sweep_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "condition", "axis", "angle_deg", "n",
                                          "accuracy", "mean_drift_deg", "sd_drift_deg"])
        w.writeheader()
        for (arch, condition, axis, angle), g in sorted(groups.items()):
            drifts = np.array(g["drifts"])
            w.writerow({
                "arch": arch, "condition": condition, "axis": axis, "angle_deg": angle,
                "n": g["n"], "accuracy": round(g["correct"] / g["n"], 6),
                "mean_drift_deg": round(float(drifts.mean()), 4),
                "sd_drift_deg": round(float(drifts.std(ddof=1)) if len(drifts) > 1 else 0.0, 4),
            })
    print(f"[rotation_sweep] Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
