"""Pose/noise/occlusion robustness check via the embedding-similarity
protocol -- the same instance pool, perturbation severities
(ANGLES/SIGMAS/LEVELS), and 5-replicate embedding-averaging protocol used
throughout the benchmark, applied to any checkpoint whose architecture
resolves inside ``pcrobust.models.REGISTRY`` (baseline and
invariance-trained checkpoints alike: the checkpoint's own ``arch`` field
decides what gets built, via :func:`pcrobust.models.loading.load_model`).

Usage:
  python -m pcrobust.evaluate.robustness --arch pointnet --perturbation pose --pool confirmatory --axis yaw
  python -m pcrobust.evaluate.robustness --arch pointnet --perturbation pose --pool confirmatory --axis tilt
  python -m pcrobust.evaluate.robustness --arch pointnet --perturbation pose --pool confirmatory --axis roll
  python -m pcrobust.evaluate.robustness --arch pointnet --perturbation pose --pool confirmatory --axis extreme
  python -m pcrobust.evaluate.robustness --arch pointnet_invreg --perturbation noise --pool confirmatory
  python -m pcrobust.evaluate.robustness --arch pointnet_invreg --perturbation occlusion --pool pilot

`--axis` only applies to `--perturbation pose` (default `yaw`, the
original/default rotation protocol): `tilt` = pitch (X axis), `roll` = Z
axis, `extreme` = yaw+tilt+roll compounded simultaneously at the same
magnitude. Same `ANGLES`, same `D = Sim(80°) - Sim(120°)`, for all 4 axes
-- only the rotation function differs, so all 4 stay directly comparable.

Output (default root <PCROBUST_HOME>/results/robustness/<domain>/, or --out_dir):
        `yaw` -> <root>/pose_raw_<arch>.csv; `tilt`/`roll`/`extreme`
        -> <root>/<axis>/pose_raw_<arch>.csv. `noise`/`occlusion`
        -> <root>/<perturbation>_raw_<arch>.csv. Same columns as the benchmark's
        confirmatory_raw_<arch>.csv / noise_raw_<arch>.csv /
        occlusion_raw_<arch>.csv, plus a mean_D / sd_D summary line
        printed to stdout (D = Sim(mild) - Sim(severe)).
"""

import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from pcrobust.corruptions.occlusion import LEVELS, MILD, SEVERE, occlude, random_direction
from pcrobust.datasets import checkpoint_file, domain_module
from pcrobust.datasets import results_dir as _results_dir
from pcrobust.models.loading import base_arch, load_model
from pcrobust.utils import _sample

_rng = np.random.default_rng(42)

# Baseline and invariance-trained checkpoints go through the same pool,
# severities, and D protocol, so their drift numbers are directly
# comparable. Pools and canonical extraction are domain-specific and are
# looked up once --domain is known (pcrobust.datasets.domain_module).

ANGLES = [20, 40, 60, 80, 120]
SIGMAS = [0.005, 0.01, 0.02, 0.03, 0.05]

K_REPLICATES = 5
N_POINTS = 1024

MILD_SEVERE = {
    "pose": (80, 120),
    "noise": (0.005, 0.05),
    "occlusion": (MILD, SEVERE),
}


def rotate_yaw(pts: np.ndarray, deg: float) -> np.ndarray:
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ R.T


def rotate_tilt(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotation about the tilt (pitch, X axis) -- matches the
    benchmark's tilt study exactly."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return pts @ R.T


def rotate_roll(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotation about the roll (Z axis) -- matches the
    benchmark's roll study exactly."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return pts @ R.T


def rotate_extreme(pts: np.ndarray, deg: float) -> np.ndarray:
    """Compound yaw+tilt+roll rotation at the same magnitude -- matches the
    benchmark's extreme study exactly."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    R_tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    R_roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    R = R_yaw @ R_tilt @ R_roll
    return pts @ R.T


ROTATE_FN = {"yaw": rotate_yaw, "tilt": rotate_tilt, "roll": rotate_roll,
             "extreme": rotate_extreme}


def add_noise(pts: np.ndarray, sigma: float) -> np.ndarray:
    return pts + np.random.randn(*pts.shape).astype(np.float32) * sigma


def embed_replicates(model, pts: np.ndarray, device, k: int = K_REPLICATES):
    embeds = []
    for _ in range(k):
        p = _sample(pts.copy(), N_POINTS)
        x = torch.from_numpy(p).unsqueeze(0).transpose(1, 2).to(device)
        with torch.no_grad():
            embeds.append(model.get_embedding(x))
    return embeds


def mean_embedding(embeds):
    return F.normalize(torch.stack(embeds).mean(0), dim=1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True,
                         help="Checkpoint key: an architecture (pointnet) or an "
                              "architecture plus recipe suffix (pointnet_invreg).")
    parser.add_argument("--perturbation", required=True,
                         choices=["pose", "noise", "occlusion"])
    parser.add_argument("--pool", default="confirmatory", choices=["pilot", "confirmatory"])
    parser.add_argument("--axis", default="yaw", choices=["yaw", "tilt", "roll", "extreme"],
                        help="Rotation axis, only used when --perturbation pose.")
    parser.add_argument("--models_dir", default=None,
                         help="Directory holding <arch>.pth (default: the zoo cache "
                              "location for --domain).")
    parser.add_argument("--out_dir", default=None,
                         help="Output root (default: <PCROBUST_HOME>/results/robustness/<domain>).")
    parser.add_argument("--out_suffix", default="",
                         help="Appended to the output CSV filename so a re-run against a "
                              "different --models_dir doesn't overwrite a prior result.")
    parser.add_argument("--domain", default="modelnet40",
                         choices=["modelnet40", "shapenet", "abo", "s3dis", "scanobjectnn"],
                         help="Benchmark domain: selects the instance pools and test split.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    base_arch(args.arch)  # fail fast on an unknown key
    _t_start = time.time()

    global _rng
    np.random.seed(42)
    torch.manual_seed(42)
    _rng = np.random.default_rng(42)

    dom = domain_module(args.domain)
    extract_object_pts = dom.extract_object_pts
    confirmatory_pool, pilot_scans = dom.confirmatory_pool, dom.pilot_scans
    out_root = Path(args.out_dir) if args.out_dir else _results_dir("robustness", args.domain)
    results_base = out_root
    results_base.mkdir(parents=True, exist_ok=True)

    ckpt = (Path(args.models_dir) / f"{args.arch}.pth" if args.models_dir
            else checkpoint_file(args.arch, args.domain))
    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_robustness] arch={args.arch} domain={args.domain} perturbation={args.perturbation} "
          f"pool={args.pool}"
          + (f" axis={args.axis}" if args.perturbation == "pose" else "")
          + f" device={device}", flush=True)

    scans = confirmatory_pool() if args.pool == "confirmatory" else pilot_scans()
    print(f"[eval_robustness] N = {len(scans)} instances", flush=True)

    model, _ = load_model(str(ckpt), device)
    model.eval()

    levels = {"pose": ANGLES, "noise": SIGMAS, "occlusion": LEVELS}[args.perturbation]
    level_key = {"pose": "angle_deg", "noise": "sigma", "occlusion": "dist_mult"}[args.perturbation]

    raw_rows = []
    for i, (identity, path) in enumerate(scans, 1):
        print(f"[eval_robustness] ({i}/{len(scans)}) {identity}/{path.name}", flush=True)
        try:
            canonical_pts = extract_object_pts(str(path))
        except Exception as e:
            print(f"[eval_robustness] EXTRACTION FAILED for {path.name}: {e} -- SKIPPING", flush=True)
            continue

        canonical_mean = mean_embedding(embed_replicates(model, canonical_pts, device))

        instance_direction = random_direction(_rng) if args.perturbation == "occlusion" else None

        for level in levels:
            if args.perturbation == "pose":
                perturbed_pts = ROTATE_FN[args.axis](canonical_pts, level)
            elif args.perturbation == "noise":
                perturbed_pts = add_noise(canonical_pts, level)
            else:
                perturbed_pts = occlude(canonical_pts, level, instance_direction)

            rep_embeds = embed_replicates(model, perturbed_pts, device)
            level_mean = mean_embedding(rep_embeds)
            sim_mean = F.cosine_similarity(level_mean, canonical_mean).item()
            rep_sims = [F.cosine_similarity(e, canonical_mean).item() for e in rep_embeds]
            rep_sim_std = float(np.std(rep_sims, ddof=1))

            print(f"    {level_key}={level}   sim={sim_mean:.4f}   "
                  f"(replicate noise sd={rep_sim_std:.5f})", flush=True)

            row = {"identity": identity, "scan": path.stem, level_key: level,
                   "similarity": round(sim_mean, 6), "replicate_noise_sd": round(rep_sim_std, 6)}
            if args.perturbation == "occlusion":
                row["n_visible"] = len(perturbed_pts)
            raw_rows.append(row)

    if not raw_rows:
        sys.exit("[eval_robustness] No instances processed -- aborting.")

    fieldnames = ["identity", "scan", level_key]
    if args.perturbation == "occlusion":
        fieldnames.append("n_visible")
    fieldnames += ["similarity", "replicate_noise_sd"]

    if args.perturbation == "pose" and args.axis != "yaw":
        results_dir = out_root / args.axis
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = results_base
    out_path = results_dir / f"{args.perturbation}_raw_{args.arch}{args.out_suffix}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)
    print(f"[eval_robustness] Saved {out_path}", flush=True)

    mild, severe = MILD_SEVERE[args.perturbation]
    by_scan = {}
    for r in raw_rows:
        by_scan.setdefault(r["scan"], {})[r[level_key]] = r["similarity"]
    Ds = [v[mild] - v[severe] for v in by_scan.values() if mild in v and severe in v]
    if Ds:
        print(f"[eval_robustness] mean_D = {np.mean(Ds):.4f}  sd_D = {np.std(Ds, ddof=1):.4f}  "
              f"n = {len(Ds)}", flush=True)

    elapsed = time.time() - _t_start
    print(f"[eval_robustness] Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
