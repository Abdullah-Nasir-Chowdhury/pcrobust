"""
confirmatory.py — Confirmatory data collection for the ModelNet40
pose-robustness study: per-instance embedding self-similarity at each
angle, for the pre-registered sample size from pilot + power.

Confirmatory sample: all raw instances in data/train + data/test for the 10
chosen classes, EXCLUDING the instances already spent on the pilot,
discovered dynamically via pcrobust.datasets.modelnet40.confirmatory_pool().

Output: results/confirmatory_raw_<arch>.csv
        one row per (instance, angle), columns: identity, scan, angle_deg,
        similarity, replicate_noise_sd
"""

import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F


from pcrobust.datasets import results_dir

_RESULTS = results_dir("pose", "modelnet40")

from pcrobust.datasets import checkpoint_file
from pcrobust.datasets.modelnet40 import confirmatory_pool, extract_object_pts
from pcrobust.models.loading import load_model
from pcrobust.utils import _sample

ANGLES = [20, 40, 60, 80, 120]
K_REPLICATES = 5
N_POINTS = 1024

CONFIRMATORY_SCANS = confirmatory_pool()


def rotate_yaw(pts: np.ndarray, deg: float) -> np.ndarray:
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ R.T


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
    parser.add_argument("--arch", default="pointnetpp",
                        choices=["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv", "pointmlp", "pointtransformer"])
    args = parser.parse_args(argv)
    np.random.seed(42)
    torch.manual_seed(42)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_file(args.arch, "modelnet40")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[confirmatory] arch: {args.arch}  device: {device}")
    print(f"[confirmatory] N = {len(CONFIRMATORY_SCANS)} instances")
    model, _ = load_model(str(ckpt), device)
    model.eval()

    raw_rows = []

    for i, (identity, path) in enumerate(CONFIRMATORY_SCANS, 1):
        print(f"\n[confirmatory] ({i}/{len(CONFIRMATORY_SCANS)}) {identity}/{path.name}")
        try:
            canonical_pts = extract_object_pts(str(path))
        except Exception as e:
            print(f"[confirmatory] EXTRACTION FAILED for {path.name}: {e} -- SKIPPING")
            continue

        canonical_mean = mean_embedding(embed_replicates(model, canonical_pts, device))

        for theta in ANGLES:
            rotated_pts = rotate_yaw(canonical_pts, theta)

            rep_embeds = embed_replicates(model, rotated_pts, device)
            rot_mean = mean_embedding(rep_embeds)
            sim_mean = F.cosine_similarity(rot_mean, canonical_mean).item()

            rep_sims = [F.cosine_similarity(e, canonical_mean).item() for e in rep_embeds]
            rep_sim_std = float(np.std(rep_sims, ddof=1))

            print(f"    theta={theta:4d} deg   sim={sim_mean:.4f}   "
                  f"(replicate noise sd={rep_sim_std:.5f})")

            raw_rows.append({
                "identity": identity,
                "scan": path.stem,
                "angle_deg": theta,
                "similarity": round(sim_mean, 6),
                "replicate_noise_sd": round(rep_sim_std, 6),
            })

    if not raw_rows:
        sys.exit("[confirmatory] No instances processed -- aborting.")

    out_path = _RESULTS / f"confirmatory_raw_{args.arch}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["identity", "scan", "angle_deg",
                                           "similarity", "replicate_noise_sd"])
        w.writeheader()
        w.writerows(raw_rows)

    n_scans = len(set(r["scan"] for r in raw_rows))
    print(f"\n[confirmatory] Saved {out_path}")
    print(f"[confirmatory] {n_scans} instances x {len(ANGLES)} angles = {len(raw_rows)} rows")


if __name__ == "__main__":
    main()
