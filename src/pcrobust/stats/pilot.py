"""
pilot.py — Pilot study for the ModelNet40 replication of the pose-robustness
experiment (the original protocol ran on face scans; here the data are
rigid ModelNet40 instances, so extraction is just canonicalization via
pcrobust.datasets.modelnet40.extract_object_pts()).

Research question
--------------------------------------------------------------------
Does PointNet++ embedding self-similarity (cosine similarity between a
rotated instance's embedding and its own canonical/0deg embedding) degrade
more beyond the trained yaw-augmentation range (+-80deg) than at the edge of
it -- now on rigid ModelNet40 object instances instead of faces?

Pilot sample: one instance per included class (the lowest-numbered instance
in data/train/<class>/), discovered dynamically via pcrobust.datasets.modelnet40.pilot_scans().

IMPORTANT: this pilot's sd_D is used FRESH to
recompute the required sample size (see power.py) and is read at runtime by
analysis.py (via results/pilot_deltasigma.json) -- it is NOT the face
study's hardcoded pilot_sd_D=0.25424. Different dataset, different classes,
different instance geometry -> different variance; reusing the face
constant here would be wrong.

Outputs
-------
results/pilot_raw.csv     -- one row per (instance, angle)
results/pilot_summary.csv -- per-angle mean/sd across the pilot instances,
                              plus the D = Sim(80)-Sim(120) paired-difference
                              stats used to set delta/sigma.
"""

import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F


from pcrobust.datasets import results_dir

_RESULTS = results_dir("pose", "modelnet40")

from pcrobust.datasets import checkpoint_file
from pcrobust.datasets.modelnet40 import extract_object_pts
from pcrobust.datasets.modelnet40 import pilot_scans as _pilot_scans
from pcrobust.models.loading import load_model
from pcrobust.utils import _sample

ANGLES = [20, 40, 60, 80, 120]
K_REPLICATES = 5
N_POINTS = 1024
CKPT = checkpoint_file("pointnetpp", "modelnet40")

PILOT_SCANS = _pilot_scans()


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


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pilot] device: {device}")
    print(f"[pilot] N classes = {len(PILOT_SCANS)}")
    model, _ = load_model(str(CKPT), device)
    model.eval()

    raw_rows = []

    for identity, path in PILOT_SCANS:
        if not path.exists():
            print(f"[pilot] MISSING {path}, skipping")
            continue

        print(f"\n[pilot] === {identity} ({path.name}) ===")
        try:
            canonical_pts = extract_object_pts(str(path))
        except Exception as e:
            print(f"[pilot] EXTRACTION FAILED for {path.name}: {e}")
            continue

        canonical_embeds = embed_replicates(model, canonical_pts, device)
        canonical_mean = mean_embedding(canonical_embeds)

        for theta in ANGLES:
            rotated_pts = rotate_yaw(canonical_pts, theta)
            rot_embeds = mean_embedding(embed_replicates(model, rotated_pts, device))
            sim_mean = F.cosine_similarity(rot_embeds, canonical_mean).item()

            rot_embeds_raw = embed_replicates(model, rotated_pts, device)
            rep_sims = [F.cosine_similarity(e, canonical_mean).item() for e in rot_embeds_raw]
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
        sys.exit("[pilot] No instances processed -- aborting.")

    raw_path = _RESULTS / "pilot_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["identity", "scan", "angle_deg",
                                           "similarity", "replicate_noise_sd"])
        w.writeheader()
        w.writerows(raw_rows)
    print(f"\n[pilot] Saved {raw_path}")

    scans = sorted(set(r["scan"] for r in raw_rows))
    by_scan_angle = {(r["scan"], r["angle_deg"]): r["similarity"] for r in raw_rows}

    summary_rows = []
    for theta in ANGLES:
        vals = [by_scan_angle[(s, theta)] for s in scans if (s, theta) in by_scan_angle]
        summary_rows.append({
            "angle_deg": theta,
            "n": len(vals),
            "mean_similarity": round(float(np.mean(vals)), 6),
            "sd_similarity": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else "",
        })

    D = [by_scan_angle[(s, 80)] - by_scan_angle[(s, 120)]
         for s in scans if (s, 80) in by_scan_angle and (s, 120) in by_scan_angle]

    summary_path = _RESULTS / "pilot_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["angle_deg", "n", "mean_similarity", "sd_similarity"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[pilot] Saved {summary_path}")

    print("\n[pilot] ── Per-angle mean similarity across pilot instances ──")
    for row in summary_rows:
        print(f"    theta={row['angle_deg']:4d} deg   "
              f"mean={row['mean_similarity']:.4f}   sd={row['sd_similarity']}")

    print(f"\n[pilot] ── D_j = Sim_j(80deg) - Sim_j(120deg), n={len(D)} ──")
    for s, d in zip(scans, D):
        print(f"    {s:<12s}  D = {d:+.4f}")
    print(f"    mean(D) = {np.mean(D):+.5f}")
    print(f"    sd(D)   = {np.std(D, ddof=1):.5f}")
    print(f"    (this pilot's sd(D) -- NOT another dataset's -- is what "
          f"power.py/analysis.py use downstream)")


if __name__ == "__main__":
    main()
