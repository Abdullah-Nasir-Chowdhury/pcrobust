"""
Turn the seed campaign's tagged drift raws
(noise/occlusion/pose-yaw/tilt/roll, *_rep<k>.csv) into per-cell stats
CSVs in the SAME shape as the published confirmatory pipeline's
<pert>_stats.csv (e.g. noise_stats.csv): one row per model key with n, mean similarity per
severity, mean_degradation (D = sim(mild) - sim(severe)), sd, paired
one-sided t (H0: D <= 0), Cohen's d_z, Wilcoxon p, Friedman across
severities. The finished study's pipeline does not read _rep tags;
this adapter closes that gap so every ModelNet40 table can be
re-derived from replicate data.

Severity pairs (identical to the paper's primary tests):
  noise:      sigma      0.005 (mild) vs 0.05 (severe)
  occlusion:  dist_mult  0.3        vs 6.0
  pose (all axes): angle_deg 80     vs 120

Inputs live in --data (tilt/roll raws use tilt__/roll__ prefixed names).
Missing files are skipped and reported, so the adapter runs cleanly on
partial data.

Usage:
  python -m pcrobust.stats.seed_stats --reps 1 2 3 --data DIR

Output: results/<pert>_stats_rep<k>.csv per perturbation-axis, plus
results/seed_stats_all.csv (long format: one row per key x pert x rep)
for the seed appendix and aggregation.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

from pcrobust.datasets import results_dir

ARCHS = ["pointnetpp", "riconvpp", "risurconv", "pointmlp", "pointnet",
         "dgcnn", "pointtransformer", "riconv", "vn_pointnet", "vn_dgcnn"]

# pert -> (local filename template, remote subpath template, severity column, mild, severe)
PERTS = {
    "noise":     ("noise_raw_{key}_rep{r}.csv",        "results/noise_raw_{key}_rep{r}.csv",        "sigma",     "0.005", "0.05"),
    "occlusion": ("occlusion_raw_{key}_rep{r}.csv",    "results/occlusion_raw_{key}_rep{r}.csv",    "dist_mult", "0.3",   "6.0"),
    "pose_yaw":  ("pose_raw_{key}_rep{r}.csv",         "results/pose_raw_{key}_rep{r}.csv",         "angle_deg", "80",    "120"),
    "pose_tilt": ("tilt__pose_raw_{key}_rep{r}.csv",   "tilt/results/pose_raw_{key}_rep{r}.csv",    "angle_deg", "80",    "120"),
    "pose_roll": ("roll__pose_raw_{key}_rep{r}.csv",   "roll/results/pose_raw_{key}_rep{r}.csv",    "angle_deg", "80",    "120"),
}


def keys():
    for a in ARCHS:
        yield a
        yield f"{a}_invreg"


def load_sims(path, sev_col):
    """raw csv -> {severity: {scan: similarity}}"""
    out = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row[sev_col]][row["scan"]] = float(row["similarity"])
    return out


def stats_row(key, sims, sev_col, mild, severe):
    scans = sorted(set(sims[mild]) & set(sims[severe]))
    d = np.array([sims[mild][s] - sims[severe][s] for s in scans])
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    t = mean / (sd / np.sqrt(n))
    p = sps.t.sf(t, n - 1)  # one-sided, H0: D <= 0
    try:
        wil = sps.wilcoxon(d, alternative="greater").pvalue
    except ValueError:
        wil = float("nan")
    sevs = sorted(sims, key=float)
    common = sorted(set.intersection(*(set(sims[s]) for s in sevs)))
    fr_chi2, fr_p = sps.friedmanchisquare(
        *[[sims[s][sc] for sc in common] for s in sevs]) if len(sevs) >= 3 else (float("nan"),) * 2
    row = {"arch": key, "n": n}
    for s in sevs:
        row[f"mean_sim_{sev_col}{s}"] = np.mean([sims[s][sc] for sc in common])
    row.update({"mean_degradation": mean, "sd_degradation": sd, "t_stat": t,
                "p_one_sided": p, "d_z": mean / sd, "wilcoxon_p": wil,
                "friedman_chi2": fr_chi2, "friedman_p": fr_p})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(results_dir("seeds")))
    ap.add_argument("--reps", nargs="+", type=int, default=[1])
    ap.add_argument("--out_dir", default=str(results_dir("seeds", "stats")))
    args = ap.parse_args(argv)
    data = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows, missing = [], []
    for r in args.reps:
        for pert, (local, _remote, sev_col, mild, severe) in PERTS.items():
            rows = []
            for key in keys():
                path = data / local.format(key=key, r=r)
                if not path.exists():
                    missing.append(f"rep{r}/{pert}/{key}")
                    continue
                row = stats_row(key, load_sims(path, sev_col), sev_col, mild, severe)
                rows.append(row)
                long_rows.append({"rep": r, "pert": pert, **{k: row[k] for k in
                                  ("arch", "n", "mean_degradation", "sd_degradation",
                                   "t_stat", "p_one_sided", "d_z")}})
            if rows:
                out = out_dir / f"{pert}_stats_rep{r}.csv"
                with open(out, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                print(f"[stats] rep{r} {pert:<10s} {len(rows):2d} models -> {out.name}", flush=True)

    if long_rows:
        with open(out_dir / "seed_stats_all.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0]))
            w.writeheader()
            w.writerows(long_rows)
        print(f"[stats] long format -> seed_stats_all.csv ({len(long_rows)} rows)", flush=True)
    if missing:
        print(f"[stats] {len(missing)} cells not yet available "
              f"(campaign still running) -- first few: {missing[:6]}", flush=True)


if __name__ == "__main__":
    main()
