"""
Evaluation-set bootstrap CIs for the ModelNet40-C
benchmark (review priority #4: Sec. 6 currently reports point estimates).

Resamples the ~1000 test instances with replacement, B draws, with the SAME
resample applied to every model (paired), and recomputes clean accuracy,
mean corruption accuracy, per-corruption CE, and mCE per draw. Reports
percentile 95% CIs for each model and for the paired invreg-baseline
deltas. mCE follows the standard protocol: CE_c = sum_s err_c,s(model) /
sum_s err_c,s(reference), reference = plain DGCNN, mCE = mean_c CE_c; the
reference errors are resampled with the same draw, so the CE ratio's
uncertainty is honest about both numerator and denominator.

This quantifies EVALUATION uncertainty (finite test set) only -- training
-seed uncertainty is the seed study's job; the two are complementary and
both go in the paper. When campaign replicate raws land, point --data at
them via --suffix (e.g. --suffix _rep1 reads raw_<key>_rep1.csv).

Usage: python -m pcrobust.stats.bootstrap   (B=2000)
Output: <out_dir>/bootstrap_summary<suffix>.csv + printed table
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from pcrobust.datasets import results_dir

ARCHS = ["pointnet", "pointnetpp", "dgcnn", "pointmlp", "pointtransformer",
         "riconv", "riconvpp", "risurconv", "vn_pointnet", "vn_dgcnn"]
REFERENCE = "dgcnn"  # plain DGCNN baseline, as in the paper's mCE


def load_matrix(path):
    """raw_<key>.csv -> (conditions list, correct matrix [n_cond, n_inst])."""
    by_cond = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["corruption"], int(row["severity"]))
            by_cond.setdefault(key, {})[int(row["instance_idx"])] = int(row["correct"])
    conds = sorted(by_cond)
    n_inst = {len(v) for v in by_cond.values()}
    if len(n_inst) != 1:
        raise ValueError(f"{path}: unequal instance counts per condition {n_inst}")
    n = n_inst.pop()
    mat = np.empty((len(conds), n), dtype=np.float64)
    for i, c in enumerate(conds):
        d = by_cond[c]
        mat[i] = [d[j] for j in sorted(d)]
    return conds, mat


def metrics(mat, ref_mat, conds, counts):
    """Per-draw clean acc, corruption acc, mCE. counts: [B, n] resample weights."""
    n = mat.shape[1]
    acc = mat @ counts.T / n            # [n_cond, B]
    ref_acc = ref_mat @ counts.T / n
    clean_i = [i for i, (c, s) in enumerate(conds) if c == "none"]
    corr_i = [i for i, (c, s) in enumerate(conds) if c != "none"]
    clean = acc[clean_i].mean(axis=0)
    corr = acc[corr_i].mean(axis=0)
    # mCE: per corruption, err summed over severities, ratio vs reference
    groups = {}
    for i in corr_i:
        groups.setdefault(conds[i][0], []).append(i)
    ces = []
    for c, idxs in sorted(groups.items()):
        err = (1.0 - acc[idxs]).sum(axis=0)
        ref_err = (1.0 - ref_acc[idxs]).sum(axis=0)
        ces.append(err / ref_err)
    mce = np.mean(ces, axis=0)
    return clean, corr, mce


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(results_dir("modelnet40c")),
                    help="Directory holding raw_<key><suffix>.csv.")
    ap.add_argument("--out_dir", default=str(results_dir("bootstrap")))
    ap.add_argument("--suffix", default="",
                    help="raw_<key><suffix>.csv, e.g. _rep1 for campaign replicates")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    data = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = []
    for a in ARCHS:
        for k in (a, f"{a}_invreg"):
            if (data / f"raw_{k}{args.suffix}.csv").exists():
                keys.append(k)
    print(f"[bootstrap] {len(keys)} models, B={args.B}", flush=True)

    conds = None
    mats = {}
    for k in keys:
        c, m = load_matrix(data / f"raw_{k}{args.suffix}.csv")
        if conds is None:
            conds = c
        elif c != conds:
            raise ValueError(f"{k}: condition grid mismatch")
        mats[k] = m

    if REFERENCE not in mats:
        # partial replicate: reference model not retrained/evaluated yet --
        # fall back to the published reference so CE ratios stay computable
        c, m = load_matrix(data / f"raw_{REFERENCE}.csv")
        if c != conds:
            raise ValueError("published reference: condition grid mismatch")
        mats[REFERENCE] = m
        print(f"[bootstrap] WARNING: no raw_{REFERENCE}{args.suffix}.csv -- "
              f"mCE referenced to PUBLISHED {REFERENCE} (mixed-provenance, "
              f"interim only)", flush=True)

    n = mats[keys[0]].shape[1]
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.B, n))
    counts = np.zeros((args.B, n))
    for b in range(args.B):
        counts[b] = np.bincount(idx[b], minlength=n)

    ref = mats[REFERENCE]
    ident = np.ones((1, n))  # point estimate = draw with uniform weights
    rows = []
    per_model = {}
    for k in keys:
        cl, co, mce = metrics(mats[k], ref, conds, counts)
        cl0, co0, mce0 = metrics(mats[k], ref, conds, ident)
        per_model[k] = (cl, co, mce)
        rows.append({"model": k, "metric_set": "absolute",
                     "clean": f"{cl0[0]:.4f}", "clean_lo": f"{ci(cl)[0]:.4f}", "clean_hi": f"{ci(cl)[1]:.4f}",
                     "corr_acc": f"{co0[0]:.4f}", "corr_lo": f"{ci(co)[0]:.4f}", "corr_hi": f"{ci(co)[1]:.4f}",
                     "mce": f"{mce0[0]:.4f}", "mce_lo": f"{ci(mce)[0]:.4f}", "mce_hi": f"{ci(mce)[1]:.4f}"})
        print(f"{k:<24s} clean {cl0[0]:.3f} [{ci(cl)[0]:.3f},{ci(cl)[1]:.3f}]  "
              f"corr {co0[0]:.3f} [{ci(co)[0]:.3f},{ci(co)[1]:.3f}]  "
              f"mCE {mce0[0]:.3f} [{ci(mce)[0]:.3f},{ci(mce)[1]:.3f}]", flush=True)

    print("\npaired deltas (invreg - baseline):", flush=True)
    for a in ARCHS:
        if a in per_model and f"{a}_invreg" in per_model:
            d_co = per_model[f"{a}_invreg"][1] - per_model[a][1]
            d_mce = per_model[f"{a}_invreg"][2] - per_model[a][2]
            lo, hi = ci(d_co)
            mlo, mhi = ci(d_mce)
            sig = "*" if lo > 0 or hi < 0 else " "
            rows.append({"model": a, "metric_set": "delta_invreg",
                         "clean": "", "clean_lo": "", "clean_hi": "",
                         "corr_acc": f"{d_co.mean():.4f}", "corr_lo": f"{lo:.4f}", "corr_hi": f"{hi:.4f}",
                         "mce": f"{d_mce.mean():.4f}", "mce_lo": f"{mlo:.4f}", "mce_hi": f"{mhi:.4f}"})
            print(f"{a:<24s} d_corr {d_co.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{sig}  "
                  f"d_mCE {d_mce.mean():+.3f} [{mlo:+.3f},{mhi:+.3f}]", flush=True)

    out = out_dir / f"bootstrap_summary{args.suffix}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
