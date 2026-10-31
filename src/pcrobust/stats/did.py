"""
Architecture-family x training interaction test. Asks the question behind "fragility is trainable":
is the accuracy improvement from the invariance loss significantly
DIFFERENT between standard and rotation-invariant families, per
perturbation type?

Difference-in-differences on trained-head accuracy (16-condition eval,
300 paired queries):
    delta_a(p)  = acc_invreg(a, p) - acc_baseline(a, p)      per arch a
    DiD(p)      = mean_{a in standard} delta_a - mean_{a in RI} delta_a
for p in {clean, pose, noise, occlusion} (levels pooled).

Inference, two complementary layers:
 1. Paired instance bootstrap (B draws, same resample across all models):
    CI for DiD(p) -- uncertainty from the finite evaluation set.
 2. Exact family-label permutation: reassign the 8 backbones to a
    5-vs-3 split in all C(8,3)=56 ways; p-value = share of assignments
    with |DiD| >= observed -- does the FAMILY line matter beyond
    arch-to-arch variation? (Small-n honest: with 5+3 archs this is the
    strongest arch-level claim the data supports.)

VN-PointNet is equivariant, NOT invariant -- excluded from both families;
its delta is reported as a separate row. When campaign replicates land,
run per rep (--suffix _rep1 ...) and report the DiD spread across seeds.

Usage: python -m pcrobust.stats.did
Output: <out_dir>/did_summary<suffix>.csv + printed table
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from pcrobust.datasets import results_dir

STANDARD = ["pointnet", "pointnetpp", "dgcnn", "pointmlp", "pointtransformer"]
RI = ["riconv", "riconvpp", "risurconv"]
EQUIVARIANT = ["vn_pointnet", "vn_dgcnn"]  # reported, never pooled into a family
PERTS = ["clean", "pose", "noise", "occlusion"]


def load(path):
    """accuracy raw -> {pert: correct matrix [n_rows_for_pert]} keyed by scan order."""
    per_pert = {p: {} for p in PERTS}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_pert[row["condition"]].setdefault(row["scan"], []).append(int(row["correct"]))
    out = {}
    scans = None
    for p, d in per_pert.items():
        if scans is None:
            scans = sorted(d)
        # per scan, mean over the levels of this perturbation (1 for clean, 5 else)
        out[p] = np.array([np.mean(d[s]) for s in scans], dtype=np.float64)
    return scans, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(results_dir("classification_accuracy", "modelnet40")),
                    help="Directory holding accuracy_raw_<arch>[_invreg]<suffix>.csv.")
    ap.add_argument("--out_dir", default=str(results_dir("did")))
    ap.add_argument("--suffix", default="",
                    help="campaign tag: baseline accuracy_raw_<a><suffix>.csv, "
                         "invreg accuracy_raw_<a>_invreg<suffix>.csv")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="backbones to drop (sensitivity runs, e.g. risurconv "
                         "to see how much the DiD leans on its collapse)")
    args = ap.parse_args(argv)
    data = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def base_file(a):
        return data / f"accuracy_raw_{a}{args.suffix}.csv"

    def inv_file(a):
        # published invreg raws carry the eval-protocol tag "_baseline"
        tagged = data / f"accuracy_raw_{a}_invreg{args.suffix}.csv"
        return tagged if tagged.exists() else data / f"accuracy_raw_{a}_invreg_baseline.csv"

    archs = [a for a in STANDARD + RI + EQUIVARIANT
             if a not in args.exclude
             and base_file(a).exists() and inv_file(a).exists()]
    std = [a for a in archs if a in STANDARD]
    ri = [a for a in archs if a in RI]
    print(f"[did] {len(archs)} backbones: standard={std} RI={ri} "
          f"equivariant={[a for a in archs if a in EQUIVARIANT]}", flush=True)

    scans = None
    delta = {}  # arch -> {pert: per-instance delta vector [n]}
    for a in archs:
        s_b, base = load(base_file(a))
        s_i, inv = load(inv_file(a))
        if s_b != s_i:
            raise ValueError(f"{a}: scan sets differ between arms")
        if scans is None:
            scans = s_b
        elif s_b != scans:
            raise ValueError(f"{a}: scan set differs from other archs")
        delta[a] = {p: inv[p] - base[p] for p in PERTS}
    n = len(scans)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.B, n))
    counts = np.zeros((args.B, n))
    for b in range(args.B):
        counts[b] = np.bincount(idx[b], minlength=n)

    rows = []
    print(f"\nper-arch delta (invreg - baseline), n={n} paired queries:", flush=True)
    for a in archs:
        fam = "standard" if a in STANDARD else ("RI" if a in RI else "equivariant")
        line = f"{a:<18s} [{fam:<11s}]"
        for p in PERTS:
            d0 = delta[a][p].mean()
            boot = counts @ delta[a][p] / n
            lo, hi = np.percentile(boot, [2.5, 97.5])
            line += f"  {p} {d0:+.3f} [{lo:+.3f},{hi:+.3f}]"
            rows.append({"row": f"delta_{a}", "family": fam, "pert": p,
                         "value": f"{d0:.4f}", "lo": f"{lo:.4f}", "hi": f"{hi:.4f}", "p_perm": ""})
        print(line, flush=True)

    print("\nDiD = mean(standard delta) - mean(RI delta):", flush=True)
    fam_deltas = {p: np.array([delta[a][p].mean() for a in std + ri]) for p in PERTS}
    labels = np.array([1] * len(std) + [0] * len(ri))  # 1 = standard
    splits = list(itertools.combinations(range(len(std) + len(ri)), int(labels.sum())))
    for p in PERTS:
        std_vec = np.mean([delta[a][p] for a in std], axis=0)
        ri_vec = np.mean([delta[a][p] for a in ri], axis=0)
        did_vec = std_vec - ri_vec           # per-instance DiD
        did0 = did_vec.mean()
        boot = counts @ did_vec / n
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # exact permutation over family labels (arch-level inference)
        vals = fam_deltas[p]
        obs = abs(vals[labels == 1].mean() - vals[labels == 0].mean())
        cnt = sum(1 for c in splits
                  if abs(vals[list(c)].mean()
                         - vals[[i for i in range(len(vals)) if i not in c]].mean()) >= obs - 1e-12)
        p_perm = cnt / len(splits)
        sig = "*" if lo > 0 or hi < 0 else " "
        rows.append({"row": "DiD", "family": "standard-RI", "pert": p,
                     "value": f"{did0:.4f}", "lo": f"{lo:.4f}", "hi": f"{hi:.4f}",
                     "p_perm": f"{p_perm:.4f}"})
        print(f"  {p:<10s} DiD {did0:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}  "
              f"perm p={p_perm:.3f} ({len(splits)} splits)", flush=True)

    excl = "".join(f"_no_{a}" for a in args.exclude)
    out = out_dir / f"did_summary{args.suffix}{excl}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row", "family", "pert", "value", "lo", "hi", "p_perm"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
