"""FR1: TOST equivalence test for the headline two-family tie.

Paired bootstrap on the difference in mean corrupted accuracy between the
top models, from per-instance ModelNet40-C raws. Shared instance resamples
(2,000 draws), 90% percentile CI; equivalence at alpha=0.05 iff the CI
lies inside +/-0.02 (margin = seed-level sd of the top models' corrupted
accuracy: pointnetpp_inv 0.020 is the larger of the pair).

Pairs:
  riconvpp_inv vs pointmlp_inv   (rep1 -- the Table 2/6 headline tie)
  riconvpp_inv vs pointnetpp_inv (reps 1,2,3 -- the across-seed tie)

Reads raw_<arch>_invreg_rep<k>.csv from --data (default: where
pcrobust.evaluate.modelnet40c writes, <PCROBUST_HOME>/results/modelnet40c).

Usage: python -m pcrobust.stats.tost [--data DIR]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from pcrobust.datasets import results_dir

DATA = None  # set by main()
N_DRAWS = 2000
MARGIN = 0.02
SEED = 0


def per_instance_corrupted(arch, rep):
    hits = np.zeros(1000)
    cells = np.zeros(1000, dtype=int)
    with open(Path(DATA) / f"raw_{arch}_invreg_rep{rep}.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["corruption"] == "none":
                continue
            i = int(row["instance_idx"])
            hits[i] += int(row["correct"])
            cells[i] += 1
    assert cells.min() == cells.max() == 75, f"{arch} rep{rep}: unbalanced cells"
    return hits / cells


def tost(a, b, idx):
    diff_point = a.mean() - b.mean()
    draws = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [5.0, 95.0])
    equivalent = (lo > -MARGIN) and (hi < MARGIN)
    return diff_point, lo, hi, equivalent


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(results_dir("modelnet40c")),
                    help="Directory holding raw_<arch>_invreg_rep<k>.csv.")
    DATA = ap.parse_args(argv).data
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 1000, size=(N_DRAWS, 1000))

    pairs = [("riconvpp", "pointmlp", [1])] + [
        ("riconvpp", "pointnetpp", [r]) for r in (1, 2, 3)
    ]
    print(f"margin +/-{MARGIN}, {N_DRAWS} shared draws, seed {SEED}, 90% percentile CI")
    print()
    for a_name, b_name, reps in pairs:
        for rep in reps:
            a = per_instance_corrupted(a_name, rep)
            b = per_instance_corrupted(b_name, rep)
            d, lo, hi, eq = tost(a, b, idx)
            verdict = "EQUIVALENT" if eq else "not shown equivalent"
            print(
                f"{a_name}_inv vs {b_name}_inv rep{rep}: "
                f"corr acc {a.mean():.4f} vs {b.mean():.4f}, "
                f"diff {d:+.4f}, 90% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}"
            )


if __name__ == "__main__":
    main()
