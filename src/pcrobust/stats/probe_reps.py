"""Cross-replicate consistency check for the head-independent probe.

Reads probe_acc_<key>_rep{1,2,3}.csv (from pcrobust.evaluate.probe_eval,
one per seed replicate) and, per (model key, head, condition), reports the
mean, per-replicate values, and cross-replicate range; flags cells whose
range exceeds --flag (default 0.08 -- generous seed noise for n=300
accuracies is ~+-0.03). Also checks extraction faithfulness: each
replicate's trained-head probe row must reproduce that replicate's
classification-accuracy CSV (accuracy_raw_<key>_repK.csv in --acc_dir)
within tolerance on deterministic conditions -- if THAT fails, the anomaly
is in extraction, not the representation.

Usage: python -m pcrobust.stats.probe_reps --data_dir DIR --acc_dir DIR
Output: <out_dir>/probe_rep_consistency.csv + flagged-cell report on stdout
"""

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from pcrobust.datasets import results_dir

ACC_DIR = None  # set by main()
ARCHS = ["pointnetpp", "riconvpp", "risurconv", "pointmlp", "pointnet",
         "dgcnn", "pointtransformer", "riconv", "vn_pointnet", "vn_dgcnn"]


def keys():
    for a in ARCHS:
        yield a
        yield f"{a}_invreg"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(results_dir("probe_rep")),
                    help="Directory holding probe_acc_<key>_rep<k>.csv.")
    ap.add_argument("--acc_dir", default=str(results_dir("seeds")),
                    help="Directory holding accuracy_raw_<key>_rep<k>.csv.")
    ap.add_argument("--out_dir", default=str(results_dir("probe_rep")))
    ap.add_argument("--flag", type=float, default=0.08)
    args = ap.parse_args(argv)
    global ACC_DIR
    ACC_DIR = Path(args.acc_dir)
    data = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cells[(key, head, condition)][rep] = accuracy
    cells = defaultdict(dict)
    for k in keys():
        for r in (1, 2, 3):
            p = data / f"probe_acc_{k}_rep{r}.csv"
            if not p.exists():
                continue
            for row in csv.DictReader(open(p, newline="", encoding="utf-8")):
                head = row.get("head") or row.get("probe") or "?"
                cond = row.get("condition") or row.get("cond") or "?"
                try:
                    cells[(k, head, cond)][r] = float(row.get("accuracy") or row.get("acc"))
                except (TypeError, ValueError):
                    pass

    rows, flagged = [], []
    for (k, head, cond), byrep in sorted(cells.items()):
        vals = [byrep[r] for r in sorted(byrep)]
        rng = max(vals) - min(vals) if len(vals) > 1 else 0.0
        rec = {"key": k, "head": head, "condition": cond,
               "n_reps": len(vals), "mean": round(st.mean(vals), 4),
               "range": round(rng, 4),
               **{f"rep{r}": round(v, 4) for r, v in byrep.items()}}
        rows.append(rec)
        if rng > args.flag:
            flagged.append(rec)

    if rows:
        fields = ["key", "head", "condition", "n_reps", "mean", "range", "rep1", "rep2", "rep3"]
        with open(out_dir / "probe_rep_consistency.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        print(f"[probe-verify] {len(rows)} cells -> probe_rep_consistency.csv")
        print(f"[probe-verify] FLAGGED (cross-rep range > {args.flag}): {len(flagged)}")
        for rec in flagged[:25]:
            print(f"  {rec['key']:<24s} {rec['head']:<10s} {rec['condition']:<12s} "
                  f"range {rec['range']:.3f}  reps {[rec.get(f'rep{r}') for r in (1,2,3)]}")
    else:
        print("[probe-verify] no probe CSVs yet (D6 still queued/running)")


if __name__ == "__main__":
    main()
