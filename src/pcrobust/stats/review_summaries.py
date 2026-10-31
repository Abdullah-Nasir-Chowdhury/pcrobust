"""Summaries for two reviewer-requested ModelNet40-C controls.

W4: augmentation-only arm (raw_<arch>_so3aug.csv) next to the replicate-1
baseline and invreg checkpoints; mCE against the replicate-1 DGCNN
baseline. W5: the second 10-class subset (modelnet40b) pair; its mCE is
internally comparable only within the W5 pair.

Reports, per model: clean acc, mean corrupted acc, mCE.

Usage: python -m pcrobust.stats.review_summaries --data_dir DIR --rep1_dir DIR
Output: printed tables + <out_dir>/w4_summary.csv, <out_dir>/w5_summary.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from pcrobust.datasets import results_dir

DATA_DIR = REP1_DIR = None  # set by main()
W4_ARCHS = ["pointnetpp", "pointnet", "dgcnn", "pointmlp", "pointtransformer"]
W5_KEYS = ["pointnetpp_modelnet40b", "pointnetpp_invreg_modelnet40b",
           "riconvpp_modelnet40b", "riconvpp_invreg_modelnet40b"]


def load(path):
    acc = defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        k = (r["corruption"], int(r["severity"]))
        acc[k][0] += int(r["correct"]); acc[k][1] += 1
    return {k: v[0] / v[1] for k, v in acc.items()}


def metrics(a, ref=None):
    clean = a[("none", 0)]
    corr_keys = [k for k in a if k[0] != "none"]
    corr = sum(a[k] for k in corr_keys) / len(corr_keys)
    mce = ""
    if ref:
        groups = defaultdict(list)
        for c, s in corr_keys:
            groups[c].append((c, s))
        ces = []
        for c, ks in groups.items():
            err = sum(1 - a[k] for k in ks)
            rerr = sum(1 - ref[k] for k in ks)
            if rerr > 0:
                ces.append(err / rerr)
        mce = round(sum(ces) / len(ces), 3)
    return round(clean, 3), round(corr, 3), mce


def main(argv=None):
    global DATA_DIR, REP1_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(results_dir("modelnet40c")),
                    help="Directory holding the W4/W5 raw_*.csv files.")
    ap.add_argument("--rep1_dir", default=None,
                    help="Directory holding raw_<arch>_rep1.csv (default: --data_dir).")
    ap.add_argument("--out_dir", default=str(results_dir("review")))
    args = ap.parse_args(argv)
    DATA_DIR = Path(args.data_dir)
    REP1_DIR = Path(args.rep1_dir) if args.rep1_dir else DATA_DIR
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # W4: aug-only arm vs rep1 baseline + rep1 invreg, mCE vs rep1 DGCNN
    ref = load(REP1_DIR / "raw_dgcnn_rep1.csv")
    rows = []
    print("== W4 (aug-only vs baseline vs invreg, rep1 recipes; mCE ref = rep1 DGCNN)")
    for a in W4_ARCHS:
        line = {"arch": a}
        for tag, fname in (("so3aug", f"raw_{a}_so3aug.csv"),
                           ("baseline", f"raw_{a}_rep1.csv"),
                           ("invreg", f"raw_{a}_invreg_rep1.csv")):
            p = (REP1_DIR if "_rep1" in fname else DATA_DIR) / fname
            if p.exists():
                cl, co, mce = metrics(load(p), ref)
                line[f"{tag}_clean"], line[f"{tag}_corr"], line[f"{tag}_mce"] = cl, co, mce
            else:
                line[f"{tag}_clean"] = line[f"{tag}_corr"] = line[f"{tag}_mce"] = ""
        rows.append(line)
        print(f"{a:<18s} " + "  ".join(f"{t}: {line[f'{t}_clean']}/{line[f'{t}_corr']}/{line[f'{t}_mce']}"
                                       for t in ("baseline", "so3aug", "invreg")))
    with open(out / "w4_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # W5: modelnet40b pair (candidate tag spellings tried defensively)
    print("\n== W5 (modelnet40b subset)")
    w5_rows = []
    for k in W5_KEYS:
        for cand in (f"raw_{k}_w5.csv", f"raw_{k}.csv"):
            p = DATA_DIR / cand
            if p.exists():
                a = load(p)
                cl, co, _ = metrics(a)
                w5_rows.append({"key": k, "file": cand, "clean": cl, "corr": co})
                print(f"{k:<34s} clean {cl}  corr {co}   ({cand})")
                break
        else:
            print(f"{k:<34s} NOT AVAILABLE YET")
    if w5_rows:
        with open(out / "w5_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(w5_rows[0])); w.writeheader(); w.writerows(w5_rows)
    print(f"\nwrote {out}/w4_summary.csv" + (" + w5_summary.csv" if w5_rows else ""))


if __name__ == "__main__":
    main()
