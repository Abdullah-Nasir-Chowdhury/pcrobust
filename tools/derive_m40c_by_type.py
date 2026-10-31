"""Collapse per-sample ModelNet40-C predictions into a table-level CSV.

Dev tool, not part of the installed package. Reads every per-instance
prediction file ``raw_<model>.csv`` (corruption, severity, instance_idx,
true_label, pred_label, correct) in a results directory and writes one
row per (model, corruption, severity) with its accuracy. ``<model>`` keeps
its run suffix: ``pointnet_rep1``, ``pointnet_invreg_warmup``,
``pointnetpp_so3aug``, ``pointnetpp_w5``. Clean accuracy, mean corrupted
accuracy and mCE all follow exactly from these cells, so the per-sample
rows are never shipped.

    python tools/derive_m40c_by_type.py RESULTS_DIR \
        src/pcrobust/paper/data/modelnet40c/m40c_by_type.csv
"""

import sys
from pathlib import Path

import pandas as pd


def main(raw_dir: str, out: str) -> None:
    rows = []
    for path in sorted(Path(raw_dir).glob("raw_*.csv")):
        df = pd.read_csv(path)
        g = df.groupby(["corruption", "severity"])["correct"].agg(["size", "mean"]).reset_index()
        g.insert(0, "model", path.stem.removeprefix("raw_"))
        rows.append(g.rename(columns={"size": "n", "mean": "accuracy"}))
    out_df = pd.concat(rows, ignore_index=True)
    out_df["accuracy"] = out_df["accuracy"].round(6)
    out_df.to_csv(out, index=False)
    print(f"{out_df.model.nunique()} models, {len(out_df)} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
