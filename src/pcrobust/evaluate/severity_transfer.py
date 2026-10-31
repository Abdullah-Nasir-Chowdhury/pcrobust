"""
"Severity resolves the law" figure. Breaks the narrow-transfer law's two
headroom-matched corruption types (lidar, occlusion) down by
ModelNet40-C severity level (1-5), averaged over all 8 architectures,
to test the prediction that gains grow with severity because headroom
(1 - baseline accuracy) does too.

Reads raw_<arch>.csv / raw_<arch>_invreg.csv as written by
pcrobust.evaluate.modelnet40c (<PCROBUST_HOME>/results/modelnet40c/) and
writes a summary CSV + one figure to
<PCROBUST_HOME>/results/figures/severity_resolved/.

Usage: python -m pcrobust.evaluate.severity_transfer
"""
import csv
from collections import defaultdict

import numpy as np

from pcrobust.datasets import results_dir

RESULTS = results_dir("modelnet40c")
OUT = results_dir("figures", "severity_resolved")


def _pyplot():
    """matplotlib is an optional extra (pip install pcrobust[figures])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp",
         "risurconv", "pointmlp", "pointtransformer"]
CORRUPTIONS = ["lidar", "occlusion"]
SEVERITIES = [1, 2, 3, 4, 5]


def acc_by_severity(path, corruption):
    """Returns {severity: accuracy} for one (arch, condition) raw CSV,
    restricted to one corruption type."""
    buckets = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corruption"] != corruption:
                continue
            buckets[int(row["severity"])].append(int(row["correct"]))
    return {s: float(np.mean(v)) for s, v in buckets.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    per_corruption = {c: {"baseline": defaultdict(list), "invreg": defaultdict(list)}
                       for c in CORRUPTIONS}

    for arch in ARCHS:
        base_path = RESULTS / f"raw_{arch}.csv"
        inv_path = RESULTS / f"raw_{arch}_invreg.csv"
        if not base_path.exists() or not inv_path.exists():
            print(f"[skip] {arch}: missing raw CSV(s)")
            continue
        for corruption in CORRUPTIONS:
            base_acc = acc_by_severity(base_path, corruption)
            inv_acc = acc_by_severity(inv_path, corruption)
            for sev in SEVERITIES:
                if sev not in base_acc or sev not in inv_acc:
                    continue
                b, i = base_acc[sev], inv_acc[sev]
                delta = i - b
                headroom = 1.0 - b
                rows.append([corruption, sev, arch, round(b, 5), round(i, 5),
                             round(delta, 5), round(headroom, 5)])
                per_corruption[corruption]["baseline"][sev].append(b)
                per_corruption[corruption]["invreg"][sev].append(i)

    csv_path = OUT / "severity_resolved_raw.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "severity", "arch", "baseline_acc", "invreg_acc",
                     "delta_acc", "headroom"])
        w.writerows(rows)
    print(f"[saved] {csv_path}")

    # mean-over-8-architectures summary, per corruption x severity
    summary_path = OUT / "severity_resolved_summary.csv"
    summary_rows = []
    for corruption in CORRUPTIONS:
        for sev in SEVERITIES:
            b_vals = per_corruption[corruption]["baseline"][sev]
            i_vals = per_corruption[corruption]["invreg"][sev]
            if not b_vals:
                continue
            mean_b = float(np.mean(b_vals))
            mean_i = float(np.mean(i_vals))
            mean_delta = mean_i - mean_b
            mean_headroom = 1.0 - mean_b
            summary_rows.append([corruption, sev, len(b_vals), round(mean_b, 5),
                                  round(mean_i, 5), round(mean_delta, 5),
                                  round(mean_headroom, 5)])
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "severity", "n_arch", "mean_baseline_acc",
                     "mean_invreg_acc", "mean_delta_acc", "mean_headroom"])
        w.writerows(summary_rows)
    print(f"[saved] {summary_path}")
    for r in summary_rows:
        print(f"  {r[0]:<10s} sev={r[1]}  n={r[2]}  base={r[3]:.3f}  "
              f"invreg={r[4]:.3f}  delta={r[5]:+.3f}  headroom={r[6]:.3f}")

    # correlation check: does delta track headroom across severities?
    for corruption in CORRUPTIONS:
        sevs = [r[1] for r in summary_rows if r[0] == corruption]
        deltas = [r[5] for r in summary_rows if r[0] == corruption]
        headrooms = [r[6] for r in summary_rows if r[0] == corruption]
        if len(deltas) >= 2:
            r_delta_sev = np.corrcoef(sevs, deltas)[0, 1]
            r_delta_headroom = np.corrcoef(headrooms, deltas)[0, 1]
            print(f"[corr] {corruption}: corr(delta, severity)={r_delta_sev:.3f}  "
                  f"corr(delta, headroom)={r_delta_headroom:.3f}")

    # figure: delta_acc vs severity, one line per corruption, headroom as dashed
    plt = _pyplot()
    fig, ax1 = plt.subplots(figsize=(5.5, 4.0))
    colors = {"lidar": "#c1440e", "occlusion": "#2b6cb0"}
    for corruption in CORRUPTIONS:
        sevs = [r[1] for r in summary_rows if r[0] == corruption]
        deltas = [r[5] for r in summary_rows if r[0] == corruption]
        ax1.plot(sevs, deltas, "o-", color=colors[corruption], label=f"{corruption} $\\Delta$ acc")
    ax1.set_xlabel("ModelNet40-C severity")
    ax1.set_ylabel("Mean accuracy delta (invreg $-$ baseline)")
    ax1.set_xticks(SEVERITIES)
    # Explicit ticks/limits -- autoscale placed a top tick right at the
    # data's max value, and bbox_inches="tight" then clipped that tick's
    # label off the saved PNG since there was no margin above it.
    ax1.set_ylim(0.0, 0.26)
    ax1.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    ax2 = ax1.twinx()
    for corruption in CORRUPTIONS:
        sevs = [r[1] for r in summary_rows if r[0] == corruption]
        headrooms = [r[6] for r in summary_rows if r[0] == corruption]
        ax2.plot(sevs, headrooms, "s--", color=colors[corruption], alpha=0.45,
                  label=f"{corruption} headroom")
    ax2.set_ylabel("Headroom (1 $-$ baseline accuracy)")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig_path = OUT / "severity_resolved_transfer.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fig_path}")


if __name__ == "__main__":
    main()
