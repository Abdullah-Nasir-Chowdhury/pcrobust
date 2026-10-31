"""
cross_architecture.py — compares PointNet, PointNet++, and DGCNN on the
SAME ModelNet40 pose-robustness protocol, reading each architecture's
confirmatory_raw_<arch>.csv from <PCROBUST_HOME>/results/pose/modelnet40/.
"""

import csv
import numpy as np
from scipy import stats

from pcrobust.datasets import results_dir

_RESULTS = results_dir("pose", "modelnet40")


def _pyplot():
    """matplotlib is an optional extra (pip install pcrobust[figures])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv", "pointmlp", "pointtransformer"]
ALPHA = 0.05


def load(arch):
    rows = list(csv.DictReader(open(_RESULTS / f"confirmatory_raw_{arch}.csv", encoding="utf-8")))
    scans = set(r["scan"] for r in rows)
    by = {(r["scan"], int(r["angle_deg"])): float(r["similarity"]) for r in rows}
    return scans, by


def main():
    _RESULTS.mkdir(parents=True, exist_ok=True)
    data = {arch: load(arch) for arch in ARCHS}
    common = set.intersection(*(scans for scans, _ in data.values()))
    common = sorted(common)
    print(f"[cross-arch] instances present in ALL {len(ARCHS)} architectures: n={len(common)}")
    dropped = set.union(*(scans for scans, _ in data.values())) - set(common)
    print(f"[cross-arch] dropped (not common to all {len(ARCHS)}): {sorted(dropped)}")

    D = {}
    print(f"\n[cross-arch] Per-architecture results on the common n={len(common)} instances:")
    print(f"{'arch':<12} {'mean(D)':>10} {'sd(D)':>10} {'d_z':>8} {'t':>8} {'p(one-sided)':>14}")
    for arch in ARCHS:
        _, by = data[arch]
        d = np.array([by[(s, 80)] - by[(s, 120)] for s in common])
        D[arch] = d
        mean_d, sd_d = float(np.mean(d)), float(np.std(d, ddof=1))
        d_z = mean_d / sd_d
        t_stat, p_two = stats.ttest_1samp(d, popmean=0.0)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        print(f"{arch:<12} {mean_d:>10.4f} {sd_d:>10.4f} {d_z:>8.3f} {t_stat:>8.3f} {p_one:>14.5f}")

    friedman_df = len(ARCHS) - 1
    friedman_stat, p_friedman = stats.friedmanchisquare(*[D[a] for a in ARCHS])
    print(f"\n[cross-arch] Friedman test (does D differ across architectures?): "
          f"chi2({friedman_df})={friedman_stat:.4f}  p={p_friedman:.5f}  "
          f"({'REJECT H0 -- architectures differ' if p_friedman < ALPHA else 'FAIL TO REJECT H0 -- no evidence architectures differ'})")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    means = [np.mean(D[a]) for a in ARCHS]
    n = len(common)
    cis = []
    for a in ARCHS:
        sd = np.std(D[a], ddof=1)
        sem = sd / np.sqrt(n)
        lo = np.mean(D[a]) + stats.t.ppf(ALPHA / 2, df=n - 1) * sem
        hi = np.mean(D[a]) + stats.t.ppf(1 - ALPHA / 2, df=n - 1) * sem
        cis.append((np.mean(D[a]) - lo, hi - np.mean(D[a])))
    yerr = np.array(cis).T
    colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]
    ax.bar(ARCHS, means, yerr=yerr, capsize=6, color=colors, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("mean D = Sim(80°) − Sim(120°)  (95% CI)")
    ax.set_title(f"Edge-of-range vs. beyond-range gap by architecture (ModelNet40, n={n})", fontsize=11)
    fig.tight_layout()
    out_path = _RESULTS / "cross_architecture_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[cross-arch] Saved {out_path}")

    return {"common_n": len(common), "D": D, "friedman_stat": friedman_stat,
            "p_friedman": p_friedman}


if __name__ == "__main__":
    main()
