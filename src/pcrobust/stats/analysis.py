"""
Assumption checks + statistical analysis of the ModelNet40 confirmatory
pose-robustness dataset (statistical rationale: primary paired t-test on
D_j = Sim(80) - Sim(120), Shapiro-Wilk + Wilcoxon fallback, secondary
randomized-block ANOVA with Fligner-Killeen/Friedman fallback).

pilot_sd_D is read from
results/pilot_deltasigma.json (written by power.py, itself derived from
THIS dataset's own pilot.py run) instead of being a hardcoded literal
copied from another dataset. Different dataset -> different variance ->
different achieved power at the same n.
"""

import sys
import csv
import json
import argparse
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


def power_paired_t(n: int, delta_std: float, alpha: float = 0.05) -> float:
    """Power of a one-sided paired/one-sample t-test at given n (noncentral-t).
    Same solver as pcrobust.stats.power.power_paired_t."""
    if n < 2:
        return 0.0
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df=df)
    nc = delta_std * np.sqrt(n)
    return float(1 - stats.nct.cdf(t_crit, df=df, nc=nc))


ANGLES = [20, 40, 60, 80, 120]
ALPHA = 0.05
DELTA = 0.05  # pre-registered minimal interesting effect, same threshold as the original pose study


def load_data(arch: str = "pointnetpp"):
    rows = list(csv.DictReader(open(_RESULTS / f"confirmatory_raw_{arch}.csv", encoding="utf-8")))
    scans = sorted(set(r["scan"] for r in rows))
    by_scan_angle = {(r["scan"], int(r["angle_deg"])): float(r["similarity"]) for r in rows}
    return scans, by_scan_angle


def load_pilot_sd_D() -> float:
    json_path = _RESULTS / "pilot_deltasigma.json"
    if not json_path.exists():
        sys.exit(f"[analysis] {json_path} not found -- run pilot.py then power.py first")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return float(data["sd_D"])


def descriptive_stats(scans, by_scan_angle, arch: str = "pointnetpp"):
    print("\n[analysis] === Exploratory Data Analysis: descriptive stats per angle ===")
    rows = []
    for theta in ANGLES:
        vals = np.array([by_scan_angle[(s, theta)] for s in scans])
        rows.append({
            "angle_deg": theta, "n": len(vals),
            "mean": np.mean(vals), "median": np.median(vals),
            "sd": np.std(vals, ddof=1), "min": np.min(vals), "max": np.max(vals),
            "q1": np.percentile(vals, 25), "q3": np.percentile(vals, 75),
        })
        r = rows[-1]
        print(f"    theta={theta:4d}  mean={r['mean']:.4f}  median={r['median']:.4f}  "
              f"sd={r['sd']:.4f}  min={r['min']:.4f}  max={r['max']:.4f}  "
              f"Q1={r['q1']:.4f}  Q3={r['q3']:.4f}")

    stats_path = _RESULTS / f"descriptive_stats_{arch}.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[analysis] Saved {stats_path}")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [[by_scan_angle[(s, theta)] for s in scans] for theta in ANGLES]
    ax.boxplot(data, tick_labels=[f"{a}°" for a in ANGLES])
    ax.set_xlabel("Rotation angle (yaw)")
    ax.set_ylabel("Cosine similarity to canonical (0°) embedding")
    ax.set_title(f"Embedding similarity vs. rotation angle ({arch}, ModelNet40, N={len(scans)})")
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    box_path = _RESULTS / f"boxplot_by_angle_{arch}.png"
    fig.savefig(box_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved {box_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    means = [r["mean"] for r in rows]
    n = len(scans)
    cis = []
    for theta_vals in data:
        m, sd = np.mean(theta_vals), np.std(theta_vals, ddof=1)
        sem = sd / np.sqrt(n)
        lo = m + stats.t.ppf(ALPHA / 2, df=n - 1) * sem
        hi = m + stats.t.ppf(1 - ALPHA / 2, df=n - 1) * sem
        cis.append((m - lo, hi - m))
    yerr = np.array(cis).T
    ax.errorbar(ANGLES, means, yerr=yerr, fmt="o-", capsize=5, color="#4C72B0")
    ax.axvline(80, color="green", linestyle=":", label="edge of trained range (80°)")
    ax.set_xlabel("Rotation angle (yaw, degrees)")
    ax.set_ylabel("Mean cosine similarity (95% CI)")
    ax.set_title(f"Angle-trend curve: similarity vs. rotation angle ({arch}, ModelNet40)")
    ax.legend()
    fig.tight_layout()
    curve_path = _RESULTS / f"angle_trend_curve_{arch}.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved {curve_path}")

    return rows


def primary_test(scans, by_scan_angle, arch: str = "pointnetpp"):
    print("\n[analysis] === Primary confirmatory test: D_j = Sim(80) - Sim(120) ===")
    D = np.array([by_scan_angle[(s, 80)] - by_scan_angle[(s, 120)] for s in scans])
    n = len(D)
    mean_D, sd_D = float(np.mean(D)), float(np.std(D, ddof=1))
    print(f"    n = {n}")
    print(f"    mean(D) = {mean_D:+.5f}   sd(D) = {sd_D:.5f}")

    W, p_norm = stats.shapiro(D)
    print(f"\n[analysis] Shapiro-Wilk on D: W={W:.4f}  p={p_norm:.4f}  "
          f"({'fails to reject normality' if p_norm >= ALPHA else 'REJECTS normality'})")

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(D, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot of D_j = Sim(80°) - Sim(120°)  ({arch}, ModelNet40)")
    fig.tight_layout()
    qq_path = _RESULTS / f"qq_plot_D_{arch}.png"
    fig.savefig(qq_path, dpi=150)
    plt.close(fig)
    print(f"[analysis] Saved {qq_path}")

    t_stat, p_two_sided = stats.ttest_1samp(D, popmean=0.0)
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    d_z = mean_D / sd_D

    sem = sd_D / np.sqrt(n)
    ci_low = mean_D + stats.t.ppf(ALPHA / 2, df=n - 1) * sem
    ci_high = mean_D + stats.t.ppf(1 - ALPHA / 2, df=n - 1) * sem

    print(f"\n[analysis] Paired t-test (one-sided, H1: mu_D > 0):")
    print(f"    t({n-1}) = {t_stat:.4f}   one-sided p = {p_one_sided:.5f}")
    print(f"    95% CI on mu_D (two-sided) = [{ci_low:+.5f}, {ci_high:+.5f}]")
    print(f"    observed standardized effect size d_z = mean(D)/sd(D) = {d_z:+.4f}")
    print(f"    decision at alpha={ALPHA}: "
          f"{'REJECT H0' if p_one_sided < ALPHA else 'FAIL TO REJECT H0'}")

    w_stat, p_wilcoxon = stats.wilcoxon(D, alternative="greater")
    print(f"\n[analysis] Wilcoxon signed-rank (one-sided, fallback/cross-check):")
    print(f"    W = {w_stat:.4f}   one-sided p = {p_wilcoxon:.5f}")

    # Achieved power at the actual n, using THIS dataset's own pilot-derived
    # delta/sigma (results/pilot_deltasigma.json) -- NOT another dataset's
    # hardcoded pilot_sd_D=0.25424, and NOT the post-hoc observed effect.
    pilot_sd_D = load_pilot_sd_D()
    delta_std_planned = DELTA / pilot_sd_D
    achieved_power = power_paired_t(n, delta_std_planned, alpha=ALPHA)
    print(f"\n[analysis] Achieved power at n={n} "
          f"(pre-registered delta={DELTA}, ModelNet40 pilot sigma_D={pilot_sd_D:.5f}): "
          f"power = {achieved_power:.3f}")
    print(f"    (for context only, NOT used for the decision above: observed d_z "
          f"in this sample was {d_z:+.4f} vs. planned d={delta_std_planned:.4f})")

    return {
        "n": n, "mean_D": mean_D, "sd_D": sd_D, "shapiro_W": W, "shapiro_p": p_norm,
        "t_stat": t_stat, "p_one_sided": p_one_sided, "ci_low": ci_low, "ci_high": ci_high,
        "d_z": d_z, "wilcoxon_p": p_wilcoxon, "achieved_power": achieved_power,
        "pilot_sd_D": pilot_sd_D,
    }


def secondary_anova(scans, by_scan_angle):
    print("\n[analysis] === Secondary/exploratory: randomized-block one-way ANOVA "
          "(angle factor, instance = block) ===")
    a, b = len(ANGLES), len(scans)
    Y = np.array([[by_scan_angle[(s, theta)] for theta in ANGLES] for s in scans])  # (b, a)

    grand_mean = Y.mean()
    angle_means = Y.mean(axis=0)
    block_means = Y.mean(axis=1)

    SS_total = np.sum((Y - grand_mean) ** 2)
    SS_angle = b * np.sum((angle_means - grand_mean) ** 2)
    SS_block = a * np.sum((block_means - grand_mean) ** 2)
    SS_error = SS_total - SS_angle - SS_block

    df_angle, df_block = a - 1, b - 1
    df_error = df_angle * df_block

    MS_angle = SS_angle / df_angle
    MS_error = SS_error / df_error
    F_stat = MS_angle / MS_error
    p_anova = 1 - stats.f.cdf(F_stat, df_angle, df_error)

    print(f"    SS_angle={SS_angle:.4f} (df={df_angle})  SS_block={SS_block:.4f} (df={df_block})  "
          f"SS_error={SS_error:.4f} (df={df_error})")
    print(f"    F({df_angle},{df_error}) = {F_stat:.4f}   p = {p_anova:.6f}")

    residuals = Y - angle_means[np.newaxis, :] - block_means[:, np.newaxis] + grand_mean
    W, p_norm = stats.shapiro(residuals.ravel())
    print(f"\n    Shapiro-Wilk on ANOVA residuals: W={W:.4f}  p={p_norm:.5f}  "
          f"({'fails to reject normality' if p_norm >= ALPHA else 'REJECTS normality'})")

    fligner_stat, p_fligner = stats.fligner(*[Y[:, i] for i in range(a)])
    print(f"    Fligner-Killeen (homoscedasticity across angles): "
          f"stat={fligner_stat:.4f}  p={p_fligner:.5f}  "
          f"({'fails to reject equal variance' if p_fligner >= ALPHA else 'REJECTS equal variance'})")

    print(f"\n    ANOVA decision at alpha={ALPHA}: "
          f"{'REJECT H0 (angle affects similarity)' if p_anova < ALPHA else 'FAIL TO REJECT H0'}")

    friedman_stat, p_friedman = stats.friedmanchisquare(*[Y[:, i] for i in range(a)])
    print(f"\n[analysis] Friedman test (non-parametric fallback, reported regardless):")
    print(f"    chi2({df_angle}) = {friedman_stat:.4f}   p = {p_friedman:.6f}   "
          f"({'REJECT H0' if p_friedman < ALPHA else 'FAIL TO REJECT H0'})")

    posthoc = []
    if p_friedman < ALPHA:
        print(f"\n[analysis] Friedman significant -> Bonferroni-corrected pairwise "
              f"Wilcoxon signed-rank tests (10 comparisons, alpha_adj = {ALPHA/10:.4f}):")
        from itertools import combinations
        for i, j in combinations(range(a), 2):
            diff = Y[:, i] - Y[:, j]
            w_ij, p_ij = stats.wilcoxon(diff)
            sig = "  *SIGNIFICANT*" if p_ij < ALPHA / 10 else ""
            print(f"    {ANGLES[i]:>3} vs {ANGLES[j]:>3} deg : "
                  f"W={w_ij:.3f}  p={p_ij:.5f}{sig}")
            posthoc.append((ANGLES[i], ANGLES[j], w_ij, p_ij))

    return {"F": F_stat, "p_anova": p_anova, "shapiro_p_resid": p_norm,
            "fligner_p": p_fligner, "friedman_stat": friedman_stat,
            "p_friedman": p_friedman, "posthoc": posthoc}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="pointnetpp",
                        choices=["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv", "pointmlp", "pointtransformer"])
    args = parser.parse_args(argv)
    _RESULTS.mkdir(parents=True, exist_ok=True)

    scans, by_scan_angle = load_data(args.arch)
    print(f"[analysis] arch={args.arch}  Loaded {len(scans)} instances "
          f"from confirmatory_raw_{args.arch}.csv")

    descriptive_stats(scans, by_scan_angle, args.arch)
    primary_results = primary_test(scans, by_scan_angle, args.arch)
    secondary_results = secondary_anova(scans, by_scan_angle)

    print("\n[analysis] === Done ===")
    return primary_results, secondary_results


if __name__ == "__main__":
    main()
