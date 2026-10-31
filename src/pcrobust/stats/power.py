"""
power.py — Sample size / power calculation for the ModelNet40 replication.

The underlying math (noncentral-t power solver) is dataset-agnostic, so it's
the same solver the original pose study used.

sd_D (and therefore the required
sample size) is computed FRESH from THIS pilot's data
(results/pilot_raw.csv, produced by this directory's pilot.py), not reused
from another dataset. Also writes results/pilot_deltasigma.json so
analysis.py can read mean_D/sd_D/n_pilot at runtime instead of a hardcoded
literal.
"""

import sys
import csv
import json
import numpy as np
from scipy import stats

from pcrobust.datasets import results_dir

_RESULTS = results_dir("pose", "modelnet40")


def power_paired_t(n: int, delta_std: float, alpha: float = 0.05) -> float:
    """Power of a one-sided paired/one-sample t-test at given n (noncentral-t).
    Dataset-agnostic: the same solver the original pose study used."""
    if n < 2:
        return 0.0
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df=df)
    nc = delta_std * np.sqrt(n)
    return float(1 - stats.nct.cdf(t_crit, df=df, nc=nc))


def required_n_paired_t(delta_std: float, alpha: float = 0.05,
                        power: float = 0.8, n_max: int = 5000) -> int:
    """Smallest n achieving the target power (one-sided paired t-test)."""
    for n in range(2, n_max + 1):
        if power_paired_t(n, delta_std, alpha) >= power:
            return n
    raise RuntimeError(f"Required n exceeds n_max={n_max}")


ALPHA = 0.05
TARGET_POWER = 0.80
DELTA = 0.05  # minimal interesting effect -- same pre-registered threshold as the original pose study,
              # kept identical so the two studies stay comparable on the same effect-size scale


def load_pilot_D() -> np.ndarray:
    raw_path = _RESULTS / "pilot_raw.csv"
    if not raw_path.exists():
        sys.exit(f"[power] {raw_path} not found -- run pilot.py first")
    rows = list(csv.DictReader(open(raw_path, encoding="utf-8")))
    scans = sorted(set(r["scan"] for r in rows))
    by_scan_angle = {(r["scan"], int(r["angle_deg"])): float(r["similarity"]) for r in rows}
    D = np.array([by_scan_angle[(s, 80)] - by_scan_angle[(s, 120)]
                  for s in scans if (s, 80) in by_scan_angle and (s, 120) in by_scan_angle])
    return D


def main():
    D = load_pilot_D()
    sd_D = float(np.std(D, ddof=1))
    mean_D = float(np.mean(D))
    print(f"[power] Pilot D_j values: {D.round(4).tolist()}")
    print(f"[power] pilot mean(D) = {mean_D:+.5f}   pilot sd(D) = {sd_D:.5f}   (n_pilot={len(D)})")

    delta_std = DELTA / sd_D
    print(f"\n[power] delta (minimal interesting effect)   = {DELTA}")
    print(f"[power] sigma_D (THIS pilot, ModelNet40, n={len(D)})  = {sd_D:.5f}")
    print(f"[power] standardized effect size d = delta/sd = {delta_std:.4f}")

    n_req = required_n_paired_t(delta_std, alpha=ALPHA, power=TARGET_POWER)
    print(f"\n[power] alpha={ALPHA}, target power={TARGET_POWER}, one-sided paired t-test")
    print(f"[power] REQUIRED sample size (pairs): n = {n_req}")

    print(f"\n[power] Power at various achievable sample sizes:")
    for n in [10, 30, 60, 100, 150, 162, 200, 250, 300, n_req]:
        p = power_paired_t(n, delta_std, alpha=ALPHA)
        print(f"    n={n:4d}   power={p:.3f}")

    out = {
        "mean_D": mean_D, "sd_D": sd_D, "n_pilot": len(D),
        "delta": DELTA, "delta_std": delta_std,
        "alpha": ALPHA, "target_power": TARGET_POWER, "n_required": n_req,
    }
    out_path = _RESULTS / "pilot_deltasigma.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[power] Saved {out_path}")


if __name__ == "__main__":
    main()
