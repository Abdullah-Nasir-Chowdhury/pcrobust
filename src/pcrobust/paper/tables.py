"""Generators for the frozen summary tables shipped in :mod:`pcrobust.paper`.

Three table builders, folded from the benchmark's standalone build scripts
with their statistics unchanged:

- :func:`main_summary` -- pose + noise primary paired-test and exploratory
  ANOVA summary tables.
- :func:`main_occlusion_summary` -- the same two tables for occlusion.
- :func:`main_training_metrics` -- training-run summary parsed from train
  logs.

All three are pure functions of already-saved raw per-scan CSVs / train
logs, which are NOT shipped with this package. They expect the benchmark
results tree (``experiments/...``, ``train_logs/...``) under the pcrobust
cache root -- the ``PCROBUST_HOME`` environment variable, defaulting to
``~/.cache/pcrobust`` -- and write their outputs to
``<root>/results_summary/``. The frozen copies of these outputs are bundled
at ``pcrobust/paper/data/summary/``.

Run as::

    python -m pcrobust.paper.tables [summary|occlusion|training-metrics|all]
"""

import csv
import os
import re
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(os.environ.get("PCROBUST_HOME", os.path.expanduser("~/.cache/pcrobust")))
_OUT = _ROOT / "results_summary"
_LOGDIR = _ROOT / "train_logs"


# ---------------------------------------------------------------------------
# Pose + noise summary tables
# ---------------------------------------------------------------------------

"""
main_summary -- aggregates every domain x architecture x experiment
combination's confirmatory raw data into the benchmark's 2 summary-table
formats (primary paired-test, exploratory ANOVA), once for pose robustness
(all 4 rotation-axis variants: yaw/tilt/roll/extreme) and once for noise
robustness.

Pure function of already-saved confirmatory_raw_<arch>.csv / noise_raw_<arch>.csv
files -- safe to re-run any time, matching the benchmark's established
"recompute from saved raw data" pattern. Independently re-derives the exact
same statistics the per-run analysis scripts already validated, but emits
one row per (domain, architecture[, rotation_type]) instead of printing to
stdout, in these 2 table shapes:
    primary-test table: Domain | Architecture | mean(D) | sd(D) | d_z |
        Shapiro-Wilk (W, p) | test statistic, p | 95% CI on mean(D) | Decision
    exploratory-ANOVA table: Dataset | Architecture | ANOVA (F, p) |
        Shapiro-Wilk (residuals) | Fligner-Killeen | Friedman | Decision

The primary table shape predates the tilt/roll/extreme rotation-axis
expansion, so pose's tables carry an extra Rotation_Type column
(yaw/tilt/roll/extreme) rather than 4 separate near-identical file sets;
noise has no rotation axis, so its tables omit that column.

The exploratory table's "Decision" is driven by the Friedman test, not the
parametric ANOVA -- similarity is bounded in [-1, 1] and its variance is
expected to grow with rotation/noise severity (that's the substantive
finding this whole study is testing for), so the ANOVA's normality/
homoscedasticity assumptions are routinely violated by design. Friedman is
the assumption-free test, so it's treated as the test of record; the
parametric ANOVA/Shapiro/Fligner numbers are reported alongside for
transparency, exactly as the per-run analysis already does for each
individual run.

Usage: python -m pcrobust.paper.tables summary
Outputs (relative to PCROBUST_HOME):
    results_summary/pose_primary_test.csv
    results_summary/pose_exploratory_anova.csv
    results_summary/noise_primary_test.csv
    results_summary/noise_exploratory_anova.csv
"""

DOMAINS = ["modelnet40", "shapenet", "abo", "scanobjectnn", "s3dis"]
ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv",
         "pointmlp", "pointtransformer"]
POSE_VARIANTS = ["yaw", "tilt", "roll", "extreme"]
ALPHA = 0.05

POSE_LEVELS = [20, 40, 60, 80, 120]
NOISE_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05]
POSE_MILD, POSE_SEVERE = 80, 120
NOISE_MILD, NOISE_SEVERE = 0.005, 0.05


def _pose_csv_path(domain: str, arch: str, variant: str) -> Path:
    base = _ROOT / "experiments" / "pose_robustness" / domain
    if variant == "yaw":
        return base / "results" / f"confirmatory_raw_{arch}.csv"
    return base / variant / "results" / f"confirmatory_raw_{arch}.csv"


def _noise_csv_path(domain: str, arch: str) -> Path:
    return _ROOT / "experiments" / "noise_robustness" / domain / "results" / f"noise_raw_{arch}.csv"


def _load(path: Path, level_col: str, level_cast):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scans = sorted(set(r["scan"] for r in rows))
    by_scan_level = {(r["scan"], level_cast(r[level_col])): float(r["similarity"]) for r in rows}
    return scans, by_scan_level


def primary_test(scans, by_scan_level, mild, severe) -> dict:
    D = np.array([by_scan_level[(s, mild)] - by_scan_level[(s, severe)] for s in scans])
    n = len(D)
    mean_D, sd_D = float(np.mean(D)), float(np.std(D, ddof=1))
    W, p_norm = stats.shapiro(D)
    t_stat, p_two_sided = stats.ttest_1samp(D, popmean=0.0)
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2
    d_z = mean_D / sd_D if sd_D > 0 else float("nan")
    sem = sd_D / np.sqrt(n)
    ci_low = mean_D + stats.t.ppf(ALPHA / 2, df=n - 1) * sem
    ci_high = mean_D + stats.t.ppf(1 - ALPHA / 2, df=n - 1) * sem
    decision = "REJECT H0" if p_one_sided < ALPHA else "FAIL TO REJECT H0"
    return {
        "n": n, "mean_D": mean_D, "sd_D": sd_D, "d_z": d_z,
        "shapiro_W": W, "shapiro_p": p_norm,
        "t_stat": t_stat, "p_one_sided": p_one_sided,
        "ci_low": ci_low, "ci_high": ci_high, "decision": decision,
    }


def secondary_anova(scans, by_scan_level, levels) -> dict:
    a, b = len(levels), len(scans)
    Y = np.array([[by_scan_level[(s, lvl)] for lvl in levels] for s in scans])
    grand_mean = Y.mean()
    level_means = Y.mean(axis=0)
    block_means = Y.mean(axis=1)
    SS_total = np.sum((Y - grand_mean) ** 2)
    SS_level = b * np.sum((level_means - grand_mean) ** 2)
    SS_block = a * np.sum((block_means - grand_mean) ** 2)
    SS_error = SS_total - SS_level - SS_block
    df_level, df_block = a - 1, b - 1
    df_error = df_level * df_block
    MS_level = SS_level / df_level
    MS_error = SS_error / df_error
    F_stat = MS_level / MS_error
    p_anova = 1 - stats.f.cdf(F_stat, df_level, df_error)
    residuals = Y - level_means[np.newaxis, :] - block_means[:, np.newaxis] + grand_mean
    W_resid, p_resid = stats.shapiro(residuals.ravel())
    fligner_stat, p_fligner = stats.fligner(*[Y[:, i] for i in range(a)])
    friedman_stat, p_friedman = stats.friedmanchisquare(*[Y[:, i] for i in range(a)])
    decision = "REJECT H0 (level affects similarity)" if p_friedman < ALPHA else "FAIL TO REJECT H0"
    return {
        "F_stat": F_stat, "anova_p": p_anova,
        "shapiro_W_resid": W_resid, "shapiro_p_resid": p_resid,
        "fligner_stat": fligner_stat, "fligner_p": p_fligner,
        "friedman_stat": friedman_stat, "friedman_p": p_friedman, "decision": decision,
    }


def _primary_row(extra: dict, pt: dict) -> dict:
    row = dict(extra)
    row.update({
        "n": pt["n"], "mean_D": round(pt["mean_D"], 5), "sd_D": round(pt["sd_D"], 5),
        "d_z": round(pt["d_z"], 4), "shapiro_W": round(pt["shapiro_W"], 4),
        "shapiro_p": round(pt["shapiro_p"], 5),
        "t_stat": round(pt["t_stat"], 4), "p_one_sided": round(pt["p_one_sided"], 6),
        "ci_low": round(pt["ci_low"], 5), "ci_high": round(pt["ci_high"], 5),
        "Decision": pt["decision"],
    })
    return row


def _anova_row(extra: dict, an: dict) -> dict:
    row = dict(extra)
    row.update({
        "F_stat": round(an["F_stat"], 4), "anova_p": round(an["anova_p"], 6),
        "shapiro_W_resid": round(an["shapiro_W_resid"], 4),
        "shapiro_p_resid": round(an["shapiro_p_resid"], 5),
        "fligner_stat": round(an["fligner_stat"], 4), "fligner_p": round(an["fligner_p"], 5),
        "friedman_stat": round(an["friedman_stat"], 4), "friedman_p": round(an["friedman_p"], 6),
        "Decision": an["decision"],
    })
    return row


def _write(rows, name, tag="[build_summary_csvs]"):
    if not rows:
        print(f"{tag} no rows for {name}, skipping")
        return
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{tag} saved {path} ({len(rows)} rows)")


def main_summary():
    _OUT.mkdir(parents=True, exist_ok=True)
    pose_primary_rows, pose_anova_rows = [], []
    noise_primary_rows, noise_anova_rows = [], []
    missing = []

    for domain in DOMAINS:
        for variant in POSE_VARIANTS:
            for arch in ARCHS:
                path = _pose_csv_path(domain, arch, variant)
                if not path.exists():
                    missing.append(str(path))
                    continue
                scans, by_scan_level = _load(path, "angle_deg", int)
                extra = {"Domain": domain, "Architecture": arch, "Rotation_Type": variant}
                pt = primary_test(scans, by_scan_level, POSE_MILD, POSE_SEVERE)
                pose_primary_rows.append(_primary_row(extra, pt))
                an = secondary_anova(scans, by_scan_level, POSE_LEVELS)
                pose_anova_rows.append(_anova_row(extra, an))

    for domain in DOMAINS:
        for arch in ARCHS:
            path = _noise_csv_path(domain, arch)
            if not path.exists():
                missing.append(str(path))
                continue
            scans, by_scan_level = _load(path, "sigma", float)
            extra = {"Domain": domain, "Architecture": arch}
            pt = primary_test(scans, by_scan_level, NOISE_MILD, NOISE_SEVERE)
            noise_primary_rows.append(_primary_row(extra, pt))
            an = secondary_anova(scans, by_scan_level, NOISE_LEVELS)
            noise_anova_rows.append(_anova_row(extra, an))

    _write(pose_primary_rows, "pose_primary_test.csv")
    _write(pose_anova_rows, "pose_exploratory_anova.csv")
    _write(noise_primary_rows, "noise_primary_test.csv")
    _write(noise_anova_rows, "noise_exploratory_anova.csv")

    if missing:
        print(f"\n[build_summary_csvs] {len(missing)} missing raw CSVs (skipped):")
        for m in missing:
            print(f"    {m}")

# ---------------------------------------------------------------------------
# Occlusion summary tables
# ---------------------------------------------------------------------------

"""
build_occlusion_summary_csvs.py -- aggregates every domain x architecture x
camera-variant combination's confirmatory occlusion raw data into the same
2 proposal-slide-format summary tables used for pose/noise
(main_summary above), for occlusion robustness.

Pure function of already-saved occlusion_raw_<arch>.csv files (both the
random-azimuth primary variant and the fixed_view variant) -- mirrors
build_summary_csvs.py's structure and statistics exactly, with "dist_mult"
(camera distance, mild=0.3/severe=6.0) as the manipulated factor instead of
rotation angle or noise sigma.

Usage: python -m pcrobust.paper.tables occlusion
Outputs (relative to project root):
    results_summary/occlusion_primary_test.csv
    results_summary/occlusion_exploratory_anova.csv
"""

from pathlib import Path

_OUT = _ROOT / "results_summary"

DOMAINS = ["modelnet40", "shapenet", "abo", "scanobjectnn", "s3dis"]
ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv",
         "pointmlp", "pointtransformer"]
VARIANTS = ["random", "fixed_view"]
ALPHA = 0.05

LEVELS = [0.3, 0.6, 1.2, 2.5, 6.0]
MILD, SEVERE = 0.3, 6.0


def _csv_path(domain: str, arch: str, variant: str) -> Path:
    base = _ROOT / "experiments" / "occlusion_robustness" / domain
    if variant == "fixed_view":
        base = base / "fixed_view"
    return base / "results" / f"occlusion_raw_{arch}.csv"


def _load_occlusion(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scans = sorted(set(r["scan"] for r in rows))
    by_scan_level = {(r["scan"], float(r["dist_mult"])): float(r["similarity"]) for r in rows}
    return scans, by_scan_level


# primary_test / secondary_anova / _primary_row / _anova_row / _write are
# shared with the pose+noise builder above (the two source scripts carried
# byte-identical copies).


def main_occlusion_summary():
    _OUT.mkdir(parents=True, exist_ok=True)
    primary_rows, anova_rows = [], []
    missing = []

    for domain in DOMAINS:
        for variant in VARIANTS:
            for arch in ARCHS:
                path = _csv_path(domain, arch, variant)
                if not path.exists():
                    missing.append(str(path))
                    continue
                scans, by_scan_level = _load(path)
                extra = {"Domain": domain, "Architecture": arch, "Camera_Variant": variant}
                pt = primary_test(scans, by_scan_level, MILD, SEVERE)
                primary_rows.append(_primary_row(extra, pt))
                an = secondary_anova(scans, by_scan_level, LEVELS)
                anova_rows.append(_anova_row(extra, an))

    _write(primary_rows, "occlusion_primary_test.csv")
    _write(anova_rows, "occlusion_exploratory_anova.csv")

    if missing:
        print(f"\n[build_occlusion_summary_csvs] {len(missing)} missing raw CSVs (skipped):")
        for m in missing:
            print(f"    {m}")

# ---------------------------------------------------------------------------
# Training-metrics table
# ---------------------------------------------------------------------------

"""
build_training_metrics_csv.py -- parses train_logs/<arch>_<domain>.log (+
matching .time files) into one row per (domain, architecture) training-run
summary CSV.

Pure text-parsing of already-written logs -- safe to re-run any time. Only
reports fields that are actually logged by run_training_gpu.sh / train.py's
own print statements (see a sample log for the exact format this parses):
    === [GPU N] starting <arch> on <domain> at <ISO timestamp> ===
    [data]  <n_id> identities  |  <n_scans> scans  |  <n_samples> samples (...)
      epoch  E/TOTAL  loss=L  acc=A  (best=B)          (repeated, ~every 25 epochs)
    [train] Best acc : B
    === [GPU N] finished <arch> on <domain>: <elapsed>s (exit <status>) ===
and the matching .time file: elapsed_sec=<N> exit_status=<N>

6 (architecture, domain) combinations -- pointnet/pointnetpp/dgcnn on
modelnet40/shapenet -- predate this session's expansion work (they were
already-trained checkpoints carried over from the original, pre-expansion
project) and were never run through this pipeline's logging, so they have
no train_logs entry. They're reported with epochs_run/train_time_sec/etc.
left blank and a note, rather than fabricated.

No held-out test-set classification accuracy is logged anywhere in this
pipeline for the general case (the confirmatory/noise scripts measure
embedding self-similarity under transformation, not classification
accuracy; only the separate ModelNet40-C evaluation reports classification
accuracy, and only for the modelnet40 domain -- see
experiments/noise_robustness/modelnet40c/results/corruption_summary.csv for
that). So this CSV reports training-time metrics only; it does not invent a
"test_accuracy" column that no script in this pipeline actually produces.

Usage: python -m pcrobust.paper.tables training-metrics
Output: results_summary/training_metrics.csv
"""

from pathlib import Path

_LOGDIR = _ROOT / "train_logs"
_OUT = _ROOT / "results_summary"

DOMAINS = ["modelnet40", "shapenet", "abo", "scanobjectnn", "s3dis"]
ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv",
         "pointmlp", "pointtransformer"]

_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+acc=([\d.]+)\s+\(best=([\d.]+)\)")
_DATA_RE = re.compile(
    r"\[data\]\s+(\d+)\s+identities\s+\|\s+([\d,]+)\s+scans\s+\|\s+([\d,]+)\s+samples")
_BESTACC_RE = re.compile(r"\[train\] Best acc\s*:\s*([\d.]+)")
_START_RE = re.compile(r"starting .+? at (\S+)")
# Driver scripts vary slightly in how they print the finish line (the
# standard run_training_gpu.sh prints "finished X on Y: <N>s (exit <M>)";
# a couple of one-off retrain scripts used during this session's riconv
# bugfix print "finished X on Y (parallel retrain): exit <M>"
# without an inline elapsed time) -- the matching .time file
# ("elapsed_sec=<N> exit_status=<M>") is written by every driver variant
# and is the authoritative source, so it's used instead of regexing the
# log's finish line at all.
_TIME_RE = re.compile(r"elapsed_sec=(\d+)\s+exit_status=(-?\d+)")


def parse_log(path: Path, time_path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    epoch_matches = _EPOCH_RE.findall(text)
    last_epoch = epoch_matches[-1] if epoch_matches else None

    data_m = _DATA_RE.search(text)
    bestacc_m = _BESTACC_RE.search(text)
    start_m = _START_RE.search(text)

    train_time_sec = exit_status = None
    if time_path.exists():
        time_m = _TIME_RE.search(time_path.read_text(encoding="utf-8"))
        if time_m:
            train_time_sec = int(time_m.group(1))
            exit_status = int(time_m.group(2))

    row = {
        "epochs_run": int(last_epoch[0]) if last_epoch else None,
        "epochs_configured": int(last_epoch[1]) if last_epoch else None,
        "final_train_loss": float(last_epoch[2]) if last_epoch else None,
        "final_train_acc": float(last_epoch[3]) if last_epoch else None,
        "best_train_acc": float(bestacc_m.group(1)) if bestacc_m else
                           (float(last_epoch[4]) if last_epoch else None),
        "n_identities": int(data_m.group(1)) if data_m else None,
        "n_scans": int(data_m.group(2).replace(",", "")) if data_m else None,
        "n_train_samples": int(data_m.group(3).replace(",", "")) if data_m else None,
        "train_time_sec": train_time_sec,
        "exit_status": exit_status,
        "start_time_utc": start_m.group(1) if start_m else None,
    }
    return row


def main_training_metrics():
    rows = []
    for domain in DOMAINS:
        for arch in ARCHS:
            log_path = _LOGDIR / f"{arch}_{domain}.log"
            row = {"Domain": domain, "Architecture": arch}
            if not log_path.exists():
                row.update({
                    "epochs_run": "", "epochs_configured": "",
                    "final_train_loss": "", "final_train_acc": "",
                    "best_train_acc": "", "n_identities": "", "n_scans": "",
                    "n_train_samples": "", "train_time_sec": "",
                    "train_time_min": "", "exit_status": "", "start_time_utc": "",
                    "note": "pre-existing checkpoint, predates this session's "
                            "pipeline logging (no train_logs entry)",
                })
                rows.append(row)
                continue

            time_path = _LOGDIR / f"{arch}_{domain}.time"
            parsed = parse_log(log_path, time_path)
            parsed["train_time_min"] = (
                round(parsed["train_time_sec"] / 60, 1)
                if parsed["train_time_sec"] is not None else ""
            )
            parsed["note"] = "" if parsed["exit_status"] == 0 else \
                f"exit_status={parsed['exit_status']} -- check log"
            row.update(parsed)
            rows.append(row)

    fieldnames = ["Domain", "Architecture", "epochs_run", "epochs_configured",
                  "train_time_sec", "train_time_min", "final_train_loss",
                  "final_train_acc", "best_train_acc", "n_identities",
                  "n_scans", "n_train_samples", "exit_status",
                  "start_time_utc", "note"]
    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / "training_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[build_training_metrics_csv] saved {out_path} ({len(rows)} rows)")

    missing = [r for r in rows if r.get("note", "").startswith("pre-existing")]
    if missing:
        print(f"[build_training_metrics_csv] {len(missing)} rows have no train_logs "
              f"entry (pre-existing checkpoints, left blank not fabricated):")
        for r in missing:
            print(f"    {r['Architecture']} / {r['Domain']}")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="pcrobust.paper.tables",
        description="Rebuild the frozen summary tables from raw benchmark "
                    "outputs under PCROBUST_HOME (default ~/.cache/pcrobust).",
    )
    parser.add_argument(
        "which", nargs="?", default="all",
        choices=["summary", "occlusion", "training-metrics", "all"],
        help="which table set to rebuild (default: all)",
    )
    args = parser.parse_args(argv)
    if args.which in ("summary", "all"):
        main_summary()
    if args.which in ("occlusion", "all"):
        main_occlusion_summary()
    if args.which in ("training-metrics", "all"):
        main_training_metrics()


if __name__ == "__main__":
    main()
