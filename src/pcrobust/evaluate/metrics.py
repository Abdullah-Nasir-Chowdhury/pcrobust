"""
Official-style ModelNet40-C corruption-robustness metrics, computed from
pcrobust.evaluate.modelnet40c's per-checkpoint raw CSVs for BOTH training
conditions the paper compares, plus a direct with-vs-without-invariance-loss
comparison table.

Definitions
------------
For architecture f, corruption c, severity s: Acc_{c,s}(f) = accuracy on that
(corruption, severity) subset; Err = 1 - Acc. Acc_clean(f) = accuracy on the
"none" (uncorrupted) reference.

- mCE(f): CE_c(f) = [sum_s Err_{c,s}(f)] / [sum_s Err_{c,s}(baseline)],
  mCE(f) = mean_c CE_c(f). Baseline = the PLAIN (non-invreg) DGCNN checkpoint
  -- the benchmark's standard mCE reference, used for BOTH conditions so mCE stays
  comparable across the with/without-invariance-loss split (an invreg-vs-
  invreg-baseline mCE would answer a different, not-asked question).
- Relative Resilience(f) = mean_{c,s} [Acc_{c,s}(f) / Acc_clean(f)]
  (self-normalized; 1.0 = no degradation).
- OOD generalization gap(f) = Acc_clean(f) - mean_{c,s} Acc_{c,s}(f).

Outputs (written next to the raw CSVs, default <PCROBUST_HOME>/results/modelnet40c/):
- corruption_summary.csv   -- one row per arch (all 16: 8 baseline + 8 invreg)
- corruption_by_type.csv   -- per (arch, corruption) breakdown, all 16
- invreg_vs_baseline.csv   -- one row per base architecture, baseline vs.
                               invreg side by side + deltas, the direct
                               answer to "does the invariance loss help on
                               ModelNet40-C"

Usage:
    python -m pcrobust.evaluate.metrics [--results_dir DIR]
"""

import csv
from pathlib import Path
from collections import defaultdict

from pcrobust.datasets import results_dir

_RESULTS = results_dir("modelnet40c")

BASELINE_ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp",
                   "risurconv", "pointmlp", "pointtransformer"]
INVREG_ARCHS = [f"{a}_invreg" for a in BASELINE_ARCHS]
# lambda_inv-warmup retrain of every base arch (fixes a
# representation-collapse convergence stall found in risurconv_invreg,
# generalized to every architecture).
WARMUP_ARCHS = [f"{a}_invreg_warmup" for a in BASELINE_ARCHS]
ALL_ARCHS = BASELINE_ARCHS + INVREG_ARCHS + WARMUP_ARCHS
MCE_BASELINE_ARCH = "dgcnn"   # plain, non-invreg -- shared reference for both conditions


def load_raw(arch: str):
    path = _RESULTS / f"raw_{arch}.csv"
    if not path.exists():
        return None
    return list(csv.DictReader(open(path, encoding="utf-8")))


def acc_by_corruption_severity(rows):
    buckets = defaultdict(list)
    for r in rows:
        key = (r["corruption"], int(r["severity"]))
        buckets[key].append(int(r["correct"]))
    return {k: sum(v) / len(v) for k, v in buckets.items()}


def summarize(per_arch_acc, baseline_acc, corruptions, severities):
    summary_rows = []
    for arch, accs in per_arch_acc.items():
        acc_clean = accs.get(("none", 0))
        if acc_clean is None:
            print(f"[metrics] SKIP {arch}: no clean ('none') accuracy found")
            continue

        corrupted_accs = [v for (c, s), v in accs.items() if c != "none"]
        mean_corrupted_acc = sum(corrupted_accs) / len(corrupted_accs) if corrupted_accs else float("nan")
        resilience = sum(a / acc_clean for a in corrupted_accs) / len(corrupted_accs) if corrupted_accs else float("nan")
        ood_gap = acc_clean - mean_corrupted_acc

        mce = float("nan")
        if baseline_acc is not None:
            ce_per_corruption = []
            for c in corruptions:
                err_f = sum(1 - accs.get((c, s), float("nan")) for s in severities)
                err_base = sum(1 - baseline_acc.get((c, s), float("nan")) for s in severities)
                if err_base > 0:
                    ce_per_corruption.append(err_f / err_base)
            if ce_per_corruption:
                mce = sum(ce_per_corruption) / len(ce_per_corruption)

        summary_rows.append({
            "architecture": arch,
            "invariance_loss": "_invreg" in arch,
            "clean_accuracy": round(acc_clean, 6),
            "mean_corrupted_accuracy": round(mean_corrupted_acc, 6),
            "mCE": round(mce, 6) if mce == mce else "",
            "relative_resilience": round(resilience, 6),
            "ood_generalization_gap": round(ood_gap, 6),
        })
    return summary_rows


def main(argv=None):
    global _RESULTS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=None,
                        help="Directory holding raw_<arch>.csv (default: "
                             "<PCROBUST_HOME>/results/modelnet40c).")
    args = parser.parse_args(argv)
    if args.results_dir:
        _RESULTS = Path(args.results_dir)
    per_arch_acc = {}
    for arch in ALL_ARCHS:
        rows = load_raw(arch)
        if rows is None:
            print(f"[metrics] SKIP {arch}: no raw_{arch}.csv yet (run modelnet40c_eval.py --arch {arch})")
            continue
        per_arch_acc[arch] = acc_by_corruption_severity(rows)

    if not per_arch_acc:
        print("[metrics] nothing to summarize -- run modelnet40c_eval.py first")
        return

    if MCE_BASELINE_ARCH not in per_arch_acc:
        print(f"[metrics] mCE reference arch '{MCE_BASELINE_ARCH}' not evaluated yet -- mCE cannot be computed")
        baseline_acc = None
    else:
        baseline_acc = per_arch_acc[MCE_BASELINE_ARCH]

    corruptions = sorted(set(c for accs in per_arch_acc.values() for (c, s) in accs if c != "none"))
    severities = sorted(set(s for accs in per_arch_acc.values() for (c, s) in accs if c != "none"))

    summary_rows = summarize(per_arch_acc, baseline_acc, corruptions, severities)
    summary_by_arch = {r["architecture"]: r for r in summary_rows}

    print(f"\n{'architecture':<20s} {'inv_loss':<9s} {'clean':>7s} {'mCE':>7s} {'resil':>7s} {'ood_gap':>8s}")
    for r in summary_rows:
        mce_str = f"{r['mCE']:.4f}" if r["mCE"] != "" else "  n/a "
        print(f"  {r['architecture']:<18s} {str(r['invariance_loss']):<9s} "
              f"{r['clean_accuracy']:.4f} {mce_str} {r['relative_resilience']:.4f} {r['ood_generalization_gap']:+.4f}")

    summary_path = _RESULTS / "corruption_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["architecture", "invariance_loss", "clean_accuracy",
                                           "mean_corrupted_accuracy", "mCE",
                                           "relative_resilience", "ood_generalization_gap"])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[metrics] saved {summary_path}")

    # Per-(arch, corruption-type) breakdown, all evaluated archs.
    detail_rows = []
    for arch, accs in per_arch_acc.items():
        acc_clean = accs.get(("none", 0))
        for c in corruptions:
            sev_accs = [accs.get((c, s)) for s in severities if (c, s) in accs]
            if not sev_accs:
                continue
            mean_acc = sum(sev_accs) / len(sev_accs)
            detail_rows.append({
                "architecture": arch,
                "invariance_loss": "_invreg" in arch,
                "corruption": c,
                "mean_accuracy": round(mean_acc, 6),
                "clean_accuracy": round(acc_clean, 6) if acc_clean is not None else "",
                "accuracy_drop": round((acc_clean - mean_acc), 6) if acc_clean is not None else "",
            })
    detail_path = _RESULTS / "corruption_by_type.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["architecture", "invariance_loss", "corruption", "mean_accuracy",
                                           "clean_accuracy", "accuracy_drop"])
        w.writeheader()
        w.writerows(detail_rows)
    print(f"[metrics] saved {detail_path}")

    # Direct with-vs-without-invariance-loss comparison, one row per base arch.
    compare_rows = []
    for base in BASELINE_ARCHS:
        invreg = f"{base}_invreg"
        b = summary_by_arch.get(base)
        i = summary_by_arch.get(invreg)
        if b is None or i is None:
            print(f"[compare] SKIP {base}: missing {'baseline' if b is None else 'invreg'} summary row")
            continue

        def _delta(key):
            bv, iv = b[key], i[key]
            if bv == "" or iv == "":
                return ""
            return round(iv - bv, 6)

        compare_rows.append({
            "base_architecture": base,
            "baseline_clean_acc": b["clean_accuracy"],
            "invreg_clean_acc": i["clean_accuracy"],
            "delta_clean_acc": _delta("clean_accuracy"),
            "baseline_mean_corrupted_acc": b["mean_corrupted_accuracy"],
            "invreg_mean_corrupted_acc": i["mean_corrupted_accuracy"],
            "delta_mean_corrupted_acc": _delta("mean_corrupted_accuracy"),
            "baseline_mCE": b["mCE"],
            "invreg_mCE": i["mCE"],
            "delta_mCE": _delta("mCE"),
            "baseline_relative_resilience": b["relative_resilience"],
            "invreg_relative_resilience": i["relative_resilience"],
            "delta_relative_resilience": _delta("relative_resilience"),
            "baseline_ood_gap": b["ood_generalization_gap"],
            "invreg_ood_gap": i["ood_generalization_gap"],
            "delta_ood_gap": _delta("ood_generalization_gap"),
        })

    if compare_rows:
        print(f"\n{'arch':<16s} {'clean(base->inv)':<20s} {'meanCorr(base->inv)':<22s} "
              f"{'mCE(base->inv)':<18s} {'resil(base->inv)':<20s}")
        for r in compare_rows:
            print(f"  {r['base_architecture']:<14s} "
                  f"{r['baseline_clean_acc']:.3f}->{r['invreg_clean_acc']:.3f} ({r['delta_clean_acc']:+}) "
                  f"  {r['baseline_mean_corrupted_acc']:.3f}->{r['invreg_mean_corrupted_acc']:.3f} ({r['delta_mean_corrupted_acc']:+}) "
                  f"  {r['baseline_mCE']}->{r['invreg_mCE']} ({r['delta_mCE']}) "
                  f"  {r['baseline_relative_resilience']:.3f}->{r['invreg_relative_resilience']:.3f} ({r['delta_relative_resilience']:+})")

        compare_path = _RESULTS / "invreg_vs_baseline.csv"
        with open(compare_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
            w.writeheader()
            w.writerows(compare_rows)
        print(f"\n[compare] saved {compare_path}")
    else:
        print("\n[compare] no complete baseline/invreg pairs yet")

    # Three-way comparison: plain baseline vs. unwarmed invreg vs.
    # warmup-fixed invreg, one row per base arch -- the direct answer to
    # "does the lambda_inv warmup help, and by how much" for every
    # architecture, not just risurconv. Only includes archs where all
    # three have been evaluated; others are skipped with a note so a
    # partial run (e.g. warmup training still in progress for some archs)
    # doesn't crash the whole script.
    warmup_rows = []
    for base in BASELINE_ARCHS:
        b = summary_by_arch.get(base)
        i = summary_by_arch.get(f"{base}_invreg")
        w_ = summary_by_arch.get(f"{base}_invreg_warmup")
        if b is None or i is None or w_ is None:
            missing = [n for n, v in [("baseline", b), ("invreg", i), ("warmup", w_)] if v is None]
            print(f"[warmup-compare] SKIP {base}: missing {', '.join(missing)}")
            continue

        def _d(key, a, c):
            av, cv = a[key], c[key]
            if av == "" or cv == "":
                return ""
            return round(cv - av, 6)

        warmup_rows.append({
            "base_architecture": base,
            "baseline_clean_acc": b["clean_accuracy"],
            "invreg_clean_acc": i["clean_accuracy"],
            "warmup_clean_acc": w_["clean_accuracy"],
            "delta_warmup_vs_invreg_clean_acc": _d("clean_accuracy", i, w_),
            "delta_warmup_vs_baseline_clean_acc": _d("clean_accuracy", b, w_),
            "baseline_mean_corrupted_acc": b["mean_corrupted_accuracy"],
            "invreg_mean_corrupted_acc": i["mean_corrupted_accuracy"],
            "warmup_mean_corrupted_acc": w_["mean_corrupted_accuracy"],
            "delta_warmup_vs_invreg_mean_corrupted_acc": _d("mean_corrupted_accuracy", i, w_),
            "delta_warmup_vs_baseline_mean_corrupted_acc": _d("mean_corrupted_accuracy", b, w_),
            "baseline_mCE": b["mCE"],
            "invreg_mCE": i["mCE"],
            "warmup_mCE": w_["mCE"],
            "delta_warmup_vs_invreg_mCE": _d("mCE", i, w_),
            "delta_warmup_vs_baseline_mCE": _d("mCE", b, w_),
            "baseline_relative_resilience": b["relative_resilience"],
            "invreg_relative_resilience": i["relative_resilience"],
            "warmup_relative_resilience": w_["relative_resilience"],
            "delta_warmup_vs_invreg_relative_resilience": _d("relative_resilience", i, w_),
            "delta_warmup_vs_baseline_relative_resilience": _d("relative_resilience", b, w_),
        })

    if warmup_rows:
        print(f"\n{'arch':<16s} {'clean (base/invreg/warmup)':<28s} {'mCE (base/invreg/warmup)':<26s} verdict")
        for r in warmup_rows:
            mce_ok = r["warmup_mCE"] != "" and r["baseline_mCE"] != ""
            if mce_ok and r["warmup_mCE"] < r["baseline_mCE"]:
                verdict = "warmup BEATS baseline"
            elif mce_ok and r["delta_warmup_vs_invreg_mCE"] != "" and r["delta_warmup_vs_invreg_mCE"] < 0:
                verdict = "warmup improves over unwarmed invreg, still below baseline"
            else:
                verdict = "no improvement / worse"
            print(f"  {r['base_architecture']:<14s} "
                  f"{r['baseline_clean_acc']:.3f} / {r['invreg_clean_acc']:.3f} / {r['warmup_clean_acc']:.3f}      "
                  f"{r['baseline_mCE']} / {r['invreg_mCE']} / {r['warmup_mCE']}      {verdict}")

        warmup_path = _RESULTS / "warmup_vs_invreg_vs_baseline.csv"
        with open(warmup_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(warmup_rows[0].keys()))
            w.writeheader()
            w.writerows(warmup_rows)
        print(f"\n[warmup-compare] saved {warmup_path}")
    else:
        print("\n[warmup-compare] no complete baseline/invreg/warmup triples yet")


if __name__ == "__main__":
    main()
