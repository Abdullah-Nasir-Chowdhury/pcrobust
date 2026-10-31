"""Crossover analysis of the fine-grained rotation sweep.

``analyze_sweep`` reads ``results/rotation_sweep/rotation_sweep_summary.csv``
under the given root and reports, per rotation axis, the sustained-crossover
angle at which the best rotation-invariant architecture starts leading the
focus architecture, plus trough/max-gap statistics, writing
``crossover_report.json`` and ``yaw_curve_data.json``.

``analyze_sweep_pose270`` runs the same curve-data extraction against the
pose270 sweep (``results/rotation_sweep_pose270/rotation_sweep_summary.csv``,
``condition == "pose270"``), producing ``pose270_curve_data.json`` alongside
``yaw_curve_data.json`` so the pose270 figure can plot both on the same axes.

Usage:
  python -m pcrobust.evaluate.rotation_analysis [--root ROOT]
  python -m pcrobust.evaluate.rotation_analysis --pose270 [--root ROOT]
"""
import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = Path(os.environ.get("PCROBUST_HOME", os.path.expanduser("~/.cache/pcrobust")))


def analyze_sweep(root=None):
    ROOT = Path(root) if root is not None else _ROOT
    rows = list(csv.DictReader(open(ROOT / "results/rotation_sweep/rotation_sweep_summary.csv")))

    RI_ARCHS = ["riconv", "riconvpp", "risurconv"]
    FOCUS_ARCH = "pointnetpp"   # Sec 6's crowned "best overall model" on ModelNet40-C
    AXES = ["yaw", "tilt", "roll", "so3"]

    acc = defaultdict(dict)  # (arch, axis) -> {angle: accuracy}
    for r in rows:
        if r["condition"] != "invreg":
            continue
        acc[(r["arch"], r["axis"])][int(r["angle_deg"])] = float(r["accuracy"])

    angles = sorted(set(int(r["angle_deg"]) for r in rows))

    report = {}
    for axis in AXES:
        ri_best = {a: max(acc[(ra, axis)][a] for ra in RI_ARCHS) for a in angles}
        focus = {a: acc[(FOCUS_ARCH, axis)][a] for a in angles}
        gaps = {a: round(ri_best[a] - focus[a], 4) for a in angles}

        # Sustained-crossover definition: first angle (scanning outward from 0)
        # after which RI leads the focus arch by >= 0.05 for at least 3
        # consecutive sampled angles (10-degree steps) -- a deliberately lenient
        # threshold given n=30/angle-cell sampling noise (SE ~ 0.05-0.09 per
        # cell at p~0.9).
        crossover = None
        for i in range(len(angles) - 2):
            a0, a1, a2 = angles[i], angles[i + 1], angles[i + 2]
            if gaps[a0] >= 0.05 and gaps[a1] >= 0.05 and gaps[a2] >= 0.05:
                crossover = a0
                break

        # Trough: worst angle for the focus arch, and the RI-lead there.
        worst_angle = min(angles, key=lambda a: focus[a])
        max_gap_angle = max(angles, key=lambda a: gaps[a])

        report[axis] = {
            "crossover_deg": crossover,
            "worst_angle_for_focus": worst_angle,
            "focus_acc_at_worst": focus[worst_angle],
            "ri_best_acc_at_worst": ri_best[worst_angle],
            "max_gap_deg": max_gap_angle,
            "max_gap": gaps[max_gap_angle],
            "focus_acc_at_0": focus[0],
            "ri_best_acc_at_0": ri_best[0],
        }

    print(json.dumps(report, indent=2))
    with open(ROOT / "results/rotation_sweep/crossover_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Also dump the full per-angle table for a couple of exemplar archs (for the figure)
    exemplars = ["riconvpp", "pointnetpp", "dgcnn", "pointmlp", "pointnet"]
    curve_data = {axis: {a: {ang: acc[(a, axis)][ang] for ang in angles} for a in exemplars}
                  for axis in AXES}
    with open(ROOT / "results/rotation_sweep/yaw_curve_data.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    print("Saved crossover_report.json and yaw_curve_data.json")


def analyze_sweep_pose270(root=None):
    """Same curve-data extraction as analyze_sweep(), run against the
    pose270 sweep (results/rotation_sweep_pose270/rotation_sweep_summary.csv,
    condition=="pose270") instead of the original invreg sweep. Produces
    pose270_curve_data.json alongside yaw_curve_data.json so the pose270
    figure can plot them on the same axes.

    RIConv++'s curve is carried over from yaw_curve_data.json rather than
    re-extracted here: RIConv++ wasn't retrained under pose270 (the RI family
    doesn't need pose augmentation, its invariance is architectural), so its
    accuracy curve is identical in both sweeps -- same checkpoint, same
    condition=="invreg" rows. Re-using it keeps the reference ceiling anchored
    to real data instead of a second copy that could silently drift out of
    sync with yaw_curve_data.json."""
    ROOT = Path(root) if root is not None else _ROOT
    rows = list(csv.DictReader(
        open(ROOT / "results/rotation_sweep_pose270/rotation_sweep_summary.csv")))

    AXES = ["yaw", "tilt", "roll", "so3"]
    EXEMPLARS = ["pointnet", "dgcnn", "pointmlp", "pointnetpp", "pointtransformer"]

    acc = {}
    for r in rows:
        if r["condition"] != "pose270":
            continue
        acc[(r["arch"], r["axis"], int(r["angle_deg"]))] = float(r["accuracy"])

    angles = sorted(set(int(r["angle_deg"]) for r in rows))

    curve_data = {axis: {a: {ang: acc[(a, axis, ang)] for ang in angles} for a in EXEMPLARS}
                  for axis in AXES}

    # Carry over RIConv++'s invreg curve from the yaw sweep as the RI-family
    # reference ceiling (see the module docstring for why).
    yaw_curve_data = json.load(open(ROOT / "results/rotation_sweep/yaw_curve_data.json"))
    for axis in AXES:
        curve_data[axis]["riconvpp"] = yaw_curve_data[axis]["riconvpp"]

    out_path = ROOT / "results/rotation_sweep_pose270/pose270_curve_data.json"
    with open(out_path, "w") as f:
        json.dump(curve_data, f, indent=2)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None,
                        help="Directory holding results/rotation_sweep*/ "
                             "(default: $PCROBUST_HOME, i.e. ~/.cache/pcrobust).")
    parser.add_argument("--pose270", action="store_true",
                        help="Run the pose270 curve-data extraction instead of "
                             "the crossover analysis.")
    args = parser.parse_args()
    root = Path(args.root) if args.root else None
    if args.pose270:
        analyze_sweep_pose270(root)
    else:
        analyze_sweep(root)


if __name__ == "__main__":
    main()
