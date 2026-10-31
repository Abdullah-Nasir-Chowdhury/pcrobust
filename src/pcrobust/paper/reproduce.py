"""Tier-1 reproduction: rebuild the paper's tables and figures from the
frozen result data bundled in :mod:`pcrobust.paper` -- offline, no GPU,
no downloads.

Every target is named by its LaTeX label in the paper source::

    from pcrobust.paper import reproduce
    reproduce.list_targets()                    # every table and figure
    df = reproduce.build("tab:noise")           # rows/columns as printed
    report = reproduce.compare("tab:noise")     # vs. the printed numbers
    reproduce.plot("fig:teaser", "teaser.png")  # needs the [figures] extra

A target's frame is laid out like the printed table: one row per printed
row (indexed by the same key ``compare`` reads from the paper, e.g.
``"PointNet|no"``) and one column per printed column. A cell is a float,
a tuple of floats when the paper prints several numbers in one cell
(``0.143 [0.130, 0.157]``), or ``None`` where the bundled data does not
cover it. Figure targets are long frames indexed ``"series|x"`` holding
the plotted values in column ``y``.

``compare`` checks each number against the snapshot of the paper source
in ``pcrobust/paper/expected/`` (written by ``tools/sync_paper.py``): a
rebuilt number matches when the paper's printed value is a correct
rounding of it.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING

from pcrobust import paper
from pcrobust.paper.latex import matches, numbers

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "Comparison",
    "Target",
    "build",
    "compare",
    "expected",
    "get_target",
    "list_targets",
    "plot",
    "to_csv",
]

ARCHS = ["pointnet", "pointnetpp", "dgcnn", "riconv", "riconvpp", "risurconv",
         "pointmlp", "pointtransformer", "vn_pointnet", "vn_dgcnn"]
STANDARD = ["pointnet", "pointnetpp", "dgcnn", "pointmlp", "pointtransformer"]
RI = ["riconv", "riconvpp", "risurconv"]
VN = ["vn_pointnet", "vn_dgcnn"]
DOMAINS = ["modelnet40", "shapenet", "abo", "scanobjectnn", "s3dis"]
# Short names the paper's tables print (registry display names are longer).
LABEL = {
    "pointnet": "PointNet", "pointnetpp": "PointNet++", "dgcnn": "DGCNN",
    "riconv": "RIConv", "riconvpp": "RIConv++", "risurconv": "RISurConv",
    "pointmlp": "PointMLP", "pointtransformer": "PointTransf.",
    "vn_pointnet": "VN-PointNet", "vn_dgcnn": "VN-DGCNN",
}
POSE_AXES = ["yaw", "tilt", "roll"]


@dataclass(frozen=True)
class Target:
    id: str
    kind: str                       # "table" | "figure"
    title: str
    where: str                      # "Sec. 4", "Supp."
    build: Callable[[], pd.DataFrame]
    key_cols: int = 1               # leading printed columns that form the row key
    draw: Callable | None = None    # figures: (frame, matplotlib Figure) -> None
    missing: str = ""               # what the bundled data does not cover
    tolerance: float | None = None  # absolute; None = the paper's printed precision

    @property
    def short(self) -> str:
        return self.id.split(":", 1)[1]


@dataclass
class Comparison:
    target: str
    checked: int = 0
    matched: int = 0
    mismatches: list[tuple[str, int, str, str]] = field(default_factory=list)
    uncovered: list[tuple[str, int, str]] = field(default_factory=list)
    paper_commit: str = ""

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        line = (f"{self.target}: {self.matched}/{self.checked} printed numbers "
                f"reproduced")
        if self.uncovered:
            line += f", {len(self.uncovered)} cells not covered by bundled data"
        return line


# ── data helpers ──────────────────────────────────────────────────────────────

# Builders share the bundled frames read-only, so each file is parsed once.
_load = functools.lru_cache(maxsize=None)(paper.load)

def _frame(rows: dict, columns: list[str]):
    import pandas as pd

    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns).astype(object)
    return df.where(df.notna(), None)


def _t_ci(mean: float, sd: float, n: int) -> tuple[float, float, float]:
    from scipy import stats

    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def _key(x) -> str:
    """Canonical text for a numeric x in a figure key (``120``, ``0.005``)."""
    v = float(x)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def _drift_cells(pert: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    """(arch, domain) -> (mean D, 95% CI low, high) as the paper reports them.

    ModelNet40 comes from replicate 1 of the seed study (all 10 archs); the
    other domains from the original study (8 archs) and the VN cross-domain
    runs.
    """
    stats_name = {"noise": "noise", "occlusion": "occlusion"}.get(pert, f"pose_{pert}")
    out = {}
    rep1 = _load(f"seeds/seed_stats/{stats_name}_stats_rep1.csv")
    for r in rep1[rep1.arch.isin(ARCHS)].itertuples():
        out[(r.arch, "modelnet40")] = _t_ci(r.mean_degradation, r.sd_degradation, r.n)
    if pert in POSE_AXES:
        prim = _load("summary/pose_primary_test.csv")
        prim = prim[prim.Rotation_Type == pert]
    else:
        prim = _load(f"summary/{pert}_primary_test.csv")
        if "Camera_Variant" in prim:
            prim = prim[prim.Camera_Variant == "random"]
    for r in prim[prim.Domain != "modelnet40"].itertuples():
        out[(r.Architecture, r.Domain)] = (r.mean_D, r.ci_low, r.ci_high)
    for arch in VN:
        cells = _load(f"vn/{arch}_cells.csv")
        for r in cells[(cells.pert == pert) & (cells.domain != "modelnet40")].itertuples():
            out[(arch, r.domain)] = (r.mean_D, r.ci_lo, r.ci_hi)
    return out


def _bootstrap_rep1():
    df = _load("bootstrap/bootstrap_summary_rep1.csv")
    return (df[df.metric_set == "absolute"].set_index("model"),
            df[df.metric_set == "delta_invreg"].set_index("model"))


def _m40c_metrics(model: str, ref: str) -> tuple[float, float, float]:
    """ModelNet40-C clean accuracy, mean corrupted accuracy, and mCE.

    mCE is the mean over corruption types of the model's error summed over
    the five severities, divided by the reference model's.
    """
    df = _load("modelnet40c/m40c_by_type.csv")
    cells = df[df.model == model].set_index(["corruption", "severity"]).accuracy
    base = df[df.model == ref].set_index(["corruption", "severity"]).accuracy
    corrupted = cells.drop("none", level=0)
    err = (1 - corrupted).groupby(level=0).sum()
    ref_err = (1 - base.drop("none", level=0)).groupby(level=0).sum()
    return (float(cells.loc["none"].mean()), float(corrupted.mean()),
            float((err / ref_err).mean()))


def _accuracy(frame, condition: str, level=None) -> float:
    sel = frame[frame.condition == condition]
    if level is not None:
        sel = sel[(sel.level - level).abs() < 1e-9]
    return float(sel.accuracy.iloc[0])


# ── table builders ────────────────────────────────────────────────────────────

def _domains():
    from pcrobust.datasets import abo, modelnet40, s3dis, scanobjectnn, shapenet

    rows = {}
    for name, mod in [("ModelNet40", modelnet40), ("ShapeNet", shapenet),
                      ("ScanObjectNN", scanobjectnn), ("ABO", abo), ("S3DIS", s3dis)]:
        k = len(mod.CLASSES)
        g, q = k * mod.N_TRAIN_PER_CLASS, k * mod.N_TEST_PER_CLASS
        rows[name] = [k, g, q, g + q - k]
    return _frame(rows, ["classes", "gallery", "query", "confirm_pool"])


def _drift_table(pert: str, with_ci: bool):
    cells = _drift_cells(pert)
    rows = {}
    for arch in ARCHS:
        row = []
        for d in DOMAINS:
            c = cells.get((arch, d))
            row.append(None if c is None else (c if with_ci else c[0]))
        rows[LABEL[arch]] = row
    return _frame(rows, DOMAINS)


def _m40c():
    rows = {}
    for suffix, loss in [("", "no"), ("_invreg", "yes")]:
        for arch in ARCHS:
            clean, corr, mce = _m40c_metrics(f"{arch}{suffix}_rep1", "dgcnn_rep1")
            rows[f"{LABEL[arch]}|{loss}"] = [clean, corr, mce, clean - corr]
    return _frame(rows, ["clean", "corrupted", "mCE", "gap"])


def _bootstrap_ci():
    absolute, _ = _bootstrap_rep1()
    rows = {}
    for suffix, loss in [("", "no"), ("_invreg", "yes")]:
        for arch in ARCHS:
            r = absolute.loc[arch + suffix]
            mce = 1.0 if arch + suffix == "dgcnn" else (r.mce, r.mce_lo, r.mce_hi)
            rows[f"{LABEL[arch]}|{loss}"] = [(r.clean, r.clean_lo, r.clean_hi),
                                             (r.corr_acc, r.corr_lo, r.corr_hi), mce]
    return _frame(rows, ["clean", "corrupted", "mCE"])


def _bootstrap_delta():
    _, delta = _bootstrap_rep1()
    rows = {LABEL[a]: [(delta.loc[a].corr_acc, delta.loc[a].corr_lo, delta.loc[a].corr_hi),
                       (delta.loc[a].mce, delta.loc[a].mce_lo, delta.loc[a].mce_hi)]
            for a in ARCHS}
    return _frame(rows, ["delta_corrupted", "delta_mCE"])


def _type_means():
    """model -> per-corruption-type accuracy, averaged over the 5 severities."""
    df = _load("modelnet40c/m40c_by_type.csv")
    return df[df.corruption != "none"].groupby(["model", "corruption"]).accuracy.mean().unstack()


def _type_deltas(arm: str = "invreg"):
    """Replicate-1 baseline accuracy and arm-minus-baseline delta, per type."""
    t = _type_means()
    base = t.loc[[f"{a}_rep1" for a in ARCHS]].set_axis(ARCHS)
    name = "{a}_invreg_rep1" if arm == "invreg" else "{a}_" + arm
    models = [name.format(a=a) for a in ARCHS]
    if not all(m in t.index for m in models):
        return base, None
    return base, t.loc[models].set_axis(ARCHS) - base


_TYPE_ORDER = ["lidar", "occlusion", "gaussian", "cutout", "rotation", "background",
               "impulse", "upsampling", "uniform", "density", "density_inc",
               "distortion_rbf", "distortion_rbf_inv", "shear", "distortion"]


def _transfer():
    base, delta = _type_deltas()
    rows = {c: [delta[c].mean(), base[c].mean()] for c in _TYPE_ORDER}
    return _frame(rows, ["delta_acc", "base_acc"])


def _w4_types():
    _, aug = _type_deltas("so3aug")
    _, loss = _type_deltas()

    def cell(delta, c):
        return None if delta is None else (delta[c].mean(), float((delta[c] > 0).sum()))

    rows = {c: [cell(aug, c), cell(loss, c)] for c in _TYPE_ORDER}
    return _frame(rows, ["delta_aug_only", "delta_loss"])


def _sweeps():
    """(arch -> {(axis, angle): accuracy}) for the n=300 rotation sweep curves."""
    curves: dict[str, dict] = {}
    sources = [
        ("rotation_sweep/rotation_sweep_n300/rotation_sweep_summary_n300.csv", "pose270", STANDARD),
        ("rotation_sweep/rotation_sweep_n300_rep1/ri_invreg_summary.csv", "invreg", RI),
        ("rotation_sweep/rotation_sweep_n300_rep1/vn_invreg_summary.csv", "invreg", VN),
    ]
    for path, condition, archs in sources:
        df = _load(path)
        for r in df[(df.condition == condition) & df.arch.isin(archs)].itertuples():
            curves.setdefault(r.arch, {})[(r.axis, int(r.angle_deg))] = r.accuracy
    return curves


def _rotation_gap():
    curves = _sweeps()
    ref = curves["riconvpp"]
    rows = {}
    for arch in [*STANDARD, *VN]:
        gaps = [ref[cell] - acc for cell, acc in curves[arch].items()]
        # Plain float comparison, as the paper computed it: many gaps are
        # exactly 15/300 = 0.05, and float error decides which of those count.
        rows[LABEL[arch]] = [sum(gaps) / len(gaps), max(gaps),
                             sum(g >= 0.05 for g in gaps) / len(gaps)]
    return _frame(rows, ["mean_gap", "widest_gap", "frac_cells_ge_0.05"])


def _yaw_gap():
    curves = _sweeps()
    ref, other = curves["riconvpp"], curves["pointnetpp"]
    rows = {}
    for i in range(12):
        row = []
        for block in range(3):
            angle = block * 120 + i * 10
            row += [float(angle), ref[("yaw", angle)] - other[("yaw", angle)]]
        rows[str(i)] = row
    return _frame(rows, ["angle", "gap", "angle_2", "gap_2", "angle_3", "gap_3"])


def _did():
    rows = {}
    perts = ["clean", "pose", "noise", "occlusion"]
    full = _load("did/did_summary_rep1.csv").set_index(["row", "pert"])
    no_ris = _load("did/did_summary_rep1_no_risurconv.csv").set_index(["row", "pert"])
    for arch in [*STANDARD, *RI, *VN]:
        rows[LABEL[arch]] = [full.loc[(f"delta_{arch}", p), "value"] for p in perts]
    for archs, label in [(STANDARD, "Mean, standard"), (RI, "Mean, RI")]:
        rows[label] = [sum(full.loc[(f"delta_{a}", p), "value"] for a in archs) / len(archs)
                       for p in perts]
    for frame, label in [(full, "DiD"), (no_ris, "DiD w/o RISur.")]:
        rows[label] = [frame.loc[("DiD", p), "value"] for p in perts]
        rows[f"{label}|95% CI"] = [(frame.loc[("DiD", p), "lo"], frame.loc[("DiD", p), "hi"])
                                   for p in perts]
        rows[f"{label}|perm. p"] = [frame.loc[("DiD", p), "p_perm"] for p in perts]
    return _frame(rows, perts)


def _lambda():
    rows = {}
    for lam, tag in [("0.1", "0p1"), ("0.25", "0p25"), ("0.5", "0p5"), ("1.0", "1p0"),
                     ("2.0", "2p0")]:
        row = []
        for arch in ["pointnetpp", "pointnet", "riconv"]:
            acc = _load(f"lambda/acc_{arch}_lam{tag}.csv")
            drift = _load(f"lambda/drift_{arch}_lam{tag}.csv").set_index("perturbation")
            row += [_accuracy(acc, "clean"), _accuracy(acc, "noise", 0.05),
                    drift.loc["noise", "mean_D"]]
        rows[lam] = row
    cols = [f"{a}_{m}" for a in ["pointnetpp", "pointnet", "riconv"]
            for m in ["clean", "noise05", "D_noise"]]
    return _frame(rows, cols)


def _cost():
    df = _load("cost/model_cost.csv").set_index("arch")
    order = ["pointnet", "pointnetpp", "dgcnn", "pointmlp", "pointtransformer", *RI, *VN]
    rows = {LABEL[a]: [df.loc[a, "params_M"], df.loc[a, "flops_G"], None] for a in order}
    return _frame(rows, ["params_M", "flops_G", "latency_ms"])


def _w2():
    df = _load("w2/w2_drift_vs_accuracy.csv").set_index(["perturbation", "architecture"])
    rows = {}
    for arch in ["pointnet", "pointnetpp", "dgcnn", *RI, "pointmlp", "pointtransformer"]:
        rows[LABEL[arch]] = [df.loc[(p, arch), c] for p in ["pose", "noise", "occlusion"]
                             for c in ["mean_D", "acc_delta"]]
    return _frame(rows, [f"{p}_{c}" for p in ["pose", "noise", "occlusion"]
                         for c in ["D", "delta_acc"]])


def _w4():
    # The aug-only arm ships as a summary: its per-type results for the RI
    # and VN models are not in the bundled ModelNet40-C cells.
    aug = _load("seeds/seed_stats/w4_summary.csv").set_index("arch")
    order = ["pointnet", "pointnetpp", "dgcnn", "pointmlp", "pointtransformer", *RI, *VN]
    metrics = {}
    for a in order:
        base = _m40c_metrics(f"{a}_rep1", "dgcnn_rep1")
        loss = _m40c_metrics(f"{a}_invreg_rep1", "dgcnn_rep1")
        so3 = [aug.loc[a, f"so3aug_{m}"] for m in ["clean", "corr", "mce"]]
        metrics[a] = [x for i in range(3) for x in (base[i], so3[i], loss[i])]
    rows = {LABEL[a]: metrics[a] for a in order}
    rows["Mean"] = [sum(metrics[a][j] for a in order) / len(order) for j in range(9)]
    return _frame(rows, [f"{m}_{arm}" for m in ["clean", "corr", "mce"]
                         for arm in ["base", "aug", "loss"]])


def _w5():
    rows = {}
    for arch in ["pointnetpp", "riconvpp"]:
        for model, loss in [(f"{arch}_w5", "no"), (f"{arch}_invreg_modelnet40b_w5", "yes")]:
            clean, corr, _ = _m40c_metrics(model, model)
            rows[f"{LABEL[arch]}|{loss}"] = [clean, corr]
    return _frame(rows, ["clean", "corrupted"])


def _seeds():
    import statistics

    rows = {}
    for suffix, loss in [("", "no"), ("_invreg", "yes")]:
        for arch in ARCHS:
            # each replicate's mCE is normalized to that replicate's DGCNN baseline
            reps = [_m40c_metrics(f"{arch}{suffix}_rep{k}", f"dgcnn_rep{k}") for k in (1, 2, 3)]
            cells = [(statistics.mean(v), statistics.stdev(v)) for v in zip(*reps)]
            if arch + suffix == "dgcnn":
                cells[2] = 1.0
            rows[f"{LABEL[arch]}|{loss}"] = cells
    return _frame(rows, ["clean", "corrupted", "mCE"])


def _warmup():
    # The original single-seed study, mCE against its own DGCNN baseline.
    rows = {}
    order = ["pointnet", "pointnetpp", "dgcnn", *RI, "pointmlp", "pointtransformer"]
    for arch in order:
        loss = _m40c_metrics(f"{arch}_invreg", "dgcnn")
        warm = _m40c_metrics(f"{arch}_invreg_warmup", "dgcnn")
        rows[LABEL[arch]] = [x for i in range(3) for x in (loss[i], warm[i])]
    return _frame(rows, ["clean_loss", "clean_warmup", "corr_loss", "corr_warmup",
                         "mce_loss", "mce_warmup"])


def _canon():
    rows = {}
    conds = [("clean", None), ("pose", 120.0), ("noise", 0.05), ("occlusion", 6.0)]
    for arch in ARCHS:
        df = _load(f"canonicalization/canon_summary_{arch}.csv")
        df = df[df["head"] == "ncm"]
        std, canon = df[df.arm == "std"], df[df.arm == "canon"]
        rows[LABEL[arch]] = [(_accuracy(std, c, lv), _accuracy(canon, c, lv)) for c, lv in conds]
    return _frame(rows, ["clean", "pose120", "noise05", "occl6"])


_PROBE_HEADS = [("head", "trained"), ("sm", "subspace"), ("knn", "kNN"), ("ncm", "NCM"),
                ("ncm_eucl", "NCM (Eucl.)")]
_PROBE_CONDS = [("clean", None), ("noise", 0.05), ("occlusion", 6.0), ("pose", 120.0)]


@functools.cache
def _probe_acc():
    """Replicate 1 of the repaired multi-rep probe run, as the paper reports it."""
    df = _load("probe/probe_acc_seeds.csv")
    df = df[df.run == "rep1"]
    return {(r.arch, r.head, r.condition, None if math.isnan(r.level) else round(r.level, 6)):
            r.accuracy for r in df.itertuples()}


def _probe(archs: list[str]):
    acc = _probe_acc()
    rows = {}
    for arch in archs:
        for suffix, tag in [("", ""), ("_invreg", " (inv.)")]:
            for head, label in _PROBE_HEADS:
                rows[f"{LABEL[arch]}{tag}|{label}"] = [
                    acc[(arch + suffix, head, c, lv)] for c, lv in _PROBE_CONDS]
    return _frame(rows, ["clean", "noise05", "occl6", "pose120"])


def _probe_delta():
    acc = _probe_acc()
    heads = ["head", "sm", "knn", "ncm"]
    rows = {}
    for arch in [*STANDARD, *RI, *VN]:
        rows[LABEL[arch]] = [100 * (acc[(f"{arch}_invreg", h, c, lv)] - acc[(arch, h, c, lv)])
                             for c, lv in [("occlusion", 6.0), ("noise", 0.05)] for h in heads]
    return _frame(rows, [f"{c}_{h}" for c in ["occl6", "noise05"] for h in heads])


def _probe_pose():
    acc = _probe_acc()
    rows = {LABEL[a]: [100 * (acc[(f"{a}_invreg", h, "pose", 120.0)] - acc[(a, h, "pose", 120.0)])
                       for h in ["head", "sm", "knn", "ncm"]] for a in [*RI, *VN]}
    return _frame(rows, ["head", "sm", "knn", "ncm"])


# ── figure builders ───────────────────────────────────────────────────────────

def _long(points: dict[str, float]):
    return _frame({k: [v] for k, v in points.items()}, ["y"])


def _teaser():
    pose = {a: [] for a in ARCHS}
    for axis in POSE_AXES:
        for (arch, _d), cell in _drift_cells(axis).items():
            pose[arch].append(cell[0])
    noise = _drift_cells("noise")
    pts = {}
    for a in ARCHS:
        pts[f"{a}|pose"] = sum(pose[a]) / len(pose[a])
        pts[f"{a}|noise"] = sum(noise[(a, d)][0] for d in DOMAINS) / len(DOMAINS)
    return _long(pts)


def _severity():
    df = _load("severity/severity_domain_curves.csv")
    return _long({f"{r.perturbation}|{r.family}|{r.domain}|{_key(r.level)}": r.mean_similarity
                  for r in df.itertuples()})


def _severity_resolved():
    df = _load("modelnet40c/m40c_by_type.csv")
    df = df[df.corruption.isin(["lidar", "occlusion"])]
    p = df.pivot_table(index=["corruption", "severity"], columns="model", values="accuracy")
    pts = {}
    for (corr, sev), row in p.iterrows():
        base = [row[f"{a}_rep1"] for a in ARCHS]
        pts[f"gain|{corr}|{sev}"] = sum(row[f"{a}_invreg_rep1"] - row[f"{a}_rep1"]
                                       for a in ARCHS) / len(ARCHS)
        pts[f"headroom|{corr}|{sev}"] = 1 - sum(base) / len(base)
    return _long(pts)


def _rotation_sweep():
    curves = _sweeps()
    return _long({f"{arch}|{axis}|{angle}": acc
                  for arch in [*RI, *STANDARD] for (axis, angle), acc in curves[arch].items()})


# ── figure drawing (matplotlib, [figures] extra) ──────────────────────────────

_FAMILY_STYLE = {"standard": ("#0072B2", "o", "Standard"),
                 "rotation-invariant": ("#009E73", "s", "RI family"),
                 "equivariant": ("#E69F00", "^", "Equivariant")}


def _split(df):
    return [(k.split("|"), v) for k, v in df["y"].items()]


def _draw_teaser(df, fig):
    from pcrobust.registry import family_of

    ax = fig.add_subplot(111)
    vals = {}
    for (arch, what), v in _split(df):
        vals.setdefault(arch, {})[what] = v
    for fam, (color, marker, label) in _FAMILY_STYLE.items():
        archs = [a for a in ARCHS if family_of(a) == fam]
        ax.scatter([vals[a]["pose"] for a in archs], [vals[a]["noise"] for a in archs],
                   color=color, marker=marker, s=34, edgecolor="white", label=label, zorder=3)
        for a in archs:
            ax.annotate(LABEL[a], (vals[a]["pose"], vals[a]["noise"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.margins(x=0.15, y=0.08)
    ax.set_xlabel("Mean pose (yaw, tilt, roll) drift across 5 domains")
    ax.set_ylabel("Mean noise drift across 5 domains")
    ax.legend(frameon=False, loc="upper right")


def _draw_severity(df, fig):
    colors = {"non-RI": "#0072B2", "RI": "#009E73", "equivariant": "#E69F00"}
    series: dict = {}
    for (pert, fam, dom, level), v in _split(df):
        series.setdefault(pert, {}).setdefault(fam, {}).setdefault(dom, []).append((float(level), v))
    axes = fig.subplots(1, 3)
    for ax, pert in zip(axes, ["pose", "noise", "occlusion"]):
        for fam, doms in series[pert].items():
            curves = [sorted(pts) for pts in doms.values()]
            for c in curves:
                ax.plot([x for x, _ in c], [y for _, y in c], color=colors[fam], lw=0.5, alpha=0.5)
            xs = [x for x, _ in curves[0]]
            mean = [sum(c[i][1] for c in curves) / len(curves) for i in range(len(xs))]
            ax.plot(xs, mean, "-o", color=colors[fam], lw=1.6, ms=3, label=fam)
        ax.set_title(pert)
        ax.set_xlabel("severity level")
    axes[0].set_ylabel("mean cosine similarity")
    axes[0].legend(frameon=False, fontsize=7)


def _draw_severity_resolved(df, fig):
    series: dict = {}
    for (panel, corr, sev), v in _split(df):
        series.setdefault(panel, {}).setdefault(corr, []).append((int(sev), v))
    for ax, panel in zip(fig.subplots(1, 2, sharex=True), ["gain", "headroom"]):
        for corr, pts in series[panel].items():
            pts.sort()
            ax.plot([s for s, _ in pts], [v for _, v in pts], "-o", ms=3, label=corr)
        ax.set_title(panel)
        ax.set_xlabel("severity")
    fig.axes[0].legend(frameon=False)


def _draw_rotation_sweep(df, fig):
    series: dict = {}
    for (arch, axis, angle), v in _split(df):
        series.setdefault(axis, {}).setdefault(arch, []).append((int(angle), v))
    axes = fig.subplots(2, 2, sharex=True, sharey=True).flatten()
    for ax, axis in zip(axes, ["yaw", "tilt", "roll", "so3"]):
        for arch, pts in series[axis].items():
            pts.sort()
            xs, ys = [a for a, _ in pts], [v for _, v in pts]
            ax.plot(xs + [360], ys + [ys[0]], lw=1.5 if arch == "riconvpp" else 0.9,
                    color="#1a1a1a" if arch in RI else None,
                    ls={"riconv": "--", "risurconv": "-."}.get(arch, "-"), label=LABEL[arch])
        ax.set_title(axis)
        ax.set_xticks([0, 90, 180, 270, 360])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.08))


# ── registry ──────────────────────────────────────────────────────────────────

_TARGETS = [
    # The paper's scatter script averages the 3-decimal table values, so its
    # points sit up to 5e-4 from the unrounded means.
    Target("fig:teaser", "figure", "Pose/noise trade-off scatter", "Sec. 1", _teaser,
           draw=_draw_teaser, tolerance=1e-3),
    Target("tab:domains", "table", "Per-domain classes and split sizes", "Sec. 3", _domains),
    Target("tab:noise", "table", "Noise robustness, mean embedding drift D", "Sec. 4",
           lambda: _drift_table("noise", with_ci=False)),
    Target("fig:severity", "figure", "Severity scaling, pose/noise/occlusion", "Sec. 4",
           _severity, draw=_draw_severity),
    Target("tab:m40c", "table", "ModelNet40-C generalization, all 20 models", "Sec. 6",
           _m40c, key_cols=2),
    Target("tab:transfer", "table", "Invariance-loss effect by corruption type", "Sec. 6",
           _transfer),
    Target("fig:severity_resolved", "figure", "Severity-resolved transfer", "Sec. 6",
           _severity_resolved, draw=_draw_severity_resolved),
    Target("fig:rotation_sweep", "figure", "360-degree rotation sweep, pose270 backbones",
           "Sec. 7", _rotation_sweep, draw=_draw_rotation_sweep),
    Target("tab:rotation_gap", "table", "Accuracy deficit against RIConv++", "Sec. 7",
           _rotation_gap),
    Target("tab:supp-warmup", "table", "Warmup schedule ablation", "Supp.", _warmup),
    Target("tab:supp-w4", "table", "Augmentation-only control", "Supp.", _w4),
    Target("tab:supp-w4-types", "table", "Augmentation-only vs loss, by corruption type",
           "Supp.", _w4_types,
           missing="aug-only column: per-type results for the RI and VN aug-only runs "
                   "are not bundled"),
    Target("tab:supp-bootstrap", "table", "Bootstrap intervals for the ModelNet40-C table",
           "Supp.", _bootstrap_ci, key_cols=2),
    Target("tab:supp-bootstrap-delta", "table", "Paired invariance-loss deltas", "Supp.",
           _bootstrap_delta),
    Target("tab:supp-did", "table", "Architecture x training interaction (DiD)", "Supp.", _did),
    Target("tab:supp-lambda", "table", "Loss-weight ablation", "Supp.", _lambda),
    Target("tab:supp-seeds", "table", "Seed replication study", "Supp.", _seeds, key_cols=2),
    Target("tab:supp-w5", "table", "Disjoint class-subset check", "Supp.", _w5, key_cols=2),
    Target("tab:supp-yaw-gap", "table", "Per-angle yaw gap, RIConv++ vs PointNet++", "Supp.",
           _yaw_gap, key_cols=0),
    Target("tab:supp-pose", "table", "Pose (yaw) drift with 95% CIs", "Supp.",
           lambda: _drift_table("yaw", with_ci=True)),
    Target("tab:supp-pose-tilt", "table", "Pose (tilt) drift with 95% CIs", "Supp.",
           lambda: _drift_table("tilt", with_ci=True)),
    Target("tab:supp-pose-roll", "table", "Pose (roll) drift with 95% CIs", "Supp.",
           lambda: _drift_table("roll", with_ci=True)),
    Target("tab:supp-noise", "table", "Noise drift with 95% CIs", "Supp.",
           lambda: _drift_table("noise", with_ci=True)),
    Target("tab:supp-occlusion", "table", "Occlusion drift with 95% CIs", "Supp.",
           lambda: _drift_table("occlusion", with_ci=True)),
    Target("tab:supp-w2", "table", "Drift-accuracy commensurability", "Supp.", _w2),
    Target("tab:supp-canon", "table", "Test-time PCA canonicalization", "Supp.", _canon),
    Target("tab:probe-delta", "table", "Probe accuracy deltas, occlusion and noise", "Supp.",
           _probe_delta),
    Target("tab:supp-probe-pose", "table", "Probe accuracy deltas, pose 120 yaw", "Supp.",
           _probe_pose),
    Target("tab:supp-probe", "table", "Head-independent probe, standard backbones", "Supp.",
           lambda: _probe(STANDARD), key_cols=2),
    Target("tab:supp-probe2", "table", "Head-independent probe, RI and VN", "Supp.",
           lambda: _probe([*RI, *VN]), key_cols=2),
    Target("tab:supp-cost", "table", "Model cost", "Supp.", _cost,
           missing="latency column: the latency benchmark results are not bundled"),
]
_BY_ID = {t.id: t for t in _TARGETS}


def list_targets(kind: str | None = None) -> list[Target]:
    """Every reproducible target in paper order (optionally only tables/figures)."""
    return [t for t in _TARGETS if kind is None or t.kind == kind]


def get_target(name: str) -> Target:
    """Look a target up by label (``tab:noise``) or bare name (``noise``)."""
    if name in _BY_ID:
        return _BY_ID[name]
    hits = [t for t in _TARGETS if t.short == name]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"unknown target {name!r}; see `pcrobust reproduce --list`")


def build(name: str) -> pd.DataFrame:
    """Rebuild one target from the bundled frozen data."""
    return get_target(name).build()


# ── paper comparison ──────────────────────────────────────────────────────────

def _expected_root():
    return resources.files("pcrobust.paper") / "expected"


def paper_source() -> dict:
    """Which paper revision the expected snapshot was taken from."""
    node = _expected_root() / "_source.json"
    return json.loads(node.read_text(encoding="utf-8")) if node.is_file() else {}


def expected(name: str) -> pd.DataFrame:
    """The printed numbers for one target, as snapshotted from the paper source."""
    import pandas as pd

    t = get_target(name)
    node = _expected_root() / f"{t.id.replace(':', '_')}.csv"
    if not node.is_file():
        raise FileNotFoundError(f"no paper snapshot for {t.id}; run tools/sync_paper.py")
    with node.open("r", encoding="utf-8") as f:
        return pd.read_csv(f, dtype=str, keep_default_na=False).set_index("row")


def _values(cell) -> list[float] | None:
    if cell is None:
        return None
    if isinstance(cell, (tuple, list)):
        return [float(v) for v in cell]
    return [float(cell)]


def compare(name: str, rebuilt: pd.DataFrame | None = None) -> Comparison:
    """Check a rebuilt target against the numbers printed in the paper."""
    t = get_target(name)
    exp = expected(t.id)
    got = t.build() if rebuilt is None else rebuilt
    rep = Comparison(t.id, paper_commit=paper_source().get("commit", ""))
    for key, row in exp.iterrows():
        for j, text in enumerate(row.tolist()):
            printed = numbers(text)
            if not printed:
                continue
            cell = got.iat[got.index.get_loc(key), j] if key in got.index and j < got.shape[1] \
                else None
            vals = _values(cell)
            if vals is None:
                rep.uncovered.append((key, j, text))
                continue
            rep.checked += len(printed)
            if len(vals) != len(printed):
                rep.mismatches.append((key, j, text, _fmt_cell(cell, 5)))
                continue
            if t.tolerance is None:
                ok = [matches(p, v) for p, v in zip(printed, vals)]
            else:
                ok = [abs(float(p) - v) <= t.tolerance for p, v in zip(printed, vals)]
            rep.matched += sum(ok)
            if not all(ok):
                rep.mismatches.append((key, j, text, _fmt_cell(cell, 5)))
    return rep


# ── output ────────────────────────────────────────────────────────────────────

def _fmt_cell(cell, decimals: int = 4) -> str:
    vals = _values(cell)
    if vals is None:
        return "--"
    parts = [f"{v:.{decimals}f}" for v in vals]
    if len(parts) == 3:
        return f"{parts[0]} [{parts[1]}, {parts[2]}]"
    return " / ".join(parts)


def format_frame(df, decimals: int = 3) -> str:
    """Plain-text rendering of a target frame."""
    import pandas as pd

    shown = df.apply(lambda col: col.map(lambda c: _fmt_cell(c, decimals)))
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 200):
        return shown.to_string()


def to_csv(df, path) -> None:
    """Write a target frame with multi-number cells split into extra columns."""
    import pandas as pd

    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        cells = [_values(c) or [] for c in df[col]]
        width = max(1, *(len(v) for v in cells))
        suffixes = ["", "_lo", "_hi"] if width == 3 else ["", *(f"_{i}" for i in range(1, width))]
        for i, suffix in enumerate(suffixes):
            out[col + suffix] = [v[i] if i < len(v) else None for v in cells]
    out.index.name = "row"
    out.to_csv(path)


def plot(name: str, path, df=None) -> None:
    """Draw a figure target to ``path`` (needs matplotlib: ``pip install pcrobust[figures]``)."""
    t = get_target(name)
    if t.draw is None:
        raise ValueError(f"{t.id} is a table, not a figure")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = {"fig:severity": (7.0, 2.4), "fig:severity_resolved": (5.0, 2.2),
            "fig:rotation_sweep": (5.0, 4.4)}.get(t.id, (4.0, 3.4))
    fig = plt.figure(figsize=size)
    t.draw(t.build() if df is None else df, fig)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
