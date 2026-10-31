"""
Severity-scaling figure, 4 variants: {pose+noise, pose+noise+occlusion} x
{pooled across all 5 domains, ModelNet40 only}.

Reads the raw per-instance similarity CSVs from the benchmark results tree
under PCROBUST_HOME (``experiments/<pert>_robustness/<domain>/results/``,
the layout :mod:`pcrobust.paper.tables` also reads) and writes PNGs plus a
small aggregated CSV to ``<PCROBUST_HOME>/results/figures/severity_curves/``.

Usage: python -m pcrobust.evaluate.severity
"""
import csv
from collections import defaultdict
import numpy as np

from pcrobust.datasets import pcrobust_home, results_dir

ROOT = pcrobust_home()
OUT = results_dir("figures", "severity_curves")


def _pyplot():
    """matplotlib is an optional extra (pip install pcrobust[figures])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

DOMAINS = ["modelnet40", "shapenet", "scanobjectnn", "abo", "s3dis"]
RI_FAMILY = ["riconv", "riconvpp", "risurconv"]
NONRI_FAMILY = ["pointnet", "pointnetpp", "dgcnn", "pointmlp", "pointtransformer"]
ALL_ARCHS = RI_FAMILY + NONRI_FAMILY

PERTURBATIONS = {
    "pose": {
        "dir": "pose_robustness", "file_prefix": "confirmatory_raw_",
        "level_col": "angle_deg", "levels": [20, 40, 60, 80, 120],
        "xlabel": "Rotation angle (deg)",
    },
    "noise": {
        "dir": "noise_robustness", "file_prefix": "noise_raw_",
        "level_col": "sigma", "levels": [0.005, 0.01, 0.02, 0.03, 0.05],
        "xlabel": "Noise sigma",
    },
    "occlusion": {
        "dir": "occlusion_robustness", "file_prefix": "occlusion_raw_",
        "level_col": "dist_mult", "levels": [0.3, 0.6, 1.2, 2.5, 6.0],
        "xlabel": "Camera distance multiplier",
    },
}


def load_domain_arch(pert_key, domain, arch):
    """Returns {level: [similarity, ...]} for one (perturbation, domain, arch)."""
    cfg = PERTURBATIONS[pert_key]
    path = ROOT / "experiments" / cfg["dir"] / domain / "results" / f"{cfg['file_prefix']}{arch}.csv"
    if not path.exists():
        return None
    by_level = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            level = float(row[cfg["level_col"]])
            by_level[level].append(float(row["similarity"]))
    return by_level


def compute_all():
    """Returns raw[pert][domain][arch][level] = mean similarity (scalar)."""
    raw = {}
    for pert_key in PERTURBATIONS:
        raw[pert_key] = {}
        for domain in DOMAINS:
            raw[pert_key][domain] = {}
            for arch in ALL_ARCHS:
                by_level = load_domain_arch(pert_key, domain, arch)
                if by_level is None:
                    print(f"  [skip] {pert_key}/{domain}/{arch}: no raw CSV")
                    continue
                raw[pert_key][domain][arch] = {
                    lvl: float(np.mean(vals)) for lvl, vals in by_level.items()
                }
    return raw


def family_curve_pooled(raw, pert_key, family):
    """Pool across all domains: for each severity level, mean + standard
    deviation (sigma_D) across the 5 domain-level family means (each
    domain-level mean = mean over architectures in family and their
    instances in that domain)."""
    cfg = PERTURBATIONS[pert_key]
    means, bands = [], []
    for lvl in cfg["levels"]:
        domain_means = []
        for domain in DOMAINS:
            arch_vals = [raw[pert_key][domain][a][lvl] for a in family
                         if a in raw[pert_key][domain] and lvl in raw[pert_key][domain][a]]
            if arch_vals:
                domain_means.append(np.mean(arch_vals))
        means.append(np.mean(domain_means))
        bands.append(np.std(domain_means, ddof=1) if len(domain_means) > 1 else 0.0)
    return np.array(means), np.array(bands)


def family_curve_single_domain(raw, pert_key, family, domain):
    """Single domain: for each severity level, mean + standard deviation
    (sigma_D) across architectures in the family (each arch's own mean
    over its instances in that domain)."""
    cfg = PERTURBATIONS[pert_key]
    means, bands = [], []
    for lvl in cfg["levels"]:
        arch_vals = [raw[pert_key][domain][a][lvl] for a in family
                     if a in raw[pert_key][domain] and lvl in raw[pert_key][domain][a]]
        means.append(np.mean(arch_vals))
        bands.append(np.std(arch_vals, ddof=1) if len(arch_vals) > 1 else 0.0)
    return np.array(means), np.array(bands)


def plot_panel(ax, pert_key, ri_curve, nonri_curve, x_is_log=False):
    cfg = PERTURBATIONS[pert_key]
    x = np.array(cfg["levels"])
    (ri_m, ri_b), (nr_m, nr_b) = ri_curve, nonri_curve
    # drift = 1 - similarity, "lower is better", consistent with the paper's mean_D convention
    ri_d, nr_d = 1 - ri_m, 1 - nr_m
    ax.plot(x, ri_d, "o-", color="#c1440e", label="RI family")
    ax.fill_between(x, ri_d - ri_b, ri_d + ri_b, color="#c1440e", alpha=0.18)
    ax.plot(x, nr_d, "s-", color="#2b6cb0", label="Non-RI family")
    ax.fill_between(x, nr_d - nr_b, nr_d + nr_b, color="#2b6cb0", alpha=0.18)
    if x_is_log:
        ax.set_xscale("log")
        ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in cfg["levels"]])
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("1 - similarity to clean")
    ax.set_title(pert_key.capitalize())
    ax.grid(alpha=0.25)


def make_figure(raw, perts, domain_scope, out_name):
    n = len(perts)
    plt = _pyplot()
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0))
    if n == 1:
        axes = [axes]
    for ax, pert_key in zip(axes, perts):
        if domain_scope == "pooled":
            ri_curve = family_curve_pooled(raw, pert_key, RI_FAMILY)
            nonri_curve = family_curve_pooled(raw, pert_key, NONRI_FAMILY)
        else:
            ri_curve = family_curve_single_domain(raw, pert_key, RI_FAMILY, "modelnet40")
            nonri_curve = family_curve_single_domain(raw, pert_key, NONRI_FAMILY, "modelnet40")
        x_log = pert_key == "noise"
        plot_panel(ax, pert_key, ri_curve, nonri_curve, x_is_log=x_log)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06),
               frameon=False, fontsize=10)
    scope_title = r"pooled across all 5 domains (band = standard deviation $\sigma_D$ across domains)" if domain_scope == "pooled" \
        else r"ModelNet40 only (band = standard deviation $\sigma_D$ across architectures in family)"
    fig.suptitle(f"Severity scaling, {scope_title}", y=1.14, fontsize=11)
    fig.tight_layout()
    out_path = OUT / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[severity_curves] loading raw per-instance similarity for all "
          f"{len(ALL_ARCHS)} architectures x {len(DOMAINS)} domains x "
          f"{len(PERTURBATIONS)} perturbations ...")
    raw = compute_all()

    variants = [
        (["pose", "noise"], "pooled", "severity_pose_noise_pooled.png"),
        (["pose", "noise"], "modelnet40", "severity_pose_noise_modelnet40.png"),
        (["pose", "noise", "occlusion"], "pooled", "severity_pose_noise_occlusion_pooled.png"),
        (["pose", "noise", "occlusion"], "modelnet40", "severity_pose_noise_occlusion_modelnet40.png"),
    ]
    for perts, scope, name in variants:
        make_figure(raw, perts, scope, name)

    # also dump the raw aggregated numbers so the paper text/caption can cite exact values
    csv_path = OUT / "severity_curve_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "level", "family", "scope", "mean_similarity", "sd_band"])
        for pert_key in PERTURBATIONS:
            for family, fname in [(RI_FAMILY, "RI"), (NONRI_FAMILY, "non-RI")]:
                m_pool, b_pool = family_curve_pooled(raw, pert_key, family)
                m_m40, b_m40 = family_curve_single_domain(raw, pert_key, family, "modelnet40")
                for i, lvl in enumerate(PERTURBATIONS[pert_key]["levels"]):
                    w.writerow([pert_key, lvl, fname, "pooled", round(float(m_pool[i]), 5), round(float(b_pool[i]), 5)])
                    w.writerow([pert_key, lvl, fname, "modelnet40", round(float(m_m40[i]), 5), round(float(b_m40[i]), 5)])
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
