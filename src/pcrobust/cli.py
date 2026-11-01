"""The ``pcrobust`` command.

Own subcommands:

    pcrobust list models|corruptions|domains|checkpoints
    pcrobust corrupt IN --corruption pose|noise|occlusion --severity 1-5 [-o OUT]
    pcrobust prepare DOMAIN [--download]
    pcrobust paper list | pcrobust paper show NAME
    pcrobust reproduce [--list] | TARGET... | all [--out DIR] [--compare-paper]

Pass-through subcommands forward their remaining arguments unchanged to the
module that implements them (``pcrobust evaluate robustness --help`` is
``python -m pcrobust.evaluate.robustness --help``):

    pcrobust evaluate robustness|accuracy|modelnet40c|metrics|rotation-sweep|
                      probe|probe-eval|severity|severity-transfer ...
    pcrobust train baseline|invreg ...
    pcrobust stats pilot|power|confirmatory|analysis|cross-architecture|
                   seed-stats|probe-reps|review|bootstrap|did|tost ...
    pcrobust tables summary|occlusion|training-metrics|all
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys

PASS_THROUGH = {
    "evaluate": {
        "robustness": "pcrobust.evaluate.robustness",
        "accuracy": "pcrobust.evaluate.accuracy",
        "modelnet40c": "pcrobust.evaluate.modelnet40c",
        "metrics": "pcrobust.evaluate.metrics",
        "rotation-sweep": "pcrobust.evaluate.rotation_sweep",
        "probe": "pcrobust.evaluate.probe",
        "probe-eval": "pcrobust.evaluate.probe_eval",
        "severity": "pcrobust.evaluate.severity",
        "severity-transfer": "pcrobust.evaluate.severity_transfer",
    },
    "train": {
        "baseline": "pcrobust.training.train",
        "invreg": "pcrobust.training.invariance",
    },
    "stats": {
        "pilot": "pcrobust.stats.pilot",
        "power": "pcrobust.stats.power",
        "confirmatory": "pcrobust.stats.confirmatory",
        "analysis": "pcrobust.stats.analysis",
        "cross-architecture": "pcrobust.stats.cross_architecture",
        "seed-stats": "pcrobust.stats.seed_stats",
        "probe-reps": "pcrobust.stats.probe_reps",
        "review": "pcrobust.stats.review_summaries",
        "bootstrap": "pcrobust.stats.bootstrap",
        "did": "pcrobust.stats.did",
        "tost": "pcrobust.stats.tost",
    },
}

DOMAINS = ("modelnet40", "shapenet", "scanobjectnn", "abo", "s3dis")
CORRUPTIONS = ("pose", "noise", "occlusion")
_ALIASES = {"rotation": "pose"}


def _run_module(module_name: str, prog: str, argv: list[str]) -> int:
    """Call ``module.main`` with ``argv``, as if run via ``python -m``."""
    module = importlib.import_module(module_name)
    main = module.main
    # Some entry points read sys.argv themselves, so set it either way.
    sys.argv = [prog, *argv]
    if inspect.signature(main).parameters:
        main(argv)
    elif any(a in ("-h", "--help") for a in argv):
        # Argument-free entry point: its docstring is its help.
        print(f"usage: {prog}\n\n{(module.__doc__ or '').strip()}")
    else:
        main()
    return 0


def _pass_through(group: str, argv: list[str]) -> int:
    targets = PASS_THROUGH[group]
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: pcrobust {group} {{{','.join(targets)}}} [args ...]\n\n"
              f"Each command forwards its arguments to the module below; "
              f"add --help after a command for its options.\n")
        for name, mod in targets.items():
            print(f"  {name:20s} {mod}")
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in targets:
        print(f"pcrobust {group}: unknown command {name!r}; "
              f"choose from {', '.join(targets)}", file=sys.stderr)
        return 2
    return _run_module(targets[name], f"pcrobust {group} {name}", rest)


# ── own subcommands ───────────────────────────────────────────────────────────

def _cmd_list(args) -> int:
    if args.what == "models":
        from pcrobust.registry import list_models, model_info

        for key in list_models():
            info = model_info(key)
            print(f"{key:18s} {info.family:20s} {info.display_name}")
    elif args.what == "corruptions":
        from pcrobust.corruptions.severity import ANGLES, LEVELS, SIGMAS

        print(f"pose       yaw angle (deg)          severities 1-5: {list(ANGLES)}")
        print(f"noise      Gaussian sigma           severities 1-5: {list(SIGMAS)}")
        print(f"occlusion  camera distance (x diag) severities 1-5: {list(LEVELS)}")
    elif args.what == "domains":
        for d in DOMAINS:
            print(d)
    elif args.what == "checkpoints":
        from pcrobust.models.zoo import list_checkpoints

        names = list_checkpoints()
        print("\n".join(names) if names else
              "(no checkpoints in the manifest yet -- populated at release)")
    return 0


def _read_points(path: str):
    import numpy as np

    if path.endswith(".npy"):
        pts = np.load(path)
    else:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            raise SystemExit(f"pcrobust corrupt: could not read points from {path}")
        pts = np.asarray(pcd.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise SystemExit(f"pcrobust corrupt: expected (N, 3) points, got {pts.shape}")
    return pts.astype(np.float32)


def _write_points(path: str, pts) -> None:
    import numpy as np

    if path.endswith(".npy"):
        np.save(path, pts)
        return
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if not o3d.io.write_point_cloud(path, pcd):
        raise SystemExit(f"pcrobust corrupt: could not write {path}")


def corrupt_points(pts, corruption: str, severity: int, seed: int = 0):
    """Apply one benchmark corruption at severity 1-5 (the paper's ladders)."""
    import numpy as np

    from pcrobust.corruptions import add_noise, occlude, random_direction, rotate_yaw
    from pcrobust.corruptions.severity import ANGLES, LEVELS, SIGMAS

    corruption = _ALIASES.get(corruption, corruption)
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in 1..5")
    i = severity - 1
    if corruption == "pose":
        return rotate_yaw(pts, ANGLES[i])
    if corruption == "noise":
        np.random.seed(seed)
        return add_noise(pts, SIGMAS[i])
    if corruption == "occlusion":
        return occlude(pts, LEVELS[i], random_direction(np.random.default_rng(seed)))
    raise ValueError(f"unknown corruption {corruption!r}; choose from {', '.join(CORRUPTIONS)}")


def _cmd_corrupt(args) -> int:
    pts = _read_points(args.input)
    out = corrupt_points(pts, args.corruption, args.severity, seed=args.seed)
    dest = args.output
    if dest is None:
        stem, dot, ext = args.input.rpartition(".")
        dest = f"{stem}_{_ALIASES.get(args.corruption, args.corruption)}{args.severity}.{ext}" \
            if dot else f"{args.input}_{args.corruption}{args.severity}"
    _write_points(dest, out)
    print(f"{len(pts)} -> {len(out)} points, written to {dest}")
    return 0


def _cmd_prepare(args) -> int:
    from pcrobust.datasets import domain_module

    mod = domain_module(args.domain)
    if args.download:
        fetch = getattr(mod, f"download_{args.domain}", None)
        if fetch is None:
            print(f"pcrobust prepare: {args.domain} has no automatic download "
                  f"(license- or request-gated); place the archive under "
                  f"{mod.ROOT / '_raw_download'} and rerun without --download.",
                  file=sys.stderr)
            return 2
        fetch()
    mod.prepare_dataset()
    return 0


def _cmd_paper(args) -> int:
    from pcrobust import paper

    if args.action == "list":
        for name in paper.list_data():
            print(name)
        return 0
    if not args.name:
        print("pcrobust paper show: give a data file name (see `pcrobust paper list`)",
              file=sys.stderr)
        return 2
    if args.name.endswith(".json"):
        import json

        print(json.dumps(paper.load_json(args.name), indent=2))
    else:
        import pandas as pd

        with pd.option_context("display.max_rows", None, "display.max_columns", None,
                               "display.width", 200):
            print(paper.load(args.name))
    return 0


def _print_comparison(rep, verbose: bool) -> None:
    print(("  ok    " if rep.ok else "  DIFFS ") + rep.summary())
    if not verbose:
        return
    for key, col, printed, rebuilt in rep.mismatches:
        print(f"          {key} [col {col + 1}]: paper {printed}, rebuilt {rebuilt}")
    for key, col, printed in rep.uncovered:
        print(f"          {key} [col {col + 1}]: paper {printed}, not in bundled data")


def _cmd_reproduce(args) -> int:
    from pathlib import Path

    from pcrobust.paper import reproduce

    if args.list or not args.targets:
        for t in reproduce.list_targets():
            note = f"  (partial: {t.missing})" if t.missing else ""
            print(f"{t.id:28s} {t.where:7s} {t.title}{note}")
        return 0
    try:
        targets = (reproduce.list_targets() if args.targets == ["all"]
                   else [reproduce.get_target(n) for n in args.targets])
    except KeyError as e:
        print(f"pcrobust reproduce: {e.args[0]}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    single = len(targets) == 1
    if args.compare_paper:
        src = reproduce.paper_source()
        print(f"comparing against paper commit {src.get('commit', '?')[:10]} "
              f"({src.get('commit_date', '?')[:10]})")
    failed = 0
    for t in targets:
        df = t.build()
        if single and not out:
            print(f"{t.id}: {t.title} ({t.where})")
            if t.kind == "table":
                print(reproduce.format_frame(df))
            else:
                print(f"  {len(df)} plotted values; --out DIR writes the data and the figure")
            if t.missing:
                print(f"  partial: {t.missing}")
        if out:
            stem = out / t.id.replace(":", "_")
            reproduce.to_csv(df, stem.with_suffix(".csv"))
            written = [stem.with_suffix(".csv").name]
            if t.kind == "figure":
                try:
                    reproduce.plot(t.id, stem.with_suffix(".png"), df)
                    written.append(stem.with_suffix(".png").name)
                except ImportError:
                    written.append("(no figure: pip install pcrobust[figures])")
            print(f"{t.id:28s} -> {', '.join(written)}")
        if args.compare_paper:
            rep = reproduce.compare(t.id, df)
            failed += not rep.ok
            _print_comparison(rep, verbose=single or args.verbose)
    if args.compare_paper and len(targets) > 1:
        print(f"{len(targets) - failed}/{len(targets)} targets match the paper"
              + ("" if not failed else "; rerun with -v for the differing cells"))
    return 1 if failed else 0


def _cmd_tables(args) -> int:
    return _run_module("pcrobust.paper.tables", "pcrobust tables", [args.which])


def build_parser() -> argparse.ArgumentParser:
    from pcrobust import __version__

    p = argparse.ArgumentParser(
        prog="pcrobust",
        description="Statistical robustness benchmark for point cloud classification.",
        epilog="Pass-through groups: " + ", ".join(
            f"{g} {{{','.join(t)}}}" for g, t in PASS_THROUGH.items()),
    )
    p.add_argument("--version", action="version", version=f"pcrobust {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")

    s = sub.add_parser("list", help="list models, corruptions, domains, or checkpoints")
    s.add_argument("what", choices=["models", "corruptions", "domains", "checkpoints"])
    s.set_defaults(func=_cmd_list)

    s = sub.add_parser("corrupt", help="apply one corruption to a point cloud file")
    s.add_argument("input", help=".ply/.pcd/.xyz (via open3d) or .npy of shape (N, 3)")
    s.add_argument("--corruption", required=True, choices=[*CORRUPTIONS, *_ALIASES])
    s.add_argument("--severity", required=True, type=int, choices=range(1, 6),
                   metavar="{1..5}")
    s.add_argument("-o", "--output", default=None,
                   help="default: <input>_<corruption><severity>.<ext>")
    s.add_argument("--seed", type=int, default=0,
                   help="seeds the noise draw / occlusion camera direction")
    s.set_defaults(func=_cmd_corrupt)

    s = sub.add_parser("prepare", help="build a domain's prepared data under PCROBUST_HOME")
    s.add_argument("domain", choices=DOMAINS)
    s.add_argument("--download", action="store_true",
                   help="fetch the raw archive first (where the license allows)")
    s.set_defaults(func=_cmd_prepare)

    s = sub.add_parser("paper", help="browse the frozen result tables behind the paper")
    s.add_argument("action", choices=["list", "show"])
    s.add_argument("name", nargs="?", default=None)
    s.set_defaults(func=_cmd_paper)

    s = sub.add_parser(
        "reproduce", help="rebuild the paper's tables and figures from the frozen data",
        description="Tier-1 reproduction: rebuild paper tables and figures from the result "
                    "data bundled with the package (offline, no GPU). Targets are the "
                    "paper's LaTeX labels (tab:noise, fig:teaser) or their bare names.")
    s.add_argument("targets", nargs="*", metavar="target",
                   help="one or more targets, or `all`; none lists them")
    s.add_argument("--list", action="store_true", help="list every target")
    s.add_argument("--out", metavar="DIR",
                   help="write <target>.csv (and <target>.png for figures) into DIR")
    s.add_argument("--compare-paper", action="store_true",
                   help="check every rebuilt number against the numbers printed in the paper")
    s.add_argument("-v", "--verbose", action="store_true",
                   help="with several targets, list each differing cell")
    s.set_defaults(func=_cmd_reproduce)

    s = sub.add_parser("tables", help="rebuild summary tables from raw results")
    s.add_argument("which", choices=["summary", "occlusion", "training-metrics", "all"])
    s.set_defaults(func=_cmd_tables)

    for group in PASS_THROUGH:
        sub.add_parser(group, help=f"{group} commands (pass-through; "
                                   f"`pcrobust {group} --help` lists them)", add_help=False)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        # Bare ``pcrobust``: always greet, unless the import just did or the
        # user silenced it.
        import os

        from pcrobust import __version__, _welcome

        if not _welcome._shown and not os.environ.get("PCROBUST_QUIET"):
            print(_welcome.message(__version__))
    if argv and argv[0] in PASS_THROUGH:
        return _pass_through(argv[0], argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
