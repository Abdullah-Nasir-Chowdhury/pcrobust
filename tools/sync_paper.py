"""Snapshot the paper's printed numbers into ``src/pcrobust/paper/expected/``.

Dev tool, not part of the installed package. Run it whenever the paper
source changes; ``pcrobust reproduce --compare-paper`` and the golden tests
then check the bundled data against the new revision.

    python tools/sync_paper.py PATH/TO/PAPER/REPO

Tables are read from the LaTeX (``sec/*.tex``, ``supp_table_*.tex``) by
label. Figures are read from the inputs their plotting scripts consume:
the CSVs under ``figures/`` and ``rotsweep/``, and the literal value
tables in the ``rotsweep/figpack`` scripts. Only the git commit of the
paper repo is recorded, never a local path.
"""

from __future__ import annotations

import ast
import csv
import json
import operator
import subprocess
import sys
from pathlib import Path

from pcrobust.paper import reproduce
from pcrobust.paper.latex import find_tables, parse_tabular

OUT = Path(__file__).resolve().parents[1] / "src" / "pcrobust" / "paper" / "expected"


def _write(target_id: str, rows: list[tuple[str, list[str]]]) -> None:
    width = max(len(cells) for _, cells in rows)
    path = OUT / f"{target_id.replace(':', '_')}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row", *[f"c{i + 1}" for i in range(width)]])
        for key, cells in rows:
            w.writerow([key, *cells, *[""] * (width - len(cells))])


def _script_values(path: Path) -> dict:
    """Module-level ``NAME = <dict>`` assignments of a figure script.

    Literal dicts are read with :func:`ast.literal_eval`; dicts of plain
    arithmetic over numbers (``(0.143 + 0.170) / 5``) are evaluated with a
    numbers-only evaluator. Nothing in the script is executed.
    """
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.USub: operator.neg}

    def arith(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](arith(node.left), arith(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](arith(node.operand))
        raise ValueError(ast.dump(node))

    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Dict)):
            continue
        name = node.targets[0].id
        try:
            out[name] = ast.literal_eval(node.value)
        except ValueError:
            try:
                out[name] = {ast.literal_eval(k): arith(v)
                             for k, v in zip(node.value.keys, node.value.values)}
            except ValueError:
                pass
    return out


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _teaser(repo: Path) -> list[tuple[str, list[str]]]:
    vals = _script_values(repo / "rotsweep" / "figpack" / "fig1_tradeoff_scatter.py")
    m40, noise = vals["M40_POSE_REP1"], vals["NOISE"]
    cells = {(r["Architecture"], r["Domain"], r["Rotation_Type"]): float(r["mean_D"])
             for r in _csv_rows(repo / "rotsweep" / "point-cloud-robustness" / "results"
                                / "pose_primary_test.csv")}
    rows = []
    for arch in reproduce.ARCHS:
        if arch in m40:
            pose = [m40[arch][ax] if d == "modelnet40" else cells[(arch, d, ax)]
                    for d in reproduce.DOMAINS for ax in reproduce.POSE_AXES]
            pose_mean = sum(pose) / len(pose)
        else:
            pose_mean = 0.0     # the VN models: all 15 pose cells |D| < 1e-4
        rows.append((f"{arch}|pose", [f"{pose_mean:.4f}"]))
        rows.append((f"{arch}|noise", [f"{noise[arch]:.4f}"]))
    return rows


def _severity(repo: Path):
    return [(f"{r['perturbation']}|{r['family']}|{r['domain']}|{reproduce._key(r['level'])}",
             [r["mean_similarity"]]) for r in _csv_rows(repo / "figures" / "severity_domain_curves.csv")]


def _severity_resolved(repo: Path):
    vals = _script_values(repo / "rotsweep" / "figpack" / "fig7_severity_resolved_v2.py")
    rows = []
    for panel, name in [("gain", "DELTA"), ("headroom", "HEADROOM")]:
        for corr, series in vals[name].items():
            rows += [(f"{panel}|{corr}|{sev}", [repr(v)]) for sev, v in enumerate(series, 1)]
    return rows


def _rotation_sweep(repo: Path):
    results = repo / "rotsweep" / "point-cloud-robustness" / "results"
    rows = []
    for path, condition, archs in [
            (results / "rotation_sweep_n300_rep1" / "ri_invreg_summary.csv", "invreg", reproduce.RI),
            (results / "rotation_sweep_n300" / "rotation_sweep_summary_n300.csv", "pose270",
             reproduce.STANDARD)]:
        rows += [(f"{r['arch']}|{r['axis']}|{r['angle_deg']}", [r["accuracy"]])
                 for r in _csv_rows(path) if r["condition"] == condition and r["arch"] in archs]
    return rows


_FIGURES = {"fig:teaser": _teaser, "fig:severity": _severity,
            "fig:severity_resolved": _severity_resolved, "fig:rotation_sweep": _rotation_sweep}


def main(repo_arg: str) -> int:
    repo = Path(repo_arg)
    tables = {}
    for tex in [*sorted(repo.glob("sec/*.tex")), *sorted(repo.glob("supp_table_*.tex"))]:
        tables.update(find_tables(tex.read_text(encoding="utf-8")))

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.csv"):
        old.unlink()
    registered = {t.id for t in reproduce.list_targets()}
    for t in reproduce.list_targets():
        if t.kind == "table":
            if t.id not in tables:
                print(f"warning: {t.id} is registered but not in the paper source")
                continue
            rows = parse_tabular(tables[t.id], key_cols=t.key_cols)
        else:
            rows = _FIGURES[t.id](repo)
        _write(t.id, rows)
        print(f"{t.id:28s} {len(rows):4d} rows")
    for label in sorted(set(tables) - registered):
        print(f"warning: paper table {label} has no reproduce target")

    git = ["git", "-C", str(repo), "log", "-1", "--format=%H%n%cI"]
    commit, date = subprocess.run(git, capture_output=True, text=True, check=True).stdout.split()
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--",
                                 "sec", "supp_table_*.tex", "figures", "rotsweep"],
                                capture_output=True, text=True, check=True).stdout.strip())
    (OUT / "_source.json").write_text(json.dumps(
        {"commit": commit, "commit_date": date, "uncommitted_changes": dirty}, indent=2) + "\n",
        encoding="utf-8")
    print(f"paper commit {commit[:10]} ({date}){' + uncommitted changes' if dirty else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
