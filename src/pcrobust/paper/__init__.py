"""Frozen table-level result data behind the paper's tables and figures.

Every file under ``pcrobust/paper/data/`` is a table-level summary CSV (or a
small JSON curve/report file) copied verbatim from the benchmark's frozen
results. No per-trial or per-sample data is shipped.

Usage::

    from pcrobust import paper
    paper.list_data()                     # all bundled files, relative paths
    df = paper.load("summary/pose_primary_test.csv")
    df = paper.load("pose_primary_test")  # bare name works when unique
    curves = paper.load_json("rotation_sweep/rotation_sweep/yaw_curve_data.json")

The generators that rebuilt the ``summary/`` tables from raw per-scan data
live in :mod:`pcrobust.paper.tables` (the raw inputs are not part of this
package). :mod:`pcrobust.paper.reproduce` rebuilds the paper's tables and
figures from these files and checks them against the printed numbers.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["list_data", "load", "load_json"]


def _data_root():
    return resources.files(__name__) / "data"


def _walk(node, prefix: str = "") -> Iterator[str]:
    for child in sorted(node.iterdir(), key=lambda c: c.name):
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            yield from _walk(child, rel + "/")
        else:
            yield rel


def list_data() -> list[str]:
    """Relative POSIX-style paths of every bundled data file.

    Paths are relative to the package's ``data/`` directory, e.g.
    ``"summary/pose_primary_test.csv"``; pass any of them to :func:`load`
    (CSV) or :func:`load_json` (JSON).
    """
    return list(_walk(_data_root()))


def _node_for(rel: str):
    node = _data_root()
    for part in rel.split("/"):
        node = node / part
    return node


def _resolve(name: str) -> tuple[str, object]:
    """Map a user-supplied name to (relative path, Traversable)."""
    name = name.replace("\\", "/").strip("/")
    candidates = [name]
    if "." not in name.rsplit("/", 1)[-1]:
        candidates += [name + ".csv", name + ".json"]
    for cand in candidates:
        node = _node_for(cand)
        if node.is_file():
            return cand, node

    # Fall back to a bare-filename lookup anywhere in the tree, but only
    # when it is unambiguous.
    base = name.rsplit("/", 1)[-1]
    wanted = {base, base + ".csv", base + ".json"}
    matches = [rel for rel in list_data() if rel.rsplit("/", 1)[-1] in wanted]
    if len(matches) == 1:
        return matches[0], _node_for(matches[0])
    if len(matches) > 1:
        raise KeyError(
            f"data name {name!r} is ambiguous; matches: {', '.join(matches)}"
        )
    raise KeyError(
        f"unknown data file {name!r}; see pcrobust.paper.list_data() for "
        f"available names"
    )


def load(name: str) -> "pd.DataFrame":
    """Load a bundled CSV table as a :class:`pandas.DataFrame`.

    ``name`` is a relative path from :func:`list_data` (the ``.csv``
    extension may be omitted); a bare filename is accepted when it is
    unique across the data tree.
    """
    rel, node = _resolve(name)
    if rel.endswith(".json"):
        raise ValueError(
            f"{rel} is a JSON curve/report file, not a CSV table; use "
            f"pcrobust.paper.load_json() instead"
        )
    import pandas as pd

    with node.open("r", encoding="utf-8") as f:
        return pd.read_csv(f)


def load_json(name: str):
    """Load one of the bundled JSON curve/report files (parsed, as-is)."""
    rel, node = _resolve(name)
    if not rel.endswith(".json"):
        raise ValueError(f"{rel} is not a JSON file; use pcrobust.paper.load()")
    with node.open("r", encoding="utf-8") as f:
        return json.load(f)
