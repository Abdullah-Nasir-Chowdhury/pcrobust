"""Read the printed numbers out of the paper's LaTeX tables.

Used in two places: ``tools/sync_paper.py`` snapshots every reproducible
table of the paper source into ``pcrobust/paper/expected/``, and
:mod:`pcrobust.paper.reproduce` compares rebuilt numbers against that
snapshot (``pcrobust reproduce --compare-paper``).

A table is reduced to rows of cleaned cell text. The leading ``key_cols``
cells of a row form its key (``"PointNet|no"``); the remaining cells are
data, compared number by number at the precision the paper prints.
"""

from __future__ import annotations

import re

__all__ = ["clean_cell", "find_tables", "matches", "numbers", "parse_tabular"]

_COMMENT = re.compile(r"(?<!\\)%.*")
_SUPERSCRIPT = re.compile(r"\^\{(?:[^{}]|\{[^{}]*\})*\}|\^\\[A-Za-z]+|\^[*\w]")
_WRAPPERS = re.compile(r"\\(?:textbf|emph|textit|mathbf|text|mathrm)\{([^{}]*)\}")
_RULES = re.compile(r"\\(?:toprule|midrule|bottomrule|hline)|\\cmidrule(?:\([a-z]*\))?\{[^}]*\}")
_MULTICOL = re.compile(r"\\multicolumn\{\d+\}\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\.\d+|\d+)(?![\w.])")


def clean_cell(text: str) -> str:
    """LaTeX cell -> plain text: markup, superscript markers and math dropped."""
    s = text
    s = s.replace("$-$", "-").replace("$+$", "+").replace("\\pm", "+/-")
    s = s.replace("\\rightarrow", "->").replace("\\_", "_").replace("\\%", "%")
    s = re.sub(r"\\quad|\\,|\\ ", " ", s).replace("{,}", ",").replace("~", " ")
    s = s.replace("$\\geq", ">=").replace("\\geq", ">=")
    s = _SUPERSCRIPT.sub("", s)
    while _WRAPPERS.search(s):
        s = _WRAPPERS.sub(r"\1", s)
    s = s.replace("$", "").replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    return " ".join(s.split())


def numbers(text: str) -> list[str]:
    """Numeric tokens of a cleaned cell, as printed (``".90"``, ``"-0.013"``)."""
    return _NUMBER.findall(text)


def matches(printed: str, value: float) -> bool:
    """True when ``printed`` is a correct rounding of ``value``.

    The tolerance is half a unit in the last printed place, so a value that
    sits exactly on a rounding tie passes either way.
    """
    decimals = len(printed.split(".", 1)[1]) if "." in printed else 0
    return abs(float(printed) - value) <= 0.5 * 10 ** -decimals + 1e-9


def find_tables(tex: str) -> dict[str, str]:
    """Map each ``\\label{...}`` inside a table environment to its tabular body."""
    tex = "\n".join(_COMMENT.sub("", line) for line in tex.splitlines())
    out = {}
    for env in re.finditer(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", tex, re.DOTALL):
        body = env.group(2)
        label = re.search(r"\\label\{([^}]*)\}", body)
        tab = re.search(r"\\begin\{tabular\}\{.*?\}\s*(.*?)\\end\{tabular\}", body, re.DOTALL)
        if label and tab:
            out[label.group(1)] = tab.group(1)
    return out


def parse_tabular(body: str, key_cols: int = 1) -> list[tuple[str, list[str]]]:
    """Data rows of a tabular body as ``(key, [cell text, ...])``.

    Everything before the first ``\\midrule`` is header. Rows without a
    digit (group captions) are skipped. With ``key_cols=0`` the key is the
    data row's 0-based position. A row whose first key cell is empty
    inherits the previous row's (the probe tables' continuation rows), and
    an indented ``\\quad`` row is prefixed with the last unindented row's
    key (``"DiD|95% CI"``).
    """
    head, sep, rest = body.partition("\\midrule")
    if not sep:
        rest = head
    rows = []
    first, parent = "", ""
    for raw in re.split(r"\\\\", rest):
        raw = _RULES.sub("", raw)
        raw = _MULTICOL.sub(r"\1", raw)
        if not raw.strip():
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)&", raw)]
        indented = cells[0].startswith("\\quad")
        cells = [clean_cell(c) for c in cells]
        if not any(numbers(c) for c in cells[key_cols:]):
            continue
        if key_cols == 0:
            key = str(len(rows))
        else:
            if cells[0]:
                first = cells[0]
            else:
                cells[0] = first
            key = "|".join(cells[:key_cols])
            if indented:
                key = f"{parent}|{key}"
            else:
                parent = key
        rows.append((key, cells[key_cols:]))
    return rows
