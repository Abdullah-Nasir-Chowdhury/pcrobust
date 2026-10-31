"""One-time welcome message, printed to stderr on the first ``import pcrobust``.

A marker file under ``$PCROBUST_HOME`` (default ``~/.cache/pcrobust``) keeps
it from repeating; ``PCROBUST_QUIET=1`` silences it entirely. Uses only the
standard library so ``import pcrobust`` stays light, and never raises: if the
marker cannot be written, the message is simply skipped.
"""

from __future__ import annotations

import os
import sys

_MESSAGE = """\
pcrobust {version}: statistical robustness benchmark for point cloud classification
Companion to "Do We Still Need Rotation-Invariant Architectures?" (WACV 2027)

  Try:     pcrobust reproduce tab:m40c --compare-paper
  Models:  pcrobust.load_model("modelnet40/riconvpp_invreg_rep1.pth")
  Docs:    https://github.com/Abdullah-Nasir-Chowdhury/pcrobust
  Cite:    https://doi.org/10.5281/zenodo.23847450

(Shown once. Set PCROBUST_QUIET=1 to silence.)
"""


def show_once(version: str) -> None:
    """Print the welcome message unless it was shown before or is silenced."""
    if os.environ.get("PCROBUST_QUIET"):
        return
    home = os.environ.get("PCROBUST_HOME", os.path.expanduser("~/.cache/pcrobust"))
    marker = os.path.join(home, ".welcome_shown")
    try:
        if os.path.exists(marker):
            return
        os.makedirs(home, exist_ok=True)
        with open(marker, "w", encoding="utf-8") as f:
            f.write(version + "\n")
    except OSError:
        return
    print(_MESSAGE.format(version=version), file=sys.stderr)
