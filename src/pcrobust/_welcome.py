"""Welcome message, shown once per installation on ``import pcrobust``.

A marker file under ``$PCROBUST_HOME`` (default ``~/.cache/pcrobust``) records
that it was shown; ``PCROBUST_QUIET=1`` silences it entirely. Nothing is
printed or recorded when nobody can read it (pipes, CI, test collection), so
those imports do not use up the message. Uses only the standard library so
``import pcrobust`` stays light, and never raises: if the marker cannot be
written, the message is simply skipped. The bare ``pcrobust`` command prints
it every time unless ``PCROBUST_QUIET`` is set (see ``pcrobust.cli.main``).
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
  Quiet:   export PCROBUST_QUIET=1   (Windows: setx PCROBUST_QUIET 1)
"""

# Set when this process has already printed the message, so the bare
# ``pcrobust`` command does not repeat what the import just showed.
_shown = False


def message(version: str) -> str:
    """The welcome text for ``version``."""
    return _MESSAGE.format(version=version)


def _stream():
    """Return the stream a person will read, or None if nobody is watching."""
    # Notebooks render stderr like an error, and their stdout is not piped anywhere.
    if "ipykernel" in sys.modules:
        return sys.stdout
    try:
        return sys.stderr if sys.stderr.isatty() else None
    except (AttributeError, ValueError):
        return None


def show_once(version: str) -> None:
    """Print the welcome message unless it was already shown or it is silenced."""
    if os.environ.get("PCROBUST_QUIET"):
        return
    stream = _stream()
    if stream is None:
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
    global _shown
    _shown = True
    print(message(version), file=stream)
