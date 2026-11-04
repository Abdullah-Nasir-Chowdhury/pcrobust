# Changelog

Corrections and extensions land as tagged point releases. Raw CSVs behind
published tables are never overwritten in place: a corrected table ships
alongside its superseded version.

## 1.0.5 (2026-11-04)

- Citations now point to the paper on arXiv,
  [arXiv:2611.03952](https://arxiv.org/abs/2611.03952)
  (DOI 10.48550/arXiv.2611.03952): the README BibTeX, `CITATION.cff`, the
  Zenodo record metadata and the `Cite:` line of the welcome message.
  The Zenodo DOI still identifies the software release.
- Checkpoints, results, and numbers are unchanged from 1.0.0.

## 1.0.4 (2026-11-01)

- `import pcrobust` shows the welcome message once per installation
  again, instead of once per version, so upgrading no longer repeats it.
- The "(Shown once per version ...)" note is gone. The message now ends
  with the command that silences it: `export PCROBUST_QUIET=1`
  (Windows: `setx PCROBUST_QUIET 1`).
- Typing `pcrobust` with no arguments still shows the message every time,
  unless `PCROBUST_QUIET` is set.
- Checkpoints, results, and numbers are unchanged from 1.0.0; the model
  zoo still downloads the v1.0.0 release assets.

## 1.0.3 (2026-11-01)

- Typing `pcrobust` with no arguments now shows the welcome message every
  time, followed by the usage. Other commands and `import pcrobust` keep
  the once-per-version behavior, and the message is not printed twice on
  the first run of a new version.
- Checkpoints, results, and numbers are unchanged from 1.0.0; the model
  zoo still downloads the v1.0.0 release assets.

## 1.0.2 (2026-11-01)

- The welcome message now shows once per version instead of once ever, so
  upgrading shows it again.
- Imports nobody can read (pipes, CI, test collection) no longer use up the
  message, which could leave it unseen in a later notebook or terminal.
- In a Jupyter notebook the message goes to stdout, so it does not render
  like an error.
- Checkpoints, results, and numbers are unchanged from 1.0.0; the model
  zoo still downloads the v1.0.0 release assets.

## 1.0.1 (2026-11-01)

- One-time welcome message on the first `import pcrobust`, printed to
  stderr. Set `PCROBUST_QUIET=1` to silence it.
- Checkpoints, results, and numbers are unchanged from 1.0.0; the model
  zoo still downloads the v1.0.0 release assets.

## 1.0.0 (2026-11-01)

The version evaluated in the WACV 2027 paper. Archived on Zenodo:
[10.5281/zenodo.23847450](https://doi.org/10.5281/zenodo.23847450).

- Corruption generators, the 10 benchmark architectures, evaluation
  protocol, statistical pipeline, and training recipes.
- Frozen paper data and `pcrobust reproduce` for every table and figure.
- Model zoo: 193 trained checkpoints as assets of the v1.0.0 GitHub
  release, downloaded on first use and verified against their SHA256.
