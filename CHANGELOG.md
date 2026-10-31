# Changelog

Corrections and extensions land as tagged point releases. Raw CSVs behind
published tables are never overwritten in place: a corrected table ships
alongside its superseded version.

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
