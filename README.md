# pcrobust

Companion package to *Do We Still Need Rotation-Invariant Architectures?
A Statistical Robustness Benchmark for Point Cloud Classification*
(WACV 2027), [arXiv:2611.03952](https://arxiv.org/abs/2611.03952)
(DOI [10.48550/arXiv.2611.03952](https://doi.org/10.48550/arXiv.2611.03952)).
The version evaluated in the paper is release v1.0.0, archived on Zenodo:
[10.5281/zenodo.23847450](https://doi.org/10.5281/zenodo.23847450).

It packages the paper's benchmark as reusable parts:

- **Corruption generators**: SO(3) pose, Gaussian noise, and
  hidden-point-removal occlusion, with the paper's exact five-step severity
  ladders, as pure functions over `(N, 3)` arrays.
- **Models**: the 10 benchmark architectures (5 standard, 3
  rotation-invariant, 2 rotation-equivariant), one `Embedder` API.
- **Evaluation protocol**: embedding-drift and accuracy sweeps,
  ModelNet40-C, the fine-grained rotation sweep, and the head-independent
  probe.
- **Statistical pipeline**: pilot, power analysis, confirmatory test,
  exploratory ANOVA, bootstrap CIs, difference-in-differences, TOST.
- **Training recipes**: the baseline recipe and the invariance-regularized
  (`invreg`) recipe, including the warmup and pose270 variants.
- **Frozen paper data**: the table-level result CSVs behind the paper
  (123 files), readable offline with no GPU.
- **Paper reproduction (Tier 1)**: every table and figure of the paper
  (27 tables, 4 figures) rebuilt from the frozen data in seconds, and
  checked number by number against what the paper prints.

- **Model zoo**: all 193 trained checkpoints behind the paper, hosted as
  assets of the v1.0.0 GitHub release and downloaded on first use.

## Install

```bash
pip install pcrobust
```

From a clone, for development:

```bash
pip install -e ".[figures,dev]"
```

Python 3.10–3.13. Extras: `figures` (matplotlib), `download` (gdown, for the
ShapeNet archive), `dev` (pytest, ruff).

## Quickstart

```python
import numpy as np
import pcrobust
from pcrobust import paper

pts = np.random.rand(1024, 3).astype(np.float32)
occluded = pcrobust.corrupt(pts, "occlusion", severity=3)   # paper ladder, step 3

net = pcrobust.get_model("dgcnn", num_classes=10)           # untrained
pcrobust.list_models()                                       # all 10 keys

df = paper.load("summary/pose_primary_test.csv")             # frozen table data
```

```bash
pcrobust list models
pcrobust corrupt cloud.ply --corruption occlusion --severity 3
pcrobust paper list
pcrobust paper show pose_primary_test
pcrobust reproduce --list
pcrobust reproduce tab:noise --compare-paper
pcrobust reproduce all --out repro/ --compare-paper
pcrobust prepare modelnet40 --download
pcrobust evaluate robustness --arch pointnet --perturbation pose
pcrobust train invreg --arch riconvpp --device cuda
pcrobust stats --help
```

`pcrobust evaluate|train|stats <command> --help` lists each command's
options. Each command is also runnable as `python -m <module>`.

## Reproducing the paper

`pcrobust reproduce` rebuilds a table or figure from the bundled frozen
data, offline and without a GPU. Targets are the paper's LaTeX labels
(`tab:noise`, `fig:rotation_sweep`) or their bare names (`noise`).
`--out DIR` writes each target as CSV, plus a PNG for figures (needs the
`figures` extra). `--compare-paper` checks every rebuilt number against
the number printed in the paper. A number matches when the printed value
is a correct rounding of it. The command exits with status 1 if any cell
differs.

```python
from pcrobust.paper import reproduce

df = reproduce.build("tab:m40c")         # laid out like the printed table
rep = reproduce.compare("tab:m40c")      # rep.mismatches, rep.uncovered
reproduce.plot("fig:teaser", "teaser.png")
```

The printed numbers live in `src/pcrobust/paper/expected/`. That is a
snapshot of the paper source, keyed to its git commit. After the paper
changes, refresh it with:

```bash
python tools/sync_paper.py PATH/TO/PAPER/REPO
```

Two targets are partial because some of their inputs are not bundled:
the latency column of `tab:supp-cost`, and the aug-only column of
`tab:supp-w4-types`. `pcrobust reproduce --list` names them.

## Data and checkpoints

Everything downloaded or produced lives under `PCROBUST_HOME` (default
`~/.cache/pcrobust`):

```
<PCROBUST_HOME>/
  <domain>/_raw_download/                 raw archives
  <domain>/data/{train,test,gallery}/     prepared point clouds
  checkpoints/wacv2027/<domain>/<key>.pth checkpoints (zoo download target)
  results/...                             evaluation and statistics outputs
```

A checkpoint key is an architecture plus an optional recipe suffix:
`pointnet`, `pointnet_invreg`, `pointnet_invreg_warmup`,
`pointnet_invreg_pose270`. Every evaluator accepts a directory override
(`--models_dir` / `--ckpt_root`) for checkpoints kept elsewhere.

Released checkpoints download on first use and are verified against the
SHA256 in `src/pcrobust/models/zoo_manifest.json`:

```python
model, classes = pcrobust.load_model("modelnet40/riconvpp_invreg_rep1.pth")
```

`pcrobust.models.zoo.list_checkpoints()` lists all 193 names.

ModelNet40 downloads automatically (`pcrobust prepare modelnet40 --download`).
ABO fetches its meshes during `pcrobust prepare abo` once `listings.json.gz`
and `3dmodels.csv.gz` from the public ABO bucket are in its `_raw_download/`.
ShapeNet, ScanObjectNN, and S3DIS are registration- or request-gated: place
the archive under `<PCROBUST_HOME>/<domain>/_raw_download/` and run
`pcrobust prepare <domain>`.

## Tests

```bash
pytest tests
```

Covers every module importing standalone, a CPU forward pass for all 10
architectures, checkpoint loading, corruption properties and determinism,
every frozen data file, and the CLI. The golden tests rebuild
every paper table and figure and compare each printed number.

## Citation

If you use pcrobust or the benchmark, please cite the paper:

```bibtex
@article{chowdhury2026rotation,
  title   = {Do We Still Need Rotation-Invariant Architectures? A Statistical
             Robustness Benchmark for Point Cloud Classification},
  author  = {Chowdhury, Abdulla Nasir and Pratamasunu, Gulpi Qorik Oktagalu
             and Fukui, Kazuhiro},
  journal = {arXiv preprint arXiv:2611.03952},
  doi     = {10.48550/arXiv.2611.03952},
  year    = {2026},
  note    = {Accepted to WACV 2027}
}
```

The Zenodo DOI above identifies the software release; cite the paper for
the work itself.

## License

MIT for code. Checkpoints and result CSVs: CC BY 4.0.
