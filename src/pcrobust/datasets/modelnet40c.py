"""ModelNet40-C (Sun et al., 2022) filtered to the benchmark's 10-class subset.

The benchmark's ModelNet40 domain trains on 10 of the standard 40 classes,
so every checkpoint has a 10-way head. Rather than train 40-class models,
ModelNet40-C is filtered to those 10 classes and relabelled, then scored
with the existing checkpoints (inference only).

Standard 40-class index -> benchmark label (alphabetical class order):
    airplane 0->0, bed 2->1, car 7->2, chair 8->3, guitar 17->4,
    monitor 22->5, piano 25->6, sofa 30->7, table 33->8, toilet 35->9

Expected files under ``<PCROBUST_HOME>/modelnet40c/_raw_download/``:

    modelnet40_c/data_<corruption>_<severity>.npy, modelnet40_c/label.npy
        the ModelNet40-C release (Zenodo)
    modelnet40_ply_hdf5_2048/ply_data_test*.h5
        the standard clean test split ModelNet40-C was generated from; used
        as the uncorrupted reference, since the release ships none

Evaluation lives in :mod:`pcrobust.evaluate.modelnet40c`.
"""

import glob
import sys

import h5py
import numpy as np

from pcrobust.datasets._paths import domain_root

ROOT = domain_root("modelnet40c")
_DATA_DIR = ROOT / "_raw_download" / "modelnet40_c"
_CLEAN_HDF5_DIR = ROOT / "_raw_download" / "modelnet40_ply_hdf5_2048"

CORRUPTIONS = [
    "occlusion", "lidar", "density_inc", "density", "cutout", "uniform",
    "gaussian", "impulse", "upsampling", "background", "rotation", "shear",
    "distortion", "distortion_rbf", "distortion_rbf_inv",
]
SEVERITIES = [1, 2, 3, 4, 5]

STD40_TO_OURS = {0: 0, 2: 1, 7: 2, 8: 3, 17: 4, 22: 5, 25: 6, 30: 7, 33: 8, 35: 9}


def _load_clean_reference():
    """Concatenate every ply_data_test*.h5 shard of the standard
    modelnet40_ply_hdf5_2048 test split -- the exact instance set
    ModelNet40-C's corruptions were generated from."""
    shards = sorted(glob.glob(str(_CLEAN_HDF5_DIR / "ply_data_test*.h5")))
    if not shards:
        sys.exit(f"[eval] no ply_data_test*.h5 found under {_CLEAN_HDF5_DIR} -- "
                  f"download and extract modelnet40_ply_hdf5_2048.zip there first")
    all_data, all_label = [], []
    for shard in shards:
        with h5py.File(shard, "r") as f:
            all_data.append(f["data"][:, :1024, :])
            all_label.append(f["label"][:].reshape(-1))
    return np.concatenate(all_data), np.concatenate(all_label).astype(int)


def _fix_axis_convention(data: np.ndarray) -> np.ndarray:
    """The hdf5 release and ModelNet40-C are Y-up; the benchmark's own
    ModelNet40 pipeline (sampled from the raw .off meshes) is Z-up. Swapping
    Y and Z took PointNet from 28.5% to 95.5% on the clean hdf5 reference
    (near-ceiling, as on its own test split), and uniform/severity=1 from
    22.4% to 77.2%, so the swap applies to the corrupted data too."""
    return data[:, :, [0, 2, 1]]


def load_and_filter(corruption: str, severity, class_map: dict = None):
    """Return ``(points (N, P, 3), labels (N,))`` for one (corruption,
    severity) file, filtered to the 10 benchmark classes and relabelled.
    ``corruption="none"`` returns the clean reference. ``class_map``
    overrides :data:`STD40_TO_OURS` (standard-40 index -> model label)."""
    class_map = STD40_TO_OURS if class_map is None else class_map
    if corruption == "none":
        data, labels = _load_clean_reference()
    else:
        data_path = _DATA_DIR / f"data_{corruption}_{severity}.npy"
        label_path = _DATA_DIR / "label.npy"
        data = np.load(data_path)
        labels = np.load(label_path).astype(int).reshape(-1)

    mask = np.isin(labels, list(class_map.keys()))
    data = _fix_axis_convention(data[mask])
    labels = np.array([class_map[l] for l in labels[mask]], dtype=int)
    return data, labels
