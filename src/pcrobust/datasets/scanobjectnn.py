"""
prepare_dataset.py — builds the ScanObjectNN subset used by the
pose-robustness replication, from the official h5 release
(http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip -- direct download, no
registration/account needed, unlike OmniObject3D).

Unlike ModelNet40/ShapeNet (CAD models / clean synthetic samples),
ScanObjectNN is real-world RGB-D scanned objects with background clutter and
occlusion left in by default -- this is deliberately the "hardest"/most
realistic domain in the study. Uses the "main split" (`training_
objectdataset.h5` / `test_objectdataset.h5`), the non-augmented, standard
benchmark variant (not the "hardest" perturbed-and-rotated variant, which
would confound this project's own pose-robustness rotation sweep with the
dataset's own pre-baked rotation augmentation).

Pipeline:
  1. Load the official h5 arrays: `data` (N, 2048, 3) point clouds, `label`
     (N,) int class indices, for train and test splits separately.
  2. Class names come from the dataset's standard 15-class OBJ_CLASS order
     (bag, bin, box, cabinet, chair, desk, display, door, shelf, table, bed,
     pillow, sink, sofa, toilet) -- label i -> OBJ_CLASSES[i].
  3. Center each point cloud to the origin (unit-sphere scaling left to
     extract.py's extract_object_pts(), same convention as every other
     domain here) and write to data/train/<class>/<class><i>.ply or
     data/test/<class>/<class>_test<i>.ply.

Classes were narrowed from the full 15 to the 10 most rigid/non-flat/non-
symmetric ones for the same reason ModelNet40/ShapeNet narrowed theirs
(bag/pillow are soft-body/deformable, box/sink are close to shape-symmetric
about at least one axis -- weak pose-robustness testbeds).

Usage:
    python -m pcrobust.datasets.scanobjectnn   (or: pcrobust prepare scanobjectnn)
"""

import sys
import argparse
import h5py
import numpy as np
import open3d as o3d
from pathlib import Path

from pcrobust.utils import normalize_and_sample
from pcrobust.datasets._paths import domain_root

ROOT = domain_root("scanobjectnn")
_SON_ROOT = ROOT
_DOWNLOAD_DIR = _SON_ROOT / "_raw_download"
_DATA_DIR = _SON_ROOT / "data"


# Standard ScanObjectNN OBJ_CLASS label order (15 classes, label index = position).
OBJ_CLASSES = [
    "bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
    "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet",
]
CLASSES = [
    "cabinet", "chair", "desk", "display", "door",
    "shelf", "table", "bed", "sofa", "toilet",
]
N_TRAIN_PER_CLASS = 20
N_TEST_PER_CLASS = 12


def find_h5(split: str) -> Path:
    """Locate training_objectdataset.h5 / test_objectdataset.h5 (main split,
    not the augmented/hardest variant) wherever the zip happened to extract
    it -- the official archive's internal directory naming isn't guaranteed
    stable across releases."""
    fname = f"{split}_objectdataset.h5"
    candidates = [p for p in _DOWNLOAD_DIR.rglob(fname)
                  if "hardest" not in p.name and "augmented" not in p.name]
    if not candidates:
        # fall back to any matching file if the strict (non-augmented) one isn't found
        candidates = list(_DOWNLOAD_DIR.rglob(fname))
    if not candidates:
        sys.exit(f"[prepare] could not find {fname} under {_DOWNLOAD_DIR} -- "
                  f"download+extract h5_files.zip first")
    return candidates[0]


def load_split(split: str):
    h5 = h5py.File(find_h5(split), "r")
    pts = np.array(h5["data"], dtype=np.float32)
    labels = np.array(h5["label"], dtype=int)
    h5.close()
    return pts, labels


def write_ply(pts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)


def process_split(split: str, classes: list[str], n_wanted: int, out_root: Path, stem_fn):
    pts, labels = load_split(split)
    totals = {}
    for cls in classes:
        cls_idx = OBJ_CLASSES.index(cls)
        idx = np.where(labels == cls_idx)[0][:n_wanted]
        i = 0
        for j in idx:
            p = pts[j] - pts[j].mean(axis=0, keepdims=True)
            i += 1
            write_ply(p.astype(np.float32), out_root / cls / f"{stem_fn(cls, i)}.ply")
        totals[cls] = i
        print(f"[prepare] {cls}/{split}: {i}/{n_wanted} instances written "
              f"({len(idx)} available)")
    return totals


def prepare_dataset(classes=None):
    """Build the prepared subset (data/train, data/test, data/gallery)."""
    classes = list(classes) if classes is not None else list(CLASSES)

    train_totals = process_split("training", classes, N_TRAIN_PER_CLASS,
                                  _DATA_DIR / "train", lambda cls, i: f"{cls}{i}")
    test_totals = process_split("test", classes, N_TEST_PER_CLASS,
                                 _DATA_DIR / "test", lambda cls, i: f"{cls}_test{i}")

    print("\n[prepare] === Summary ===")
    for cls in classes:
        print(f"    {cls:<10s}  train={train_totals.get(cls, 0):3d}  test={test_totals.get(cls, 0):3d}")

    print("\n[prepare] Building training gallery (data/gallery/*_gallery.ply) ...")
    n_gallery = build_gallery(_DATA_DIR / "train", _DATA_DIR / "gallery")
    print(f"[prepare] wrote {n_gallery} gallery files")


TRAIN_DIR = _DATA_DIR / "train"
TEST_DIR = _DATA_DIR / "test"


# ── Canonical extraction ───────────────────────────────────────────────────────
# Each raw .ply produced by prepare_dataset() already IS a single clean object,
# so there is no segmentation to do -- the only step needed is canonicalization:
# centering to origin, scaling to unit sphere, and farthest-point-sampling to
# n_points, reused directly via pcrobust.utils.normalize_and_sample(), not
# reimplemented. extract_object_pts() returns a fresh (N, 3) canonical point
# array given a raw .ply path; the pose evaluation calls this once per raw
# scan, then applies pure yaw rotation to the result.


def extract_object_pts(path: str, n_points: int = 1024) -> np.ndarray:
    """Load a raw ScanObjectNN instance .ply and return a canonical, centered,
    unit-sphere-normalized, n_points-point cloud."""
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise RuntimeError(f"Could not load point cloud from: {path}")
    canonical = normalize_and_sample(pcd, n_points=n_points)
    pts = np.asarray(canonical.points, dtype=np.float32)
    return pts


def build_gallery(train_dir: Path, gallery_dir: Path, n_points: int = 1024) -> int:
    """For every raw instance .ply under train_dir/<class>/, write a
    normalized <class><i>_gallery.ply into gallery_dir (flat), matching the
    naming convention pcrobust.datasets.gallery.MeshGalleryDataset expects.
    Returns the number of files written."""
    gallery_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.glob("*.ply")):
            pts = extract_object_pts(str(f), n_points=n_points)
            out = o3d.geometry.PointCloud()
            out.points = o3d.utility.Vector3dVector(pts)
            out_path = gallery_dir / f"{f.stem}_gallery.ply"
            o3d.io.write_point_cloud(str(out_path), out)
            n += 1
    return n


# ── Object pools ───────────────────────────────────────────────────────────────
# "identity" = object CLASS, "scan" = individual mesh INSTANCE within that
# class:
#
#   - The pilot uses exactly one instance per class: the lowest-numbered
#     instance in data/train/<class>/. Reserved for the pilot and excluded
#     from the confirmatory pool (avoid double-dipping: the pilot sets
#     delta/sigma; reusing the same data to test H0 would invalidate that
#     separation).
#   - The confirmatory pool is every remaining train+test instance for all
#     classes.
#
# EXCLUDED_IDENTITIES is empty here: the ScanObjectNN classes were
# deliberately chosen (see prepare_dataset()) for having ample instance
# counts, so none need to be dropped.

EXCLUDED_IDENTITIES: dict[str, str] = {}


def list_identities() -> list[str]:
    """All classes in data/train, minus EXCLUDED_IDENTITIES."""
    if not TRAIN_DIR.exists():
        return []
    return sorted(p.name for p in TRAIN_DIR.iterdir()
                  if p.is_dir() and p.name not in EXCLUDED_IDENTITIES)


def pilot_scans() -> list[tuple[str, Path]]:
    """One instance per included class: the lowest-numbered instance in data/train."""
    scans = []
    for identity in list_identities():
        files = sorted((TRAIN_DIR / identity).glob("*.ply"))
        if files:
            scans.append((identity, files[0]))
    return scans


def confirmatory_pool() -> list[tuple[str, Path]]:
    """All train+test instances for included classes, minus the pilot instances."""
    pilot_paths = {p.resolve() for _, p in pilot_scans()}
    pool = []
    for identity in list_identities():
        for f in sorted((TRAIN_DIR / identity).glob("*.ply")):
            if f.resolve() not in pilot_paths:
                pool.append((identity, f))
        test_dir = TEST_DIR / identity
        if test_dir.exists():
            for f in sorted(test_dir.glob("*.ply")):
                pool.append((identity, f))
    return pool


# ── Pool feasibility check ─────────────────────────────────────────────────────
# check_pool(): does extract_object_pts() produce a valid 1024-point canonical
# cloud for each candidate instance in the confirmatory pool? Every raw
# instance was already produced cleanly by prepare_dataset(), so this is
# expected to report ~100% success; running it anyway catches any genuinely
# corrupt .ply files before they'd silently shrink the confirmatory N.


def check_pool():
    pool = confirmatory_pool()
    print(f"[check] candidate pool: {len(pool)} instances\n")

    ok, failed = [], []
    for identity, path in pool:
        try:
            pts = extract_object_pts(str(path))
            ok.append((identity, path.stem, len(pts)))
            print(f"  OK    {identity:<10s} {path.stem:<16s} ({len(pts)} pts)")
        except Exception as e:
            failed.append((identity, path.stem, str(e)))
            print(f"  FAIL  {identity:<10s} {path.stem:<16s} -> {e}")

    print(f"\n[check] usable: {len(ok)} / {len(pool)}")
    if failed:
        print(f"[check] failed:")
        for identity, stem, err in failed:
            print(f"    {identity}/{stem}: {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    prepare_dataset(classes=parser.parse_args().classes)
