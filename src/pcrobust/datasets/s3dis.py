"""
prepare_dataset.py — builds the S3DIS subset used by the pose-robustness
replication, from the raw annotated release
(Stanford3dDataset_v1.2_Aligned_Version, mirrored at
https://cvg-data.inf.ethz.ch/s3dis/ -- direct download, no registration,
unlike the canonical Stanford portal).

S3DIS is a whole-room indoor scene dataset (6 areas, dozens of rooms each),
not an object-classification dataset -- there is no standard "one object per
file" split like ModelNet40/ShapeNet/ScanObjectNN ship natively. But the raw
annotated release already segments every room into per-instance object files
(`Annotations/<class>_<instance>.txt`, one row per point: "x y z r g b", no
header) -- exactly the same "extract individual object instances out of a
richer scanned-scene dataset" approach ScanObjectNN itself is built on. This
script does that extraction directly from the raw annotations (no room-level
parsing, no re-segmentation needed -- the instance boundaries are already
given).

Class selection: S3DIS's 13 semantic classes are mostly large architectural
surfaces (ceiling, floor, wall, beam, column) or a catch-all "clutter" bin --
not meaningful discrete rotatable objects, so they're excluded (same
rigid/non-flat/non-symmetric criterion used to pick ModelNet40/ShapeNet/ABO's
classes). Checked actual per-class instance counts across all 6 areas (after
full extraction -- an earlier count taken mid-extraction badly undercounted
everything, e.g. showed chair=135 when the true figure is chair=1363; always
verify counts against a *complete* extraction, not a still-in-progress one).
Final set: **chair, table, bookcase, door, sofa, board, window** (counts:
1363, 455, 583, 543, 55, 137, 168 respectively) -- 7 classes, still fewer
than the 10 used for the other domains since S3DIS's remaining categories
(ceiling/floor/wall/beam/column/clutter) are architectural surfaces or a
junk bin, not discrete rotatable objects.

Instances are pooled across all 6 areas (no area-based split -- S3DIS's
official benchmark split is for semantic segmentation cross-validation, not
relevant here) and given an arbitrary but deterministic train/test split
(first N_TRAIN_PER_CLASS sorted instances -> train, next N_TEST_PER_CLASS ->
test), same convention as ABO.

Usage:
    python -m pcrobust.datasets.s3dis   (or: pcrobust prepare s3dis)
"""

import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

from pcrobust.utils import normalize_and_sample
from pcrobust.datasets._paths import domain_root

ROOT = domain_root("s3dis")
_S3DIS_ROOT = ROOT
_DOWNLOAD_DIR = _S3DIS_ROOT / "_raw_download"
_DATA_DIR = _S3DIS_ROOT / "data"


CLASSES = ["chair", "table", "bookcase", "door", "sofa", "board", "window"]
N_TRAIN_PER_CLASS = 20
N_TEST_PER_CLASS = 12


def find_extract_root() -> Path:
    """Locate the extracted 'Stanford3dDataset_v1.2_Aligned_Version' root,
    wherever the zip happened to place it."""
    candidates = [p for p in _DOWNLOAD_DIR.rglob("Area_1") if p.is_dir()]
    if not candidates:
        sys.exit(f"[prepare] could not find an extracted 'Area_1' folder under "
                  f"{_DOWNLOAD_DIR} -- extract the S3DIS zip first")
    return candidates[0].parent


def find_instance_files(root: Path, cls: str) -> list[Path]:
    """All Annotations/<cls>_<i>.txt files across every Area/room, sorted for
    determinism."""
    return sorted(root.glob(f"Area_*/*/Annotations/{cls}_*.txt"))


def txt_to_points(path: Path) -> np.ndarray:
    """Parses 'x y z r g b' rows (no header); drops color, centers to origin."""
    # A handful of S3DIS annotation files have a stray non-numeric byte
    # (known dataset quirk in a few Area_5/Area_6 files) -- skip bad rows
    # rather than fail the whole instance.
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"no valid points parsed from {path}")
    pts = np.asarray(rows, dtype=np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts


def write_ply(pts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)


def process_class(root: Path, cls: str) -> tuple[int, int]:
    files = find_instance_files(root, cls)
    train_files = files[:N_TRAIN_PER_CLASS]
    test_files = files[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_TEST_PER_CLASS]

    n_train = n_test = 0
    for split, split_files, out_subdir, stem_fn in (
        ("train", train_files, _DATA_DIR / "train" / cls, lambda i: f"{cls}{i}"),
        ("test", test_files, _DATA_DIR / "test" / cls, lambda i: f"{cls}_test{i}"),
    ):
        i = 0
        for f in split_files:
            try:
                pts = txt_to_points(f)
            except Exception as e:
                print(f"[prepare] SKIP {f}: {e}")
                continue
            i += 1
            write_ply(pts, out_subdir / f"{stem_fn(i)}.ply")
        if split == "train":
            n_train = i
        else:
            n_test = i
        print(f"[prepare] {cls}/{split}: {i} instances written "
              f"({len(files)} total instances found across all areas)")

    return n_train, n_test


def prepare_dataset(classes=None):
    """Build the prepared subset (data/train, data/test, data/gallery)."""
    classes = list(classes) if classes is not None else list(CLASSES)

    root = find_extract_root()
    print(f"[prepare] extract root: {root}")

    totals = {}
    for cls in classes:
        n_train, n_test = process_class(root, cls)
        totals[cls] = (n_train, n_test)
        if n_train < N_TRAIN_PER_CLASS:
            print(f"[prepare] WARNING: {cls} only got {n_train}/{N_TRAIN_PER_CLASS} "
                  f"train instances -- rare class in S3DIS")

    print("\n[prepare] === Summary ===")
    for cls, (n_train, n_test) in totals.items():
        print(f"    {cls:<10s}  train={n_train:3d}  test={n_test:3d}")

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
    """Load a raw S3DIS instance .ply and return a canonical, centered,
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
# EXCLUDED_IDENTITIES is empty here: the S3DIS classes were
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
