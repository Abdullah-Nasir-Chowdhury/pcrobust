"""
prepare_dataset.py — builds the ABO (Amazon-Berkeley Objects) subset used by
the pose-robustness replication, from individual glTF meshes fetched
directly off the public ABO S3 bucket (no login/account required, unlike
OmniObject3D's OpenXLab-gated download -- see the main README's note on why
ABO was substituted in as the "recent dataset" pick).

Source: https://amazon-berkeley-objects.s3.amazonaws.com/ (Amazon-Berkeley
Objects, CVPR 2022 workshop paper, CC BY-NC 4.0). 7,953 products ship
high-quality glTF 2.0 meshes under `3dmodels/original/<bucket>/<id>.glb`,
indexed by `3dmodels/metadata/3dmodels.csv.gz` (id -> path) and categorized
via `listings/listings.json.gz` (id -> product_type). Individual .glb files
are fetched directly (2-50MB each) -- never the full 154GB archive.

Pipeline:
  1. From the cached listings/3dmodels metadata (downloaded once, ~85MB,
     kept under <PCROBUST_HOME>/abo/_raw_download/), pick the 10 chosen
     classes' 3dmodel_ids, sorted for determinism.
  2. Download each needed .glb (skipped if already cached locally).
  3. Load with Open3D, uniformly sample points off the mesh surface, center
     to the origin (unit-sphere scaling left to extract.py's
     extract_object_pts(), same convention as ModelNet40/ShapeNet).
  4. No official train/test split exists for ABO (it isn't an ML benchmark
     dataset) -- an arbitrary but deterministic split is used instead: the
     first N_TRAIN_PER_CLASS ids (by sorted 3dmodel_id) go to train, the next
     N_TEST_PER_CLASS to test.

Classes chosen for rigid, non-flat, non-rotationally-symmetric geometry and
ample instance counts (same criteria as ModelNet40/ShapeNet's own class
selection) -- excludes flat/deformable categories like RUG/WALL_ART/PILLOW
that would make a weak pose-robustness testbed.

Usage:
    python -m pcrobust.datasets.abo   (or: pcrobust prepare abo)
"""

import sys
import gzip
import json
import csv
import argparse
import urllib.request
import numpy as np
import open3d as o3d
from pathlib import Path

from pcrobust.utils import normalize_and_sample
from pcrobust.datasets._paths import domain_root

ROOT = domain_root("abo")
_ABO_ROOT = ROOT
_DOWNLOAD_DIR = _ABO_ROOT / "_raw_download"
_GLB_CACHE = _DOWNLOAD_DIR / "glb_cache"
_DATA_DIR = _ABO_ROOT / "data"
_S3_BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/"


CLASSES = [
    "CHAIR", "SOFA", "TABLE", "LAMP", "STOOL_SEATING",
    "OTTOMAN", "HEADBOARD", "BED", "DESK", "CABINET",
]
N_TRAIN_PER_CLASS = 20
N_TEST_PER_CLASS = 12
RAW_SAMPLE_POINTS = 2048


def load_metadata():
    """Returns (id -> relative glb path) and (class -> sorted list of ids)."""
    listings_path = _DOWNLOAD_DIR / "listings.json.gz"
    models_csv_path = _DOWNLOAD_DIR / "3dmodels.csv.gz"
    if not listings_path.exists() or not models_csv_path.exists():
        sys.exit(f"[prepare] metadata not found under {_DOWNLOAD_DIR} -- "
                  f"expected listings.json.gz and 3dmodels.csv.gz")

    id_to_path = {}
    with gzip.open(models_csv_path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_path[row["3dmodel_id"]] = row["path"]

    by_class = {c: [] for c in CLASSES}
    with gzip.open(listings_path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            mid = d.get("3dmodel_id")
            pt = d.get("product_type")
            if not mid or not pt or mid not in id_to_path:
                continue
            cls = pt[0]["value"]
            if cls in by_class:
                by_class[cls].append(mid)

    for cls in by_class:
        by_class[cls] = sorted(set(by_class[cls]))
    return id_to_path, by_class


def fetch_glb(model_id: str, rel_path: str) -> Path:
    out_path = _GLB_CACHE / f"{model_id}.glb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = _S3_BASE + "3dmodels/original/" + rel_path
    urllib.request.urlretrieve(url, out_path)
    return out_path


def glb_to_points(glb_path: Path, n_points: int = RAW_SAMPLE_POINTS) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(glb_path))
    mesh.remove_degenerate_triangles()
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"no valid triangles in {glb_path}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    pts = np.asarray(pcd.points, dtype=np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts.astype(np.float32)


def write_ply(pts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)


def process_class(cls: str, ids: list[str], id_to_path: dict) -> tuple[int, int]:
    cls_lower = cls.lower()
    n_train = n_test = 0
    train_ids = ids[:N_TRAIN_PER_CLASS]
    test_ids = ids[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_TEST_PER_CLASS]

    for split, split_ids, out_subdir, stem_fn in (
        ("train", train_ids, _DATA_DIR / "train" / cls_lower, lambda i: f"{cls_lower}{i}"),
        ("test", test_ids, _DATA_DIR / "test" / cls_lower, lambda i: f"{cls_lower}_test{i}"),
    ):
        i = 0
        for mid in split_ids:
            try:
                glb_path = fetch_glb(mid, id_to_path[mid])
                pts = glb_to_points(glb_path)
            except Exception as e:
                print(f"[prepare] SKIP {mid}: {e}")
                continue
            i += 1
            write_ply(pts, out_subdir / f"{stem_fn(i)}.ply")
        if split == "train":
            n_train = i
        else:
            n_test = i
        print(f"[prepare] {cls_lower}/{split}: {i}/{len(split_ids)} instances written")

    return n_train, n_test


def prepare_dataset(classes=None):
    """Build the prepared subset (data/train, data/test, data/gallery)."""
    classes = list(classes) if classes is not None else list(CLASSES)

    id_to_path, by_class = load_metadata()

    totals = {}
    for cls in classes:
        ids = by_class.get(cls, [])
        if len(ids) < N_TRAIN_PER_CLASS + N_TEST_PER_CLASS:
            print(f"[prepare] WARNING: {cls} has only {len(ids)} candidates "
                  f"(need {N_TRAIN_PER_CLASS + N_TEST_PER_CLASS})")
        n_train, n_test = process_class(cls, ids, id_to_path)
        totals[cls] = (n_train, n_test)

    print("\n[prepare] === Summary ===")
    for cls, (n_train, n_test) in totals.items():
        print(f"    {cls.lower():<15s}  train={n_train:3d}  test={n_test:3d}")

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
    """Load a raw ABO instance .ply and return a canonical, centered,
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
# EXCLUDED_IDENTITIES is empty here: the ABO classes were
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
