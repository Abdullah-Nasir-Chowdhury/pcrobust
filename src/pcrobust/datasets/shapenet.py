"""ShapeNet pose-robustness domain: download, preparation, and object pools.

download_shapenet() fetches the ShapeNet-Part point-cloud benchmark
(shapenetcore_partanno_segmentation_benchmark_v0_normal.zip), the standard
per-shape point-cloud dataset used by reference PointNet/PointNet++ PyTorch
implementations (16 categories, .txt point clouds with x y z nx ny nz label
per line, official train/val/test split included).

Source note: the official host (shapenet.cs.stanford.edu) has been
unreachable (connection refused) since at least Oct 2024, and ShapeNetCore's
gated site / HuggingFace mirror both require account login, which this
package does not automate. The download instead fetches the same file from
a long-standing public Google Drive mirror (file ID
1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS) -- the same ID hardcoded into well-known
open-source ShapeNet dataloaders (e.g. Pointnet2_PyTorch) -- via `gdown`,
which handles Google Drive's large-file confirmation-token flow.

Note: despite the ".zip" filename (kept for parity with the well-known
filename everyone refers to this dataset by), the Google Drive mirror
actually serves a POSIX tar archive, not a zip -- confirmed by inspecting
the downloaded bytes (`ustar` header). verify()/prepare_dataset() use
`tarfile`, not `zipfile`, accordingly.

prepare_dataset() pipeline:
  1. Selectively extract only the chosen classes' shape .txt files from the
     downloaded archive (not the whole 3.1GB).
  2. For each selected shape .txt: parse whitespace-separated
     "x y z nx ny nz label" rows (no header), drop normals/label, center to
     the origin. Unlike ModelNet40's meshes, these are ALREADY point clouds
     (no mesh-sampling step needed) -- write directly to
     data/train/<class>/<class><i>.ply or data/test/<class>/<class>_test<i>.ply.
     Uses the official ShapeNet-Part train/val/test split (train+val treated
     as "train" here; the benchmark has no other use for the val split) -- no
     custom splitting, avoids leakage.
  3. Build the flat training gallery (data/gallery/<class><i>_gallery.ply)
     from the train split via build_gallery(), reusing
     pcrobust.utils.normalize_and_sample() directly.

Classes were chosen for having ample official instance counts AND rigid,
non-rotationally-symmetric geometry (meaningful yaw sensitivity) -- unlike
e.g. cap/mug/rocket, which are close to rotation-invariant about their own
axis (or too rare) and would make a weak pose-robustness testbed.

All paths live under <PCROBUST_HOME>/shapenet/ (see
pcrobust.datasets._paths).

Usage:
    python -m pcrobust.datasets.shapenet
"""

import sys
import json
import argparse
import tarfile
import numpy as np
import open3d as o3d
from pathlib import Path

from pcrobust.utils import normalize_and_sample
from pcrobust.datasets._paths import domain_root

ROOT = domain_root("shapenet")
_DOWNLOAD_DIR = ROOT / "_raw_download"
_ZIP_PATH = _DOWNLOAD_DIR / "shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"
_ARCHIVE_PATH = _ZIP_PATH
_ARCHIVE_ROOT_NAME = "shapenetcore_partanno_segmentation_benchmark_v0_normal"
_EXTRACT_DIR = _DOWNLOAD_DIR / "extracted"
_DATA_DIR = ROOT / "data"
TRAIN_DIR = _DATA_DIR / "train"
TEST_DIR = _DATA_DIR / "test"
_GDRIVE_ID = "1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS"


def download(out_path: Path) -> None:
    import gdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] gdown id={_GDRIVE_ID} -> {out_path}")
    gdown.download(id=_GDRIVE_ID, output=str(out_path), quiet=False)


def verify(zip_path: Path) -> None:
    print(f"[verify] checking {zip_path} is a valid (tar) archive ...")
    with tarfile.open(zip_path, mode="r:*") as tf:
        names = tf.getnames()
        print(f"[verify] OK -- {len(names)} entries")


def download_shapenet():
    """Fetch and verify the ShapeNet-Part archive (skipped if already cached)."""
    if _ZIP_PATH.exists() and _ZIP_PATH.stat().st_size > 0:
        print(f"[download] {_ZIP_PATH} already exists "
              f"({_ZIP_PATH.stat().st_size/1e6:.1f} MB), skipping download")
    else:
        download(_ZIP_PATH)
    verify(_ZIP_PATH)


# ── Dataset preparation ────────────────────────────────────────────────────────

# class name (lowercase, used throughout this benchmark) -> ShapeNetCore synset ID
CLASS_TO_SYNSET = {
    "airplane":  "02691156",
    "car":       "02958343",
    "chair":     "03001627",
    "guitar":    "03467517",
    "knife":     "03624134",
    "laptop":    "03642806",
    "motorbike": "03790512",
    "pistol":    "03948459",
    "skateboard": "04225987",
    "table":     "04379243",
}
CLASSES = list(CLASS_TO_SYNSET)
N_TRAIN_PER_CLASS = 45
N_TEST_PER_CLASS = 25


# ── Step 1: selective extraction from the archive ──────────────────────────────

def load_split(archive: tarfile.TarFile, split: str) -> list[str]:
    """Returns a list of 'shape_data/<synset>/<shapeid>' entries for the given
    official split ('train', 'val', or 'test')."""
    member = archive.getmember(
        f"{_ARCHIVE_ROOT_NAME}/train_test_split/shuffled_{split}_file_list.json")
    f = archive.extractfile(member)
    return json.loads(f.read().decode("utf-8"))


def selective_extract(classes: list[str]) -> Path:
    """Extract only the .txt shape files needed for `classes`' train+val+test
    pools from the archive. Returns the directory containing per-synset
    folders (<extract_dir>/<synset>/<shapeid>.txt)."""
    if not _ARCHIVE_PATH.exists():
        sys.exit(f"[prepare] {_ARCHIVE_PATH} not found -- run download_shapenet.py first")

    wanted_synsets = {CLASS_TO_SYNSET[c] for c in classes}

    with tarfile.open(_ARCHIVE_PATH, mode="r:*") as tf:
        train_ids = load_split(tf, "train")
        val_ids = load_split(tf, "val")
        test_ids = load_split(tf, "test")

        def synset_of(entry: str) -> str:
            return entry.split("/")[1]

        wanted_entries = [e for e in (train_ids + val_ids + test_ids)
                          if synset_of(e) in wanted_synsets]
        wanted_members = [f"{_ARCHIVE_ROOT_NAME}/{e.split('/', 1)[1]}.txt"
                          for e in wanted_entries]

        print(f"[prepare] extracting {len(wanted_members)} shape files for "
              f"{len(classes)} classes -> {_EXTRACT_DIR}")
        _EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        members_by_name = {m.name: m for m in tf.getmembers()}
        missing = 0
        for name in wanted_members:
            m = members_by_name.get(name)
            if m is None:
                missing += 1
                continue
            tf.extract(m, _EXTRACT_DIR)
        if missing:
            print(f"[prepare] WARNING: {missing} listed shape files not found in archive")

    return _EXTRACT_DIR / _ARCHIVE_ROOT_NAME, train_ids, val_ids, test_ids


# ── Step 2: shape .txt -> point cloud ───────────────────────────────────────────

def txt_to_points(txt_path: Path) -> np.ndarray:
    """Parse 'x y z nx ny nz label' rows (no header), drop normals/label,
    center to origin. Already a point cloud -- no mesh sampling needed."""
    arr = np.loadtxt(txt_path, dtype=np.float32)
    pts = arr[:, :3]
    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts.astype(np.float32)


def write_ply(pts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)


# ── Step 3: per-class processing ───────────────────────────────────────────────

def process_class(extract_root: Path, cls: str,
                   train_ids: list[str], val_ids: list[str], test_ids: list[str]) -> tuple[int, int]:
    synset = CLASS_TO_SYNSET[cls]

    def shape_ids_for(entries: list[str]) -> list[str]:
        ids = sorted(e.split("/")[2] for e in entries if e.split("/")[1] == synset)
        return ids

    # ShapeNet-Part's own val split isn't used elsewhere in this repo --
    # folded into "train" here (mirrors how ModelNet40's official train split
    # alone fed this study's train pool).
    train_shape_ids = shape_ids_for(train_ids + val_ids)
    test_shape_ids = shape_ids_for(test_ids)

    n_train = n_test = 0
    for split, shape_ids, n_wanted, out_subdir, stem_fn in (
        ("train", train_shape_ids, N_TRAIN_PER_CLASS, _DATA_DIR / "train" / cls, lambda i: f"{cls}{i}"),
        ("test", test_shape_ids, N_TEST_PER_CLASS, _DATA_DIR / "test" / cls, lambda i: f"{cls}_test{i}"),
    ):
        chosen = shape_ids[:n_wanted]
        i = 0
        for shape_id in chosen:
            txt_path = extract_root / synset / f"{shape_id}.txt"
            if not txt_path.exists():
                print(f"[prepare] SKIP {synset}/{shape_id}.txt: not extracted")
                continue
            try:
                pts = txt_to_points(txt_path)
            except Exception as e:
                print(f"[prepare] SKIP {synset}/{shape_id}.txt: {e}")
                continue
            i += 1
            out_path = out_subdir / f"{stem_fn(i)}.ply"
            write_ply(pts, out_path)
        if split == "train":
            n_train = i
        else:
            n_test = i
        print(f"[prepare] {cls}/{split}: {i}/{len(chosen)} instances written "
              f"(pool had {len(shape_ids)} available)")

    return n_train, n_test


def prepare_dataset(classes=None, skip_extract: bool = False):
    """Build the prepared subset (data/train, data/test, data/gallery)."""
    classes = list(classes) if classes is not None else list(CLASSES)

    if skip_extract and _EXTRACT_DIR.exists():
        extract_root = _EXTRACT_DIR / _ARCHIVE_ROOT_NAME
        with tarfile.open(_ARCHIVE_PATH, mode="r:*") as tf:
            train_ids = load_split(tf, "train")
            val_ids = load_split(tf, "val")
            test_ids = load_split(tf, "test")
    else:
        extract_root, train_ids, val_ids, test_ids = selective_extract(classes)

    print(f"[prepare] extract root: {extract_root}")

    totals = {}
    for cls in classes:
        try:
            n_train, n_test = process_class(extract_root, cls, train_ids, val_ids, test_ids)
            totals[cls] = (n_train, n_test)
        except Exception as e:
            print(f"[prepare] ERROR processing class {cls!r}: {e}")

    print("\n[prepare] === Summary ===")
    for cls, (n_train, n_test) in totals.items():
        print(f"    {cls:<10s}  train={n_train:3d}  test={n_test:3d}")

    print("\n[prepare] Building training gallery (data/gallery/*_gallery.ply) ...")
    n_gallery = build_gallery(_DATA_DIR / "train", _DATA_DIR / "gallery")
    print(f"[prepare] wrote {n_gallery} gallery files")


# ── Canonical extraction ───────────────────────────────────────────────────────
# Each raw .ply produced by prepare_dataset() already IS a single clean object,
# so there is no segmentation to do -- the only step needed is canonicalization:
# centering to origin, scaling to unit sphere, and farthest-point-sampling to
# n_points, reused directly via pcrobust.utils.normalize_and_sample(), not
# reimplemented. extract_object_pts() returns a fresh (N, 3) canonical point
# array given a raw .ply path; the pose evaluation calls this once per raw
# scan, then applies pure yaw rotation to the result.


def extract_object_pts(path: str, n_points: int = 1024) -> np.ndarray:
    """Load a raw ShapeNet instance .ply and return a canonical, centered,
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
# EXCLUDED_IDENTITIES is empty here: the ShapeNet classes were
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
    parser.add_argument("--skip_extract", action="store_true")
    args = parser.parse_args()
    prepare_dataset(classes=args.classes, skip_extract=args.skip_extract)
