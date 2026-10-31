"""ModelNet40 pose-robustness domain: download, preparation, and object pools.

Source: https://modelnet.cs.princeton.edu/ (Wu et al., 2015, "3D ShapeNets: A
Deep Representation for Volumetric Shapes"). ~1.9GB, .off mesh format,
organized as <class>/{train,test}/<class>_####.off with the official
train/test split baked in.

download_modelnet40() only downloads + verifies the archive; prepare_dataset()
does the selective extraction and preprocessing (no need to unzip the whole
1.9GB when only ~10 of the 40 classes are used):

  1. Selectively extract only the chosen classes' train/test .off meshes from
     the downloaded zip.
  2. For each selected .off: load as a triangle mesh, uniformly sample
     RAW_SAMPLE_POINTS points off the surface, center to the origin (unit-
     sphere scaling is deliberately left to extract_object_pts(), keeping
     "raw scan" and "canonical extraction" as separate pipeline stages), and
     write to data/train/<class>/<class><i>.ply or
     data/test/<class>/<class>_test<i>.ply. Uses the official ModelNet40
     train/test split (no custom splitting -- avoids leakage).
  3. Build the flat training gallery (data/gallery/<class><i>_gallery.ply)
     from the train split via build_gallery(), reusing
     pcrobust.utils.normalize_and_sample() directly.

Classes were chosen for having ample official instance counts AND rigid,
non-rotationally-symmetric geometry (meaningful yaw sensitivity) -- unlike
e.g. bowl/vase/cone, which are close to rotation-invariant about their own
axis and would make a weak pose-robustness testbed.

All paths live under <PCROBUST_HOME>/modelnet40/ (see
pcrobust.datasets._paths).

Usage:
    python -m pcrobust.datasets.modelnet40
"""

import sys
import argparse
import zipfile
import numpy as np
import open3d as o3d
from pathlib import Path
from urllib.request import urlopen

from pcrobust.utils import normalize_and_sample
from pcrobust.datasets._paths import domain_root

ROOT = domain_root("modelnet40")
_DOWNLOAD_DIR = ROOT / "_raw_download"
_ZIP_PATH = _DOWNLOAD_DIR / "ModelNet40.zip"
_EXTRACT_DIR = _DOWNLOAD_DIR / "extracted"
_DATA_DIR = ROOT / "data"
TRAIN_DIR = _DATA_DIR / "train"
TEST_DIR = _DATA_DIR / "test"
_URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"


def download(url: str, out_path: Path, chunk_size: int = 1 << 20) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {out_path}")
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        with open(out_path, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r[download] {written/1e6:8.1f} / {total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
    print(f"\n[download] done: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


def verify(zip_path: Path) -> None:
    print(f"[verify] checking {zip_path} is a valid zip ...")
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            sys.exit(f"[verify] CORRUPT entry in archive: {bad}")
        names = zf.namelist()
        print(f"[verify] OK -- {len(names)} entries")
        classes = sorted({n.split("/")[1] for n in names
                          if n.count("/") >= 1 and not n.split("/")[1].startswith("__MACOSX")})
        print(f"[verify] classes found: {len(classes)}")


def download_modelnet40():
    """Fetch and verify the official ModelNet40.zip (skipped if already cached)."""
    if _ZIP_PATH.exists() and _ZIP_PATH.stat().st_size > 0:
        print(f"[download] {_ZIP_PATH} already exists ({_ZIP_PATH.stat().st_size/1e6:.1f} MB), "
              f"skipping download")
    else:
        download(_URL, _ZIP_PATH)
    verify(_ZIP_PATH)


# ── Dataset preparation ────────────────────────────────────────────────────────

CLASSES = [
    "airplane", "bed", "car", "chair", "guitar",
    "monitor", "piano", "sofa", "table", "toilet",
]
N_TRAIN_PER_CLASS = 45
N_TEST_PER_CLASS = 30
RAW_SAMPLE_POINTS = 2048


# ── Step 1: selective extraction from the zip ──────────────────────────────────

def selective_extract(classes: list[str]) -> Path:
    """Extract only train/test .off files for `classes` from ModelNet40.zip.
    Returns the root directory containing per-class folders."""
    if not _ZIP_PATH.exists():
        sys.exit(f"[prepare] {_ZIP_PATH} not found -- run download_modelnet40.py first")

    with zipfile.ZipFile(_ZIP_PATH) as zf:
        names = zf.namelist()
        wanted = []
        for cls in classes:
            train_prefix = f"/{cls}/train/"
            test_prefix = f"/{cls}/test/"
            hits = [n for n in names
                    if (train_prefix in f"/{n}" or test_prefix in f"/{n}")
                    and n.lower().endswith(".off")]
            if not hits:
                print(f"[prepare] WARNING: no .off files found for class {cls!r} in zip")
            wanted.extend(hits)

        print(f"[prepare] extracting {len(wanted)} files for {len(classes)} classes "
              f"-> {_EXTRACT_DIR}")
        _EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        for n in wanted:
            zf.extract(n, _EXTRACT_DIR)

    # Find the actual root: the first path component common to all extracted files
    sample = wanted[0] if wanted else None
    if sample is None:
        sys.exit("[prepare] nothing extracted -- check CLASSES against the zip contents")
    root_component = sample.split("/")[0]
    return _EXTRACT_DIR / root_component


def find_class_dir(root: Path, cls: str) -> Path:
    """Locate the extracted <root>/.../<cls> directory (handles the zip's
    actual nesting, whatever it turns out to be)."""
    matches = [p for p in root.rglob(cls) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"class dir for {cls!r} not found under {root}")
    return matches[0]


# ── Step 2: mesh -> point cloud ────────────────────────────────────────────────

def read_off_robust(off_path: Path) -> tuple[np.ndarray, list[list[int]]]:
    """Custom .off parser. A known chunk of the Princeton ModelNet40 archive
    (observed in e.g. table/, sofa/, monitor/) has a malformed header where
    the "OFF" tag and the vertex/face counts are jammed onto one line
    ("OFF12636 8652 0" instead of "OFF\\n12636 8652 0"), which Open3D's
    strict OFF reader rejects outright. Parsed manually here so both the
    normal and malformed variants load correctly."""
    lines = [ln for ln in off_path.read_text(encoding="ascii", errors="ignore").splitlines()
             if ln.strip() != ""]
    if not lines:
        raise ValueError(f"empty file: {off_path}")
    first = lines[0].strip()
    if first == "OFF":
        header_line, data_lines = lines[1], lines[2:]
    elif first.startswith("OFF"):
        header_line, data_lines = first[3:], lines[1:]
    else:
        raise ValueError(f"not an OFF file (bad magic): {off_path}")

    n_verts, n_faces = (int(v) for v in header_line.split()[:2])
    verts = np.array([[float(v) for v in data_lines[i].split()[:3]]
                      for i in range(n_verts)], dtype=np.float64)
    faces = []
    for i in range(n_faces):
        vals = data_lines[n_verts + i].split()
        k = int(vals[0])
        faces.append([int(v) for v in vals[1:1 + k]])
    return verts, faces


def off_to_points(off_path: Path, n_points: int = RAW_SAMPLE_POINTS) -> np.ndarray:
    verts, faces = read_off_robust(off_path)

    triangles = []
    for f in faces:
        for i in range(1, len(f) - 1):          # fan-triangulate polygons
            triangles.append([f[0], f[i], f[i + 1]])

    if triangles:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
        mesh.remove_degenerate_triangles()
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            pts = np.asarray(pcd.points, dtype=np.float32)
        else:
            triangles = []

    if not triangles:
        # Degenerate mesh (no valid triangles) -- fall back to vertices directly.
        pts = verts.astype(np.float32)
        if len(pts) == 0:
            raise RuntimeError(f"no triangles or vertices in {off_path}")
        if len(pts) > n_points:
            idx = np.random.choice(len(pts), n_points, replace=False)
            pts = pts[idx]

    pts = pts - pts.mean(axis=0, keepdims=True)
    return pts.astype(np.float32)


def write_ply(pts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(out_path), pcd)


# ── Step 3: per-class processing ───────────────────────────────────────────────

def process_class(extract_root: Path, cls: str) -> tuple[int, int]:
    class_dir = find_class_dir(extract_root, cls)
    n_train = n_test = 0

    # Test files are named "<class>_test<i>.ply" (not "<class><i>.ply") so their
    # stems never collide with train instance stems -- object_pool.py's "scan"
    # identifier is just path.stem, and pilot.py/confirmatory.py/analysis.py key
    # data by (scan, angle), so a train/test stem collision (e.g. both splits'
    # first file being "chair1") would silently merge two different physical
    # instances into one confirmatory row. Mirrors the original pose study's own
    # data/test/<identity>/<identity>_test<i>.ply naming for the same reason.
    for split, n_wanted, out_subdir, stem_fn in (
        ("train", N_TRAIN_PER_CLASS, _DATA_DIR / "train" / cls, lambda i: f"{cls}{i}"),
        ("test", N_TEST_PER_CLASS, _DATA_DIR / "test" / cls, lambda i: f"{cls}_test{i}"),
    ):
        split_dir = class_dir / split
        if not split_dir.exists():
            print(f"[prepare] WARNING: {split_dir} missing, skipping {split} split for {cls}")
            continue
        off_files = sorted(split_dir.glob("*.off"))[:n_wanted]
        i = 0
        for off_path in off_files:
            try:
                pts = off_to_points(off_path)
            except Exception as e:
                print(f"[prepare] SKIP {off_path.name}: {e}")
                continue
            i += 1
            out_path = out_subdir / f"{stem_fn(i)}.ply"
            write_ply(pts, out_path)
        if split == "train":
            n_train = i
        else:
            n_test = i
        print(f"[prepare] {cls}/{split}: {i}/{len(off_files)} instances written")

    return n_train, n_test


def prepare_dataset(classes=None, skip_extract: bool = False):
    """Build the prepared ModelNet40 subset (data/train, data/test, data/gallery)."""
    classes = list(classes) if classes is not None else list(CLASSES)

    if skip_extract and _EXTRACT_DIR.exists():
        candidates = [p for p in _EXTRACT_DIR.iterdir() if p.is_dir()]
        extract_root = candidates[0] if candidates else _EXTRACT_DIR
    else:
        extract_root = selective_extract(classes)

    print(f"[prepare] extract root: {extract_root}")

    totals = {}
    for cls in classes:
        try:
            n_train, n_test = process_class(extract_root, cls)
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
    """Load a raw ModelNet40 instance .ply and return a canonical, centered,
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
# EXCLUDED_IDENTITIES is empty here: the 10 ModelNet40 classes were
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
    parser.add_argument("--skip_extract", action="store_true",
                        help="Reuse an already-extracted _raw_download/extracted/")
    args = parser.parse_args()
    prepare_dataset(classes=args.classes, skip_extract=args.skip_extract)
