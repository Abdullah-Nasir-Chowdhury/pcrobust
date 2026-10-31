"""Shared augmentation, sampling, and dataset utilities.

utils_pose270yaw.py -- fork of utils.py (2026-08-12) with ONLY the yaw
augmentation range extended, from +/-80deg to +/-270deg (step 30 instead
of 20). Pitch (X) and roll (Z) ranges are UNCHANGED from utils.py
(+/-40deg, +/-30deg) -- this is the yaw-only counterpart to
utils_pose270.py (which extends all 3 axes to +/-270deg), for isolating
the effect of the yaw axis alone. Used only by the *_pose270yaw training
script variants; utils.py and utils_pose270.py are both untouched."""

import numpy as np
import torch
import open3d as o3d
from pathlib import Path
from torch.utils.data import Dataset

_DEG    = np.pi / 180
_Y_STEPS = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, -30, -60, -90, -120, -150, -180, -210, -240, -270], dtype=np.float32) * _DEG
_X_STEPS = np.array([0, 20, 40, -20, -40],                    dtype=np.float32) * _DEG
_Z_STEPS = np.array([0, 15, 30, -15, -30],                    dtype=np.float32) * _DEG


def _augment(pts: np.ndarray) -> np.ndarray:
    jitter = 8 * _DEG

    t = np.random.choice(_Y_STEPS) + np.random.uniform(-jitter, jitter)
    c, s = np.cos(t), np.sin(t)
    pts = pts @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32).T

    t = np.random.choice(_X_STEPS) + np.random.uniform(-jitter, jitter)
    c, s = np.cos(t), np.sin(t)
    pts = pts @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32).T

    t = np.random.choice(_Z_STEPS) + np.random.uniform(-jitter, jitter)
    c, s = np.cos(t), np.sin(t)
    pts = pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32).T

    pts += np.random.randn(*pts.shape).astype(np.float32) * 0.01
    np.clip(pts, -1.5, 1.5, out=pts)
    pts *= np.float32(np.random.uniform(0.9, 1.1))

    keep = np.random.uniform(0.7, 0.9)
    idx  = np.random.choice(len(pts), int(len(pts) * keep), replace=False)
    return pts[idx]


def _sample(pts: np.ndarray, n: int) -> np.ndarray:
    m   = len(pts)
    idx = np.random.choice(m, n, replace=(m < n))
    return pts[idx]


def normalize_and_sample(pcd: o3d.geometry.PointCloud,
                         n_points: int = 1024) -> o3d.geometry.PointCloud:
    """Center to origin, scale to unit sphere, FPS-sample to n_points."""
    pts = np.asarray(pcd.points).copy().astype(np.float64)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1).max()
    if r > 1e-6:
        pts /= r
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    n = len(pts)
    if n > n_points:
        out = out.farthest_point_down_sample(n_points)
    elif n < n_points:
        rng   = np.random.default_rng(0)
        extra = rng.integers(0, n, n_points - n)
        pad   = np.vstack([pts, pts[extra] + rng.standard_normal((n_points - n, 3)) * 1e-4])
        out.points = o3d.utility.Vector3dVector(pad)
    return out


class MeshGalleryDataset(Dataset):
    """
    Loads *_gallery.ply files from gallery_dir.
    Category parsed from stem: airplane1_gallery → airplane.
    Files with purely numeric stems (from data/test/) are skipped.
    """

    def __init__(self, gallery_dir: str, n_points: int = 1024,
                 augment: bool = True, repeats: int = 100):
        gallery_dir = Path(gallery_dir)
        ply_files = sorted(gallery_dir.glob("*_gallery.ply"))
        if not ply_files:
            raise FileNotFoundError(
                f"No *_gallery.ply files found in {gallery_dir}\n"
                f"Run the relevant domain's prepare_dataset.py first "
                f"(e.g. pcrobust.datasets.modelnet40.prepare_dataset())"
            )

        self.n_points   = n_points
        self.do_augment = augment
        self.repeats    = repeats if augment else 1

        raw, categories = [], []
        for f in ply_files:
            name = f.stem.replace("_gallery", "").rstrip("0123456789")
            if not name:
                continue
            pcd = o3d.io.read_point_cloud(str(f))
            pts = np.asarray(pcd.points, dtype=np.float32)
            raw.append(pts)
            categories.append(name)

        self.classes      = sorted(set(categories))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.raw          = raw
        self.labels       = [self.class_to_idx[n] for n in categories]

        print(f"[data]  {len(self.classes)} categories  |  {len(raw)} scans  |  "
              f"{len(self):,} samples ({repeats}x aug)")

    def __len__(self):
        return len(self.raw) * self.repeats

    def __getitem__(self, idx):
        i   = idx % len(self.raw)
        pts = self.raw[i].copy()
        if self.do_augment:
            pts = _augment(pts)
        pts = _sample(pts, self.n_points)
        return torch.from_numpy(pts), self.labels[i]
