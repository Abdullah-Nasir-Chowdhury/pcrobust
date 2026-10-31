"""Sampling utilities: random resampling to a fixed point count and
normalize-then-FPS preprocessing."""

import numpy as np
import open3d as o3d


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
