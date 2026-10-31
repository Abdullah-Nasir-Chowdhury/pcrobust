"""Training-time augmentation: stepped yaw/tilt/roll rotations with jitter,
Gaussian noise, clipping, scaling, and random point dropout."""

import numpy as np

_DEG    = np.pi / 180
_Y_STEPS = np.array([0, 20, 40, 60, 80, -20, -40, -60, -80], dtype=np.float32) * _DEG
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
