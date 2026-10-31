"""Rotation corruptions.

Fixed-angle rotations (the pose studies' four regimes -- yaw is the primary
study; tilt, roll, and extreme reuse the exact same angle ladder on a
different axis / compound axis for direct comparability) plus the uniform
random SO(3) rotation used at training time by invariance-regularized
training (:mod:`pcrobust.corruptions.invreg`).

Angle ladder for the fixed-angle regimes:
:data:`pcrobust.corruptions.severity.ANGLES` -- [20, 40, 60, 80, 120]
degrees, with the confirmatory contrast D_j = Sim(80 deg) - Sim(120 deg).
"""

import numpy as np


def rotate_yaw(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotation about the yaw (Y axis) -- the pose studies' primary regime."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ R.T


def rotate_tilt(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotation about the tilt (pitch, X axis) -- same angle levels as the
    yaw study, different axis, for direct comparability."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return pts @ R.T


def rotate_roll(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotation about the roll (Z axis) -- same angle levels as the
    yaw study, different axis, for direct comparability."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return pts @ R.T


def rotate_extreme(pts: np.ndarray, deg: float) -> np.ndarray:
    """Compound rotation applying yaw, tilt, and roll simultaneously at the
    same magnitude -- the "extreme" combined-axis case, using the same
    angle levels as the single-axis studies (not a separate/larger angle
    range) so all 4 rotation studies stay directly comparable."""
    t = deg * np.pi / 180.0
    c, s = np.cos(t), np.sin(t)
    R_yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    R_tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    R_roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    R = R_yaw @ R_tilt @ R_roll
    return pts @ R.T


def random_rotation(pts: np.ndarray) -> np.ndarray:
    """Uniform random rotation matrix over SO(3) via axis-angle (Rodrigues)."""
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
    axis = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
    angle = np.random.uniform(0, 2 * np.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return (pts @ R.T).astype(np.float32)
