"""
Training-time perturbation generators for invariance-regularized (invreg)
training. Reuses the exact perturbation definitions used by the benchmark's
evaluation pipeline, so the training objective matches what a model is
actually evaluated against later:
    - random_rotation: full SO(3) (uniform axis-angle), deliberately not
      restricted to yaw/tilt/roll individually -- the invariance loss should
      target the general rotation-robustness claim the pose studies test,
      not any one axis convention.
    - add_noise: identical to the noise studies' generator
      (re-exported from pcrobust.corruptions.noise).
    - occlude: Open3D Hidden Point Removal (Katz, Tal & Basri, 2007).
      Operates directly on a point cloud (no mesh needed, unlike
      ModelNet40-C's own mesh-render occlusion generator) so it works
      uniformly across every domain in the benchmark, including
      ScanObjectNN/S3DIS which have no clean meshes to render from.
      Severity is controlled via camera distance -- CORRECTED here from an
      earlier, wrong assumption: with HPR's radius scaled proportionally to
      camera distance (radius = camera_distance * 100, the standard/
      recommended Open3D usage -- see this function's own radius line), a
      CLOSE camera has an extremely wide effective field of view relative to
      a compact object and can see almost fisheye-style around much of its
      surface, while a FAR camera converges toward a strict single-
      hemisphere orthographic view. So FARTHER camera = MORE occlusion, not
      closer -- empirically verified across all 10 ModelNet40 pilot classes
      (see pcrobust.corruptions.occlusion's module docstring for the full
      derivation). For this module's *training-time* use this correction
      doesn't change functional behavior either way (each call draws one
      candidate distance to perturb with; there is no mild-vs-severe
      *comparison* the way the confirmatory evaluation pipeline's D_j
      statistic needs), but OCCLUSION_CAMERA_DISTANCES is still ordered
      mild-to-severe for documentation clarity and in case a future caller
      (e.g. a curriculum that starts mild and increases severity) relies on
      the ordering.

Unlike the evaluation generator (pcrobust.corruptions.occlusion.occlude),
this module's occlude takes an ABSOLUTE camera distance, draws its own
random camera direction, and resamples back to a fixed point count.
"""
import numpy as np
import open3d as o3d

from pcrobust.corruptions.noise import add_noise  # noqa: F401  (re-exported)
from pcrobust.corruptions.rotation import random_rotation  # noqa: F401  (re-exported)
from pcrobust.corruptions.severity import (  # noqa: F401  (re-exported)
    OCCLUSION_CAMERA_DISTANCES,
    SIGMAS,
)


def occlude(pts: np.ndarray, camera_distance: float = None, n_points: int = None) -> np.ndarray:
    """Hidden Point Removal from a random camera direction at camera_distance.
    Resamples back to n_points (default: len(pts)) with replacement, matching
    pcrobust.utils._sample()'s convention, so occlusion's point-count drop
    needs no special-casing downstream."""
    n_points = n_points or len(pts)
    camera_distance = camera_distance if camera_distance is not None else float(np.random.choice(OCCLUSION_CAMERA_DISTANCES))

    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    camera = (direction * camera_distance).tolist()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    radius = camera_distance * 100.0   # HPR's own recommended scale (Open3D docs)
    _, visible_idx = pcd.hidden_point_removal(camera, radius)
    visible = pts[np.asarray(visible_idx)]

    if len(visible) == 0:              # degenerate camera angle; fall back to no occlusion
        visible = pts
    idx = np.random.choice(len(visible), n_points, replace=(len(visible) < n_points))
    return visible[idx]
