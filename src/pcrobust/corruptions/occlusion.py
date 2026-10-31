"""
Point-cloud-native occlusion generator using Open3D's Hidden Point Removal
(HPR, Katz, Tal & Basri 2007), the same physical definition ModelNet40-C's
own "occlusion" corruption uses (points visible from one viewpoint) but
operating directly on a point cloud instead of a mesh render. This is what
makes it usable uniformly across all 5 domains, including
ScanObjectNN/S3DIS, which have no clean source mesh to render an occlusion
mask from -- HPR only ever needs the point cloud itself.

Severity = camera distance from the object, expressed as a multiple of the
point cloud's own bounding-box diagonal (dimensionless, so the same LEVELS
work across every domain regardless of that domain's absolute scale -- every
domain's extraction step already unit-sphere-normalizes anyway, but the
multiplier design doesn't depend on that).

IMPORTANT, empirically-verified correction to the naive assumption that
"closer camera -> more occlusion" (the assumption the occlusion study
started from): with radius scaled proportionally to camera distance (radius =
camera_dist * 100, the standard/recommended HPR usage per Open3D's own
hidden_point_removal tutorial), a CLOSE camera has an extremely wide
effective field of view relative to a compact object and can see almost
fisheye-style around much of its surface, while a FAR camera converges
toward a strict single-hemisphere orthographic view showing only what
directly faces it. Verified empirically across all 10 ModelNet40 pilot
classes before committing to this direction, not assumed: visible fraction
at dist_mult=0.3 ranged ~0.32-0.72 across classes; at dist_mult=6.0 it
consistently dropped to ~0.20-0.47, monotonically decreasing in between for
every class tested. So in the occlusion study, LARGER camera distance = MORE
severe occlusion -- the opposite of the initial design assumption --
documented here explicitly so nobody re-derives (or silently assumes the
wrong direction) later.

LEVELS (camera-distance multipliers of the point cloud's own bounding-box
diagonal) chosen to span a meaningful mild-to-severe range, mirroring how
ANGLES/SIGMAS were chosen for the rotation/noise studies (a fixed, a priori
design choice checked for a sane spread across all 10 ModelNet40 classes,
not literally re-derived from the pilot every time -- the pilot's job is to
estimate sigma_D for the power calculation, same convention those two
studies already established):
    LEVELS = [0.3, 0.6, 1.2, 2.5, 6.0]
mild = LEVELS[0] (0.3, most of the object visible), severe = LEVELS[-1]
(6.0, most of the object hidden) -- D = Sim(mild) - Sim(severe), same
structure as noise's D = Sim(sigma=0.005) - Sim(sigma=0.05) and the rotation
studies' D = Sim(80deg) - Sim(120deg).

Two azimuth (camera-direction) variants, mirroring the tilt/roll/extreme
pattern from the rotation-axis expansion:
- random (default/primary): a fresh random direction is drawn per call,
  matching the noise study's philosophy of averaging over random
  perturbations, and matching ModelNet40-C's own occlusion corruption
  (random viewpoint per instance).
- fixed ("from behind", the occlusion study's fixed-view variant): the same
  camera direction every time, answering a different, more specific
  question -- can the architecture recognize the object from one consistent
  viewpoint only, rather than "on average over random viewpoints."
"""

import numpy as np
import open3d as o3d

from pcrobust.corruptions.severity import LEVELS, MILD, SEVERE  # noqa: F401  (re-exported)

# "From behind" for the fixed-azimuth variant -- an arbitrary but fixed unit
# vector, consistent across every instance/replicate/domain in that variant.
FIXED_DIRECTION = np.array([0.0, 0.0, 1.0])


def random_direction(rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def occlude(pts: np.ndarray, dist_mult: float, direction: np.ndarray) -> np.ndarray:
    """Removes points hidden from a camera placed at `direction * dist_mult *
    diagonal`, using Open3D's hidden_point_removal. `direction` must be a
    unit vector -- callers decide random vs. fixed (see module docstring).
    Returns the occluded point cloud WITHOUT resampling back to a fixed
    point count -- callers must use pcrobust.utils._sample() afterward, as
    every domain's pilot/confirmatory scripts already do for every other
    perturbation type (rotation, noise); this is not a new requirement."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    if diameter == 0:
        return pts  # degenerate (all points identical) -- nothing to occlude

    cam_dist = diameter * dist_mult
    camera = (direction * cam_dist).tolist()
    radius = cam_dist * 100.0

    _, pt_map = pcd.hidden_point_removal(camera, radius)
    if len(pt_map) == 0:
        return pts  # pathological -- keep all points rather than returning empty
    return pts[pt_map]
