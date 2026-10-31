"""Corruption generators and severity ladders for the benchmark.

Evaluation-side generators (the ones the robustness studies use):

- rotation: :func:`rotate_yaw` (primary), :func:`rotate_tilt`,
  :func:`rotate_roll`, :func:`rotate_extreme`
- noise: :func:`add_noise`
- occlusion: :func:`occlude` (+ :func:`random_direction` /
  :data:`FIXED_DIRECTION`)

Severity ladders (:mod:`pcrobust.corruptions.severity`): :data:`ANGLES`,
:data:`SIGMAS`, :data:`LEVELS` / :data:`MILD` / :data:`SEVERE`,
:data:`OCCLUSION_CAMERA_DISTANCES`.

``occlude`` here is the evaluation generator (camera distance as a multiple
of the cloud's bounding-box diagonal, caller-supplied direction, no
resampling).  The training-time variant used by invariance-regularized
training -- absolute camera distance, self-drawn random direction,
resampling back to a fixed point count -- keeps its original name in
:mod:`pcrobust.corruptions.invreg` and is deliberately NOT re-exported
here, to avoid shadowing.
"""

from pcrobust.corruptions import invreg  # noqa: F401  (training-time variants)
from pcrobust.corruptions.noise import add_noise
from pcrobust.corruptions.occlusion import (
    FIXED_DIRECTION,
    LEVELS,
    MILD,
    SEVERE,
    occlude,
    random_direction,
)
from pcrobust.corruptions.rotation import (
    random_rotation,
    rotate_extreme,
    rotate_roll,
    rotate_tilt,
    rotate_yaw,
)
from pcrobust.corruptions.severity import ANGLES, OCCLUSION_CAMERA_DISTANCES, SIGMAS

__all__ = [
    "ANGLES",
    "FIXED_DIRECTION",
    "LEVELS",
    "MILD",
    "OCCLUSION_CAMERA_DISTANCES",
    "SEVERE",
    "SIGMAS",
    "add_noise",
    "invreg",
    "occlude",
    "random_direction",
    "random_rotation",
    "rotate_extreme",
    "rotate_roll",
    "rotate_tilt",
    "rotate_yaw",
]
