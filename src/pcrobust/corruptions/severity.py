"""The benchmark's severity ladders, exactly as the underlying studies define them.

Every corruption type uses a fixed, a-priori five-level ladder ordered mild
(index 0) -> severe (index 4).  Each study's confirmatory statistic contrasts
one mild and one severe rung of its ladder.

Rotation (pose studies) -- ``ANGLES``, degrees.  The yaw (primary), tilt,
roll, and extreme (compound yaw+tilt+roll) regimes all reuse this same
ladder, differing only in rotation axis, for direct comparability (see
:mod:`pcrobust.corruptions.rotation`)::

    index:   0    1    2    3     4
    angle:  20   40   60   80   120        (degrees)

    D_j = Sim(80 deg) - Sim(120 deg)       mild = 80, severe = 120

Gaussian noise (noise studies) -- ``SIGMAS``, standard deviation of isotropic
per-point jitter on unit-sphere-normalized clouds::

    index:    0      1      2      3      4
    sigma:  0.005   0.01   0.02   0.03   0.05

    D_j = Sim(sigma=0.005) - Sim(sigma=0.05)   mild = 0.005, severe = 0.05

Occlusion (occlusion studies, evaluation-side) -- ``LEVELS``, hidden-point-
removal camera distance expressed as a multiple of the point cloud's own
bounding-box diagonal (dimensionless, so the same ladder works across every
domain regardless of absolute scale).  LARGER camera distance = MORE severe
occlusion -- see :mod:`pcrobust.corruptions.occlusion` for the empirically
verified derivation.  ``MILD``/``SEVERE`` are the ladder's endpoints::

    index:       0     1     2     3     4
    dist_mult:  0.3   0.6   1.2   2.5   6.0

    D_j = Sim(MILD) - Sim(SEVERE)          MILD = 0.3, SEVERE = 6.0

Occlusion (training-time, invariance-regularized training) --
``OCCLUSION_CAMERA_DISTANCES``, ABSOLUTE camera distances (clouds at
unit-sphere scale), ordered mild (close) -> severe (far).  One value is drawn
uniformly at random per call by :func:`pcrobust.corruptions.invreg.occlude`;
there is no mild-vs-severe contrast at training time, the ordering is kept
for documentation clarity and for any future caller (e.g. a severity
curriculum) that relies on it::

    index:       0     1     2      3      4
    distance:   2.0   4.0   8.0   20.0   100.0
"""

ANGLES = [20, 40, 60, 80, 120]

SIGMAS = (0.005, 0.01, 0.02, 0.03, 0.05)

LEVELS = [0.3, 0.6, 1.2, 2.5, 6.0]
MILD, SEVERE = LEVELS[0], LEVELS[-1]

OCCLUSION_CAMERA_DISTANCES = (2.0, 4.0, 8.0, 20.0, 100.0)     # mild (close) -> severe (far)
