"""Shared augmentation and sampling utilities.

Split into :mod:`pcrobust.utils.augment` (training-time augmentation) and
:mod:`pcrobust.utils.sampling` (resampling / normalize-and-FPS).  The
gallery dataset class lives in :mod:`pcrobust.datasets`.

The leading-underscore names (``_augment``, ``_sample``) are kept exactly as
the benchmark's scripts have always imported them.
"""

from pcrobust.utils.augment import _DEG, _X_STEPS, _Y_STEPS, _Z_STEPS, _augment
from pcrobust.utils.sampling import _sample, normalize_and_sample

__all__ = [
    "_DEG",
    "_X_STEPS",
    "_Y_STEPS",
    "_Z_STEPS",
    "_augment",
    "_sample",
    "normalize_and_sample",
]
