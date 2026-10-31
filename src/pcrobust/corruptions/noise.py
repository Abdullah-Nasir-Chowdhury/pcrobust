"""Gaussian per-point jitter corruption.

Identical to the noise studies' generator.  When ``sigma`` is omitted, one
value is drawn uniformly at random from the noise severity ladder
(:data:`pcrobust.corruptions.severity.SIGMAS`) -- the training-time behavior
used by invariance-regularized training.
"""

import numpy as np

from pcrobust.corruptions.severity import SIGMAS


def add_noise(pts: np.ndarray, sigma: float = None) -> np.ndarray:
    sigma = sigma if sigma is not None else float(np.random.choice(SIGMAS))
    return pts + np.random.randn(*pts.shape).astype(np.float32) * sigma
