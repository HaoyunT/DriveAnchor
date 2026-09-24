"""Small pure-Python curvature predicate for the PDM planner path.

The current PDM selector must not depend on the historical optional C++
geometry kernels.  This predicate is only used while assembling a geometric
repair candidate; the proposal decision itself is made by PDM.
"""

from __future__ import annotations

import numpy as np


def evaluate(path, max_curvature: float):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 3:
        return False, float("inf")
    if not np.isfinite(path).all() or not np.isfinite(max_curvature) or max_curvature <= 0:
        return False, float("inf")

    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-6):
        return False, float("inf")

    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    curvature = np.diff(headings) / np.maximum((lengths[:-1] + lengths[1:]) * 0.5, 1e-6)
    if not np.isfinite(curvature).all():
        return False, float("inf")
    cost = float(np.mean(curvature * curvature)) if len(curvature) else 0.0
    max_abs = float(np.max(np.abs(curvature))) if curvature.size else 0.0
    return bool(max_abs <= float(max_curvature) + 1e-6), cost
