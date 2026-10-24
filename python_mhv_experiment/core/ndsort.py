from __future__ import annotations

import numpy as np


def first_front(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask for the first nondominated front.

    PlatEMO metrics use minimization. A point p dominates q when p <= q in
    every objective and p < q in at least one objective.
    """
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        dominated_by_i = np.all(points[i] <= points, axis=1) & np.any(points[i] < points, axis=1)
        dominated_by_i[i] = False
        keep[dominated_by_i] = False

        if np.any(np.all(points[keep] <= points[i], axis=1) & np.any(points[keep] < points[i], axis=1)):
            keep[i] = False

    return keep


def nondominated(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(0, 0)
    return np.unique(points[first_front(points)], axis=0)

