"""Pareto optimality and the Pareto frontier of an outcome set."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

EPS = 1e-9


def pareto_optimal_cells(A: np.ndarray, B: np.ndarray, tol: float = EPS) -> List[Tuple[int, int]]:
    """Cells ``(i, j)`` whose payoff pair is not Pareto-dominated by another cell."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape
    pts = [(A[i, j], B[i, j], i, j) for i in range(m) for j in range(n)]
    optimal = []
    for a, b, i, j in pts:
        dominated = any(
            (a2 >= a - tol and b2 >= b - tol) and (a2 > a + tol or b2 > b + tol)
            for a2, b2, _, _ in pts
        )
        if not dominated:
            optimal.append((i, j))
    return optimal


def pareto_frontier(points: np.ndarray, tol: float = EPS) -> np.ndarray:
    """Return the Pareto-optimal subset of an ``(N, 2)`` array of payoff pairs."""
    points = np.asarray(points, dtype=float)
    keep = []
    for k, (a, b) in enumerate(points):
        dominated = np.any(
            (points[:, 0] >= a - tol) & (points[:, 1] >= b - tol)
            & ((points[:, 0] > a + tol) | (points[:, 1] > b + tol))
        )
        if not dominated:
            keep.append(k)
    return points[keep]
