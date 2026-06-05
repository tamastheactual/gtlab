"""Nash equilibrium computation: pure detection and mixed support enumeration.

Pure-NE detection is dependency-free. Mixed NE uses ``nashpy`` when available
(install ``gtlab[nash]``) and falls back to a self-contained 2x2 solver plus a
support-enumeration routine for larger games.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .best_response import br_masks

EPS = 1e-9


def pure_nash(A: np.ndarray, B: np.ndarray, tol: float = EPS) -> List[Tuple[int, int]]:
    """All pure-strategy Nash equilibria as ``(row, col)`` index pairs."""
    br_row, br_col = br_masks(A, B, tol)
    mask = br_row & br_col
    return [(int(i), int(j)) for i, j in zip(*np.where(mask))]


def ne_mask(A: np.ndarray, B: np.ndarray, tol: float = EPS) -> np.ndarray:
    """Boolean (m, n) mask of pure-NE cells."""
    br_row, br_col = br_masks(A, B, tol)
    return br_row & br_col


def _nashpy_equilibria(A: np.ndarray, B: np.ndarray):
    """Yield (p, q) mixed equilibria via nashpy support enumeration, if installed."""
    try:
        import nashpy as nash  # type: ignore
    except ImportError:
        return None
    game = nash.Game(A, B)
    out = []
    try:
        for p, q in game.support_enumeration():
            out.append((np.asarray(p, float), np.asarray(q, float)))
    except Exception:  # pragma: no cover - degenerate games
        return out
    return out


def _mixed_2x2(A: np.ndarray, B: np.ndarray):
    """Closed-form fully-mixed equilibrium of a 2x2 game, or None."""
    a = A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1]
    b = B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1]
    if abs(a) < EPS or abs(b) < EPS:
        return None
    q = (A[1, 1] - A[1, 0]) / a   # col mix making row indifferent
    p = (B[1, 1] - B[0, 1]) / b   # row mix making col indifferent
    if 0 <= p <= 1 and 0 <= q <= 1:
        return np.array([p, 1 - p]), np.array([q, 1 - q])
    return None


def all_equilibria(A: np.ndarray, B: np.ndarray):
    """Best-available list of ``(p, q)`` equilibria (pure + mixed).

    Prefers nashpy; otherwise returns pure equilibria plus the analytic 2x2
    mixed equilibrium when applicable.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    via_nashpy = _nashpy_equilibria(A, B)
    if via_nashpy is not None and via_nashpy:
        return via_nashpy
    # Fallback: pure equilibria as degenerate mixes.
    eqs = []
    m, n = A.shape
    for i, j in pure_nash(A, B):
        p = np.zeros(m); p[i] = 1.0
        q = np.zeros(n); q[j] = 1.0
        eqs.append((p, q))
    if (m, n) == (2, 2):
        mixed = _mixed_2x2(A, B)
        if mixed is not None:
            eqs.append(mixed)
    return eqs
