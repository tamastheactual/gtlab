"""Best-response computation for bimatrix games.

Shared by Normal-Form, Extensive-Form, Zero-Sum, and Correlated games, each of
which previously carried its own ``best_responses_row/col`` pair.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def best_responses_to_col(A: np.ndarray, j: int, tol: float = EPS) -> np.ndarray:
    """Row indices that best-respond to the column player's pure action ``j``."""
    col = A[:, j]
    return np.where(col >= col.max() - tol)[0]


def best_responses_to_row(B: np.ndarray, i: int, tol: float = EPS) -> np.ndarray:
    """Column indices that best-respond to the row player's pure action ``i``."""
    row = B[i, :]
    return np.where(row >= row.max() - tol)[0]


def br_masks(A: np.ndarray, B: np.ndarray, tol: float = EPS):
    """Boolean (m, n) masks marking row-BR cells and col-BR cells.

    ``br_row[i, j]`` is True when row action ``i`` is a best response to column
    ``j``; ``br_col[i, j]`` when column ``j`` best-responds to row ``i``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape
    br_row = np.zeros((m, n), dtype=bool)
    br_col = np.zeros((m, n), dtype=bool)
    for j in range(n):
        br_row[best_responses_to_col(A, j, tol), j] = True
    for i in range(m):
        br_col[i, best_responses_to_row(B, i, tol)] = True
    return br_row, br_col


def best_response_to_mixed(A: np.ndarray, q: np.ndarray, tol: float = EPS) -> np.ndarray:
    """Row actions maximizing expected payoff against column mix ``q``."""
    vals = A @ np.asarray(q, dtype=float)
    return np.where(vals >= vals.max() - tol)[0]


def exploitability(A: np.ndarray, q: np.ndarray, value: float) -> float:
    """How much the row player gains by best-responding to ``q`` over ``value``."""
    return float((A @ np.asarray(q, dtype=float)).max() - value)
