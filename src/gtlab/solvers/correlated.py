"""Correlated and coarse-correlated equilibrium via linear programming.

Extracted from the Correlated-Equilibrium notebook. The obedience (CE) and
ex-ante (CCE) constraint builders are unified by a single flag.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _constraints(A: np.ndarray, B: np.ndarray, coarse: bool) -> np.ndarray:
    """Deviation constraints as rows of ``A_ub`` for ``A_ub @ mu <= 0``.

    CE (``coarse=False``): conditional obedience for every (action, deviation).
    CCE (``coarse=True``): ex-ante, a single deviation per alternative action.
    """
    m, n = A.shape
    N = m * n
    # ``mu`` is flattened row-major: index of cell (i, j) is i*n + j.
    if coarse:
        # CCE: one ex-ante constraint per alternative action.
        # Row → ip:  sum_{i,j} mu[i,j]*(A[i,j] - A[ip,j]) >= 0   (m constraints)
        #   coef[ip, i, j] = A[i,j] - A[ip,j]
        row_dev = A[None, :, :] - A[:, None, :]          # (ip, i, j)
        ce_row = row_dev.reshape(m, N)
        # Col → jp:  sum_{i,j} mu[i,j]*(B[i,j] - B[i,jp]) >= 0   (n constraints)
        #   coef[jp, i, j] = B[i,j] - B[i,jp]
        col_dev = B[None, :, :] - B.T[:, :, None]        # (jp, i, j)
        ce_col = col_dev.reshape(n, N)
        cons = np.vstack([ce_row, ce_col])
    else:
        # CE: conditional obedience for every (recommended action, deviation).
        eye_m, eye_n = np.eye(m), np.eye(n)
        # Row told i, deviates to ip (i != ip):
        #   nonzero only in block row i; value = A[i,j] - A[ip,j]
        diff_row = A[:, None, :] - A[None, :, :]          # (i, ip, j)
        full_row = np.einsum("ik,ipj->ipkj", eye_m, diff_row)  # (i, ip, i', j)
        full_row = full_row.reshape(m * m, N)
        keep_row = (np.arange(m)[:, None] != np.arange(m)[None, :]).reshape(-1)
        ce_row = full_row[keep_row]
        # Col told j, deviates to jp (j != jp):
        #   nonzero only in block column j; value = B[i,j] - B[i,jp]
        diff_col = B.T[:, None, :] - B.T[None, :, :]      # (j, jp, i)
        full_col = np.einsum("jk,jpi->jpik", eye_n, diff_col)  # (j, jp, i, j')
        full_col = full_col.reshape(n * n, N)
        keep_col = (np.arange(n)[:, None] != np.arange(n)[None, :]).reshape(-1)
        ce_col = full_col[keep_col]
        cons = np.vstack([ce_row, ce_col])
    return -cons  # A_ub @ mu <= 0 form


def _solve(A: np.ndarray, B: np.ndarray, coarse: bool, maximize: str) -> Optional[Dict]:
    from scipy.optimize import linprog

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape
    N = m * n
    A_ub = _constraints(A, B, coarse)
    b_ub = np.zeros(len(A_ub))
    A_eq = np.ones((1, N))
    b_eq = np.array([1.0])
    bounds = [(0, None)] * N
    if maximize == "welfare":
        c = -(A + B).flatten()
    elif maximize == "row":
        c = -A.flatten()
    else:
        c = -B.flatten()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return None
    mu = np.clip(res.x.reshape(m, n), 0, None)
    mu /= mu.sum()
    eu_row = float((mu * A).sum())
    eu_col = float((mu * B).sum())
    return {"mu": mu, "eu_row": eu_row, "eu_col": eu_col,
            "welfare": eu_row + eu_col, "kind": "CCE" if coarse else "CE"}


def find_ce(A: np.ndarray, B: np.ndarray, maximize: str = "welfare") -> Optional[Dict]:
    """Welfare-maximizing correlated equilibrium (or row/col-optimal)."""
    return _solve(A, B, coarse=False, maximize=maximize)


def find_cce(A: np.ndarray, B: np.ndarray, maximize: str = "welfare") -> Optional[Dict]:
    """Welfare-maximizing coarse correlated equilibrium."""
    return _solve(A, B, coarse=True, maximize=maximize)


def check_ce(A: np.ndarray, B: np.ndarray, mu: np.ndarray, tol: float = 1e-6) -> bool:
    """True if joint distribution ``mu`` satisfies CE obedience constraints."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    cons = _constraints(A, B, coarse=False)
    return bool(np.all(cons @ mu.flatten() <= tol))


def check_cce(A: np.ndarray, B: np.ndarray, mu: np.ndarray, tol: float = 1e-6) -> bool:
    """True if joint distribution ``mu`` satisfies CCE ex-ante constraints."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    cons = _constraints(A, B, coarse=True)
    return bool(np.all(cons @ mu.flatten() <= tol))
