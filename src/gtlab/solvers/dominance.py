"""Strict dominance and iterated elimination (IESDS).

Shared by Normal-Form and Extensive-Form games, which had near-identical copies
(including a duplicated explain/strict split within Extensive-Form itself).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-9


def strictly_dominated_rows(A: np.ndarray, tol: float = EPS) -> Dict[int, int]:
    """Map each strictly dominated row to a row that dominates it."""
    A = np.asarray(A, dtype=float)
    m = A.shape[0]
    dominated: Dict[int, int] = {}
    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            if np.all(A[k] > A[i] + tol):
                dominated[i] = k
                break
    return dominated


def strictly_dominated_cols(B: np.ndarray, tol: float = EPS) -> Dict[int, int]:
    """Map each strictly dominated column to a column that dominates it."""
    B = np.asarray(B, dtype=float)
    n = B.shape[1]
    dominated: Dict[int, int] = {}
    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            if np.all(B[:, k] > B[:, j] + tol):
                dominated[j] = k
                break
    return dominated


def iesds(
    A: np.ndarray,
    B: np.ndarray,
    row_labels: List[str] | None = None,
    col_labels: List[str] | None = None,
    tol: float = EPS,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[dict]]:
    """Iterated elimination of strictly dominated strategies.

    Returns the reduced ``(A, B)``, surviving ``(row_labels, col_labels)``, and
    a step log describing each elimination round.
    """
    A = np.asarray(A, dtype=float).copy()
    B = np.asarray(B, dtype=float).copy()
    rows = list(row_labels) if row_labels else [f"r{i}" for i in range(A.shape[0])]
    cols = list(col_labels) if col_labels else [f"c{j}" for j in range(A.shape[1])]
    log: List[dict] = []

    changed = True
    while changed:
        changed = False
        dr = strictly_dominated_rows(A, tol)
        if dr:
            victim, by = next(iter(dr.items()))
            log.append({"player": "row", "removed": rows[victim], "by": rows[by]})
            keep = [i for i in range(A.shape[0]) if i != victim]
            A, B = A[keep, :], B[keep, :]
            rows = [rows[i] for i in keep]
            changed = True
            continue
        dc = strictly_dominated_cols(B, tol)
        if dc:
            victim, by = next(iter(dc.items()))
            log.append({"player": "col", "removed": cols[victim], "by": cols[by]})
            keep = [j for j in range(B.shape[1]) if j != victim]
            A, B = A[:, keep], B[:, keep]
            cols = [cols[j] for j in keep]
            changed = True
    return A, B, rows, cols, log
