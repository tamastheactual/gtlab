"""Extra pure algorithms for the correlated-equilibrium module.

These complement :mod:`gtlab.solvers.correlated` (whose ``check_ce`` /
``check_cce`` return a plain boolean) with constraint-by-constraint detail
needed for the styled verification walkthroughs, plus payoff-region sampling
for the CE / CCE achievable-payoff scatter.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

TOL = 1e-8


def ce_obedience_detail(A: np.ndarray, B: np.ndarray, mu: np.ndarray) -> List[Dict]:
    """Per-constraint CE obedience report.

    For every (recommended action, deviation) pair returns the obedience gain
    ``sum_j mu[i,j] (A[i,j] - A[ip,j])`` for the row player (analogously for the
    column player). A non-negative value means the recommendation is obeyed.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    mu = np.asarray(mu, float)
    m, n = A.shape
    out: List[Dict] = []
    for i in range(m):
        for ip in range(m):
            if ip == i:
                continue
            val = float(sum(mu[i, j] * (A[i, j] - A[ip, j]) for j in range(n)))
            out.append({"player": "row", "told": i, "deviation": ip,
                        "gain": val, "ok": val >= -TOL})
    for j in range(n):
        for jp in range(n):
            if jp == j:
                continue
            val = float(sum(mu[i, j] * (B[i, j] - B[i, jp]) for i in range(m)))
            out.append({"player": "col", "told": j, "deviation": jp,
                        "gain": val, "ok": val >= -TOL})
    return out


def cce_exante_detail(A: np.ndarray, B: np.ndarray, mu: np.ndarray) -> Dict:
    """Per-constraint CCE ex-ante deviation report.

    Compares each player's expected payoff under ``mu`` against the payoff of
    committing to a fixed action before the signal. A non-positive gain on every
    alternative action means ``mu`` is a CCE.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    mu = np.asarray(mu, float)
    m, n = A.shape
    eu_row = float((mu * A).sum())
    eu_col = float((mu * B).sum())
    rows: List[Dict] = []
    for ip in range(m):
        dev = float((mu * A[ip, :]).sum())
        gain = dev - eu_row
        rows.append({"player": "row", "deviation": ip, "dev_payoff": dev,
                     "current": eu_row, "gain": gain, "ok": gain <= TOL})
    for jp in range(n):
        dev = float((mu * B[:, jp]).sum())
        gain = dev - eu_col
        rows.append({"player": "col", "deviation": jp, "dev_payoff": dev,
                     "current": eu_col, "gain": gain, "ok": gain <= TOL})
    return {"eu_row": eu_row, "eu_col": eu_col, "constraints": rows,
            "ok": all(r["ok"] for r in rows)}


def _constraints(A: np.ndarray, B: np.ndarray, coarse: bool) -> np.ndarray:
    """Deviation constraints as rows of ``A_ub`` for ``A_ub @ mu <= 0``.

    Mirrors :func:`gtlab.solvers.correlated._constraints` but kept local so this
    module imports cleanly when used directly by the core class.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    m, n = A.shape
    N = m * n
    if coarse:
        row_dev = A[None, :, :] - A[:, None, :]
        ce_row = row_dev.reshape(m, N)
        col_dev = B[None, :, :] - B.T[:, :, None]
        ce_col = col_dev.reshape(n, N)
        cons = np.vstack([ce_row, ce_col])
    else:
        eye_m, eye_n = np.eye(m), np.eye(n)
        diff_row = A[:, None, :] - A[None, :, :]
        full_row = np.einsum("ik,ipj->ipkj", eye_m, diff_row).reshape(m * m, N)
        keep_row = (np.arange(m)[:, None] != np.arange(m)[None, :]).reshape(-1)
        ce_row = full_row[keep_row]
        diff_col = B.T[:, None, :] - B.T[None, :, :]
        full_col = np.einsum("jk,jpi->jpik", eye_n, diff_col).reshape(n * n, N)
        keep_col = (np.arange(n)[:, None] != np.arange(n)[None, :]).reshape(-1)
        ce_col = full_col[keep_col]
        cons = np.vstack([ce_row, ce_col])
    return -cons


def sample_payoff_region(
    A: np.ndarray, B: np.ndarray, coarse: bool,
    n_samples: int = 400, seed: int = 0,
) -> np.ndarray:
    """Sample achievable (row, col) payoff pairs over the CE / CCE polytope.

    Each sample optimizes a random linear objective over the equilibrium
    polytope, tracing out its payoff image. Returns an ``(k, 2)`` array of
    ``(eu_row, eu_col)`` pairs (``k <= n_samples`` after dropping LP failures).
    """
    from scipy.optimize import linprog

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    m, n = A.shape
    N = m * n
    A_ub = _constraints(A, B, coarse)
    b_ub = np.zeros(len(A_ub))
    A_eq = np.ones((1, N))
    b_eq = np.array([1.0])
    bounds = [(0, None)] * N
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_samples):
        c = rng.standard_normal(N)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        if res.status == 0:
            mu = np.clip(res.x.reshape(m, n), 0, None)
            s = mu.sum()
            if s <= 0:
                continue
            mu /= s
            pairs.append((float((mu * A).sum()), float((mu * B).sum())))
    return np.array(pairs) if pairs else np.empty((0, 2))
