"""Linear-programming solvers shared across zero-sum and stochastic games.

``solve_zero_sum`` is the workhorse that appeared (re-implemented) in the
Zero-Sum, Stochastic, and Correlated notebooks. One copy now.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

EPS = 1e-9


def solve_zero_sum(M: np.ndarray) -> Dict[str, Any]:
    """Solve a two-player zero-sum matrix game by linear programming.

    ``M`` is the row player's payoff matrix (column player gets ``-M``).
    Returns ``{"p", "q", "value"}`` - optimal row mix, column mix, game value.
    """
    from scipy.optimize import linprog

    M = np.asarray(M, dtype=float)
    m, n = M.shape

    # Row's primal: max v s.t. M^T p >= v, sum p = 1, p >= 0.
    c = np.zeros(m + 1)
    c[-1] = -1.0
    A_ub = np.hstack([-M.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    A_eq = np.ones((1, m + 1))
    A_eq[0, -1] = 0.0
    b_eq = np.array([1.0])
    bounds = [(0, None)] * m + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Row LP failed: {res.message}")
    p = np.clip(res.x[:m], 0, None)
    p = p / p.sum()
    v = float(res.x[m])

    # Column's dual: min v s.t. M q <= v, sum q = 1, q >= 0.
    c2 = np.zeros(n + 1)
    c2[-1] = 1.0
    A_ub2 = np.hstack([M, -np.ones((m, 1))])
    b_ub2 = np.zeros(m)
    A_eq2 = np.ones((1, n + 1))
    A_eq2[0, -1] = 0.0
    b_eq2 = np.array([1.0])
    bounds2 = [(0, None)] * n + [(None, None)]
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                   bounds=bounds2, method="highs")
    if not res2.success:
        raise RuntimeError(f"Column LP failed: {res2.message}")
    q = np.clip(res2.x[:n], 0, None)
    q = q / q.sum()

    return {"p": p, "q": q, "value": v}


def complementary_slackness(
    M: np.ndarray, p: np.ndarray, q: np.ndarray, v: float, tol: float = 1e-6
) -> Dict[str, Any]:
    """Verify KKT/complementary-slackness conditions for a zero-sum solution."""
    M = np.asarray(M, dtype=float)
    row_payoffs = M @ q          # payoff to each row action vs q
    col_payoffs = M.T @ p        # payoff to each col action vs p
    row_ok = np.all(row_payoffs <= v + tol)
    col_ok = np.all(col_payoffs >= v - tol)
    # Actions in the support must be exactly indifferent at value v.
    row_support_tight = np.all(np.abs(row_payoffs[p > tol] - v) < tol)
    col_support_tight = np.all(np.abs(col_payoffs[q > tol] - v) < tol)
    return {
        "row_payoffs": row_payoffs,
        "col_payoffs": col_payoffs,
        "valid": bool(row_ok and col_ok and row_support_tight and col_support_tight),
    }
