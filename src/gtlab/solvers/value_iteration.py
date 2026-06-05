"""Shapley operator and value iteration for zero-sum stochastic games.

Extracted from the Stochastic-Games notebook so the same routine can back any
finite-state Markov game.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .linprog import solve_zero_sum


def stage_game(r_s: np.ndarray, P_s: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """Stage-game matrix M_s(V) = r(s) + gamma * E_{s'}[V] for one state.

    ``r_s`` is the (A, B) reward matrix for state s; ``P_s`` is the
    (A, B, S) transition tensor; ``V`` is the current value vector.
    """
    EV = (P_s * V[np.newaxis, np.newaxis, :]).sum(axis=2)
    return r_s + gamma * EV


def shapley_operator(r: np.ndarray, P: np.ndarray, V: np.ndarray, gamma: float):
    """Apply the Shapley operator T over all states; return (TV, per-state sols)."""
    nS = r.shape[0]
    TV = np.zeros(nS)
    sols: Dict[int, dict] = {}
    for s in range(nS):
        sol = solve_zero_sum(stage_game(r[s], P[s], V, gamma))
        TV[s] = sol["value"]
        sols[s] = sol
    return TV, sols


def value_iteration(
    r: np.ndarray, P: np.ndarray, gamma: float,
    tol: float = 1e-8, max_iter: int = 500, V0: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Iterate the Shapley operator to the fixed point V*.

    Returns V*, the value history, Bellman residuals, iteration count, and the
    per-state stationary policies ``{s: {"p", "q"}}``.
    """
    nS = r.shape[0]
    V = np.zeros(nS) if V0 is None else np.asarray(V0, dtype=float)
    history = [V.copy()]
    residuals = []
    sols: Dict[int, dict] = {}
    for _ in range(max_iter):
        TV, sols = shapley_operator(r, P, V, gamma)
        res = float(np.max(np.abs(TV - V)))
        residuals.append(res)
        history.append(TV.copy())
        V = TV
        if res < tol:
            break
    policies = {s: {"p": sols[s]["p"], "q": sols[s]["q"]} for s in range(nS)}
    return {
        "V_star": V,
        "history": history,
        "residuals": residuals,
        "n_iter": len(residuals),
        "policies": policies,
    }
