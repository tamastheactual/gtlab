"""Extra stochastic / communication game algorithms.

Pure-math routines ported from the Stochastic-Games-and-Communication notebook
that are not yet part of the shared solver layer:

- ``best_response_iteration`` : Gauss-Seidel best-response dynamics for a
  general-sum stochastic game (approximate Markov Perfect Equilibrium).
- ``pure_values`` : value of a fixed pure stationary joint policy.
- ``stage_values`` : isolated zero-sum stage-game values per state.
- ``bayesian_update`` : one-step POMDP belief update.
- ``check_correlated_eq`` : correlated-equilibrium incentive check on a joint
  distribution over a one-shot bimatrix.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .linprog import solve_zero_sum


def best_response_iteration(
    r1: np.ndarray, r2: np.ndarray, P: np.ndarray, gamma: float,
    tol: float = 1e-6, max_iter: int = 300,
) -> Dict[str, Any]:
    """Gauss-Seidel best-response dynamics for a general-sum stochastic game.

    Alternates per-state pure best responses while propagating the joint value
    functions. Convergence is not guaranteed; on convergence the result is an
    approximate Markov Perfect Equilibrium (a stationary Nash equilibrium of
    every stage game at the joint values).

    Returns ``{"V1", "V2", "pi", "sig", "n_iter", "converged"}``.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    P = np.asarray(P, dtype=float)
    nS, nA, nB = r1.shape
    pi = np.ones((nS, nA)) / nA
    sig = np.ones((nS, nB)) / nB
    V1 = np.zeros(nS)
    V2 = np.zeros(nS)
    converged = False
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        V1n = np.zeros(nS)
        V2n = np.zeros(nS)
        for s in range(nS):
            EV1 = (P[s] * V1[np.newaxis, np.newaxis, :]).sum(axis=2)
            EV2 = (P[s] * V2[np.newaxis, np.newaxis, :]).sum(axis=2)
            Q1 = r1[s] + gamma * EV1
            Q2 = r2[s] + gamma * EV2
            V1n[s] = float(pi[s] @ Q1 @ sig[s])
            V2n[s] = float(pi[s] @ Q2 @ sig[s])
        pi_n = np.zeros((nS, nA))
        sig_n = np.zeros((nS, nB))
        for s in range(nS):
            M1 = r1[s] + gamma * (P[s] * V1n[np.newaxis, np.newaxis, :]).sum(2)
            M2 = r2[s] + gamma * (P[s] * V2n[np.newaxis, np.newaxis, :]).sum(2)
            pi_n[s, int(np.argmax(M1 @ sig[s]))] = 1.0
            sig_n[s, int(np.argmax(pi[s] @ M2))] = 1.0
        delta = max(float(np.max(np.abs(V1n - V1))),
                    float(np.max(np.abs(V2n - V2))))
        V1, V2, pi, sig = V1n, V2n, pi_n, sig_n
        if delta < tol:
            converged = True
            break
    return {"V1": V1, "V2": V2, "pi": pi, "sig": sig,
            "n_iter": n_iter, "converged": converged}


def pure_values(
    r1: np.ndarray, r2: np.ndarray, P: np.ndarray, gamma: float,
    a1_sel: Sequence[int], a2_sel: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Discounted values (V1, V2) of a fixed pure stationary joint policy.

    ``a1_sel[s]`` / ``a2_sel[s]`` are the actions chosen deterministically at
    each state. Solves (I - gamma P_pi) V = r_pi exactly.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    P = np.asarray(P, dtype=float)
    nS = r1.shape[0]
    P_pi = np.array([P[s, a1_sel[s], a2_sel[s]] for s in range(nS)])
    r1_pi = np.array([r1[s, a1_sel[s], a2_sel[s]] for s in range(nS)])
    r2_pi = np.array([r2[s, a1_sel[s], a2_sel[s]] for s in range(nS)])
    I = np.eye(nS)
    V1 = np.linalg.solve(I - gamma * P_pi, r1_pi)
    V2 = np.linalg.solve(I - gamma * P_pi, r2_pi)
    return V1, V2


def stage_values(r: np.ndarray) -> List[float]:
    """Isolated zero-sum stage-game value for each state of a (S, A, B) tensor."""
    r = np.asarray(r, dtype=float)
    return [float(solve_zero_sum(r[s])["value"]) for s in range(r.shape[0])]


def bayesian_update(
    b: np.ndarray, obs: int, P_tr: np.ndarray, O_obs: np.ndarray, a: int, c: int,
) -> np.ndarray:
    """One-step POMDP belief update.

    ``b_{t+1}(s') ~ O(o|s') * sum_s P(s'|s,a,c) b_t(s)``.

    ``P_tr`` is (S, A, C, S); ``O_obs`` is (S, O) observation likelihoods.
    """
    b = np.asarray(b, dtype=float)
    nS = len(b)
    b_new = np.zeros(nS)
    for sp in range(nS):
        pred = sum(P_tr[s, a, c, sp] * b[s] for s in range(nS))
        b_new[sp] = O_obs[sp, obs] * pred
    total = b_new.sum()
    return b_new / total if total > 1e-12 else b_new


def check_correlated_eq(
    phi: np.ndarray, U1: np.ndarray, U2: np.ndarray,
    row_names: Sequence[str], col_names: Sequence[str], tol: float = 1e-8,
) -> Dict[str, Any]:
    """Check correlated-equilibrium obedience constraints on a joint distribution.

    ``phi`` is the recommendation distribution over a one-shot bimatrix with row
    payoffs ``U1`` and column payoffs ``U2``. Returns whether no player can gain
    by deviating from any recommended action, the expected payoffs, and a list
    of any violations.
    """
    phi = np.asarray(phi, dtype=float)
    U1 = np.asarray(U1, dtype=float)
    U2 = np.asarray(U2, dtype=float)
    nA, nB = phi.shape
    violations: List[str] = []
    for a in range(nA):
        if phi[a, :].sum() < 1e-10:
            continue
        EX_f = sum(phi[a, b] * U1[a, b] for b in range(nB))
        for ap in range(nA):
            if ap == a:
                continue
            EX_d = sum(phi[a, b] * U1[ap, b] for b in range(nB))
            if EX_d > EX_f + tol:
                violations.append(
                    f"Row: {row_names[a]}->{row_names[ap]}: gain={EX_d - EX_f:.4f}")
    for b in range(nB):
        if phi[:, b].sum() < 1e-10:
            continue
        EY_f = sum(phi[a, b] * U2[a, b] for a in range(nA))
        for bp in range(nB):
            if bp == b:
                continue
            EY_d = sum(phi[a, b] * U2[a, bp] for a in range(nA))
            if EY_d > EY_f + tol:
                violations.append(
                    f"Col: {col_names[b]}->{col_names[bp]}: gain={EY_d - EY_f:.4f}")
    EU1 = float((phi * U1).sum())
    EU2 = float((phi * U2).sum())
    return {"valid": len(violations) == 0, "EU1": EU1, "EU2": EU2,
            "violations": violations}
