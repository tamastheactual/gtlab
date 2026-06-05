"""Support-enumeration and security analysis for zero-sum matrix games.

Pure-math helpers ported from the Joint Policies / Minimax notebook engine.
They operate on plain numpy arrays so the core class only wires them to viz.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-9


def solve_support(
    A: np.ndarray, I: List[int], J: List[int], tol: float = 1e-6
) -> Optional[Dict[str, Any]]:
    """Solve a zero-sum game on a fixed support pair ``(I, J)``.

    Returns ``{"p", "q", "value", "support_row", "support_col"}`` if the
    indifference system yields a feasible equilibrium, else ``None``.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    sI, sJ = len(I), len(J)

    # Row indifference: (Aq)_i = v for i in I, plus sum_J q = 1.
    A_sys = np.zeros((sI + 1, sJ + 1))
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            A_sys[ii, jj] = A[i, j]
        A_sys[ii, sJ] = -1.0
    A_sys[sI, :sJ] = 1.0
    b_sys = np.zeros(sI + 1)
    b_sys[sI] = 1.0

    if sI + 1 == sJ + 1:
        try:
            sol = np.linalg.solve(A_sys, b_sys)
        except np.linalg.LinAlgError:
            return None
    else:
        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
    q_J = sol[:sJ]
    v_row = sol[sJ]
    if np.any(q_J < -1e-10):
        return None

    # Column indifference: (A^T p)_j = v for j in J, plus sum_I p = 1.
    B_sys = np.zeros((sJ + 1, sI + 1))
    for jj, j in enumerate(J):
        for ii, i in enumerate(I):
            B_sys[jj, ii] = A[i, j]
        B_sys[jj, sI] = -1.0
    B_sys[sJ, :sI] = 1.0
    c_sys = np.zeros(sJ + 1)
    c_sys[sJ] = 1.0

    if sJ + 1 == sI + 1:
        try:
            sol2 = np.linalg.solve(B_sys, c_sys)
        except np.linalg.LinAlgError:
            return None
    else:
        sol2, _, _, _ = np.linalg.lstsq(B_sys, c_sys, rcond=None)
    p_I = sol2[:sI]
    v_col = sol2[sI]
    if np.any(p_I < -1e-10):
        return None

    if abs(v_row - v_col) > tol:
        return None

    p = np.zeros(m)
    q = np.zeros(n)
    for ii, i in enumerate(I):
        p[i] = max(0.0, p_I[ii])
    for jj, j in enumerate(J):
        q[j] = max(0.0, q_J[jj])

    v = float(v_row)
    Aq = A @ q
    ATp = A.T @ p
    for i in range(m):
        if i not in I and Aq[i] > v + tol:
            return None
    for j in range(n):
        if j not in J and ATp[j] < v - tol:
            return None

    return {"p": p, "q": q, "value": v,
            "support_row": list(I), "support_col": list(J)}


def solve_all_supports(A: np.ndarray) -> List[Dict[str, Any]]:
    """Enumerate every feasible support pair, returning distinct equilibria."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    results: List[Dict[str, Any]] = []

    def _add(sol: Dict[str, Any]) -> None:
        for r in results:
            if np.allclose(r["p"], sol["p"]) and np.allclose(r["q"], sol["q"]):
                return
        results.append(sol)

    # Equal-size supports first (generic case), then unequal.
    for equal in (True, False):
        for sr in range(1, m + 1):
            for sc in range(1, n + 1):
                if (sr == sc) != equal:
                    continue
                for I in combinations(range(m), sr):
                    for J in combinations(range(n), sc):
                        sol = solve_support(A, list(I), list(J))
                        if sol is not None:
                            _add(sol)
    return results


def epsilon_security(
    A: np.ndarray, p: np.ndarray, q: np.ndarray, value: Optional[float] = None
) -> Dict[str, float]:
    """Epsilon-security gaps for a strategy pair ``(p, q)``.

    ``eps_row`` is Row's gain by best-responding to ``q``; ``eps_col`` is
    Column's gain by best-responding to ``p``; ``eps_max`` the worst case.
    """
    A = np.asarray(A, dtype=float)
    Aq = A @ np.asarray(q, dtype=float)
    ATp = A.T @ np.asarray(p, dtype=float)
    pAq = float(np.asarray(p) @ A @ np.asarray(q))
    eps_row = float(np.max(Aq) - pAq)
    eps_col = float(pAq - np.min(ATp))
    out = {"eps_row": eps_row, "eps_col": eps_col,
           "eps_max": max(eps_row, eps_col), "pAq": pAq}
    if value is not None:
        out["value"] = float(value)
    return out


def support_enumeration_trace(
    A: np.ndarray, tol: float = 1e-6
) -> Tuple[List[Dict[str, Any]], List[Tuple[List[int], List[int], str]], int]:
    """Trace equal-size support enumeration for pedagogy.

    Returns ``(feasible, rejected, n_tried)`` where ``feasible`` items carry the
    full indifference derivation and ``rejected`` items are ``(I, J, reason)``.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    feasible: List[Dict[str, Any]] = []
    rejected: List[Tuple[List[int], List[int], str]] = []
    seen: List[Tuple[np.ndarray, np.ndarray]] = []
    n_tried = 0

    for k in range(1, min(m, n) + 1):
        for I in combinations(range(m), k):
            for J in combinations(range(n), k):
                I, J = list(I), list(J)
                n_tried += 1
                sI, sJ = k, k

                A_sys = np.zeros((sI + 1, sJ + 1))
                for ii, iv in enumerate(I):
                    for jj, jv in enumerate(J):
                        A_sys[ii, jj] = A[iv, jv]
                    A_sys[ii, sJ] = -1.0
                A_sys[sI, :sJ] = 1.0
                b_sys = np.zeros(sI + 1)
                b_sys[sI] = 1.0
                try:
                    sol_q = np.linalg.solve(A_sys, b_sys)
                except np.linalg.LinAlgError:
                    rejected.append((I, J, "singular system"))
                    continue
                q_J = sol_q[:sJ]
                v_row = sol_q[sJ]
                if np.any(q_J < -1e-10):
                    rejected.append((I, J, "q < 0 in support"))
                    continue

                B_sys = np.zeros((sJ + 1, sI + 1))
                for jj, jv in enumerate(J):
                    for ii, iv in enumerate(I):
                        B_sys[jj, ii] = A[iv, jv]
                    B_sys[jj, sI] = -1.0
                B_sys[sJ, :sI] = 1.0
                c_sys = np.zeros(sJ + 1)
                c_sys[sJ] = 1.0
                try:
                    sol_p = np.linalg.solve(B_sys, c_sys)
                except np.linalg.LinAlgError:
                    rejected.append((I, J, "singular system"))
                    continue
                p_I = sol_p[:sI]
                v_col = sol_p[sI]
                if np.any(p_I < -1e-10):
                    rejected.append((I, J, "p < 0 in support"))
                    continue
                if abs(v_row - v_col) > tol:
                    rejected.append((I, J, "value mismatch"))
                    continue

                p_full = np.zeros(m)
                q_full = np.zeros(n)
                for ii, iv in enumerate(I):
                    p_full[iv] = max(0.0, p_I[ii])
                for jj, jv in enumerate(J):
                    q_full[jv] = max(0.0, q_J[jj])
                v_val = float(v_row)

                Aq = A @ q_full
                ATp = A.T @ p_full
                out_ok = True
                for i in range(m):
                    if i not in I and Aq[i] > v_val + tol:
                        out_ok = False
                        break
                if out_ok:
                    for j in range(n):
                        if j not in J and ATp[j] < v_val - tol:
                            out_ok = False
                            break
                if not out_ok:
                    rejected.append((I, J, "out-of-support inequality violated"))
                    continue

                if any(np.allclose(p_full, pp) and np.allclose(q_full, qq)
                       for pp, qq in seen):
                    rejected.append((I, J, "duplicate"))
                    continue

                seen.append((p_full.copy(), q_full.copy()))
                feasible.append({
                    "support_row": I, "support_col": J,
                    "p": p_full, "q": q_full, "value": v_val,
                    "q_support": q_J, "p_support": p_I,
                    "Aq": Aq, "ATp": ATp,
                })
    return feasible, rejected, n_tried
