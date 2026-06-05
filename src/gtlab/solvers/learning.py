"""No-regret learning dynamics in bimatrix games.

Hedge (multiplicative weights) is extracted from the Correlated-Equilibrium
notebook; fictitious play is the companion classic dynamic. Both return regret
trajectories and the empirical joint distribution, which converges to the
(C)CE set by Hannan's theorem.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def hedge(
    A: np.ndarray, B: np.ndarray, T: int = 2000,
    eta: float | None = None, seed: int = 0,
) -> Dict[str, Any]:
    """Run Hedge / multiplicative weights for both players for ``T`` rounds."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    rng = np.random.default_rng(seed)
    m, n = A.shape
    if eta is None:
        eta = np.sqrt(np.log(max(m, n)) / T)

    w_row = np.ones(m)
    w_col = np.ones(n)
    empirical = np.zeros((m, n))
    regret_row, regret_col = [], []
    p_hist, q_hist = [], []
    cum_row = cum_col = 0.0

    for _ in range(T):
        p = w_row / w_row.sum()
        q = w_col / w_col.sum()
        p_hist.append(p.copy())
        q_hist.append(q.copy())
        a = int(rng.choice(m, p=p))
        b = int(rng.choice(n, p=q))
        empirical[a, b] += 1
        cum_row += A[a, b]
        cum_col += B[a, b]
        regret_row.append(A[:, b].max() - A[a, b])
        regret_col.append(B[a, :].max() - B[a, b])
        w_row *= np.exp(eta * A[:, b])
        w_col *= np.exp(eta * B[a, :])

    empirical /= empirical.sum()
    t = np.arange(1, T + 1)
    return {
        "empirical_mu": empirical,
        "avg_regret_row": np.cumsum(regret_row) / t,
        "avg_regret_col": np.cumsum(regret_col) / t,
        "history_p_row": np.array(p_hist),
        "history_p_col": np.array(q_hist),
        "eu_row": cum_row / T,
        "eu_col": cum_col / T,
        "eta": float(eta),
        "T": T,
    }


def fictitious_play(
    A: np.ndarray, B: np.ndarray, T: int = 2000, seed: int = 0,
) -> Dict[str, Any]:
    """Classic fictitious play: each player best-responds to the empirical
    frequency of the opponent's past actions."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    rng = np.random.default_rng(seed)
    m, n = A.shape
    count_row = np.zeros(m)
    count_col = np.zeros(n)
    # Seed with one random action each to avoid a zero belief.
    count_row[rng.integers(m)] = 1
    count_col[rng.integers(n)] = 1
    empirical = np.zeros((m, n))

    for _ in range(T):
        q = count_col / count_col.sum()
        p_belief = count_row / count_row.sum()
        a = int(np.argmax(A @ q))
        b = int(np.argmax(B.T @ p_belief))
        count_row[a] += 1
        count_col[b] += 1
        empirical[a, b] += 1

    empirical /= empirical.sum()
    return {
        "empirical_mu": empirical,
        "freq_row": count_row / count_row.sum(),
        "freq_col": count_col / count_col.sum(),
        "T": T,
    }
