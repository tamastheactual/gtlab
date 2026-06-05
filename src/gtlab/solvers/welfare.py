"""Social-welfare objectives shared by Extensive-Form, Stochastic, Correlated.

Utilitarian and egalitarian objectives are pure numpy; the Nash bargaining
product uses scipy when an optimized allocation is requested.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def utilitarian(payoffs: np.ndarray) -> float:
    """Sum of player payoffs."""
    return float(np.sum(payoffs))


def egalitarian(payoffs: np.ndarray) -> float:
    """Minimum player payoff (maximin welfare)."""
    return float(np.min(payoffs))


def nash_welfare(payoffs: np.ndarray, baseline: float = 0.0) -> float:
    """Product of (payoff - baseline); 0 if any player is at/below baseline."""
    shifted = np.asarray(payoffs, dtype=float) - baseline
    if np.any(shifted <= 0):
        return 0.0
    return float(np.prod(shifted))


def welfare_summary(payoffs: np.ndarray, baseline: float = 0.0) -> Dict[str, float]:
    """All three welfare measures for a payoff vector."""
    return {
        "utilitarian": utilitarian(payoffs),
        "egalitarian": egalitarian(payoffs),
        "nash": nash_welfare(payoffs, baseline),
    }


def best_outcome(outcomes: np.ndarray, objective: str = "utilitarian",
                 baseline: float = 0.0) -> int:
    """Index of the welfare-maximizing outcome in an ``(N, players)`` array."""
    outcomes = np.asarray(outcomes, dtype=float)
    fns = {"utilitarian": utilitarian, "egalitarian": egalitarian,
           "nash": lambda p: nash_welfare(p, baseline)}
    fn = fns[objective]
    scores = [fn(row) for row in outcomes]
    return int(np.argmax(scores))
