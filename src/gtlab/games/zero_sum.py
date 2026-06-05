"""Canonical and applied zero-sum games as ready-made factories.

Ported from the Joint Policies / Minimax notebook so each worked example is a
one-call constructor returning a :class:`gtlab.core.ZeroSumGame`.
"""
from __future__ import annotations

import numpy as np

from ..core import ZeroSumGame


def micro_check_2x2() -> ZeroSumGame:
    """The 2x2 micro-check game with a unique mixed equilibrium."""
    A = np.array([[2, -1], [-3, 4]], dtype=float)
    return ZeroSumGame(A, ["U", "D"], ["L", "R"], name="Micro-Check 2x2")


def rock_paper_scissors() -> ZeroSumGame:
    """Symmetric Rock-Paper-Scissors (uniform equilibrium, value 0)."""
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    return ZeroSumGame(A, ["R", "P", "S"], ["R", "P", "S"],
                       name="Rock-Paper-Scissors")


def weighted_rps() -> ZeroSumGame:
    """Weighted Rock-Paper-Scissors with broken symmetry."""
    A = np.array([[0, -2, 1], [2, 0, -1], [-1, 1, 0]], dtype=float)
    return ZeroSumGame(A, ["R", "P", "S"], ["R", "P", "S"],
                       name="Weighted RPS")


def penalty_kick() -> ZeroSumGame:
    """Penalty kick: kicker vs goalkeeper scoring probabilities."""
    A = np.array([[0.55, 0.80], [0.90, 0.20]], dtype=float)
    return ZeroSumGame(A, ["L", "R"], ["L", "R"],
                       row_name="Kicker", col_name="Goalkeeper",
                       name="Penalty Kick")


def security_audit() -> ZeroSumGame:
    """Defender vs attacker monitoring-allocation game (Row payoffs are losses)."""
    A = np.array([[-1, -6], [-5, -2]], dtype=float)
    return ZeroSumGame(A, ["Defend A", "Defend B"], ["Attack A", "Attack B"],
                       row_name="Defender", col_name="Attacker",
                       name="Security Audit")
