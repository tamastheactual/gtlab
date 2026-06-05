"""Example games for the correlated-equilibrium module, as factories.

Each returns a :class:`gtlab.core.correlated.CorrelatedGame`, the engine used in
the Correlated-Equilibrium and Learning notebook.
"""
from __future__ import annotations

import numpy as np

from ..core.correlated import CorrelatedGame


def chicken_ce() -> CorrelatedGame:
    """Chicken - the classic example where CE beats every Nash equilibrium."""
    A = np.array([[0, -1], [1, -10]], dtype=float)
    B = np.array([[0, 1], [-1, -10]], dtype=float)
    return CorrelatedGame(A, B, ["Swerve", "Straight"], ["Swerve", "Straight"],
                          name="Chicken")


def battle_of_sexes_ce() -> CorrelatedGame:
    """Battle of the Sexes - coordination via a fair correlating signal."""
    A = np.array([[3, 0], [0, 1]], dtype=float)
    B = np.array([[1, 0], [0, 3]], dtype=float)
    return CorrelatedGame(A, B, ["Boxing", "Film"], ["Boxing", "Film"],
                          name="Battle of Sexes")


def stag_hunt_ce() -> CorrelatedGame:
    """Stag Hunt - correlation steers play toward the efficient outcome."""
    A = np.array([[4, 0], [2, 2]], dtype=float)
    B = np.array([[4, 2], [0, 2]], dtype=float)
    return CorrelatedGame(A, B, ["Stag", "Hare"], ["Stag", "Hare"],
                          name="Stag Hunt")


def prisoners_dilemma_ce() -> CorrelatedGame:
    """Prisoner's Dilemma - CE collapses to the unique NE (no correlation helps)."""
    A = np.array([[3, 0], [4, 1]], dtype=float)
    B = np.array([[3, 4], [0, 1]], dtype=float)
    return CorrelatedGame(A, B, ["C", "D"], ["C", "D"],
                          name="Prisoner's Dilemma")


def traffic_coordination() -> CorrelatedGame:
    """Traffic coordination - a fair signal resolves the merge conflict."""
    A = np.array([[1, 3], [2, 1]], dtype=float)
    B = np.array([[1, 2], [3, 1]], dtype=float)
    return CorrelatedGame(A, B, ["Highway", "Side-street"], ["Highway", "Side-street"],
                          name="Traffic Coordination")


def rps_ce() -> CorrelatedGame:
    """Rock-Paper-Scissors - zero-sum, so NE = CE = CCE (uniform)."""
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    labels = ["Rock", "Paper", "Scissors"]
    return CorrelatedGame(A, -A, labels, labels, name="Rock-Paper-Scissors")
