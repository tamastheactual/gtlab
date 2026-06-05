"""Classic 2x2 (and small) textbook games as ready-made factories."""
from __future__ import annotations

import numpy as np

from ..core import NormalFormGame, ZeroSumGame


def prisoners_dilemma(R=3, T=5, P=1, S=0) -> NormalFormGame:
    """Prisoner's Dilemma with reward/temptation/punishment/sucker payoffs."""
    A = np.array([[R, S], [T, P]], dtype=float)
    return NormalFormGame(A, A.T, ["Cooperate", "Defect"], ["Cooperate", "Defect"],
                          name="Prisoner's Dilemma")


def stag_hunt(stag=4, hare=2, fail=0) -> NormalFormGame:
    A = np.array([[stag, fail], [hare, hare]], dtype=float)
    return NormalFormGame(A, A.T, ["Stag", "Hare"], ["Stag", "Hare"],
                          name="Stag Hunt")


def chicken(crash=0, swerve=2, win=3, lose=1) -> NormalFormGame:
    A = np.array([[swerve, lose], [win, crash]], dtype=float)
    return NormalFormGame(A, A.T, ["Swerve", "Straight"], ["Swerve", "Straight"],
                          name="Chicken")


def battle_of_the_sexes(a=2, b=1) -> NormalFormGame:
    A = np.array([[a, 0], [0, b]], dtype=float)
    B = np.array([[b, 0], [0, a]], dtype=float)
    return NormalFormGame(A, B, ["Opera", "Football"], ["Opera", "Football"],
                          name="Battle of the Sexes")


def matching_pennies() -> ZeroSumGame:
    A = np.array([[1, -1], [-1, 1]], dtype=float)
    return ZeroSumGame(A, ["Heads", "Tails"], ["Heads", "Tails"],
                       name="Matching Pennies")


def rock_paper_scissors() -> ZeroSumGame:
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    labels = ["Rock", "Paper", "Scissors"]
    return ZeroSumGame(A, labels, labels, name="Rock–Paper–Scissors")
