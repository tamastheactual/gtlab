"""Applied normal-form example games (ported from the course notebook).

Each factory returns a :class:`gtlab.core.NormalFormGame`. Parameterised games
(``worker_checking_game``, ``driver_police_game`` etc.) are designed to be passed
directly to the comparative-statics sweeps (``sweep_mixed``, ``sweep_pure``,
``sweep_ne_regions``).
"""
from __future__ import annotations

import numpy as np

from ..core import NormalFormGame


def worker_checking_game(c: float = 1.0) -> NormalFormGame:
    """Supervisor checks a worker; ``c`` is the monitoring cost."""
    A = np.array([[8 - c, 6 - c],
                  [8,     6]])
    B = np.array([[6, 2],
                  [6, 8]])
    return NormalFormGame(A, B,
                          row_actions=["Check", "NoCheck"],
                          col_actions=["DoJob", "Shirk"],
                          row_name="Supervisor", col_name="Worker",
                          name="Checking a Worker")


def driver_police_game(p: float = 6.0) -> NormalFormGame:
    """Driver vs police enforcement; ``p`` is the speeding penalty."""
    A = np.array([[-p, 5.0],
                  [0.0, 0.0]])
    B = np.array([[2.0, -1.0],
                  [0.0, 1.0]])
    return NormalFormGame(A, B,
                          row_actions=["Speed", "NoSpeed"],
                          col_actions=["Ticket", "NoTicket"],
                          row_name="Driver", col_name="Police",
                          name="Driver and Police")


def cooperation_job(c: float = 0.3) -> NormalFormGame:
    """Two workers on a joint project; ``c`` is the effort cost."""
    A = np.array([[1 - c, -c],
                  [0.0,   0.0]])
    B = np.array([[1 - c, 0.0],
                  [-c,    0.0]])
    return NormalFormGame(A, B,
                          row_actions=["Work", "NotWork"],
                          col_actions=["Work", "NotWork"],
                          row_name="Worker 1", col_name="Worker 2",
                          name="Cooperation in a Job")


def privilege_game(b1: float = 2.0, b2: float = 4.0,
                   c1: float = 1.0, c2: float = 1.0) -> NormalFormGame:
    """Public-good provision with heterogeneous benefits/costs."""
    A = np.array([[b2 - c2, b1 - c1],
                  [b1,      0.0]])
    B = np.array([[b2 - c2, b1],
                  [b1 - c1, 0.0]])
    return NormalFormGame(A, B,
                          row_actions=["C", "N"], col_actions=["C", "N"],
                          row_name="Family 1", col_name="Family 2",
                          name="Game of Privilege")


def waste_game(C1: float = 2.0, C2: float = 6.0,
               D1: float = 5.0, D2: float = 4.0) -> NormalFormGame:
    """Two counties lobbying against shared pollution; ``C`` cost, ``D`` damage."""
    A = np.array([[-C1, -C1],
                  [0.0, -D1]])
    B = np.array([[-C2, 0.0],
                  [-C2, -D2]])
    return NormalFormGame(A, B,
                          row_actions=["L", "N"], col_actions=["L", "N"],
                          row_name="County 1", col_name="County 2",
                          name="Waste Management")


def pick_a_number_game(m: int = 50) -> NormalFormGame:
    """Two players pick an integer 1..m; ties split, higher-but-not-matched wins."""
    acts = list(range(1, m + 1))
    A = np.zeros((m, m), dtype=float)
    B = np.zeros_like(A)
    for i, x1 in enumerate(acts):
        for j, x2 in enumerate(acts):
            if x1 == x2:
                A[i, j] = 50 - x1
                B[i, j] = 50 - x2
            elif x1 > x2:
                A[i, j] = 100 - x1
                B[i, j] = 0
            else:
                A[i, j] = 0
                B[i, j] = 100 - x2
    return NormalFormGame(A, B,
                          row_actions=[str(a) for a in acts],
                          col_actions=[str(a) for a in acts],
                          row_name="Player 1", col_name="Player 2",
                          name="Pick a Number")


def transform_payoffs(g: NormalFormGame, alpha_row: float = 1.0,
                      beta_row: float = 0.0, alpha_col: float = 1.0,
                      beta_col: float = 0.0) -> NormalFormGame:
    """Apply positive affine transforms to each player's payoffs.

    Positive affine transformations (``alpha > 0``) preserve all best responses
    and hence every Nash equilibrium - a useful sanity check in class.
    """
    A2 = alpha_row * g.A + beta_row
    B2 = alpha_col * g.B + beta_col
    return NormalFormGame(A2, B2,
                          row_actions=list(g.row_actions),
                          col_actions=list(g.col_actions),
                          row_name=g.row_name, col_name=g.col_name,
                          name=g.name)
