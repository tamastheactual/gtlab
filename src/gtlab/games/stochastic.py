"""Example stochastic and general-sum stochastic games.

Factory functions ported from the Stochastic-Games-and-Communication notebook.
"""
from __future__ import annotations

import numpy as np

from ..core.stochastic import GeneralSumSG, StochasticGame


def security_game(gamma: float = 0.8, escalate: float = 0.7,
                  recover: float = 0.6) -> StochasticGame:
    """Two-state zero-sum security game.

    ``escalate`` : P(Secure -> Vulnerable) when the attacker Probes.
    ``recover``  : P(Vulnerable -> Secure) when the defender Patches.

    Designed so the equilibrium is genuinely mixed at Secure (a matching-pennies
    style monitor/probe interaction) and the induced Markov chain visits BOTH
    states: the attacker probes with positive probability (escalating), while at
    Vulnerable the defender patches and recovers. This keeps simulated
    trajectories dynamic rather than absorbing.
    """
    # Row = defender payoff (zero-sum). Secure: mixed (no saddle); Vulnerable:
    # patching dominates monitoring so the defender works to recover.
    r = np.array([
        [[2, -1], [-1, 1]],    # Secure:     Monitor/Patch x Probe/Wait
        [[-3, -3], [-1, -1]],  # Vulnerable: patching limits the damage
    ], dtype=float)
    # Transitions to [Secure, Vulnerable].
    P = np.array([
        # Secure: probing escalates (attacker-driven), waiting stays secure.
        [[[1 - escalate, escalate], [1.0, 0.0]],   # Monitor x [Probe, Wait]
         [[1 - escalate, escalate], [1.0, 0.0]]],  # Patch   x [Probe, Wait]
        # Vulnerable: patching recovers (defender-driven), monitoring stays.
        [[[0.0, 1.0], [0.0, 1.0]],                 # Monitor x [Probe, Wait]
         [[recover, 1 - recover], [recover, 1 - recover]]],  # Patch
    ], dtype=float)
    return StochasticGame(r, P, gamma=gamma,
                          state_names=["Secure", "Vulnerable"],
                          row_actions=["Monitor", "Patch"],
                          col_actions=["Probe", "Wait"],
                          name="Two-State Security Game")


def patrol_game(gamma: float = 0.9,
                miss_penalty_breach: float = -5) -> StochasticGame:
    """Three-state zero-sum patrol game with Breached as an absorbing bad state."""
    r = np.array([
        [[1, -1], [-1, 1]],                                   # far
        [[0, -2], [-2, 0]],                                   # near
        [[miss_penalty_breach] * 2, [miss_penalty_breach] * 2],  # breached
    ], dtype=float)
    P = np.array([
        # far: matching blocks (stay far), mismatching advances (far -> near)
        [[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]]],
        # near: matching returns to far, mismatching advances (near -> breached)
        [[[1, 0, 0], [0, 0, 1]], [[0, 0, 1], [1, 0, 0]]],
        # breached: absorbing
        [[[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]],
    ], dtype=float)
    return StochasticGame(r, P, gamma=gamma,
                          state_names=["far", "near", "breached"],
                          row_actions=["North", "South"],
                          col_actions=["North", "South"],
                          name="Three-State Patrol Game")


def market_entry_general_sum(gamma: float = 0.9) -> GeneralSumSG:
    """Battle-of-Sexes-across-states general-sum stochastic game.

    Actions: invest in A or B. Firm 1 prefers (A,A); Firm 2 prefers (B,B).
    State transitions depend on the joint choice.
    """
    r1 = np.array([
        [[3, 0], [0, 1]],   # Boom-A
        [[1, 0], [0, 3]],   # Boom-B (roles flip)
    ], dtype=float)
    r2 = np.array([
        [[2, 0], [0, 3]],   # Boom-A, Firm 2 prefers (B,B)
        [[3, 0], [0, 2]],   # Boom-B, Firm 2 prefers (A,A)
    ], dtype=float)
    P = np.array([
        # Boom-A
        [[[0.9, 0.1], [0.5, 0.5]],
         [[0.5, 0.5], [0.1, 0.9]]],
        # Boom-B
        [[[0.9, 0.1], [0.5, 0.5]],
         [[0.5, 0.5], [0.1, 0.9]]],
    ], dtype=float)
    return GeneralSumSG(r1, r2, P, gamma=gamma,
                        state_names=["Boom-A", "Boom-B"],
                        row_actions=["Invest A", "Invest B"],
                        col_actions=["Invest A", "Invest B"],
                        name="Market-Entry General-Sum Game")
