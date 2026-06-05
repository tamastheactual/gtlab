"""Applied / economic example games."""
from __future__ import annotations

import numpy as np

from ..core import ExtensiveFormGame, NormalFormGame


def cournot_duopoly(a=10.0, cost=1.0, quantities=(2.0, 3.0, 4.0)) -> NormalFormGame:
    """Discretized Cournot duopoly. Price P = a − (q_i + q_j); profit = (P − c)·q_i."""
    qs = np.asarray(quantities, dtype=float)
    n = len(qs)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for i, qi in enumerate(qs):
        for j, qj in enumerate(qs):
            price = max(a - (qi + qj), 0.0)
            A[i, j] = (price - cost) * qi
            B[i, j] = (price - cost) * qj
    labels = [f"q={q:g}" for q in qs]
    return NormalFormGame(A, B, labels, labels, name="Cournot Duopoly")


def entry_deterrence(incumbent_fight=0, incumbent_accommodate=2,
                     entrant_enter_fight=-1, entrant_enter_accommodate=1,
                     entrant_stay_out=0, incumbent_monopoly=3) -> ExtensiveFormGame:
    """Entrant moves first; if it enters, the incumbent fights or accommodates."""
    tree = {
        "root": {"player": 0, "actions": {"Enter": "incumbent", "Stay out": "out"}},
        "incumbent": {"player": 1, "actions": {"Fight": "fight", "Accommodate": "acc"}},
        "out": {"payoff": (entrant_stay_out, incumbent_monopoly)},
        "fight": {"payoff": (entrant_enter_fight, incumbent_fight)},
        "acc": {"payoff": (entrant_enter_accommodate, incumbent_accommodate)},
    }
    return ExtensiveFormGame(tree, root="root", players=["Entrant", "Incumbent"],
                             name="Entry Deterrence")
