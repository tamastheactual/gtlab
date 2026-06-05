"""Extensive-form example games ported from the EFG notebook.

Each factory returns a :class:`gtlab.core.ExtensiveFormGame`. Simultaneous-move
classics (Battle of the Sexes, Stag Hunt, Chicken) are represented with an
information set joining the second mover's nodes.
"""
from __future__ import annotations

from ..core import ExtensiveFormGame


def chain_store_game(incumbent_profit: float = 2, entry_profit: float = 1,
                     fight_loss: float = -1) -> ExtensiveFormGame:
    """Chain-store / entry-deterrence game.

    Entrant chooses In/Out; if In, the incumbent fights or accommodates.
    """
    tree = {
        "root": {"player": 0, "actions": ["In", "Out"],
                 "children": {"In": "incumbent", "Out": "term_out"}},
        "incumbent": {"player": 1, "actions": ["Fight", "Accommodate"],
                      "children": {"Fight": "term_fight",
                                   "Accommodate": "term_accom"}},
        "term_out": {"is_terminal": True, "payoffs": (0, incumbent_profit)},
        "term_fight": {"is_terminal": True,
                       "payoffs": (fight_loss, fight_loss + 0.5)},
        "term_accom": {"is_terminal": True,
                       "payoffs": (entry_profit, entry_profit)},
    }
    return ExtensiveFormGame(tree, players=["Entrant", "Incumbent"],
                             name="Chain Store")


def _simultaneous_2x2(a1, a2, payoffs, name, players=("Player 1", "Player 2")):
    """Build a simultaneous 2x2 game: P1 moves, then P2 in one info set."""
    tree = {
        "root": {"player": 0, "actions": [a1, a2],
                 "children": {a1: "p2_after_1", a2: "p2_after_2"}},
        "p2_after_1": {"player": 1, "actions": [a1, a2],
                       "children": {a1: "n11", a2: "n12"}},
        "p2_after_2": {"player": 1, "actions": [a1, a2],
                       "children": {a1: "n21", a2: "n22"}},
        "n11": {"is_terminal": True, "payoffs": payoffs[(a1, a1)]},
        "n12": {"is_terminal": True, "payoffs": payoffs[(a1, a2)]},
        "n21": {"is_terminal": True, "payoffs": payoffs[(a2, a1)]},
        "n22": {"is_terminal": True, "payoffs": payoffs[(a2, a2)]},
    }
    return ExtensiveFormGame(
        tree, players=list(players), name=name,
        info_sets={"P2_info": ["p2_after_1", "p2_after_2"]})


def battle_of_the_sexes_game() -> ExtensiveFormGame:
    """Battle of the Sexes (simultaneous, P2 in one information set)."""
    payoffs = {("Opera", "Opera"): (2, 1), ("Opera", "Football"): (0, 0),
               ("Football", "Opera"): (0, 0), ("Football", "Football"): (1, 2)}
    return _simultaneous_2x2("Opera", "Football", payoffs,
                             "Battle of the Sexes")


def parametric_bos(mismatch: float = 0.0) -> ExtensiveFormGame:
    """BoS where miscoordination yields ``mismatch`` for both players."""
    payoffs = {("Opera", "Opera"): (2, 1),
               ("Opera", "Football"): (mismatch, mismatch),
               ("Football", "Opera"): (mismatch, mismatch),
               ("Football", "Football"): (1, 2)}
    return _simultaneous_2x2("Opera", "Football", payoffs,
                             "Parametric BoS", players=("P1", "P2"))


def stag_hunt_game() -> ExtensiveFormGame:
    """Stag Hunt (simultaneous, P2 in one information set)."""
    payoffs = {("Stag", "Stag"): (4, 4), ("Stag", "Hare"): (0, 2),
               ("Hare", "Stag"): (2, 0), ("Hare", "Hare"): (2, 2)}
    return _simultaneous_2x2("Stag", "Hare", payoffs, "Stag Hunt")


def chicken_game() -> ExtensiveFormGame:
    """Chicken / Hawk-Dove (simultaneous, P2 in one information set)."""
    payoffs = {("Swerve", "Swerve"): (2, 2), ("Swerve", "Straight"): (1, 3),
               ("Straight", "Swerve"): (3, 1), ("Straight", "Straight"): (0, 0)}
    return _simultaneous_2x2("Swerve", "Straight", payoffs, "Chicken")
