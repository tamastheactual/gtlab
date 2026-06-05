"""Library of ready-made games. Extend by adding a factory and registering it.

    from gtlab.games import prisoners_dilemma, REGISTRY
    g = prisoners_dilemma()
    g2 = REGISTRY["chicken"]()
"""
from .applied import cournot_duopoly, entry_deterrence
from .classic import (battle_of_the_sexes, chicken, matching_pennies,
                      prisoners_dilemma, rock_paper_scissors, stag_hunt)

# Name → factory. New games belong here so they are discoverable.
REGISTRY = {
    "prisoners_dilemma": prisoners_dilemma,
    "stag_hunt": stag_hunt,
    "chicken": chicken,
    "battle_of_the_sexes": battle_of_the_sexes,
    "matching_pennies": matching_pennies,
    "rock_paper_scissors": rock_paper_scissors,
    "cournot_duopoly": cournot_duopoly,
    "entry_deterrence": entry_deterrence,
}

__all__ = [
    "prisoners_dilemma", "stag_hunt", "chicken", "battle_of_the_sexes",
    "matching_pennies", "rock_paper_scissors", "cournot_duopoly",
    "entry_deterrence", "REGISTRY",
]
