"""gtlab - Game Theory Lab for the ELTE Game Theory course.

Quick start::

    import gtlab
    gtlab.apply_rc()                       # consistent plot styling (once)

    from gtlab.games import prisoners_dilemma
    prisoners_dilemma().solve()            # annotated bimatrix in Jupyter

Build your own::

    from gtlab import NormalFormGame
    import numpy as np
    g = NormalFormGame(np.array([[3, 0], [5, 1]]), np.array([[3, 5], [0, 1]]))
    g.explain()

Layers:
  * ``gtlab.core``    - game classes (data + thin API)
  * ``gtlab.solvers`` - pure algorithms (best response, Nash, value iteration, …)
  * ``gtlab.viz``     - formatting, HTML, plots, theme
  * ``gtlab.games``   - ready-made example games
"""
from . import games, solvers, viz
from .core import (CheapTalkGame, CorrelatedGame, EntryGame,
                   ExtensiveFormGame, FirstPriceAuction, GeneralSumSG,
                   Mechanism, NormalFormGame, PostedPrice, Procurement,
                   PublicProject, SecondPriceAuction, SpenceSignaling,
                   StochasticGame, VCGAssignment, ZeroSumGame)
from .viz import apply_rc

__version__ = "0.2.0"

__all__ = [
    "NormalFormGame", "ZeroSumGame", "CorrelatedGame", "StochasticGame",
    "GeneralSumSG", "CheapTalkGame",
    "ExtensiveFormGame", "Mechanism", "PostedPrice", "FirstPriceAuction",
    "SecondPriceAuction", "SpenceSignaling", "VCGAssignment", "PublicProject",
    "Procurement", "EntryGame",
    "solvers", "viz", "games", "apply_rc", "__version__",
]
