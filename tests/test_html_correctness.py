"""Regression tests for the HTML-visualization correctness fixes (0.2.6).

Each guards a specific bug found in the visualization audit so it cannot return.
"""
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np

from gtlab import CorrelatedGame, NormalFormGame, PostedPrice
from gtlab.games import (chain_store_game, entry_deterrence,
                         market_entry_general_sum)
from gtlab.viz import capture
from gtlab.viz.format import fmt


def _html(fn, *a, **k):
    with capture() as sink:
        fn(*a, **k)
    return "".join(sink.parts)


def _has_class(html, cls):
    return bool(re.search(rf'class="[^"]*{cls}', html))


# BLOCKER-1: strict (not weak) dominance - these games have NO strictly
# dominated strategies, so solve_nf must strike nothing.
def test_strict_dominance_strikes_nothing():
    for g in (entry_deterrence(), chain_store_game()):
        assert not _has_class(_html(g.solve_nf), "gt-dom")


# BLOCKER-2: a cycling best-response iteration must not be reported as an MPE.
def test_general_sum_non_convergence_is_honest():
    gs = market_entry_general_sum()
    res = gs.best_response_iteration()
    assert res["converged"] is False
    html = _html(gs.summary).lower()
    assert "not an equilibrium" in html


# BLOCKER-3: imperfect-info backward induction must surface an in-card caveat.
def test_imperfect_info_caveat_in_card():
    from gtlab.games import battle_of_the_sexes_game
    assert "information sets" in _html(battle_of_the_sexes_game().solve)


# MAJOR-1: compare_equilibria must use the welfare-maximizing NE.
def test_compare_equilibria_uses_welfare_max_ne():
    A = np.array([[1, 0], [0, 5]], dtype=float)
    html = _html(CorrelatedGame(A, A).compare_equilibria)
    assert "max-welfare NE" in html
    # the best NE already achieves welfare 10 (same as CE/CCE here)
    row = re.search(r"max-welfare NE</td>(.*?)</tr>", html).group(1)
    assert "10" in re.sub(r"<[^>]+>", " ", row)


# MAJOR-3: PostedPrice.explain must iterate the candidate price set, not values.
def test_posted_price_explain_uses_candidate_prices():
    pp = PostedPrice(values=[10, 20], probs=[0.5, 0.5], prices=[12, 18])
    html = re.sub(r"<[^>]+>", " ", _html(pp.explain))
    assert "12" in html and "18" in html


# iesds_explain strikethrough must reflect the FULL elimination, not round 1.
def test_iesds_explain_full_elimination_strikethrough():
    A = np.array([[4, 3, 2], [0, 1, 3]], dtype=float)
    B = np.array([[2, 1, 0], [1, 2, 1]], dtype=float)
    g = NormalFormGame(A, B, ["Top", "Bottom"], ["L", "M", "R"])
    html = _html(g.iesds_explain)
    # IESDS removes R, then Bottom, then M -> every eliminated strategy struck.
    assert _has_class(html, "gt-dom")
    # the surviving cell (Top, L) is the only outcome; many cells must be struck
    assert html.count("gt-dom") >= 4


# fmt() must never render a near-zero value as "-0".
def test_fmt_no_negative_zero():
    assert fmt(-7.8e-9) == "0"
    assert fmt(-0.0) == "0"
    assert fmt(-0.00001) == "0"
