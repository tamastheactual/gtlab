"""Smoke tests for the ported correlated-equilibrium features."""
import matplotlib

matplotlib.use("Agg")

import numpy as np

from gtlab.core.correlated import CorrelatedGame
from gtlab.games.correlated import (battle_of_sexes_ce, chicken_ce,
                                    prisoners_dilemma_ce, rps_ce,
                                    stag_hunt_ce, traffic_coordination)
from gtlab.viz import capture


def _grab(fn):
    with capture() as sink:
        fn()
    out = "".join(sink.parts)
    assert "gt-wrap" in out and len(out) > 50, fn
    return out


def test_factories_build():
    for factory in (chicken_ce, battle_of_sexes_ce, stag_hunt_ce,
                    prisoners_dilemma_ce, traffic_coordination, rps_ce):
        g = factory()
        assert isinstance(g, CorrelatedGame)
        assert g.shape[0] >= 2 and g.shape[1] >= 2


def test_aliases_kept():
    g = chicken_ce()
    assert g.find_nash() == g.nash()
    a = g.simulate_hedge(T=200, seed=1)
    b = g.hedge(T=200, seed=1)
    assert np.allclose(a["empirical_mu"], b["empirical_mu"])


def test_verify_ce_and_cce_display():
    g = chicken_ce()
    mu_fair = np.array([[0.0, 0.5], [0.5, 0.0]])
    _grab(lambda: g.verify_ce(mu_fair, title="fair CE"))
    _grab(lambda: g.verify_cce(mu_fair, title="fair CCE"))


def test_verify_detects_violation():
    g = prisoners_dilemma_ce()
    # All-cooperate is not a CE in the PD: defection is profitable.
    mu_cc = np.array([[1.0, 0.0], [0.0, 0.0]])
    out = _grab(lambda: g.verify_ce(mu_cc))
    assert "Not a CE" in out


def test_lp_detail_display():
    g = battle_of_sexes_ce()
    out = _grab(lambda: g.lp_detail(title="BoS LP"))
    assert "gt-steps" in out


def test_existing_displays_still_work():
    g = stag_hunt_ce()
    for fn in (g.summary, g.compare_equilibria, g.explain):
        _grab(fn)


def test_plot_payoff_region_returns_fig():
    g = stag_hunt_ce()
    fig, ax = g.plot_payoff_region(n_samples=40, seed=0)
    assert fig is not None and ax is not None
    assert len(ax.collections) > 0


def test_plot_welfare_comparison_returns_fig():
    games = {"Chicken": chicken_ce(), "Stag Hunt": stag_hunt_ce(),
             "PD": prisoners_dilemma_ce()}
    fig, ax = CorrelatedGame.plot_welfare_comparison(games, title="welfare")
    assert fig is not None and ax is not None
    assert len(ax.patches) == 9  # 3 games x 3 concepts


def test_plot_welfare_comparison_iterable():
    fig, ax = CorrelatedGame.plot_welfare_comparison([chicken_ce(), rps_ce()])
    assert fig is not None and ax is not None


def test_plot_regret_returns_fig():
    g = chicken_ce()
    fig, ax = g.plot_regret(T=200, seed=2)
    assert fig is not None and ax is not None
