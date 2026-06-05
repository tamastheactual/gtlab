"""Smoke tests for the ported extensive-form features."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import gtlab.viz as viz
from gtlab.core import ExtensiveFormGame
from gtlab.games.extensive_form import (battle_of_the_sexes_game,
                                        chain_store_game, chicken_game,
                                        parametric_bos, stag_hunt_game)


def _wrapped(method, *args, **kwargs):
    """Call a display method and assert it produces gt-wrapped HTML."""
    with viz.capture() as sink:
        method(*args, **kwargs)
    out = "".join(sink.parts)
    assert "gt-wrap" in out
    return out


def test_factories_build():
    for f in (chain_store_game, battle_of_the_sexes_game, stag_hunt_game,
              chicken_game, parametric_bos):
        g = f()
        assert isinstance(g, ExtensiveFormGame)
        assert len(g.players) == 2


def test_backward_compat_compact_schema():
    # The original compact schema must still work.
    tree = {
        "root": {"player": 0, "actions": {"Enter": "inc", "Out": "out"}},
        "inc": {"player": 1, "actions": {"Fight": "f", "Acc": "a"}},
        "out": {"payoff": (0, 3)},
        "f": {"payoff": (-1, 0)},
        "a": {"payoff": (1, 2)},
    }
    g = ExtensiveFormGame(tree, players=["Entrant", "Incumbent"])
    res = g.backward_induction()
    assert res["strategy"]["inc"] == "Acc"
    assert np.allclose(res["value"], [1, 2])


def test_chance_node_compact():
    tree = {
        "root": {"chance": {"H": (0.5, "t1"), "T": (0.5, "t2")}},
        "t1": {"payoff": (2, 0)},
        "t2": {"payoff": (0, 2)},
    }
    g = ExtensiveFormGame(tree)
    val = g.backward_induction()["value"]
    assert np.allclose(val, [1, 1])
    sim = g.simulate(rng=np.random.default_rng(0))
    assert sim["payoffs"].shape == (2,)


def test_normal_form_analysis():
    g = chain_store_game()
    A, B = g.convert_to_normal()
    assert A.shape == B.shape
    assert isinstance(g.pure_nash_nf(), list)
    assert isinstance(g.pareto_optimal_nf(), list)
    assert isinstance(g.strictly_dominated_rows(), dict)
    assert isinstance(g.strictly_dominated_cols(), dict)
    assert g.best_responses_row(0) is not None
    df = g.as_dataframe()
    assert df.shape[0] >= 1


def test_iesds_and_properties():
    g = chain_store_game()
    A, B, rl, cl, rounds = g.iesds_strict()
    assert isinstance(rounds, list)
    assert isinstance(g.is_zero_sum(), bool)
    assert isinstance(g.is_symmetric(), bool)


def test_welfare_and_poa():
    g = chain_store_game()
    out, score = g.social_welfare("utilitarian", return_outcome=True)
    assert np.isfinite(score)
    assert np.isfinite(g.price_of_anarchy())
    assert g.terminal_payoffs().shape[1] == 2
    assert len(g.pareto_outcomes()) >= 1
    assert len(g.pareto_frontier_vertices()) >= 1


def test_imperfect_info_in_conversion():
    bos = battle_of_the_sexes_game()
    A, B = bos.convert_to_normal()
    # P2 has a single information set -> 2 strategies, not 4.
    assert A.shape == (2, 2)


def test_display_methods():
    g = chain_store_game()
    _wrapped(g.summary)
    _wrapped(g.explain)
    _wrapped(g.solve)
    _wrapped(g.solve_nf)
    _wrapped(g.show_payoff_matrix)
    _wrapped(g.iesds_explain)


def test_plots_return_fig():
    g = chain_store_game()
    bos = battle_of_the_sexes_game()
    for fig_ax in (g.plot_tree(), g.plot_frontier(), g.plot_br_map(),
                   bos.plot_mixed()):
        fig = fig_ax[0]
        assert fig is not None
        assert hasattr(fig, "savefig")


def test_sweeps_return_fig():
    fig, _ = ExtensiveFormGame.sweep_mixed(
        parametric_bos, np.linspace(-1, 1.5, 8), param_name="mismatch")
    assert hasattr(fig, "savefig")
    fig, _ = ExtensiveFormGame.sweep_pure(
        lambda fl: chain_store_game(fight_loss=fl),
        np.linspace(-3, 2, 8), param_name="fight_loss")
    assert hasattr(fig, "savefig")
    fig, _ = ExtensiveFormGame.sweep_ne_regions(
        lambda x, y: parametric_bos(x),
        (-1, 1.5), (-1, 1.5), n=9)
    assert hasattr(fig, "savefig")
    fig, _ = ExtensiveFormGame.sweep_spe_regions(
        lambda x, y: chain_store_game(fight_loss=x, entry_profit=y),
        (-3, 2), (0.5, 2), decision_node="incumbent", n=9)
    assert hasattr(fig, "savefig")


def test_simulate_spe_path():
    g = chain_store_game()
    sim = g.simulate()
    assert sim["terminal"] in g.tree
    assert len(sim["path"]) >= 1


def test_validation_rejects_dangling_child():
    bad = {"root": {"player": 0, "actions": {"L": "missing"}}}
    with pytest.raises(ValueError):
        ExtensiveFormGame(bad)
