"""Smoke tests for ported normal-form features (plots, sweeps, IESDS, displays)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np

from gtlab.core import NormalFormGame
from gtlab.games.normal_form import (cooperation_job, driver_police_game,
                                     pick_a_number_game, privilege_game,
                                     transform_payoffs, waste_game,
                                     worker_checking_game)
from gtlab.viz import capture


def _grab(fn):
    with capture() as sink:
        fn()
    out = "".join(sink.parts)
    assert "gt-wrap" in out and len(out) > 50, fn
    return out


def _bos():
    A = np.array([[2, 0], [0, 1]], dtype=float)
    B = np.array([[1, 0], [0, 2]], dtype=float)
    return NormalFormGame(A, B, ["F", "M"], ["F", "M"],
                          name="Battle of the Sexes",
                          row_name="Husband", col_name="Wife")


def test_player_names_in_headers():
    g = _bos()
    assert g.row_name == "Husband" and g.col_name == "Wife"
    out = _grab(g.summary)
    assert "Husband" in out and "Wife" in out


def test_display_and_dataframe():
    g = _bos()
    _grab(g.display)
    df = g.as_dataframe()
    assert df.shape == (2, 2)
    assert df.iloc[0, 0] == (2.0, 1.0)


def test_solve_full_flags():
    g = _bos()
    out = _grab(lambda: g.solve(show_br=True, show_ne=True, show_pareto=True,
                                show_dominated=True, show_heatmap=True,
                                show_arrows=True, show_mixed=True))
    assert "gt-wrap" in out
    # default-arg backward compatibility
    _grab(g.solve)


def test_iesds_explain():
    pd_A = np.array([[-2, -10], [-1, -5]], dtype=float)
    pd_B = np.array([[-2, -1], [-10, -5]], dtype=float)
    g = NormalFormGame(pd_A, pd_B, ["NC", "C"], ["NC", "C"],
                       row_name="P1", col_name="P2", name="PD")
    out = _grab(g.iesds_explain)
    assert "Round" in out
    # game with no dominance still renders
    _grab(_bos().iesds_explain)


def test_plot_mixed_2x2():
    fig, axes = _bos().plot_mixed(title="mix")
    assert fig is not None and axes is not None


def test_plot_mixed_3x3():
    A = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    g = NormalFormGame(A, -A, ["L", "R", "M"], ["L", "R", "M"], name="P3")
    fig, ax = g.plot_mixed()
    assert fig is not None


def test_plot_br_curves():
    fig, ax = NormalFormGame.plot_br_curves(
        br1=lambda x2: max(0, (9 - x2) / 2),
        br2=lambda x1: max(0, (9 - x1) / 2),
        ne=[(3.0, 3.0, "NE (3, 3)")], domain=(0, 5))
    assert fig is not None


def test_sweeps():
    fig1, _ = NormalFormGame.sweep_mixed(driver_police_game,
                                         np.linspace(0.5, 20.0, 30),
                                         param_name="penalty p")
    fig2, _ = NormalFormGame.sweep_pure(cooperation_job,
                                        np.linspace(0.0, 1.5, 40),
                                        param_name="cost c")
    fig3, _ = NormalFormGame.sweep_ne_regions(
        lambda c2, c1: privilege_game(b1=2, b2=4, c1=c1, c2=c2),
        x_range=(0, 3), y_range=(0, 3), n=21)
    assert fig1 is not None and fig2 is not None and fig3 is not None


def test_game_factories():
    for g in (worker_checking_game(2.0), driver_police_game(6.0),
              cooperation_job(0.3), privilege_game(), waste_game(),
              pick_a_number_game(8)):
        assert isinstance(g, NormalFormGame)
        _grab(g.summary)
    pan = pick_a_number_game(6)
    assert pan.shape == (6, 6)


def test_transform_payoffs_preserves_ne():
    g = _bos()
    g2 = transform_payoffs(g, alpha_row=10, beta_row=3, alpha_col=2, beta_col=-5)
    assert set(g.pure_nash()) == set(g2.pure_nash())
