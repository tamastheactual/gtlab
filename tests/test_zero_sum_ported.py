"""Smoke tests for the ported zero-sum features."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import gtlab.viz as viz
from gtlab.core import ZeroSumGame
from gtlab.games.zero_sum import (micro_check_2x2, penalty_kick,
                                  rock_paper_scissors, security_audit,
                                  weighted_rps)


def _ghtml(method, *args, **kwargs):
    with viz.capture() as sink:
        method(*args, **kwargs)
    out = "".join(sink.parts)
    assert "gt-wrap" in out
    return out


def test_factories_solve():
    for f in (micro_check_2x2, rock_paper_scissors, weighted_rps,
              penalty_kick, security_audit):
        g = f()
        s = g.solve_value()
        assert set(s) >= {"p", "q", "value"}
        assert np.isclose(s["p"].sum(), 1.0)
        assert np.isclose(s["q"].sum(), 1.0)


def test_rps_value_is_zero():
    g = rock_paper_scissors()
    assert np.isclose(g.solve_value()["value"], 0.0)
    np.testing.assert_allclose(g.solve_value()["p"], np.ones(3) / 3, atol=1e-7)


def test_best_responses():
    g = micro_check_2x2()
    br_r = g.best_response_row(np.array([1.0, 0.0]))
    br_c = g.best_response_col(np.array([1.0, 0.0]))
    assert len(br_r) >= 1 and len(br_c) >= 1


def test_support_solvers():
    g = micro_check_2x2()
    sols = g.solve_all_supports()
    assert any(np.isclose(s["value"], g.solve_value()["value"]) for s in sols)
    one = g.solve_support([0, 1], [0, 1])
    assert one is not None


def test_epsilon_security_at_optimum():
    g = penalty_kick()
    s = g.solve_value()
    eps = g.epsilon_security(s["p"], s["q"])
    assert eps["eps_max"] < 1e-6


def test_as_dataframe():
    g = micro_check_2x2()
    df = g.as_dataframe()
    assert df.shape == (2, 2)


def test_display_methods_emit_html():
    g = micro_check_2x2()
    _ghtml(g.summary)
    _ghtml(g.display)
    _ghtml(g.solve)
    _ghtml(g.explain)
    _ghtml(g.lp_detail)
    _ghtml(g.support_detail)
    _ghtml(g.dominance_detail)
    _ghtml(g.security_analysis)


def test_detail_methods_3x3():
    g = weighted_rps()
    _ghtml(g.support_detail)
    _ghtml(g.dominance_detail)
    _ghtml(g.security_analysis)
    _ghtml(g.lp_detail)


def test_plots_return_fig():
    g = penalty_kick()
    for plot in (g.plot_mixed, g.plot_value_surface, g.plot_security_levels,
                 g.plot_exploitability):
        fig, _ = plot()
        assert fig is not None
    fig, _ = g.plot_convergence(T=50)
    assert fig is not None


def test_plots_reject_non_2x2():
    g = rock_paper_scissors()
    with pytest.raises(ValueError):
        g.plot_mixed()


def test_sweep_value():
    def factory(theta):
        A = np.array([[theta, -1.0], [-3.0, 4.0]])
        return ZeroSumGame(A, ["U", "D"], ["L", "R"])

    fig, axes = ZeroSumGame.sweep_value(factory, np.linspace(0.5, 3.0, 6))
    assert fig is not None and len(axes) == 3
