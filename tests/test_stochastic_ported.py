"""Smoke tests for the ported stochastic / communication game features."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from gtlab.core.stochastic import CheapTalkGame, StochasticGame
from gtlab.games.stochastic import (market_entry_general_sum, patrol_game,
                                     security_game)
from gtlab.solvers.stochastic_extra import (bayesian_update, best_response_iteration,
                                            check_correlated_eq, pure_values,
                                            stage_values)
from gtlab.viz import capture


def _grab(fn, *a, **k):
    with capture() as sink:
        out = fn(*a, **k)
    html = "".join(sink.parts)
    assert "gt-wrap" in html and len(html) > 50, fn
    return out


def _isfig(obj):
    fig = obj[0] if isinstance(obj, tuple) else obj
    assert isinstance(fig, Figure), obj


# ── StochasticGame new methods ────────────────────────────────────────────────
def test_stochastic_displays_and_plots():
    g = security_game()
    res = g.solve()
    for fn in (g.summary, g.policy_summary, g.explain):
        _grab(fn)
    _grab(g.value_iteration_walkthrough, n_show=2)
    _grab(g.solve_stage_games, res)
    assert g.bellman_certificate(res) is True
    # plot+display hybrids return a figure
    _isfig(_capture_fig(g.exploitability_analysis, res))
    _isfig(_capture_fig(g.gamma_sweep, [0.5, 0.8, 0.95]))
    _isfig(_capture_fig(g.perturbation_robustness, 8))
    # pure plots
    _isfig(g.plot_trajectories(res, s0=1, T=40, K=5))
    _isfig(g.plot_q_heatmap(res))
    _isfig(g.plot_convergence())


def _capture_fig(fn, *a, **k):
    with capture():
        out = fn(*a, **k)
    return out


def test_simulate_and_qvalues():
    g = patrol_game()
    res = g.solve()
    sim = g.simulate(res, s0=0, T=30, K=4, seed=1)
    assert sim["states"].shape == (4, 30)
    assert sim["rewards"].shape == (4, 30)
    Q = g.q_values(res)
    assert Q.shape == (g.nS, g.nA, g.nB)
    assert np.allclose(Q, np.stack([g.stage_game(s, res["V_star"])
                                    for s in range(g.nS)]))


def test_value_iteration_alias():
    g = security_game()
    assert np.allclose(g.value_iteration()["V_star"], g.solve()["V_star"])


def test_compare_static():
    a = security_game()
    b = security_game(recover=0.3)
    with capture() as sink:
        results = StochasticGame.compare([a, b], ["base", "sticky"])
    out = "".join(sink.parts)
    assert "gt-wrap" in out and "base" in out and "sticky" in out
    assert len(results) == 2


# ── GeneralSumSG ──────────────────────────────────────────────────────────────
def test_general_sum_sg():
    mg = market_entry_general_sum()
    res = mg.best_response_iteration()
    assert res["V1"].shape == (mg.nS,)
    for fn in (mg.summary, mg.explain):
        _grab(fn)
    w = mg.welfare(res)
    assert "utilitarian" in w and "egalitarian" in w and "nash_product" in w
    V1, V2 = mg.pure_values([0, 0], [0, 0])
    assert V1.shape == (mg.nS,)


# ── CheapTalkGame ─────────────────────────────────────────────────────────────
def test_cheap_talk():
    ct = CheapTalkGame(name="Sender-Receiver")
    eq = ct.equilibrium(0.5)
    assert "is_pooling" in eq and "is_separating" in eq
    _grab(ct.summary)
    _grab(ct.equilibrium_analysis, 0.3)
    _isfig(ct.plot_equilibrium_regions())


# ── solver extras ─────────────────────────────────────────────────────────────
def test_solver_extras():
    mg = market_entry_general_sum()
    res = best_response_iteration(mg.r1, mg.r2, mg.P, mg.gamma)
    assert res["V1"].shape == (mg.nS,)
    V1, V2 = pure_values(mg.r1, mg.r2, mg.P, mg.gamma, [0, 0], [0, 0])
    assert V1.shape == (mg.nS,)
    sv = stage_values(security_game().r)
    assert len(sv) == 2

    P_tr = np.array([[[[0.7, 0.3]]], [[[0.3, 0.7]]]])
    O_obs = np.array([[0.8, 0.2], [0.2, 0.8]])
    b = bayesian_update(np.array([0.5, 0.5]), 0, P_tr, O_obs, 0, 0)
    assert abs(b.sum() - 1.0) < 1e-9

    phi = np.array([[0.5, 0.0], [0.0, 0.5]])
    U1 = np.array([[2.0, 0.0], [0.0, 1.0]])
    U2 = np.array([[1.0, 0.0], [0.0, 2.0]])
    chk = check_correlated_eq(phi, U1, U2, ["a0", "a1"], ["b0", "b1"])
    assert "valid" in chk and "EU1" in chk
