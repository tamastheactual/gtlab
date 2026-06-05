"""Smoke tests for the ported Bayesian-games / mechanism-design features."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gtlab.core.bayesian import (
    EntryGame, FirstPriceAuction, Mechanism, PostedPrice, Procurement,
    PublicProject, SecondPriceAuction, SpenceSignaling, VCGAssignment,
)
from gtlab.games.bayesian import (
    cautious_entrant, costly_diploma, honest_auctioneer, public_project,
    reverse_procurement, shaded_bid_auction, suspicious_buyer,
    vcg_assignment,
)
from gtlab.viz import capture


def _wrapped(method, *args, **kwargs):
    with capture() as sink:
        method(*args, **kwargs)
    out = "".join(sink.parts)
    assert "gt-wrap" in out
    return out


def _is_fig(ret):
    fig, ax = ret
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_posted_price():
    g = suspicious_buyer()
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=500, seed=1)
    assert len(df) == 500
    assert {"buyer_value", "revenue", "accepted"} <= set(df.columns)
    _is_fig(g.plot_revenue_curve())


def test_first_price_auction():
    g = shaded_bid_auction(3)
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=400, seed=2)
    assert len(df) == 400
    _is_fig(g.plot_bid_function(compare_n=[2, 3, 5]))
    _is_fig(g.plot_revenue_vs_n(n_range=[2, 3, 4], n_trials=300))


def test_second_price_auction():
    g = honest_auctioneer(3)
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=400, seed=3, strategy="truthful")
    assert "payment" in df.columns
    _is_fig(g.plot_bid_function())
    _is_fig(g.plot_revenue_vs_n(n_range=[2, 3, 4]))
    _is_fig(g.plot_utility_sweep(v_me=0.6, n_trials=300))


def test_entry_game():
    g = cautious_entrant()
    assert isinstance(g, EntryGame)
    sol = g.solve()
    assert sol["q_star"] is not None
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=300, seed=4)
    assert {"incumbent_type", "entrant_payoff"} <= set(df.columns)
    _is_fig(g.plot_entry_threshold(
        scenarios=[("baseline", (2.0, -1.0)), ("aggressive", (1.0, -2.0))]))


def test_spence_signaling():
    g = costly_diploma()
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=300, seed=5)
    assert {"type", "net_utility", "education"} <= set(df.columns)
    _is_fig(g.plot_signaling_curves())


def test_procurement():
    g = reverse_procurement(3)
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=400, seed=6)
    assert "winner_profit" in df.columns
    _is_fig(g.plot_utility_sweep(firm_true_cost=40.0, n_trials=300))
    _is_fig(g.plot_procurement_vs_n(n_range=[2, 3, 4], n_trials=300))


def test_vcg_assignment():
    g = vcg_assignment()
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=5, seed=7, report_noise=1.0)
    assert {"bidder", "utility", "payment"} <= set(df.columns)
    _is_fig(g.plot_utility_sweep(bidder_idx=0, item_idx=0))
    _is_fig(g.plot_payments())


def test_public_project():
    g = public_project()
    _wrapped(g.summary)
    _wrapped(g.explain)
    df = g.simulate(n_trials=50, seed=8, value_noise=5.0)
    assert {"build", "deficit", "total_pivot_payment"} <= set(df.columns)
    _is_fig(g.plot_payments())


def test_compare_and_sweep():
    a = honest_auctioneer(3)
    b = shaded_bid_auction(3)
    with capture() as sink:
        Mechanism.compare(a, b, method="summary")
    assert "gt-wrap" in "".join(sink.parts)

    fig, ax = Mechanism.sweep(
        factory=lambda q: PostedPrice(values=[40, 100], probs=[1 - q, q]),
        param_range=np.linspace(0.05, 0.95, 7),
        metric="optimal_revenue",
        param_name="Pr(high-value buyer)",
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
