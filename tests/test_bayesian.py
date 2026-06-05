"""Tests for the mechanism-design classes."""
import numpy as np

from gtlab import (FirstPriceAuction, PostedPrice, Procurement, PublicProject,
                   SecondPriceAuction, SpenceSignaling, VCGAssignment)


def test_posted_price_optimum():
    pp = PostedPrice(values=[10, 20], probs=[0.5, 0.5])
    sol = pp.solve()
    # Price 20 sells w.p. 0.5 → E[rev]=10; price 10 sells for sure → 10. Tie → 10 or 20.
    assert sol["optimal_revenue"] == 10.0


def test_auction_revenue_equivalence():
    fpa = FirstPriceAuction(n_bidders=3, lo=0, hi=1)
    spa = SecondPriceAuction(n_bidders=3, lo=0, hi=1)
    assert np.isclose(fpa.expected_revenue(), spa.expected_revenue())
    assert np.isclose(fpa.expected_revenue(), 0.5)  # (n-1)/(n+1) = 2/4


def test_fpa_bid_shading():
    fpa = FirstPriceAuction(n_bidders=2, lo=0, hi=1)
    assert np.isclose(fpa.bid(1.0), 0.5)  # shade by (n-1)/n = 1/2


def test_signaling_interval():
    sig = SpenceSignaling(w_low=10, w_high=20, c_low=2, c_high=1)
    s = sig.solve()
    assert np.isclose(s["e_min"], 5.0) and np.isclose(s["e_max"], 10.0)


def test_signaling_requires_single_crossing():
    try:
        SpenceSignaling(w_low=10, w_high=20, c_low=1, c_high=2)
    except ValueError:
        return
    raise AssertionError("expected single-crossing ValueError")


def test_vcg_efficiency_and_ir():
    V = np.array([[10, 4], [7, 9], [3, 2]], dtype=float)
    sol = VCGAssignment(V).solve()
    assert sol["welfare"] == 19.0
    assert np.all(sol["payments"] >= -1e-9)
    assert np.all(sol["utilities"] >= -1e-9)        # individually rational


def test_public_project_pivot():
    sol = PublicProject(values=[6, 5, 2], cost=10).solve()
    assert sol["build"] is True
    # Each of the two large citizens is pivotal; the small one is not.
    assert sol["payments"][2] == 0.0
    assert sol["deficit"] > 0                         # budget deficit


def test_procurement_second_price():
    sol = Procurement(costs=[2, 5], probs=[0.5, 0.5], n=2).solve()
    # Pay = E[max(c1,c2)] over {2,5}^2 = (2+5+5+5)/4 = 4.25
    assert np.isclose(sol["expected_payment"], 4.25)
