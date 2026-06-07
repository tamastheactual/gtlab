"""Example Bayesian games and mechanisms ported from the lecture notebook.

Each factory returns a ready-to-use mechanism instance from
:mod:`gtlab.core.bayesian`.
"""
from __future__ import annotations

import numpy as np

from ..core.bayesian import (
    EntryGame,
    FirstPriceAuction,
    PostedPrice,
    Procurement,
    PublicProject,
    SecondPriceAuction,
    SpenceSignaling,
    VCGAssignment,
)


def suspicious_buyer() -> PostedPrice:
    """The Suspicious Buyer: a low/high-value buyer facing a posted price."""
    return PostedPrice(values=[40, 100], probs=[0.6, 0.4],
                       name="Low/High buyer")


def cautious_entrant() -> EntryGame:
    """The Cautious Entrant: incumbent type private; entrant best-responds in expectation."""
    return EntryGame(payoff_weak=(2.0, -1.0), payoff_strong=(-1.0, 1.0),
                     prior_strong=0.5, stay_out=0.0, name="Cautious entrant")


def shaded_bid_auction(n_bidders: int = 3) -> FirstPriceAuction:
    """The Shaded Bid: symmetric first-price auction on Uniform[0, 1]."""
    return FirstPriceAuction(n_bidders=n_bidders, lo=0.0, hi=1.0,
                             name="First-price (shaded bid)")


def honest_auctioneer(n_bidders: int = 3) -> SecondPriceAuction:
    """The Honest Auctioneer: second-price (Vickrey) auction on Uniform[0, 1]."""
    return SecondPriceAuction(n_bidders=n_bidders, lo=0.0, hi=1.0,
                              name="Second-price (Vickrey)")


def costly_diploma() -> SpenceSignaling:
    """The Costly Diploma: Spence job-market signaling with single-crossing costs."""
    return SpenceSignaling(w_low=40.0, w_high=100.0, c_low=20.0, c_high=8.0,
                           prior_high=0.5, name="Costly diploma")


def reverse_procurement(n_firms: int = 3) -> Procurement:
    """The Reverse Auction: reverse-Vickrey procurement with discrete costs."""
    return Procurement(costs=[20.0, 40.0, 60.0], probs=[0.3, 0.4, 0.3],
                       n=n_firms, name="Reverse procurement")


def vcg_assignment() -> VCGAssignment:
    """The VCG Assignment: two bidders, two items, unit demand.

    Both bidders rank Item A first, so the winner of A (Alice) pays the
    externality she imposes on Bob, while Bob wins Item B uncontested and pays
    nothing - a clean illustration of "pay the harm you cause the others".
    """
    V = np.array([[10.0, 2.0], [8.0, 3.0]])
    return VCGAssignment(V=V, bidders=["Alice", "Bob"], items=["Item A", "Item B"],
                         name="VCG assignment")


def public_project() -> PublicProject:
    """The Public Project: Clarke-pivot mechanism for a binary public good."""
    return PublicProject(values=[60.0, 50.0, 40.0], cost=120.0,
                         name="Public project (Clarke pivot)")
