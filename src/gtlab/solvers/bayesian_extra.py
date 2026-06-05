"""Pure algorithms for the Bayesian-games / mechanism-design area.

These are the closed-form formulas and Monte-Carlo kernels ported from the
original notebook engine. They operate on plain numbers / numpy arrays so the
core classes only have to wire them into the viz layer. Nothing here imports
matplotlib or the display layer.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Posted price ────────────────────────────────────────────────────────────

def posted_price_revenue(values: Sequence[float], probs: Sequence[float],
                         price: float) -> float:
    """E[revenue] = price * Pr(value >= price)."""
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(price * probs[values >= price].sum())


def posted_price_threshold(values: Sequence[float]) -> Optional[float]:
    """Two-type threshold belief Pr(high)* making the seller indifferent.

    Seller is indifferent between charging v_low (sells always) and v_high
    (sells only to high types) when v_low = Pr(high) * v_high, i.e.
    Pr(high)* = v_low / v_high. Returns ``None`` if not a 2-type game.
    """
    vals = sorted(float(v) for v in values)
    if len(vals) != 2 or vals[1] <= 0:
        return None
    return vals[0] / vals[1]


# ── Entry game ──────────────────────────────────────────────────────────────

def entry_expected_payoff(payoff_weak: float, payoff_strong: float,
                          prior_strong: float) -> float:
    """Entrant's expected payoff from entering, taking expectation over types."""
    q = float(prior_strong)
    return q * float(payoff_strong) + (1.0 - q) * float(payoff_weak)


def entry_threshold(payoff_weak: float, payoff_strong: float,
                    stay_out: float = 0.0) -> Optional[float]:
    """Prior q* on the strong type making the entrant indifferent to entry.

    q*(pi_strong - pi_weak) + pi_weak = stay_out. Returns ``None`` when the
    two type payoffs coincide (no threshold).
    """
    denom = float(payoff_strong) - float(payoff_weak)
    if abs(denom) < 1e-12:
        return None
    return (float(stay_out) - float(payoff_weak)) / denom


# ── Auctions (symmetric IPV, Uniform[lo, hi]) ───────────────────────────────

def fpa_bid(value: float, n: int, lo: float, hi: float) -> float:
    """Symmetric first-price BNE bid: b(v) = lo + (n-1)/n * (v - lo)."""
    return float(lo) + (n - 1.0) / n * (float(value) - float(lo))


def auction_expected_revenue(n: int, lo: float, hi: float) -> float:
    """E[revenue] = lo + (n-1)/(n+1) * (hi - lo) (= E[2nd-highest value])."""
    return float(lo) + (n - 1.0) / (n + 1.0) * (float(hi) - float(lo))


def auction_expected_winner_value(n: int, lo: float, hi: float) -> float:
    """E[highest of n i.i.d. U[lo, hi]] = lo + n/(n+1) * (hi - lo)."""
    return float(lo) + n / (n + 1.0) * (float(hi) - float(lo))


def simulate_fpa(n: int, lo: float, hi: float, n_trials: int, rng,
                 strategy: str = "bne") -> Dict[str, np.ndarray]:
    """Monte-Carlo a first-price auction; return per-trial column arrays."""
    V = rng.uniform(lo, hi, size=(n_trials, n))
    if strategy == "bne":
        B = lo + (n - 1) / n * (V - lo)
    elif strategy == "truthful":
        B = V.copy()
    else:
        raise ValueError(f"unknown strategy {strategy!r}; use 'bne' or 'truthful'")
    winner = np.argmax(B, axis=1)
    idx = np.arange(n_trials)
    winning_bid = B[idx, winner]
    winning_val = V[idx, winner]
    return {"winner": winner + 1, "winning_value": winning_val,
            "winning_bid": winning_bid, "revenue": winning_bid,
            "winner_utility": winning_val - winning_bid}


def simulate_spa(n: int, lo: float, hi: float, n_trials: int, rng,
                 strategy: str = "truthful") -> Dict[str, np.ndarray]:
    """Monte-Carlo a second-price auction; return per-trial column arrays."""
    V = rng.uniform(lo, hi, size=(n_trials, n))
    if strategy == "truthful":
        B = V.copy()
    elif strategy == "overbid":
        B = V * 1.5
    elif strategy == "underbid":
        B = V * 0.5
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    winner = np.argmax(B, axis=1)
    payment = np.sort(B, axis=1)[:, -2]
    winning_val = V[np.arange(n_trials), winner]
    return {"winner": winner + 1, "winning_value": winning_val,
            "payment": payment, "revenue": payment,
            "winner_utility": winning_val - payment}


# ── Spence signaling ────────────────────────────────────────────────────────

def signaling_interval(w_low: float, w_high: float,
                       c_low: float, c_high: float) -> Tuple[float, float, float]:
    """Separating education interval [e_min, e_max] and its midpoint e_star."""
    d = float(w_high) - float(w_low)
    e_min = d / float(c_low)      # low-type IC
    e_max = d / float(c_high)     # high-type IC
    return e_min, e_max, 0.5 * (e_min + e_max)


# ── Procurement (reverse Vickrey, discrete costs) ───────────────────────────

def procurement_expectations(costs: Sequence[float], probs: Sequence[float],
                             n: int, max_exact: int = 5000,
                             n_sim: int = 20_000) -> Dict[str, float]:
    """E[payment] (2nd-lowest cost), E[winner cost], E[rent] over n i.i.d. firms."""
    costs = np.asarray(costs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    n_types = len(costs)
    if n_types ** n <= max_exact:
        e_pay = e_win = 0.0
        for profile in product(range(n_types), repeat=n):
            p = float(np.prod(probs[list(profile)]))
            srt = sorted(costs[list(profile)])
            e_pay += p * srt[1 if n >= 2 else 0]
            e_win += p * srt[0]
    else:
        draws = np.random.default_rng(0).choice(costs, size=(n_sim, n), p=probs)
        srt = np.sort(draws, axis=1)
        e_pay = float(srt[:, 1].mean()) if n >= 2 else float(srt[:, 0].mean())
        e_win = float(srt[:, 0].mean())
    return {"expected_payment": float(e_pay),
            "expected_winner_cost": float(e_win),
            "expected_rent": float(e_pay - e_win)}


def simulate_procurement(costs: Sequence[float], probs: Sequence[float], n: int,
                         n_trials: int, rng,
                         strategy: str = "truthful") -> Dict[str, np.ndarray]:
    """Monte-Carlo a reverse-Vickrey procurement auction."""
    costs = np.asarray(costs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    C_true = rng.choice(costs, size=(n_trials, n), p=probs)
    if strategy == "truthful":
        R = C_true.copy()
    elif strategy == "overstate":
        R = C_true * 1.25
    elif strategy == "understate":
        R = C_true * 0.75
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    winner = np.argmin(R, axis=1)
    srt = np.sort(R, axis=1)
    payment = srt[:, 1] if n >= 2 else srt[:, 0]
    winner_cost = C_true[np.arange(n_trials), winner]
    return {"winner": winner + 1, "winner_true_cost": winner_cost,
            "payment": payment, "winner_profit": payment - winner_cost}


# ── VCG assignment ──────────────────────────────────────────────────────────

def vcg_enumerate(V: np.ndarray) -> List[Tuple[Tuple[int, ...], float]]:
    """Every feasible assignment (item -> bidder index, or -1) with its welfare."""
    n_b, n_it = V.shape
    out: List[Tuple[Tuple[int, ...], float]] = []

    def assign(pos: int, used: set, cur: List[int]) -> None:
        if pos == n_it:
            w = sum(V[cur[j], j] if cur[j] >= 0 else 0.0 for j in range(n_it))
            out.append((tuple(cur), float(w)))
            return
        assign(pos + 1, used, cur + [-1])
        for i in range(n_b):
            if i not in used:
                assign(pos + 1, used | {i}, cur + [i])

    assign(0, set(), [])
    return out


def vcg_efficient(V: np.ndarray) -> Tuple[Tuple[int, ...], float]:
    """Welfare-maximising assignment and its welfare."""
    return max(vcg_enumerate(V), key=lambda x: x[1])


def vcg_solve(V: np.ndarray) -> Dict[str, object]:
    """Full VCG outcome: efficient assignment, payments, utilities."""
    V = np.asarray(V, dtype=float)
    n_b = V.shape[0]
    a_star, w_star = vcg_efficient(V)
    alloc = [-1] * n_b
    for j, i in enumerate(a_star):
        if i >= 0:
            alloc[i] = j
    payments = np.zeros(n_b)
    for i in range(n_b):
        own = V[i, alloc[i]] if alloc[i] >= 0 else 0.0
        _, w_without_i = vcg_efficient(np.delete(V, i, axis=0))
        payments[i] = w_without_i - (w_star - own)
    utilities = np.array([
        (V[i, alloc[i]] if alloc[i] >= 0 else 0.0) - payments[i]
        for i in range(n_b)
    ])
    return {"assignment": list(a_star), "alloc": alloc,
            "welfare": float(w_star), "payments": payments,
            "utilities": utilities}


def vcg_utility_of_report(V: np.ndarray, bidder: int, item: int,
                          report: float) -> float:
    """Bidder's TRUE-value utility when she reports ``report`` for ``item``."""
    V_rep = np.asarray(V, dtype=float).copy()
    V_rep[bidder, item] = report
    sol = vcg_solve(V_rep)
    alloc = sol["alloc"]
    j = alloc[bidder]
    v_true = float(V[bidder, j]) if j >= 0 else 0.0
    return v_true - float(sol["payments"][bidder])


# ── Public project (Clarke pivot) ───────────────────────────────────────────

def public_project_solve(values: Sequence[float], cost: float) -> Dict[str, object]:
    """Clarke-pivot outcome for a binary public good."""
    v = np.asarray(values, dtype=float)
    total = float(v.sum())
    build = bool(total >= cost)
    pay = np.zeros_like(v)
    if build:
        for i in range(len(v)):
            others = total - v[i]
            if others < cost:                     # i is pivotal
                pay[i] = max(0.0, cost - others)
    total_pay = float(pay.sum())
    return {"build": build, "total_value": total, "payments": pay,
            "total_payment": total_pay,
            "deficit": float(cost - total_pay) if build else 0.0,
            "pivotal": [build and (total - v[i] < cost) for i in range(len(v))]}
