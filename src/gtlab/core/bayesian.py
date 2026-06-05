"""Bayesian games and mechanism design.

The original notebook dispatched ~8 mechanisms through one dataclass. Here each
mechanism is a small focused class with the closed-form results from the
lecture, sharing the display layer. Add a mechanism by subclassing
:class:`Mechanism` and implementing ``solve``/``summary``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..viz import fmt, fmt_money, fmt_prob, html


class Mechanism:
    """Base class for a mechanism-design example."""

    name: str = "Mechanism"

    def solve(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def summary(self, title: Optional[str] = None) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class PostedPrice(Mechanism):
    """Single seller posts a price; buyer's value is private (discrete types)."""

    values: Sequence[float]
    probs: Sequence[float]
    name: str = "Posted price"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.probs = np.asarray(self.probs, dtype=float)
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("type probabilities must sum to 1")

    def expected_revenue(self, price: float) -> float:
        """E[revenue] = price · P(value ≥ price)."""
        return float(price * self.probs[self.values >= price].sum())

    def solve(self) -> Dict[str, Any]:
        # Optimal posted price is one of the candidate type values.
        revenues = {float(v): self.expected_revenue(v) for v in self.values}
        best = max(revenues, key=revenues.get)
        return {"revenues": revenues, "optimal_price": best,
                "optimal_revenue": revenues[best]}

    def summary(self, title: Optional[str] = None) -> None:
        rows = [[fmt_money(v), fmt_prob(p)] for v, p in zip(self.values, self.probs)]
        tbl = html.table(["value", "probability"], rows)
        sol = self.solve()
        body = (tbl + f'<p><b>Optimal price</b> {fmt_money(sol["optimal_price"])} '
                f'→ E[revenue] {fmt_money(sol["optimal_revenue"])}</p>')
        html.show(html.card(title or self.name, body))


@dataclass
class FirstPriceAuction(Mechanism):
    """Symmetric IPV first-price auction, values ~ Uniform[lo, hi]."""

    n_bidders: int
    lo: float = 0.0
    hi: float = 1.0
    name: str = "First-price auction"

    def bid(self, value: float) -> float:
        """BNE bid: shade toward lo by a factor (n-1)/n."""
        n = self.n_bidders
        return self.lo + (n - 1) / n * (value - self.lo)

    def expected_revenue(self) -> float:
        """E[revenue] = lo + (n-1)/(n+1) · (hi - lo) (= E[2nd-highest value])."""
        n = self.n_bidders
        return self.lo + (n - 1) / (n + 1) * (self.hi - self.lo)

    def solve(self) -> Dict[str, Any]:
        return {"expected_revenue": self.expected_revenue(),
                "shading_factor": (self.n_bidders - 1) / self.n_bidders}

    def summary(self, title: Optional[str] = None) -> None:
        sol = self.solve()
        body = (f"<p>{self.n_bidders} bidders, values ~ U[{fmt(self.lo)}, {fmt(self.hi)}]</p>"
                f"<p><b>BNE bid:</b> b(v) = {fmt(self.lo)} + "
                f"{fmt(sol['shading_factor'])}·(v − {fmt(self.lo)})</p>"
                f"<p><b>Expected revenue:</b> {fmt(sol['expected_revenue'])}</p>")
        html.show(html.card(title or self.name, body))


@dataclass
class SecondPriceAuction(Mechanism):
    """Vickrey auction: truthful bidding is weakly dominant; revenue-equivalent to FPA."""

    n_bidders: int
    lo: float = 0.0
    hi: float = 1.0
    name: str = "Second-price auction"

    def expected_revenue(self) -> float:
        n = self.n_bidders
        return self.lo + (n - 1) / (n + 1) * (self.hi - self.lo)

    def solve(self) -> Dict[str, Any]:
        return {"expected_revenue": self.expected_revenue(), "dominant": "truthful"}

    def summary(self, title: Optional[str] = None) -> None:
        sol = self.solve()
        body = (f"<p>{self.n_bidders} bidders, values ~ U[{fmt(self.lo)}, {fmt(self.hi)}]</p>"
                "<p><b>Weakly dominant strategy:</b> bid your true value.</p>"
                f"<p><b>Expected revenue:</b> {fmt(sol['expected_revenue'])} "
                "(equals the first-price auction — revenue equivalence).</p>")
        html.show(html.card(title or self.name, body))


@dataclass
class SpenceSignaling(Mechanism):
    """Spence job-market signaling: a worker's type is private; education is a
    costly, productivity-free signal. Single-crossing (``c_high < c_low``)
    yields a non-empty separating interval of education levels."""

    w_low: float
    w_high: float
    c_low: float            # cost of education per unit for the LOW type
    c_high: float           # cost per unit for the HIGH type (c_high < c_low)
    name: str = "Spence signaling"

    def __post_init__(self) -> None:
        if not self.c_high < self.c_low:
            raise ValueError("single-crossing requires c_high < c_low")

    def solve(self) -> Dict[str, Any]:
        d = self.w_high - self.w_low
        e_min = d / self.c_low      # low-type indifference (IC for low type)
        e_max = d / self.c_high     # high-type indifference (IC for high type)
        e_star = 0.5 * (e_min + e_max)
        return {
            "e_min": e_min, "e_max": e_max, "e_star": e_star,
            "u_high": self.w_high - self.c_high * e_star,
            "u_low": self.w_low,
        }

    def summary(self, title: Optional[str] = None) -> None:
        s = self.solve()
        rows = [
            ["e<sub>min</sub> (low-type IC)", "(w<sub>H</sub>−w<sub>L</sub>)/c<sub>L</sub>", fmt(s["e_min"])],
            ["e<sub>max</sub> (high-type IC)", "(w<sub>H</sub>−w<sub>L</sub>)/c<sub>H</sub>", fmt(s["e_max"])],
        ]
        tbl = html.table(["bound", "formula", "value"], rows)
        body = (tbl + f'<p>Any e* ∈ [{fmt(s["e_min"])}, {fmt(s["e_max"])}] supports a '
                f'separating equilibrium; midpoint e* = <b>{fmt(s["e_star"])}</b>.</p>')
        html.show(html.card(title or self.name, body))


@dataclass
class VCGAssignment(Mechanism):
    """VCG (Vickrey–Clarke–Groves) assignment of indivisible items to bidders.

    ``V[i, j]`` is bidder ``i``'s value for item ``j``. The efficient allocation
    maximizes total welfare; each winner pays the externality it imposes on the
    others. Truthful reporting is weakly dominant.
    """

    V: np.ndarray
    bidders: Optional[Sequence[str]] = None
    items: Optional[Sequence[str]] = None
    name: str = "VCG assignment"

    def __post_init__(self) -> None:
        self.V = np.asarray(self.V, dtype=float)
        n_b, n_it = self.V.shape
        self.bidders = list(self.bidders) if self.bidders else [f"B{i+1}" for i in range(n_b)]
        self.items = list(self.items) if self.items else [f"item{j+1}" for j in range(n_it)]

    @staticmethod
    def _enumerate(V: np.ndarray) -> List[Tuple[Tuple[int, ...], float]]:
        """Every feasible assignment (item → bidder index, or −1) with welfare."""
        n_b, n_it = V.shape
        results: List[Tuple[Tuple[int, ...], float]] = []

        def assign(pos: int, used: set, cur: List[int]) -> None:
            if pos == n_it:
                w = sum(V[cur[j], j] if cur[j] >= 0 else 0.0 for j in range(n_it))
                results.append((tuple(cur), float(w)))
                return
            assign(pos + 1, used, cur + [-1])         # leave item unassigned
            for i in range(n_b):
                if i not in used:
                    assign(pos + 1, used | {i}, cur + [i])

        assign(0, set(), [])
        return results

    def _efficient(self, V: np.ndarray) -> Tuple[Tuple[int, ...], float]:
        return max(self._enumerate(V), key=lambda x: x[1])

    def solve(self) -> Dict[str, Any]:
        V = self.V
        n_b = V.shape[0]
        a_star, w_star = self._efficient(V)
        alloc = [-1] * n_b                # bidder → item
        for j, i in enumerate(a_star):
            if i >= 0:
                alloc[i] = j
        payments = np.zeros(n_b)
        for i in range(n_b):
            own = V[i, alloc[i]] if alloc[i] >= 0 else 0.0
            others_with_i = w_star - own
            _, w_without_i = self._efficient(np.delete(V, i, axis=0))
            payments[i] = w_without_i - others_with_i
        utilities = np.array([
            (V[i, alloc[i]] if alloc[i] >= 0 else 0.0) - payments[i]
            for i in range(n_b)
        ])
        return {"assignment": a_star, "welfare": w_star, "alloc": alloc,
                "payments": payments, "utilities": utilities}

    def summary(self, title: Optional[str] = None) -> None:
        s = self.solve()
        rows = []
        for i in range(self.V.shape[0]):
            j = s["alloc"][i]
            won = self.items[j] if j >= 0 else "—"
            rows.append([won, fmt_money(s["payments"][i]), fmt_money(s["utilities"][i])])
        tbl = html.table(["item won", "VCG payment", "utility"], rows,
                         row_headers=self.bidders)
        body = (tbl + f'<p><b>Efficient welfare:</b> {fmt(s["welfare"])}. '
                "Each winner pays the externality it imposes; truthful bidding is "
                "weakly dominant.</p>")
        html.show(html.card(title or self.name, body))


@dataclass
class PublicProject(Mechanism):
    """Clarke pivot mechanism for a binary public good.

    Build iff the sum of reported values covers the cost. A *pivotal* citizen
    (one whose presence flips the decision) pays the externality it imposes;
    everyone else pays 0. Truthful, efficient, individually rational — but
    generally runs a budget deficit (the impossibility trilemma).
    """

    values: Sequence[float]
    cost: float
    citizens: Optional[Sequence[str]] = None
    name: str = "Public project (Clarke pivot)"

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        self.citizens = list(self.citizens) if self.citizens else \
            [f"C{i+1}" for i in range(len(self.values))]

    def solve(self) -> Dict[str, Any]:
        v = self.values
        total = float(v.sum())
        build = total >= self.cost
        pay = np.zeros_like(v)
        if build:
            for i in range(len(v)):
                others = total - v[i]
                if others < self.cost:               # i is pivotal
                    pay[i] = max(0.0, self.cost - others)
        total_pay = float(pay.sum())
        return {"build": build, "total_value": total, "payments": pay,
                "total_payment": total_pay,
                "deficit": float(self.cost - total_pay) if build else 0.0}

    def summary(self, title: Optional[str] = None) -> None:
        s = self.solve()
        rows = [[fmt(v), fmt_money(p)] for v, p in zip(self.values, s["payments"])]
        tbl = html.table(["value", "pivot payment"], rows, row_headers=self.citizens)
        verdict = "BUILD" if s["build"] else "DO NOT build"
        body = (tbl + f'<p><b>Decision:</b> {verdict} '
                f'(Σv = {fmt(s["total_value"])} vs cost {fmt_money(self.cost)}).</p>'
                + (f'<p>Total collected {fmt_money(s["total_payment"])} → '
                   f'budget deficit {fmt_money(s["deficit"])}.</p>' if s["build"] else ""))
        html.show(html.card(title or self.name, body))


@dataclass
class Procurement(Mechanism):
    """Reverse (Vickrey) procurement auction with discrete private costs.

    The firm reporting the lowest cost wins and is paid the second-lowest
    reported cost. Truthful reporting is weakly dominant; the expected payment
    is the second-order statistic of ``n`` i.i.d. cost draws.
    """

    costs: Sequence[float]
    probs: Sequence[float]
    n: int = 2
    name: str = "Procurement (reverse Vickrey)"

    def __post_init__(self) -> None:
        self.costs = np.asarray(self.costs, dtype=float)
        self.probs = np.asarray(self.probs, dtype=float)
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("cost-type probabilities must sum to 1")
        if self.n < 2:
            raise ValueError("need at least 2 firms")

    def solve(self, max_exact: int = 5000) -> Dict[str, Any]:
        costs, probs, n = self.costs, self.probs, self.n
        n_types = len(costs)
        if n_types ** n <= max_exact:
            e_pay = e_win = 0.0
            for profile in product(range(n_types), repeat=n):
                p = float(np.prod(probs[list(profile)]))
                c_sorted = sorted(costs[list(profile)])
                e_pay += p * c_sorted[1]
                e_win += p * c_sorted[0]
        else:
            rng = np.random.default_rng(0)
            draws = rng.choice(costs, size=(20_000, n), p=probs)
            srt = np.sort(draws, axis=1)
            e_pay = float(srt[:, 1].mean())
            e_win = float(srt[:, 0].mean())
        return {"expected_payment": e_pay, "expected_winner_cost": e_win,
                "expected_rent": e_pay - e_win}

    def summary(self, title: Optional[str] = None) -> None:
        s = self.solve()
        rows = [[fmt_money(c), fmt_prob(p)] for c, p in zip(self.costs, self.probs)]
        tbl = html.table(["cost type", "probability"], rows)
        body = (tbl + f"<p>{self.n} firms. <b>Expected payment</b> "
                f"{fmt_money(s['expected_payment'])} (2nd-lowest cost); winner's "
                f"expected cost {fmt_money(s['expected_winner_cost'])} → "
                f"expected rent {fmt_money(s['expected_rent'])}.</p>")
        html.show(html.card(title or self.name, body))
