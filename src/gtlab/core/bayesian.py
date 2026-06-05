"""Bayesian games and mechanism design.

The original notebook dispatched ~8 mechanisms through one dataclass. Here each
mechanism is a small focused class with the closed-form results from the
lecture, sharing the display layer. Add a mechanism by subclassing
:class:`Mechanism` and implementing ``solve``/``summary``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..viz import C, fmt, fmt_money, fmt_prob, html, rc_context
from ..solvers.bayesian_extra import (
    auction_expected_revenue,
    entry_expected_payoff,
    entry_threshold,
    posted_price_threshold,
    public_project_solve,
    signaling_interval,
    simulate_fpa,
    simulate_procurement,
    simulate_spa,
    vcg_solve,
    vcg_utility_of_report,
)


def _df(columns: Dict[str, Any]):
    """Build a pandas DataFrame, importing pandas lazily.

    Returns the raw dict if pandas is unavailable so simulate() never hard-fails
    outside a notebook.
    """
    try:
        import pandas as pd  # type: ignore
        return pd.DataFrame(columns)
    except Exception:  # pragma: no cover - pandas always present in practice
        return columns


class Mechanism:
    """Base class for a mechanism-design example."""

    name: str = "Mechanism"

    def solve(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def summary(self, title: Optional[str] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def explain(self, title: Optional[str] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    # ── Static utilities (shared across all mechanisms) ───────────────────
    @staticmethod
    def compare(*mechs: "Mechanism", method: str = "summary",
                title: Optional[str] = None) -> None:
        """Side-by-side display of several mechanisms via ``method`` (default summary)."""
        items = []
        for m in mechs:
            with html.capture() as sink:
                getattr(m, method)()
            items.append((getattr(m, "name", type(m).__name__), "".join(sink.parts)))
        if title is not None:
            html.show(html.card(title, ""))
        html.compare(items)

    @staticmethod
    def sweep(factory: Callable[[float], "Mechanism"],
              param_range: Sequence[float],
              metric: str,
              param_name: str = "parameter",
              title: Optional[str] = None,
              figsize: Tuple[float, float] = (6.5, 4.0)):
        """Plot one metric from ``.solve()`` as a parameter is swept.

        ``factory(v)`` builds a mechanism for parameter value ``v``; ``metric``
        is a key into the dict returned by that mechanism's ``.solve()``.
        Returns ``(fig, ax)``.
        """
        xs = [float(v) for v in param_range]
        ys = []
        for v in xs:
            sol = factory(v).solve()
            ys.append(float(sol.get(metric, np.nan)) if isinstance(sol, dict)
                      else np.nan)
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(xs, ys, color=C["p1"], lw=2.2, marker="o", markersize=4,
                    label=metric)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.set_title(title or f"Comparative statics in {param_name}")
            ax.legend(loc="best")
        return fig, ax


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
                f'-> E[revenue] {fmt_money(sol["optimal_revenue"])}</p>')
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        sol = self.solve()
        thr = posted_price_threshold(self.values)
        rev_items = []
        for v in self.values:
            r = self.expected_revenue(float(v))
            tag = " (optimum)" if abs(float(v) - sol["optimal_price"]) < 1e-9 else ""
            rev_items.append(f"p = {fmt_money(v)}: E[R] = {fmt_money(r)}{tag}")
        steps = [
            "<b>Buyer's rule.</b> The buyer knows her own value v and accepts "
            "the posted price p iff p &le; v (individual rationality).",
            "<b>Seller's revenue.</b> E[R | p] = &Sigma;<sub>t</sub> "
            "Pr(v = v<sub>t</sub>) &middot; p &middot; 1{p &le; v<sub>t</sub>}.",
            "<b>Tradeoff.</b> A low price sells to everyone but leaves money on "
            "the table; a high price extracts more from high types but loses the "
            "low types. The optimum depends on the prior.",
            "<b>Revenue at each candidate price.</b><br>" + "<br>".join(rev_items),
        ]
        body = html.steps(steps)
        body += (f'<p><b>Optimal posted price</b> {fmt_money(sol["optimal_price"])} '
                 f'with E[R*] = {fmt_money(sol["optimal_revenue"])}.</p>')
        if thr is not None:
            body += html.note(
                f"Two-type threshold belief for switching to the high price: "
                f"Pr(high) = {fmt_prob(thr)}.")
        html.show(html.card((title or self.name) + " - walkthrough", body))

    def simulate(self, n_trials: int = 10_000, seed: int = 0,
                 price: Optional[float] = None):
        """Monte-Carlo draws of buyer values; returns a DataFrame (and is reproducible)."""
        rng = np.random.default_rng(seed)
        if price is None:
            price = self.solve()["optimal_price"]
        idx = rng.choice(len(self.values), size=n_trials, p=self.probs)
        v = self.values[idx]
        accept = (price <= v).astype(int)
        revenue = np.where(accept == 1, price, 0.0)
        return _df({
            "trial": np.arange(1, n_trials + 1),
            "buyer_value": v, "price": float(price),
            "accepted": accept, "revenue": revenue,
        })

    def plot_revenue_curve(self, title: Optional[str] = None,
                           figsize: Tuple[float, float] = (7.0, 4.2)):
        """Seller's expected revenue as a continuous function of the posted price."""
        vals = self.values
        cand = sorted(set(float(v) for v in vals))
        cand_R = [self.expected_revenue(p) for p in cand]
        p_star = cand[int(np.argmax(cand_R))]
        R_max = max(cand_R) if cand_R else 1.0
        x_lo, x_hi = min(min(cand), float(vals.min())), max(max(cand), float(vals.max()))
        pad = 0.04 * max(x_hi - x_lo, 1.0)
        grid = np.linspace(x_lo - pad, x_hi + pad, 400)
        R = np.array([self.expected_revenue(p) for p in grid])
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(grid, R, color=C["p1"], lw=2.2, label="E[R | p]")
            ax.fill_between(grid, R, alpha=0.08, color=C["p1"], linewidth=0)
            ax.scatter(cand, cand_R, color=C["ne"], zorder=5, s=44,
                       edgecolor="white", linewidths=0.8)
            ax.set_ylim(0, R_max * 1.25)
            ax.axvline(p_star, ls=":", lw=1.1, color=C["ne"], alpha=0.7)
            ax.text(0.5, 0.96, f"optimal: p* = {fmt_money(p_star)}",
                    transform=ax.transAxes, ha="center", va="top",
                    color=C["ne"], fontweight="bold")
            ax.set_xlabel("Posted price p")
            ax.set_ylabel("Expected revenue E[R | p]")
            ax.set_title(title or f"{self.name}: seller's expected revenue")
            ax.legend(loc="lower right")
        return fig, ax


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
                f"{fmt(sol['shading_factor'])}*(v - {fmt(self.lo)})</p>"
                f"<p><b>Expected revenue:</b> {fmt(sol['expected_revenue'])}</p>")
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        n = self.n_bidders
        steps = [
            "<b>The shading tradeoff.</b> Bidding your true value v and winning "
            "earns 0; bidding b &lt; v earns v - b on a win but lowers the win "
            "probability. The equilibrium balances these.",
            "<b>Symmetric BNE derivation.</b> Assume rivals bid b(v) = &beta;v. A "
            "bidder of value v choosing bid b wins with probability "
            "(b/&beta;)<sup>n-1</sup>, so she maximises (v - b)(b/&beta;)<sup>n-1</sup>. "
            "The first-order condition gives b = (n-1)/n &middot; v.",
            f"<b>Comparative statics.</b> With n = {n} the shading factor is "
            f"(n-1)/n = {fmt((n - 1) / n)}. As n grows the bid approaches the "
            "true value.",
            "<b>Revenue equivalence.</b> The BNE is symmetric and strictly "
            "increasing, and yields the same expected revenue as the "
            "second-price auction.",
        ]
        html.show(html.card((title or self.name) + " - BNE derivation",
                            html.steps(steps)))

    def simulate(self, n_trials: int = 10_000, seed: int = 0,
                 strategy: str = "bne"):
        """Monte-Carlo the auction under ``strategy`` ('bne' or 'truthful')."""
        rng = np.random.default_rng(seed)
        cols = simulate_fpa(self.n_bidders, self.lo, self.hi, n_trials, rng,
                            strategy=strategy)
        return _df({"trial": np.arange(1, n_trials + 1), **cols})

    def plot_bid_function(self, title: Optional[str] = None, compare_n=None,
                          figsize: Tuple[float, float] = (7.0, 4.5)):
        """BNE bid b(v) vs the 45-degree truthful line, optionally for several n."""
        v = np.linspace(self.lo, self.hi, 200)
        compare_n = compare_n or [self.n_bidders]
        palette = [C["p1"], C["p2"], C["accent"], C["ce"], C["chance"]]
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(v, v, color=C["muted"], ls="--", lw=1.3, label="truthful: b = v")
            for k, nn in enumerate(compare_n):
                b = self.lo + (nn - 1) / nn * (v - self.lo)
                ax.plot(v, b, color=palette[k % len(palette)], lw=2.4,
                        label=f"FPA BNE, n = {nn}")
            ax.set_xlabel("Private value v")
            ax.set_ylabel("Bid b(v)")
            ax.set_title(title or f"{self.name}: bid shading")
            ax.legend(loc="best")
        return fig, ax

    def plot_revenue_vs_n(self, title: Optional[str] = None, n_range=None,
                          n_trials: int = 3000, seed: int = 0,
                          figsize: Tuple[float, float] = (7.5, 4.5)):
        """Closed-form FPA revenue vs simulated SPA revenue across n (revenue equivalence)."""
        n_range = list(n_range) if n_range is not None else list(range(2, 11))
        rng = np.random.default_rng(seed)
        fpa_R, spa_R = [], []
        for n in n_range:
            fpa_R.append(auction_expected_revenue(n, self.lo, self.hi))
            V = rng.uniform(self.lo, self.hi, size=(n_trials, n))
            spa_R.append(float(np.sort(V, axis=1)[:, -2].mean()))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(n_range, fpa_R, color=C["p1"], lw=2.4, marker="o",
                    label="FPA E[revenue] (closed-form)")
            ax.plot(n_range, spa_R, color=C["p2"], lw=2.4, marker="s",
                    label="SPA E[revenue] (simulated)")
            ax.set_xlabel("Number of bidders n")
            ax.set_ylabel("Expected revenue")
            ax.set_title(title or "Revenue equivalence: FPA vs SPA")
            ax.legend(loc="best")
        return fig, ax


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
                "(equals the first-price auction - revenue equivalence).</p>")
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        steps = [
            "<b>Decompose by rivals' highest bid.</b> Let m be the highest rival "
            "bid. A bidder with value v who bids b wins (paying m) iff b &gt; m, "
            "earning v - m; otherwise she earns 0.",
            "<b>Overbidding (b &gt; v) never helps.</b> If m &lt; v truthful "
            "bidding also wins and pays m. If m &gt; v overbidding can make you "
            "win and pay more than your value (v - m &lt; 0), strictly worse than "
            "the truthful loss of 0.",
            "<b>Underbidding (b &lt; v) never helps.</b> It only risks losing a "
            "prize you would have won profitably at the truthful bid.",
            "<b>Conclusion.</b> Bidding b = v is weakly dominant - no beliefs "
            "about rivals or the prior are required.",
        ]
        html.show(html.card((title or self.name) + " - why truth is dominant",
                            html.steps(steps)))

    def simulate(self, n_trials: int = 10_000, seed: int = 0,
                 strategy: str = "truthful"):
        """Monte-Carlo the auction under ``strategy`` (truthful / overbid / underbid)."""
        rng = np.random.default_rng(seed)
        cols = simulate_spa(self.n_bidders, self.lo, self.hi, n_trials, rng,
                            strategy=strategy)
        return _df({"trial": np.arange(1, n_trials + 1), **cols})

    def plot_bid_function(self, title: Optional[str] = None,
                          figsize: Tuple[float, float] = (7.0, 4.5)):
        """The weakly-dominant truthful bid b(v) = v."""
        v = np.linspace(self.lo, self.hi, 200)
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(v, v, color=C["ne"], lw=2.4,
                    label="SPA (weakly dominant): b = v")
            ax.set_xlabel("Private value v")
            ax.set_ylabel("Bid b(v)")
            ax.set_title(title or f"{self.name}: truthful bidding")
            ax.legend(loc="best")
        return fig, ax

    def plot_revenue_vs_n(self, title: Optional[str] = None, n_range=None,
                          figsize: Tuple[float, float] = (7.5, 4.5)):
        """Closed-form expected revenue (= E[2nd-highest value]) across n."""
        n_range = list(n_range) if n_range is not None else list(range(2, 11))
        R = [auction_expected_revenue(n, self.lo, self.hi) for n in n_range]
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(n_range, R, color=C["p2"], lw=2.4, marker="s",
                    label="SPA E[revenue]")
            ax.set_xlabel("Number of bidders n")
            ax.set_ylabel("Expected revenue")
            ax.set_title(title or f"{self.name}: revenue vs n")
            ax.legend(loc="best")
        return fig, ax

    def plot_utility_sweep(self, title: Optional[str] = None, v_me: float = 0.6,
                           n_trials: int = 3000, seed: int = 0,
                           figsize: Tuple[float, float] = (7.0, 4.4)):
        """Bidder 1's expected utility vs her own bid, rivals held truthful (IC check)."""
        rng = np.random.default_rng(seed)
        b_grid = np.linspace(self.lo, self.hi, 41)
        rivals_max = rng.uniform(self.lo, self.hi,
                                 size=(n_trials, self.n_bidders - 1)).max(axis=1)
        utils = []
        for b in b_grid:
            won = b > rivals_max
            utils.append(float(np.where(won, v_me - rivals_max, 0.0).mean()))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(b_grid, utils, color=C["p1"], lw=2.4, label="expected utility")
            ax.axvline(v_me, ls=":", lw=1.2, color=C["ne"],
                       label=f"truthful bid b = v = {fmt(v_me)}")
            ax.set_xlabel("Bidder 1's own bid b_1")
            ax.set_ylabel(f"Expected utility (value v_1 = {fmt(v_me)})")
            ax.set_title(title or f"{self.name}: utility is flat around the truthful bid")
            ax.legend(loc="best")
        return fig, ax


@dataclass
class SpenceSignaling(Mechanism):
    """Spence job-market signaling: a worker's type is private; education is a
    costly, productivity-free signal. Single-crossing (``c_high < c_low``)
    yields a non-empty separating interval of education levels."""

    w_low: float
    w_high: float
    c_low: float            # cost of education per unit for the LOW type
    c_high: float           # cost per unit for the HIGH type (c_high < c_low)
    prior_high: float = 0.5  # prior probability of the high type (used by simulate)
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
        body = (tbl + f'<p>Any e* in [{fmt(s["e_min"])}, {fmt(s["e_max"])}] supports a '
                f'separating equilibrium; midpoint e* = <b>{fmt(s["e_star"])}</b>.</p>')
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        e_min, e_max, _ = signaling_interval(self.w_low, self.w_high,
                                             self.c_low, self.c_high)
        steps = [
            "<b>Candidate separating profile.</b> The high type acquires "
            "education e*, the low type acquires 0. The employer pays w<sub>H</sub> "
            "on seeing e &ge; e*, else w<sub>L</sub>.",
            "<b>Incentive-compatibility constraints.</b> Low-type IC requires "
            "w<sub>L</sub> &ge; w<sub>H</sub> - c<sub>L</sub>e*, i.e. "
            "e* &ge; (w<sub>H</sub> - w<sub>L</sub>)/c<sub>L</sub>. High-type IC "
            "requires e* &le; (w<sub>H</sub> - w<sub>L</sub>)/c<sub>H</sub>.",
            f"<b>Single-crossing.</b> c<sub>H</sub> &lt; c<sub>L</sub> guarantees "
            f"a non-empty interval [e_min, e_max] = "
            f"[{fmt(e_min)}, {fmt(e_max)}] of feasible separating signals.",
            "<b>Pooling and screening.</b> Pooling equilibria also exist and are "
            "sensitive to off-path beliefs; screening flips the story - the "
            "employer posts a menu of wage-education contracts and lets workers "
            "self-select.",
        ]
        html.show(html.card((title or self.name) + " - separating equilibrium",
                            html.steps(steps)))

    def simulate(self, n_trials: int = 10_000, seed: int = 0,
                 e_star: Optional[float] = None):
        """Draw types from ``prior_high`` and play the separating equilibrium."""
        rng = np.random.default_rng(seed)
        if e_star is None:
            e_star = self.solve()["e_star"]
        types = rng.choice(["High", "Low"], size=n_trials,
                           p=[self.prior_high, 1 - self.prior_high])
        education = np.where(types == "High", e_star, 0.0)
        wage = np.where(education >= e_star - 1e-9, self.w_high, self.w_low)
        cost = np.where(types == "High", self.c_high, self.c_low) * education
        return _df({"trial": np.arange(1, n_trials + 1), "type": types,
                    "education": education, "wage": wage,
                    "signal_cost": cost, "net_utility": wage - cost})

    def plot_signaling_curves(self, title: Optional[str] = None, e_range=None,
                              figsize: Tuple[float, float] = (7.5, 4.6)):
        """Net-utility curves for both types vs the separating education band."""
        e_min, e_max, _ = signaling_interval(self.w_low, self.w_high,
                                             self.c_low, self.c_high)
        if e_range is None:
            e_range = np.linspace(0, max(2 * e_max, 1.0), 200)
        e = np.asarray(e_range, dtype=float)
        u_no = np.full_like(e, self.w_low)
        u_low_mimic = self.w_high - self.c_low * e
        u_high = self.w_high - self.c_high * e
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(e, u_no, color=C["muted"], lw=1.6, ls="--",
                    label=f"no signal: w_L = {fmt_money(self.w_low)}")
            ax.plot(e, u_low_mimic, color=C["p1"], lw=2.3,
                    label="low type if mimicking: w_H - c_L e")
            ax.plot(e, u_high, color=C["p2"], lw=2.3,
                    label="high type: w_H - c_H e")
            ax.axvspan(e_min, e_max, color=C["ne"], alpha=0.10)
            ax.axvline(e_min, ls=":", lw=0.9, color=C["ne"])
            ax.axvline(e_max, ls=":", lw=0.9, color=C["ne"])
            ax.text(0.5 * (e_min + e_max), 1.0, "separating\ninterval",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=C["ne"], fontweight="bold")
            ax.set_xlabel("Education level e")
            ax.set_ylabel("Net utility")
            ax.set_title(title or f"{self.name}: signaling utility curves")
            ax.legend(loc="lower left")
        return fig, ax


@dataclass
class VCGAssignment(Mechanism):
    """VCG (Vickrey-Clarke-Groves) assignment of indivisible items to bidders.

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
            won = self.items[j] if j >= 0 else "-"
            rows.append([won, fmt_money(s["payments"][i]), fmt_money(s["utilities"][i])])
        tbl = html.table(["item won", "VCG payment", "utility"], rows,
                         row_headers=self.bidders)
        body = (tbl + f'<p><b>Efficient welfare:</b> {fmt(s["welfare"])}. '
                "Each winner pays the externality it imposes; truthful bidding is "
                "weakly dominant.</p>")
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        steps = [
            "<b>Efficient assignment.</b> A* = argmax<sub>A</sub> "
            "&Sigma;<sub>j</sub> v<sub>A(j), j</sub> - pick the allocation that "
            "maximises total reported welfare.",
            "<b>VCG payment.</b> p<sub>i</sub> = W<sub>-i</sub><sup>without i</sup> "
            "- W<sub>-i</sub><sup>with i</sup>: each winner pays the welfare loss "
            "its presence imposes on the others. Non-winners pay 0.",
            "<b>Why truth is dominant.</b> A bidder's payment does not depend on "
            "her own report, only the allocation does. Misreporting can only flip "
            "the allocation to a less efficient one while still paying the "
            "externality, so truthful reporting is weakly dominant.",
            "<b>Special case.</b> Single-item VCG collapses to the second-price "
            "auction.",
        ]
        html.show(html.card((title or self.name) + " - pay your externality",
                            html.steps(steps)))

    def simulate(self, n_trials: int = 200, seed: int = 0,
                 report_noise: float = 0.0):
        """Perturb reports by Gaussian noise to show truthfulness is dominant.

        With ``report_noise = 0`` the mechanism is deterministic; positive noise
        demonstrates that misreports never raise a bidder's true-value utility.
        """
        rng = np.random.default_rng(seed)
        rows = []
        n_b = self.V.shape[0]
        for t in range(n_trials):
            eps = (rng.normal(0, report_noise, size=self.V.shape)
                   if report_noise > 0 else np.zeros_like(self.V))
            sol = vcg_solve(self.V + eps)
            alloc = sol["alloc"]
            pays = sol["payments"]
            for i in range(n_b):
                j = alloc[i]
                v_true = float(self.V[i, j]) if j >= 0 else 0.0
                rows.append({"trial": t + 1, "bidder": i + 1, "item_won": j,
                             "true_value": v_true, "payment": float(pays[i]),
                             "utility": float(v_true - pays[i])})
        return _df({k: [r[k] for r in rows] for k in rows[0]} if rows else {})

    def plot_utility_sweep(self, title: Optional[str] = None, bidder_idx: int = 0,
                           item_idx: int = 0, report_range=None,
                           figsize: Tuple[float, float] = (7.0, 4.5)):
        """A bidder's true-value utility vs her reported value (IC check)."""
        true_v = float(self.V[bidder_idx, item_idx])
        if report_range is None:
            report_range = np.linspace(0, 2 * max(float(self.V.max()),
                                                  true_v * 1.5 + 1), 31)
        u = [vcg_utility_of_report(self.V, bidder_idx, item_idx, float(r))
             for r in report_range]
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(report_range, u, color=C["p1"], lw=2.4, marker="o",
                    markersize=5, label=f"{self.bidders[bidder_idx]}'s utility")
            ax.axvline(true_v, ls=":", lw=1.2, color=C["ne"],
                       label=f"truthful report = {fmt(true_v)}")
            ax.set_xlabel(f"{self.bidders[bidder_idx]}'s reported value "
                          f"for {self.items[item_idx]}")
            ax.set_ylabel("True utility")
            ax.set_title(title or f"{self.name}: truthful report maximises utility")
            ax.legend(loc="best")
        return fig, ax

    def plot_payments(self, title: Optional[str] = None,
                      figsize: Tuple[float, float] = (7.5, 4.6)):
        """Bar chart of each bidder's value won vs VCG payment (utility annotated)."""
        s = self.solve()
        alloc = s["alloc"]
        pays = np.asarray(s["payments"])
        values_won = [float(self.V[i, alloc[i]]) if alloc[i] >= 0 else 0.0
                      for i in range(len(self.bidders))]
        x = np.arange(len(self.bidders))
        y_max = max(max(values_won, default=0.0), float(pays.max(initial=0.0)), 1.0)
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(x - 0.175, values_won, 0.35, color=C["p1"],
                   label="value of item won")
            ax.bar(x + 0.175, pays, 0.35, color=C["p2"], label="VCG payment")
            ax.set_ylim(0, y_max * 1.22)
            for i in range(len(self.bidders)):
                ax.text(i, max(values_won[i], pays[i]) + y_max * 0.04,
                        f"u = {fmt(values_won[i] - pays[i])}", ha="center",
                        va="bottom", color=C["ne"], fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(self.bidders)
            ax.set_ylabel("Amount")
            ax.set_title(title or f"{self.name}: values won vs VCG payments")
            ax.legend(loc="upper right")
        return fig, ax


@dataclass
class PublicProject(Mechanism):
    """Clarke pivot mechanism for a binary public good.

    Build iff the sum of reported values covers the cost. A *pivotal* citizen
    (one whose presence flips the decision) pays the externality it imposes;
    everyone else pays 0. Truthful, efficient, individually rational - but
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
                + (f'<p>Total collected {fmt_money(s["total_payment"])} -> '
                   f'budget deficit {fmt_money(s["deficit"])}.</p>' if s["build"] else ""))
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        steps = [
            "<b>Efficient rule.</b> Build iff total reported value covers the "
            "cost: &Sigma;<sub>i</sub> v<sub>i</sub> &ge; C.",
            "<b>Clarke pivot payment.</b> Citizen i is pivotal if the decision "
            "flips when she is removed. A pivotal citizen pays the externality "
            "p<sub>i</sub> = max(0, C - &Sigma;<sub>j != i</sub> v<sub>j</sub>); "
            "everyone else pays 0. Payments are individually rational.",
            "<b>The budget trilemma.</b> The mechanism is efficient and truthful "
            "but generally NOT budget-balanced - pivot payments usually fall short "
            "of the cost, running a deficit.",
            "<b>Impossibility.</b> The Green-Laffont / Myerson-Satterthwaite family "
            "shows no dominant-strategy mechanism can be efficient, IC, and "
            "budget-balanced at once for public goods with quasi-linear utility.",
        ]
        html.show(html.card((title or self.name) + " - pivot payments",
                            html.steps(steps)))

    def simulate(self, n_trials: int = 1000, seed: int = 0,
                 value_noise: float = 5.0):
        """Perturb reported values by Gaussian noise; track build rate and deficit."""
        rng = np.random.default_rng(seed)
        base = self.values
        rows = []
        for t in range(n_trials):
            v = np.clip(base + rng.normal(0, value_noise, size=base.shape), 0, None)
            sol = public_project_solve(v, self.cost)
            rows.append({"trial": t + 1, "total_value": sol["total_value"],
                         "build": sol["build"],
                         "total_pivot_payment": sol["total_payment"],
                         "deficit": sol["deficit"]})
        return _df({k: [r[k] for r in rows] for k in rows[0]} if rows else {})

    def plot_payments(self, title: Optional[str] = None,
                      figsize: Tuple[float, float] = (7.5, 4.6)):
        """Reported values vs Clarke pivot payments vs (non-IR) equal cost sharing."""
        s = self.solve()
        pay = np.asarray(s["payments"])
        n = len(self.citizens)
        equal = (np.full(n, self.cost / n) if s["build"] else np.zeros(n))
        x = np.arange(n)
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(x - 0.165, equal, 0.33, color=C["muted"], alpha=0.55,
                   label="equal sharing (not IR)")
            ax.bar(x + 0.165, pay, 0.33, color=C["p2"],
                   label="Clarke pivot payment")
            ax.plot(x, self.values, "o-", color=C["p1"], lw=2.0, markersize=8,
                    label="reported value")
            ax.set_xticks(x)
            ax.set_xticklabels(self.citizens)
            ax.set_ylabel("Money")
            ax.set_title(title or f"{self.name}: values vs pivot vs equal sharing")
            ax.legend(loc="best")
        return fig, ax


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
                f"expected cost {fmt_money(s['expected_winner_cost'])} -> "
                f"expected rent {fmt_money(s['expected_rent'])}.</p>")
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        steps = [
            "<b>The reverse Vickrey rule.</b> Each firm submits a sealed cost "
            "report; the lowest reporter wins the job and is paid the "
            "<b>second-lowest</b> report. Losers are paid 0.",
            "<b>Why truthful reporting is weakly dominant.</b> Same logic as the "
            "regular Vickrey auction, flipped: over-reporting risks losing a "
            "profitable job; under-reporting risks winning an unprofitable one. "
            "Truth leaves you with payment - cost &ge; 0 on a win, 0 otherwise.",
            "<b>Revelation principle.</b> Any equilibrium of any mechanism with "
            "the same allocation rule can be replicated by a direct truthful "
            "mechanism; procurement uses the VCG form directly.",
        ]
        html.show(html.card((title or self.name) + " - revelation principle",
                            html.steps(steps)))

    def simulate(self, n_trials: int = 10_000, seed: int = 0,
                 strategy: str = "truthful"):
        """Monte-Carlo the reverse auction (truthful / overstate / understate)."""
        rng = np.random.default_rng(seed)
        cols = simulate_procurement(self.costs, self.probs, self.n, n_trials, rng,
                                    strategy=strategy)
        return _df({"trial": np.arange(1, n_trials + 1), **cols})

    def plot_utility_sweep(self, title: Optional[str] = None,
                           firm_true_cost: float = 40.0, n_trials: int = 3000,
                           seed: int = 0, figsize: Tuple[float, float] = (7.0, 4.4)):
        """Firm 1's expected profit vs its reported cost, rivals truthful (IC check)."""
        rng = np.random.default_rng(seed)
        lo, hi = 0.5 * float(self.costs.min()), 1.5 * float(self.costs.max())
        r_grid = np.linspace(lo, hi, 41)
        rivals = rng.choice(self.costs, size=(n_trials, self.n - 1), p=self.probs)
        rivals_min = rivals.min(axis=1)
        utils = []
        for r in r_grid:
            won = r < rivals_min
            all_reports = np.column_stack([np.full(n_trials, r), rivals])
            second = np.sort(all_reports, axis=1)[:, 1]
            payment = np.where(won, second, 0.0)
            utils.append(float(np.where(won, payment - firm_true_cost, 0.0).mean()))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(r_grid, utils, color=C["p1"], lw=2.4, label="expected profit")
            ax.axvline(firm_true_cost, ls=":", lw=1.2, color=C["ne"],
                       label=f"truthful report r = c = {fmt_money(firm_true_cost)}")
            ax.set_xlabel("Firm 1's reported cost r_1")
            ax.set_ylabel(f"Expected profit (true cost = {fmt_money(firm_true_cost)})")
            ax.set_title(title or f"{self.name}: truthful reporting maximises profit")
            ax.legend(loc="best")
        return fig, ax

    def plot_procurement_vs_n(self, title: Optional[str] = None, n_range=None,
                              n_trials: int = 3000, seed: int = 0,
                              figsize: Tuple[float, float] = (7.5, 4.5)):
        """Expected payment and winner cost vs the number of firms (rent erosion)."""
        n_range = list(n_range) if n_range is not None else list(range(2, 9))
        rng = np.random.default_rng(seed)
        pay, win_cost = [], []
        for n in n_range:
            draws = rng.choice(self.costs, size=(n_trials, n), p=self.probs)
            srt = np.sort(draws, axis=1)
            pay.append(float(srt[:, 1].mean()))
            win_cost.append(float(srt[:, 0].mean()))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(n_range, pay, color=C["p1"], lw=2.4, marker="o",
                    label="E[payment to winner]")
            ax.plot(n_range, win_cost, color=C["p2"], lw=2.4, marker="s",
                    label="E[winner's true cost]")
            ax.fill_between(n_range, win_cost, pay, color=C["ne"], alpha=0.10,
                            label="winner's profit margin")
            ax.set_xlabel("Number of firms n")
            ax.set_ylabel("Money")
            ax.set_title(title or f"{self.name}: competition erodes the winner's rent")
            ax.legend(loc="best")
        return fig, ax


@dataclass
class EntryGame(Mechanism):
    """Bayesian entry deterrence: the incumbent's type (weak/strong) is private.

    The entrant cannot observe the incumbent's type. She enters iff her expected
    payoff over types exceeds the stay-out payoff. The incumbent plays her
    type-contingent (weakly dominant) action. ``payoff_weak`` /
    ``payoff_strong`` are the entrant's payoff from entering against each type;
    the second element of each pair is the incumbent's own payoff (display only).
    """

    payoff_weak: Tuple[float, float]
    payoff_strong: Tuple[float, float]
    prior_strong: float
    stay_out: float = 0.0
    entrant_name: str = "Entrant"
    incumbent_name: str = "Incumbent"
    name: str = "Entry game"

    def __post_init__(self) -> None:
        self.payoff_weak = (float(self.payoff_weak[0]), float(self.payoff_weak[1]))
        self.payoff_strong = (float(self.payoff_strong[0]), float(self.payoff_strong[1]))
        q = float(self.prior_strong)
        if not 0.0 <= q <= 1.0:
            raise ValueError("prior_strong must be in [0, 1]")
        self.prior_strong = q

    def solve(self) -> Dict[str, Any]:
        pw, ps = self.payoff_weak[0], self.payoff_strong[0]
        e_enter = entry_expected_payoff(pw, ps, self.prior_strong)
        e_out = self.stay_out
        best = ("Enter" if e_enter > e_out else
                "Stay out" if e_enter < e_out else "Indifferent")
        q_star = entry_threshold(pw, ps, e_out)
        return {"expected_payoff_enter": e_enter, "expected_payoff_out": e_out,
                "best_response": best, "q_star": q_star}

    def summary(self, title: Optional[str] = None) -> None:
        pw, ps = self.payoff_weak, self.payoff_strong
        rows = [
            [fmt(pw[0]), fmt(ps[0]), fmt(self.stay_out)],
            [fmt(pw[1]), fmt(ps[1]), "-"],
        ]
        tbl = html.table(["Weak incumbent", "Strong incumbent", "Stay out"], rows,
                         row_headers=[f"{self.entrant_name} payoff",
                                      f"{self.incumbent_name} payoff"])
        body = (tbl + f"<p>Prior: Pr(Strong) = <b>{fmt_prob(self.prior_strong)}</b>, "
                f"Pr(Weak) = <b>{fmt_prob(1 - self.prior_strong)}</b>.</p>")
        html.show(html.card(title or self.name, body))

    def explain(self, title: Optional[str] = None) -> None:
        s = self.solve()
        steps = [
            "<b>Incumbent's type-contingent strategy.</b> The weak incumbent "
            "prefers to accommodate, the strong incumbent prefers to fight; each "
            "plays her weakly dominant action given her type.",
            "<b>Entrant takes expectation over types.</b> The entrant does not see "
            "the type, so she enters iff q&middot;pi<sub>S</sub> + "
            "(1-q)&middot;pi<sub>W</sub> &gt; stay-out payoff.",
            f"<b>Bayesian Nash equilibrium.</b> The entrant best-responds "
            f"<b>in expectation</b> (E[pi_enter] = "
            f"{fmt(s['expected_payoff_enter'])} vs stay-out "
            f"{fmt(s['expected_payoff_out'])}, so she would "
            f"<b>{s['best_response']}</b>) while the incumbent best-responds "
            "pointwise per type.",
        ]
        body = html.steps(steps)
        if s["q_star"] is not None and 0 <= s["q_star"] <= 1:
            rule = ("enter if Pr(Strong) &lt; " if self.payoff_strong[0] < self.payoff_weak[0]
                    else "enter if Pr(Strong) &gt; ") + fmt_prob(s["q_star"])
            body += html.note(f"Threshold belief q* = {fmt_prob(s['q_star'])}: {rule}.")
        html.show(html.card((title or self.name) + " - BNE walkthrough", body))

    def simulate(self, n_trials: int = 10_000, seed: int = 0):
        """Draw incumbent types and play the BNE; returns a per-trial DataFrame."""
        rng = np.random.default_rng(seed)
        q = self.prior_strong
        types = rng.choice(["Strong", "Weak"], size=n_trials, p=[q, 1.0 - q])
        action = self.solve()["best_response"]
        if action == "Enter":
            payoff = np.where(types == "Strong", self.payoff_strong[0],
                              self.payoff_weak[0])
        else:
            payoff = np.full(n_trials, self.stay_out, dtype=float)
        return _df({"trial": np.arange(1, n_trials + 1), "incumbent_type": types,
                    "entrant_action": action, "entrant_payoff": payoff})

    def plot_entry_threshold(self, title: Optional[str] = None, scenarios=None,
                             figsize: Tuple[float, float] = (7.0, 4.2)):
        """Entrant's expected entry payoff vs Pr(Strong), marking the threshold q*."""
        q_grid = np.linspace(0.0, 1.0, 200)
        scenarios = scenarios or [("baseline",
                                   (self.payoff_weak[0], self.payoff_strong[0]))]
        palette = [C["p1"], C["p2"], C["accent"], C["ce"]]
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
            for k, (lbl, (a, b)) in enumerate(scenarios):
                clr = palette[k % len(palette)]
                ax.plot(q_grid, q_grid * b + (1 - q_grid) * a, color=clr, lw=2.2,
                        label=lbl)
                if a != b:
                    q_star = (self.stay_out - a) / (b - a)
                    if 0 <= q_star <= 1:
                        ax.axvline(q_star, ls=":", lw=0.9, color=clr, alpha=0.6)
                        ax.text(q_star, 0.05, f" q* = {fmt_prob(q_star)}",
                                transform=ax.get_xaxis_transform(), ha="left",
                                va="bottom", color=clr, fontweight="bold")
            ax.axhline(self.stay_out, ls="--", lw=1.0, color=C["muted"], alpha=0.8,
                       label=f"stay-out payoff = {fmt(self.stay_out)}")
            ax.set_xlabel("Pr(Strong incumbent), q")
            ax.set_ylabel("Entrant's expected payoff from entering")
            ax.set_title(title or f"{self.name}: entry threshold")
            ax.legend(loc="best")
        return fig, ax
