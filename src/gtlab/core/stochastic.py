"""Zero-sum stochastic (Markov) games solved by value iteration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import fmt, fmt_prob_vec, fmt_vec, html, plots


@dataclass
class StochasticGame:
    """A finite zero-sum stochastic game.

    Parameters
    ----------
    r      : reward tensor, shape (S, A, B) - row player's stage reward.
    P      : transition tensor, shape (S, A, B, S) - next-state distribution.
    gamma  : discount factor in [0, 1).
    """

    r: np.ndarray
    P: np.ndarray
    gamma: float = 0.9
    state_names: Optional[Sequence[str]] = None
    row_actions: Optional[Sequence[str]] = None
    col_actions: Optional[Sequence[str]] = None
    name: str = "Stochastic game"

    def __post_init__(self) -> None:
        self.r = np.asarray(self.r, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        nS, nA, nB = self.r.shape
        if self.P.shape != (nS, nA, nB, nS):
            raise ValueError(f"P shape {self.P.shape} != {(nS, nA, nB, nS)}")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        self.nS, self.nA, self.nB = nS, nA, nB
        self.state_names = list(self.state_names) if self.state_names else [f"s{i}" for i in range(nS)]
        self.row_actions = list(self.row_actions) if self.row_actions else [f"a{i}" for i in range(nA)]
        self.col_actions = list(self.col_actions) if self.col_actions else [f"b{j}" for j in range(nB)]

    @cached_method
    def solve(self, tol: float = 1e-8, max_iter: int = 500) -> dict:
        """Run value iteration; cache and return the result dict (keyed by args)."""
        return solvers.value_iteration(
            self.r, self.P, self.gamma, tol=tol, max_iter=max_iter)

    # ── display ──────────────────────────────────────────────────────────────
    def summary(self, title: Optional[str] = None) -> None:
        body = html.note(f"{self.nS} states, {self.nA}×{self.nB} stage actions, "
                         f"γ = {fmt(self.gamma)}")
        html.show(html.card(title or self.name, body))

    def policy_summary(self, title: Optional[str] = None) -> None:
        res = self.solve()
        rows = []
        for s in range(self.nS):
            pol = res["policies"][s]
            rows.append([fmt(res["V_star"][s]), fmt_prob_vec(pol["p"]),
                         fmt_prob_vec(pol["q"])])
        tbl = html.table(["V*(s)", "row mix p*", "col mix q*"], rows,
                         row_headers=self.state_names)
        html.show(html.card(title or f"{self.name} - optimal policy",
                            tbl + html.note(f"converged in {res['n_iter']} iterations")))

    def explain(self, title: Optional[str] = None) -> None:
        """Walkthrough of value iteration via the Shapley operator."""
        res = self.solve()
        items = [
            "<b>Step 1 - Stage games.</b> At a value guess V, each state s defines a "
            "matrix game M<sub>s</sub>(V) = r(s) + γ·E<sub>s'</sub>[V].",
            "<b>Step 2 - Shapley operator.</b> Solve each stage game's minimax value; "
            "collecting them gives (TV)(s). T is a γ-contraction, so it has a unique "
            "fixed point.",
            f"<b>Step 3 - Value iteration.</b> Iterating V ← TV converged in "
            f"{res['n_iter']} steps to V* = {fmt_vec(res['V_star'])}.",
            "<b>Step 4 - Stationary policy.</b> The minimax mixes of the stage games "
            "at V* form the optimal stationary policy (see <code>policy_summary</code>).",
        ]
        html.show(html.card(title or f"{self.name} - value iteration",
                            html.steps(items)))

    def plot_convergence(self, title: Optional[str] = None):
        res = self.solve()
        residuals = np.array(res["residuals"])
        return plots.convergence({"Bellman residual": residuals}, logy=True,
                                 title=title or f"{self.name} - value iteration",
                                 ylabel="‖TV − V‖∞")

    def __repr__(self) -> str:
        return f"StochasticGame({self.name!r}, S={self.nS})"
