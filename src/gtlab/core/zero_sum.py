"""Two-player zero-sum games: minimax value via linear programming."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import fmt, fmt_prob_vec, html, plots


@dataclass
class ZeroSumGame:
    """A zero-sum game defined by the row player's payoff matrix ``A``.

    The column player receives ``-A``. Value and optimal mixed strategies are
    obtained by linear programming.
    """

    A: np.ndarray
    row_actions: Optional[Sequence[str]] = None
    col_actions: Optional[Sequence[str]] = None
    name: str = "Zero-sum game"

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        if not np.all(np.isfinite(self.A)):
            raise ValueError("payoff matrix contains non-finite values")
        m, n = self.A.shape
        self.row_actions = list(self.row_actions) if self.row_actions else [f"r{i}" for i in range(m)]
        self.col_actions = list(self.col_actions) if self.col_actions else [f"c{j}" for j in range(n)]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.A.shape

    @cached_method
    def solve_value(self) -> dict:
        """Return ``{"p", "q", "value"}`` - optimal mixes and game value."""
        return solvers.solve_zero_sum(self.A)

    @cached_method
    def pure_saddle_points(self) -> List[Tuple[int, int]]:
        """Pure saddle points (maximin row meets minimax column)."""
        return solvers.pure_nash(self.A, -self.A)

    def exploitability(self, q: np.ndarray) -> float:
        return solvers.exploitability(self.A, q, self.solve_value()["value"])

    def verify(self) -> dict:
        """Complementary-slackness certificate for the LP solution."""
        s = self.solve_value()
        return solvers.complementary_slackness(self.A, s["p"], s["q"], s["value"])

    # ── display ──────────────────────────────────────────────────────────────
    def _matrix_html(self) -> str:
        m, n = self.shape
        rows = [[fmt(self.A[i, j]) for j in range(n)] for i in range(m)]
        return html.table(self.col_actions, rows, row_headers=self.row_actions)

    def _solution_html(self) -> str:
        s = self.solve_value()
        return html.kv([
            ("Game value", f"v = {fmt(s['value'])}"),
            ('<span class="gt-row">Row</span> plays', fmt_prob_vec(s["p"])),
            ('<span class="gt-col">Column</span> plays', fmt_prob_vec(s["q"])),
        ])

    def summary(self, title: Optional[str] = None) -> None:
        html.show(html.card(title or self.name, self._matrix_html()))

    def solve(self, title: Optional[str] = None) -> None:
        """Show the value and optimal mixed strategies."""
        html.show(html.card(title or f"{self.name} - minimax solution",
                            self._solution_html()))

    def explain(self, title: Optional[str] = None) -> None:
        saddles = self.pure_saddle_points()
        items = []
        if saddles:
            cells = ", ".join(f"({self.row_actions[i]}, {self.col_actions[j]})"
                              for i, j in saddles)
            items.append(f"<b>Step 1 - Pure saddle point(s):</b> {cells}. "
                         "Here maximin equals minimax already in pure strategies.")
        else:
            items.append("<b>Step 1 - No pure saddle point</b> - maximin &lt; minimax, "
                         "so the value is achieved only in mixed strategies.")
        items.append("<b>Step 2 - Minimax value.</b> Linear programming gives the "
                     "value v and the optimal mixes shown above.")
        cert = self.verify()
        items.append("<b>Step 3 - Complementary slackness:</b> "
                     + ("verified ✓ - every action in the support earns exactly v."
                        if cert["valid"] else "not satisfied."))
        body = self._solution_html() + html.steps(items)
        html.show(html.card(title or f"{self.name} - explanation", body))

    def plot_convergence(self, T: int = 500, seed: int = 0, title: Optional[str] = None):
        """Show empirical average payoff under repeated optimal play converging to v."""
        s = self.solve_value()
        rng = np.random.default_rng(seed)
        m, n = self.shape
        rewards = []
        for _ in range(T):
            a = rng.choice(m, p=s["p"])
            b = rng.choice(n, p=s["q"])
            rewards.append(self.A[a, b])
        cesaro = np.cumsum(rewards) / np.arange(1, T + 1)
        return plots.convergence({"Cesàro average": cesaro}, target=s["value"],
                                 title=title or f"{self.name} - convergence to value",
                                 ylabel="average payoff")

    @staticmethod
    def compare(*games: "ZeroSumGame") -> None:
        html.compare_via(games, "solve")

    def __repr__(self) -> str:
        return f"ZeroSumGame({self.name!r}, shape={self.shape})"
