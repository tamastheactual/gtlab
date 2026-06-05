"""Correlated equilibrium, coarse correlated equilibrium, and no-regret learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import fmt, html, plots


@dataclass
class CorrelatedGame:
    """Two-player general-sum game analyzed through the CE / CCE lens."""

    A: np.ndarray
    B: np.ndarray
    row_actions: Optional[Sequence[str]] = None
    col_actions: Optional[Sequence[str]] = None
    name: str = "Game"

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.shape != self.B.shape:
            raise ValueError("A and B must have the same shape")
        m, n = self.A.shape
        self.row_actions = list(self.row_actions) if self.row_actions else [f"r{i}" for i in range(m)]
        self.col_actions = list(self.col_actions) if self.col_actions else [f"c{j}" for j in range(n)]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.A.shape

    # ── analysis (memoized; keyed by arguments) ──────────────────────────────
    @cached_method
    def find_ce(self, maximize: str = "welfare"):
        return solvers.find_ce(self.A, self.B, maximize)

    @cached_method
    def find_cce(self, maximize: str = "welfare"):
        return solvers.find_cce(self.A, self.B, maximize)

    @cached_method
    def nash(self):
        return solvers.all_equilibria(self.A, self.B)

    @cached_method
    def hedge(self, T: int = 2000, seed: int = 0):
        return solvers.hedge(self.A, self.B, T=T, seed=seed)

    # ── display ──────────────────────────────────────────────────────────────
    def _mu_table(self, mu: np.ndarray) -> str:
        m, n = self.shape
        rows = [[fmt(mu[i, j]) for j in range(n)] for i in range(m)]
        return html.table(self.col_actions, rows, row_headers=self.row_actions)

    def summary(self, title: Optional[str] = None) -> None:
        ce = self.find_ce()
        cce = self.find_cce()
        body = ""
        if ce:
            body += "<b>CE (welfare-max)</b>" + self._mu_table(ce["mu"]) \
                    + html.note(f"welfare = {fmt(ce['welfare'])}")
        if cce:
            body += "<b>CCE (welfare-max)</b>" + self._mu_table(cce["mu"]) \
                    + html.note(f"welfare = {fmt(cce['welfare'])}")
        html.show(html.card(title or self.name, body))

    def compare_equilibria(self, title: Optional[str] = None) -> None:
        """Compare NE, CE, and CCE welfare."""
        ce = self.find_ce()
        cce = self.find_cce()
        rows = []
        nash = self.nash()
        if nash:
            p, q = nash[0]
            eu_r = float(p @ self.A @ q)
            eu_c = float(p @ self.B @ q)
            rows.append(["Nash", fmt(eu_r), fmt(eu_c), fmt(eu_r + eu_c)])
        if ce:
            rows.append(["CE", fmt(ce["eu_row"]), fmt(ce["eu_col"]), fmt(ce["welfare"])])
        if cce:
            rows.append(["CCE", fmt(cce["eu_row"]), fmt(cce["eu_col"]), fmt(cce["welfare"])])
        tbl = html.table(["concept", "E[row]", "E[col]", "welfare"], rows)
        body = tbl + html.note("welfare ordering: NE ≤ CE ≤ CCE (each relaxes the "
                               "deviation constraints of the previous).")
        html.show(html.card(title or f"{self.name} — equilibrium concepts", body))

    def explain(self, title: Optional[str] = None) -> None:
        """Walkthrough: NE ⊆ CE ⊆ CCE, plus the no-regret learning connection."""
        items = [
            "<b>Step 1 — Nash equilibrium.</b> Players randomize independently; "
            "their product distribution must be a mutual best response.",
            "<b>Step 2 — Correlated equilibrium (CE).</b> A trusted device draws a "
            "joint action and privately recommends each player's part; obeying must "
            "be optimal given the conditional belief it induces.",
            "<b>Step 3 — Coarse correlated equilibrium (CCE).</b> Players commit "
            "before seeing the recommendation; only ex-ante deviations are checked, "
            "so CCE ⊇ CE ⊇ NE.",
            "<b>Step 4 — Learning.</b> If both players run a no-regret algorithm "
            "(see <code>plot_regret</code>), the empirical play converges to the CCE "
            "set (Hannan's theorem).",
        ]
        html.show(html.card(title or f"{self.name} — equilibrium concepts",
                            html.steps(items)))

    def plot_regret(self, T: int = 2000, seed: int = 0, title: Optional[str] = None):
        """Average regret of both players under Hedge → 0 (Hannan)."""
        res = self.hedge(T=T, seed=seed)
        return plots.convergence(
            {"Row avg regret": res["avg_regret_row"],
             "Col avg regret": res["avg_regret_col"]},
            target=0.0, title=title or f"{self.name} — no-regret learning",
            ylabel="average regret")

    def __repr__(self) -> str:
        return f"CorrelatedGame({self.name!r}, shape={self.shape})"
