"""Two-player normal-form (bimatrix) games.

The class holds payoff data and delegates all computation to :mod:`gtlab.solvers`
and all rendering to :mod:`gtlab.viz`. Compare with the original notebook's
~1,800-line monolith: the math and the HTML now live elsewhere, once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import fmt, fmt_prob_vec, html, plots


@dataclass
class NormalFormGame:
    """A two-player game in normal form.

    Parameters
    ----------
    A, B          : row- and column-player payoff matrices (m x n).
    row_actions   : labels for the row player's actions (default r0, r1, ...).
    col_actions   : labels for the column player's actions.
    name          : optional display name.
    """

    A: np.ndarray
    B: np.ndarray
    row_actions: Optional[Sequence[str]] = None
    col_actions: Optional[Sequence[str]] = None
    name: str = "Game"

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.shape != self.B.shape:
            raise ValueError(f"A {self.A.shape} and B {self.B.shape} must match")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.B)):
            raise ValueError("payoff matrices contain non-finite values")
        m, n = self.A.shape
        self.row_actions = list(self.row_actions) if self.row_actions else [f"r{i}" for i in range(m)]
        self.col_actions = list(self.col_actions) if self.col_actions else [f"c{j}" for j in range(n)]

    # ── properties ──────────────────────────────────────────────────────────
    @property
    def shape(self) -> Tuple[int, int]:
        return self.A.shape

    def is_zero_sum(self, tol: float = 1e-9) -> bool:
        return bool(np.all(np.abs(self.A + self.B) < tol))

    def is_symmetric(self, tol: float = 1e-9) -> bool:
        return self.A.shape[0] == self.A.shape[1] and bool(np.all(np.abs(self.A - self.B.T) < tol))

    # ── analysis (delegates to solvers; memoized) ────────────────────────────
    @cached_method
    def best_responses(self):
        """Return ``(br_row, br_col)`` boolean masks."""
        return solvers.br_masks(self.A, self.B)

    @cached_method
    def pure_nash(self) -> List[Tuple[int, int]]:
        return solvers.pure_nash(self.A, self.B)

    @cached_method
    def equilibria(self):
        """All equilibria as ``(p, q)`` mixed-strategy pairs."""
        return solvers.all_equilibria(self.A, self.B)

    @cached_method
    def dominated(self):
        return (solvers.strictly_dominated_rows(self.A),
                solvers.strictly_dominated_cols(self.B))

    @cached_method
    def iesds(self):
        return solvers.iesds(self.A, self.B, self.row_actions, self.col_actions)

    @cached_method
    def pareto_optimal(self) -> List[Tuple[int, int]]:
        return solvers.pareto_optimal_cells(self.A, self.B)

    # ── display (delegates to viz) ───────────────────────────────────────────
    def _matrix_html(self, show_br: bool = False, show_ne: bool = False,
                     show_pareto: bool = False) -> str:
        """Build the (optionally annotated) bimatrix table as an HTML string."""
        m, n = self.shape
        br_row, br_col = self.best_responses() if show_br else (None, None)
        ne = set(self.pure_nash()) if show_ne else set()
        pareto = set(self.pareto_optimal()) if show_pareto else set()
        rows, classes = [], []
        for i in range(m):
            r, c = [], []
            for j in range(n):
                a_cls = "gt-row" + (" gt-br" if show_br and br_row[i, j] else "")
                b_cls = "gt-col" + (" gt-br" if show_br and br_col[i, j] else "")
                r.append(f'<span class="{a_cls}">{fmt(self.A[i, j])}</span>, '
                         f'<span class="{b_cls}">{fmt(self.B[i, j])}</span>')
                cell = []
                if (i, j) in ne:
                    cell.append("gt-ne")
                if (i, j) in pareto:
                    cell.append("gt-pareto")
                c.append(" ".join(cell))
            rows.append(r)
            classes.append(c)
        return html.table(self.col_actions, rows, row_headers=self.row_actions,
                          cell_classes=classes)

    def _summary_html(self) -> str:
        props = []
        if self.is_zero_sum():
            props.append("zero-sum")
        if self.is_symmetric():
            props.append("symmetric")
        props.append(f"{len(self.pure_nash())} pure NE")
        return self._matrix_html(show_ne=True) + html.legend(*props)

    def _solution_html(self, show_br=True, show_ne=True, show_pareto=True) -> str:
        tbl = self._matrix_html(show_br=show_br, show_ne=show_ne, show_pareto=show_pareto)
        return tbl + html.legend("underline = best response",
                                 "green outline = Nash equilibrium",
                                 "★ = Pareto optimal")

    def summary(self, title: Optional[str] = None) -> None:
        """Render the payoff bimatrix plus quick game properties."""
        html.show(html.card(title or self.name, self._summary_html()))

    def solve(self, title: Optional[str] = None, show_br: bool = True,
              show_ne: bool = True, show_pareto: bool = True) -> None:
        """Annotated bimatrix: best responses underlined, NE outlined, Pareto starred."""
        body = self._solution_html(show_br, show_ne, show_pareto)
        html.show(html.card(title or f"{self.name} - solution", body))

    def explain(self, title: Optional[str] = None) -> None:
        """Step-by-step walkthrough: best responses → pure NE → mixed NE."""
        items = ["<b>Step 1 - Best responses.</b> Underlines mark each player's "
                 "best reply: row's to every column, column's to every row."]
        ne = self.pure_nash()
        if ne:
            cells = ", ".join(f"({self.row_actions[i]}, {self.col_actions[j]})"
                              for i, j in ne)
            items.append(f"<b>Step 2 - Pure Nash equilibria.</b> Cells where both "
                         f"players' best responses coincide: {cells}.")
        else:
            items.append("<b>Step 2 - Pure Nash equilibria.</b> None - no cell is a "
                         "mutual best response.")
        mixed = [(p, q) for p, q in self.equilibria()
                 if np.max(p) < 1 - 1e-9 or np.max(q) < 1 - 1e-9]
        if mixed:
            p, q = mixed[0]
            items.append(f"<b>Step 3 - Mixed Nash equilibrium.</b> Row plays "
                         f"{fmt_prob_vec(p)}; Column plays {fmt_prob_vec(q)} - each "
                         "making the opponent indifferent across their support.")
        body = self._solution_html() + html.steps(items)
        html.show(html.card(title or f"{self.name} - explanation", body))

    def plot_br_map(self, title: Optional[str] = None):
        """Best-response heatmap (any size)."""
        br_row, br_col = self.best_responses()
        return plots.br_heatmap(br_row, br_col, solvers.ne_mask(self.A, self.B),
                                self.row_actions, self.col_actions,
                                title=title or f"{self.name} - best responses")

    # ── comparison ──────────────────────────────────────────────────────────
    @staticmethod
    def compare(*games: "NormalFormGame") -> None:
        """Render several games' summaries side by side."""
        html.compare_via(games, "summary")

    def __repr__(self) -> str:
        return f"NormalFormGame({self.name!r}, shape={self.shape})"
