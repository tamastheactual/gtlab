"""Two-player normal-form (bimatrix) games.

The class holds payoff data and delegates all computation to :mod:`gtlab.solvers`
and all rendering to :mod:`gtlab.viz`. Compare with the original notebook's
~1,800-line monolith: the math and the HTML now live elsewhere, once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..solvers.normal_form_extra import (envelope_crossings, expected_payoffs,
                                         mixed_equilibria, row_indifference_lines,
                                         sweep_mixed_data, sweep_pure_data,
                                         sweep_regions_data)
from ..viz import C, fmt, fmt_prob, fmt_prob_vec, html, plots, rc_context

_ref_lines = plots.ref_lines


@dataclass
class NormalFormGame:
    """A two-player game in normal form.

    Parameters
    ----------
    A, B          : row- and column-player payoff matrices (m x n).
    row_actions   : labels for the row player's actions (default r0, r1, ...).
    col_actions   : labels for the column player's actions.
    name          : optional display name.
    row_name      : display name for the row player (default "Row").
    col_name      : display name for the column player (default "Column").
    """

    A: np.ndarray
    B: np.ndarray
    row_actions: Optional[Sequence[str]] = None
    col_actions: Optional[Sequence[str]] = None
    name: str = "Game"
    row_name: str = "Row"
    col_name: str = "Column"

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

    # ── per-player labelled headers ──────────────────────────────────────────
    def _player_legend(self) -> str:
        """Muted line naming the two players and their colors."""
        return html.legend(
            f'<span class="gt-row">{self.row_name}</span> = rows',
            f'<span class="gt-col">{self.col_name}</span> = columns')

    # ── display (delegates to viz) ───────────────────────────────────────────
    def _matrix_html(self, show_br: bool = False, show_ne: bool = False,
                     show_pareto: bool = False, show_dominated: bool = False,
                     show_heatmap: bool = False,
                     dom_rows: Optional[set] = None,
                     dom_cols: Optional[set] = None) -> str:
        """Build the (optionally annotated) bimatrix table as an HTML string.

        When ``show_dominated`` is set, the struck-through cells come from
        explicit ``dom_rows``/``dom_cols`` index sets if given, otherwise from
        first-round :meth:`dominated`.
        """
        m, n = self.shape
        br_row, br_col = self.best_responses() if show_br else (None, None)
        ne = set(self.pure_nash()) if show_ne else set()
        pareto = set(self.pareto_optimal()) if show_pareto else set()
        if not show_dominated:
            dom_rows, dom_cols = set(), set()
        elif dom_rows is None and dom_cols is None:
            dr, dc = self.dominated()
            dom_rows, dom_cols = set(dr.keys()), set(dc.keys())
        else:
            dom_rows = set(dom_rows or ())
            dom_cols = set(dom_cols or ())
        rows, classes = [], []
        for i in range(m):
            r, c = [], []
            for j in range(n):
                a_cls = "gt-row" + (" gt-br" if show_br and br_row[i, j] else "")
                if i in dom_rows:
                    a_cls += " gt-dom"
                b_cls = "gt-col" + (" gt-br" if show_br and br_col[i, j] else "")
                if j in dom_cols:
                    b_cls += " gt-dom"
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
        props = [f"{self.row_name} x {self.col_name}", f"{self.shape[0]}x{self.shape[1]}"]
        if self.is_zero_sum():
            props.append("zero-sum")
        if self.is_symmetric():
            props.append("symmetric")
        props.append(f"{len(self.pure_nash())} pure NE")
        return self._matrix_html(show_ne=True) + html.legend(*props)

    def _solution_html(self, show_br=True, show_ne=True, show_pareto=True,
                       show_dominated=False, show_heatmap=False,
                       show_arrows=False, show_mixed=False) -> str:
        # When every cell is Pareto-optimal (e.g. zero-sum games) the stars carry
        # no information, so suppress them and explain why with a note instead.
        all_pareto = show_pareto and len(self.pareto_optimal()) == self.A.size
        star_pareto = show_pareto and not all_pareto
        tbl = self._matrix_html(show_br=show_br, show_ne=show_ne,
                                show_pareto=star_pareto,
                                show_dominated=show_dominated,
                                show_heatmap=show_heatmap)
        legend_parts = []
        if show_br:
            legend_parts.append("underline = best response")
        if show_ne:
            legend_parts.append("green outline = Nash equilibrium")
        if star_pareto:
            legend_parts.append("star = Pareto optimal")
        if show_dominated:
            legend_parts.append("strikethrough = strictly dominated")
        body = tbl + self._player_legend()
        if legend_parts:
            body += html.legend(*legend_parts)
        if all_pareto:
            body += html.note("all outcomes are Pareto-optimal")
        if show_arrows:
            body += self._arrows_html()
        if show_mixed:
            body += self._mixed_ne_html()
        return body

    def _arrows_html(self) -> str:
        """Best-response deviation cycle (shown when there is no pure NE)."""
        if self.pure_nash():
            return ""
        m, n = self.shape
        br_row, br_col = self.best_responses()
        arrows = []
        for i in range(m):
            for j in range(n):
                if not br_row[i, j]:
                    for i2 in np.where(br_row[:, j])[0]:
                        arrows.append(f"({self.row_actions[i]},{self.col_actions[j]}) "
                                      f"-> ({self.row_actions[i2]},{self.col_actions[j]})")
                if not br_col[i, j]:
                    for j2 in np.where(br_col[i, :])[0]:
                        arrows.append(f"({self.row_actions[i]},{self.col_actions[j]}) "
                                      f"-> ({self.row_actions[i]},{self.col_actions[j2]})")
        if not arrows:
            return ""
        text = "BR deviation cycle: " + "  ".join(arrows[:6])
        if len(arrows) > 6:
            text += f"  ... ({len(arrows)} total)"
        return html.note(text)

    def _mixed_ne_html(self) -> str:
        """Annotation listing mixed (non-pure) Nash equilibria."""
        lines = []
        for p, q in mixed_equilibria(self.A, self.B):
            eu_r, eu_c = expected_payoffs(self.A, self.B, p, q)
            r_parts = ", ".join(f"{self.row_actions[i]}={fmt_prob(p[i])}"
                                for i in range(len(p)) if p[i] > 1e-9)
            c_parts = ", ".join(f"{self.col_actions[j]}={fmt_prob(q[j])}"
                                for j in range(len(q)) if q[j] > 1e-9)
            lines.append(f"Mixed NE: {self.row_name} plays ({r_parts}), "
                         f"{self.col_name} plays ({c_parts}) - "
                         f"E[payoff] = ({fmt(eu_r)}, {fmt(eu_c)})")
        if not lines:
            return ""
        return html.note("<br>".join(lines))

    def summary(self, title: Optional[str] = None) -> None:
        """Render the payoff bimatrix plus quick game properties."""
        html.show(html.card(title or self.name, self._summary_html()))

    def solve(self, title: Optional[str] = None, show_br: bool = True,
              show_ne: bool = True, show_pareto: bool = True,
              show_dominated: bool = False, show_heatmap: bool = False,
              show_arrows: bool = False, show_mixed: bool = False) -> None:
        """Annotated bimatrix with selectable overlays.

        ``show_br`` underlines best responses, ``show_ne`` outlines pure NE,
        ``show_pareto`` stars Pareto-optimal cells, ``show_dominated`` strikes
        through strictly dominated actions, ``show_arrows`` lists the BR
        deviation cycle when there is no pure NE, and ``show_mixed`` annotates
        mixed equilibria. (``show_heatmap`` is accepted for API parity.)
        """
        body = self._solution_html(show_br, show_ne, show_pareto,
                                   show_dominated, show_heatmap,
                                   show_arrows, show_mixed)
        html.show(html.card(title or f"{self.name} - solution", body))

    def display(self, title: Optional[str] = None) -> None:
        """Clean ``(a, b)`` payoff table with no analysis overlays."""
        body = self._matrix_html() + self._player_legend()
        html.show(html.card(title or self.name, body))

    def as_dataframe(self):
        """Return the bimatrix as a pandas DataFrame of ``(a, b)`` tuples."""
        import pandas as pd
        m, n = self.shape
        idx = pd.Index(self.row_actions, name=f"{self.row_name} action")
        cols = pd.Index(self.col_actions, name=f"{self.col_name} action")
        data = [[(self.A[i, j], self.B[i, j]) for j in range(n)] for i in range(m)]
        return pd.DataFrame(data, index=idx, columns=cols)

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
        # Efficiency step: name the Pareto-optimal outcomes and relate them to NE.
        pareto = self.pareto_optimal()
        all_pareto = len(pareto) == self.A.size
        step = len(items) + 1
        if all_pareto:
            items.append(f"<b>Step {step} - Efficiency.</b> Every outcome is "
                         "Pareto-optimal, so no profile is unambiguously better "
                         "for both players.")
        else:
            cells = ", ".join(f"({self.row_actions[i]}, {self.col_actions[j]})"
                              for i, j in pareto)
            eff = f"<b>Step {step} - Efficiency.</b> Pareto-optimal outcomes: {cells}."
            if ne:
                pareto_set = set(pareto)
                eff_ne = [c for c in ne if c in pareto_set]
                if not eff_ne:
                    eff += " The Nash equilibrium is not Pareto-optimal."
                elif len(eff_ne) == len(ne):
                    eff += " The Nash equilibrium is Pareto-optimal."
                else:
                    eff += " Some Nash equilibria are Pareto-optimal."
            items.append(eff)
        body = self._solution_html() + html.steps(items)
        html.show(html.card(title or f"{self.name} - explanation", body))

    def iesds_explain(self, title: Optional[str] = None) -> None:
        """Round-by-round iterated elimination of strictly dominated strategies."""
        _, _, rows, cols, log = self.iesds()
        if not log:
            body = (self._matrix_html()
                    + html.note("No strictly dominated strategies - "
                                "IESDS removes nothing."))
            html.show(html.card(title or f"{self.name} - IESDS", body))
            return

        items = []
        for k, step in enumerate(log, 1):
            who = self.row_name if step["player"] == "row" else self.col_name
            kind = "row" if step["player"] == "row" else "column"
            items.append(f"<b>Round {k}.</b> {who}'s {kind} action "
                         f"<b>{step['removed']}</b> is strictly dominated by "
                         f"<b>{step['by']}</b> -> eliminate.")
        if len(rows) == 1 and len(cols) == 1:
            items.append(f"<b>Result.</b> Unique surviving outcome: "
                         f"({rows[0]}, {cols[0]}).")
        else:
            items.append(f"<b>Reduced game.</b> {len(rows)}x{len(cols)} remains: "
                         f"{self.row_name} in {{{', '.join(rows)}}}, "
                         f"{self.col_name} in {{{', '.join(cols)}}}.")
        # Strike through every strategy eliminated across ALL rounds (not just the
        # first), mapping the logged labels back to original matrix indices so the
        # overlay agrees with the narrated walkthrough.
        row_index = {lbl: i for i, lbl in enumerate(self.row_actions)}
        col_index = {lbl: j for j, lbl in enumerate(self.col_actions)}
        dom_rows = {row_index[s["removed"]] for s in log if s["player"] == "row"}
        dom_cols = {col_index[s["removed"]] for s in log if s["player"] == "col"}
        body = (self._matrix_html(show_dominated=True,
                                  dom_rows=dom_rows, dom_cols=dom_cols)
                + html.steps(items))
        html.show(html.card(title or f"{self.name} - IESDS", body))

    def plot_br_map(self, title: Optional[str] = None):
        """Best-response heatmap (any size)."""
        br_row, br_col = self.best_responses()
        return plots.br_heatmap(br_row, br_col, solvers.ne_mask(self.A, self.B),
                                self.row_actions, self.col_actions,
                                title=title or f"{self.name} - best responses",
                                row_player=getattr(self, "row_name", "Row"),
                                col_player=getattr(self, "col_name", "Column"))

    # ── mixed-strategy indifference plot ─────────────────────────────────────
    def plot_mixed(self, title: Optional[str] = None, figsize=(10.0, 4.0)):
        """Expected-payoff indifference lines for a 2x2 (or surfaces for 3x3) game.

        For a 2x2 game this draws, side by side, the row player's expected
        payoff to each pure action as the column player mixes, and vice versa,
        marking the indifference (mixed-NE) crossing. Returns ``(fig, ax)``.
        """
        m, n = self.shape
        if (m, n) == (3, 3):
            return self._plot_mixed_3x3(title=title, figsize=figsize)
        if (m, n) != (2, 2):
            # Indifference lines/simplex only make sense for 2x2 / 3x3. For larger
            # games fall back to bar charts of the (first) mixed equilibrium.
            return self._plot_mixed_bars(title=title, figsize=figsize)

        import matplotlib.pyplot as plt

        p = np.linspace(0, 1, 401)
        lines_row = row_indifference_lines(self.A, p)
        # Column player's perspective: swap roles via transpose.
        lines_col = row_indifference_lines(self.B.T, p)

        with rc_context():
            fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(title or f"{self.name} - mixed-strategy indifference",
                         fontweight="bold")

            for i in range(m):
                ax_r.plot(p, lines_row[i], color=(C["p1"] if i == 0 else C["accent"]),
                          lw=2.2, label=f"{self.row_name} plays {self.row_actions[i]}")
            for (ps, us) in envelope_crossings(lines_row, p):
                ax_r.plot(ps, us, "o", ms=8, color=C["ne"],
                          markeredgecolor="white", zorder=6)
                ax_r.axvline(ps, ls=":", lw=0.8, color=C["ne"], alpha=0.5)
                ax_r.annotate(f"$p^*$={fmt_prob(ps)}", (ps, us),
                              textcoords="offset points", xytext=(10, -16),
                              ha="left", va="top", color=C["ne"], fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                        ec="none", alpha=0.8))
            ax_r.set_xlabel(f"Pr({self.col_name} plays {self.col_actions[0]})")
            ax_r.set_ylabel(f"Expected payoff to {self.row_name}")
            ax_r.set_xlim(-0.02, 1.02)
            ax_r.set_title(f"{self.row_name}'s indifference")
            ax_r.legend(loc="best")

            for j in range(n):
                ax_c.plot(p, lines_col[j], color=(C["p2"] if j == 0 else C["chance"]),
                          lw=2.2, label=f"{self.col_name} plays {self.col_actions[j]}")
            for (ps, us) in envelope_crossings(lines_col, p):
                ax_c.plot(ps, us, "o", ms=8, color=C["ne"],
                          markeredgecolor="white", zorder=6)
                ax_c.axvline(ps, ls=":", lw=0.8, color=C["ne"], alpha=0.5)
                ax_c.annotate(f"$p^*$={fmt_prob(ps)}", (ps, us),
                              textcoords="offset points", xytext=(10, -16),
                              ha="left", va="top", color=C["ne"], fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                        ec="none", alpha=0.8))
            ax_c.set_xlabel(f"Pr({self.row_name} plays {self.row_actions[0]})")
            ax_c.set_ylabel(f"Expected payoff to {self.col_name}")
            ax_c.set_xlim(-0.02, 1.02)
            ax_c.set_title(f"{self.col_name}'s indifference")
            ax_c.legend(loc="best")
        return fig, (ax_r, ax_c)

    def _plot_mixed_3x3(self, title: Optional[str] = None, figsize=(7.0, 5.5)):
        """3D expected-payoff surfaces over the column player's mixing simplex."""
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation

        surf_colors = [C["p1"], C["chance"], C["p2"]]
        ra, ca = self.row_actions, self.col_actions
        ngrid = 40
        xs, ys = [], []
        for i in range(ngrid + 1):
            for j in range(ngrid + 1 - i):
                xs.append(i / ngrid)
                ys.append(j / ngrid)
        xs, ys = np.array(xs), np.array(ys)
        q0 = 1.0 - xs - ys
        tri = Triangulation(xs, ys)
        A = self.A
        surfs = [A[i, 0] * q0 + A[i, 1] * xs + A[i, 2] * ys for i in range(3)]

        with rc_context({"figure.autolayout": False}):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            for i in range(3):
                ax.plot_trisurf(xs, ys, surfs[i], triangles=tri.triangles,
                                color=surf_colors[i], alpha=0.4,
                                edgecolor=surf_colors[i], linewidth=0.08)
            ax.set_xlabel(f"Pr({ca[1]})")
            ax.set_ylabel(f"Pr({ca[2]})")
            ax.set_zlabel("Expected payoff")
            ax.set_title(title or f"{self.name} - mixed payoff surfaces")
            handles = [plt.Line2D([0], [0], color=surf_colors[i], lw=4,
                                  label=f"{self.row_name} plays {ra[i]}")
                       for i in range(3)]
            ax.legend(handles=handles, loc="upper left")
        return fig, ax

    def _plot_mixed_bars(self, title: Optional[str] = None, figsize=(10.0, 4.0)):
        """Fallback for games larger than 3x3: bar charts of the first mixed
        equilibrium's strategy probabilities for each player.

        Mixed-equilibrium enumeration is exponential, so for large action sets we
        skip it and show a note rather than hang.
        """
        import matplotlib.pyplot as plt

        m, n = self.shape
        # Support enumeration is intractable for large games; guard it.
        eqs = self.equilibria() if max(m, n) <= 8 else None
        with rc_context():
            fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=figsize)
            fig.suptitle(title or f"{self.name} - equilibrium strategies",
                         fontweight="bold")
            if not eqs:
                msg = ("no equilibrium found" if eqs == [] else
                       f"mixed-strategy view omitted for a {m}x{n} game\n"
                       "(use .summary() / .solve())")
                for ax in (ax_r, ax_c):
                    ax.text(0.5, 0.5, msg, ha="center", va="center",
                            color=C["muted"])
                    ax.axis("off")
                return fig, (ax_r, ax_c)
            p, q = eqs[0]
            ax_r.bar(range(len(p)), p, color=C["p1"], zorder=3)
            ax_r.set_xticks(range(len(p)))
            ax_r.set_xticklabels(self.row_actions, rotation=45, ha="right")
            ax_r.set_xlabel(f"{self.row_name} action")
            ax_r.set_ylabel("Equilibrium probability")
            ax_r.set_ylim(0, 1.0)
            ax_r.set_title(f"{self.row_name}'s mix $p^*$")
            ax_c.bar(range(len(q)), q, color=C["p2"], zorder=3)
            ax_c.set_xticks(range(len(q)))
            ax_c.set_xticklabels(self.col_actions, rotation=45, ha="right")
            ax_c.set_xlabel(f"{self.col_name} action")
            ax_c.set_ylabel("Equilibrium probability")
            ax_c.set_ylim(0, 1.0)
            ax_c.set_title(f"{self.col_name}'s mix $q^*$")
        return fig, (ax_r, ax_c)

    # ── continuous best-response curves (static) ─────────────────────────────
    @staticmethod
    def plot_br_curves(br1: Callable, br2: Callable, *, ne=None,
                       domain=(0.0, 5.0), br1_label="$BR_1$", br2_label="$BR_2$",
                       xlabel="$x_1$", ylabel="$x_2$",
                       title="Best-response curves", figsize=(6.0, 5.5),
                       n_points=400, hlines=None, vlines=None):
        """Plot best-response curves of a 2-player continuous-strategy game.

        ``br1(x2) -> x1`` and ``br2(x1) -> x2``. ``ne`` is an optional list of
        ``(x1, x2)`` (or ``(x1, x2, label)``) equilibria to mark. ``hlines`` /
        ``vlines`` draw dotted reference lines at the given y / x values (each a
        number or a ``(value, label)`` / ``(value, label, color)`` tuple).
        Returns ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt

        lo, hi = domain
        t = np.linspace(lo, hi, n_points)
        br1_vals = np.array([br1(v) for v in t], dtype=float)
        br2_vals = np.array([br2(v) for v in t], dtype=float)

        with rc_context():
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(br1_vals, t, lw=2.4, color=C["p1"], label=br1_label)
            ax.plot(t, br2_vals, lw=2.4, color=C["p2"], ls="--", label=br2_label)
            for pt in (ne or []):
                x1, x2 = pt[0], pt[1]
                lbl = pt[2] if len(pt) > 2 else f"NE ({fmt(x1)}, {fmt(x2)})"
                ax.plot(x1, x2, "o", ms=9, color=C["ne"],
                        markeredgecolor="white", zorder=6)
                ax.annotate(lbl, (x1, x2), textcoords="offset points",
                            xytext=(10, 10), color=C["ne"], fontweight="bold")
            _ref_lines(ax, hlines, vlines)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.legend(loc="upper right")
        return fig, ax

    # ── comparative statics sweeps (static) ──────────────────────────────────
    @staticmethod
    def sweep_mixed(factory: Callable, param_range, param_name="parameter",
                    title: Optional[str] = None, figsize=(7.0, 4.0),
                    hlines=None, vlines=None):
        """Plot first-equilibrium mixing probabilities against a swept parameter.

        ``hlines`` / ``vlines`` draw dotted reference lines at the given y / x
        values (e.g. a probability level or a parameter of interest).
        """
        import matplotlib.pyplot as plt

        prange, row_probs, col_probs = sweep_mixed_data(factory, param_range)
        g0 = factory(float(prange[0]))
        r, c = g0.shape

        def _interesting(vals):
            a = np.array(vals, dtype=float)
            return not (np.nanstd(a) < 1e-12 and
                        (abs(np.nanmean(a)) < 1e-12 or
                         abs(np.nanmean(a) - 1.0) < 1e-12))

        row_idx = [i for i in range(r - 1) if _interesting(row_probs[i])]
        col_idx = [j for j in range(c - 1) if _interesting(col_probs[j])]
        if not row_idx and not col_idx:
            row_idx, col_idx = [0], [0]
        rc = [C["p1"], C["accent"], C["ce"], C["cce"]]
        cc = [C["p2"], C["chance"], C["ce"], C["cce"]]

        with rc_context():
            fig, ax = plt.subplots(figsize=figsize)
            for k, i in enumerate(row_idx):
                ax.plot(prange, row_probs[i], color=rc[k % len(rc)], lw=2.2,
                        label=f"Pr({g0.row_name} plays {g0.row_actions[i]})")
            for k, j in enumerate(col_idx):
                ax.plot(prange, col_probs[j], color=cc[k % len(cc)], lw=2.2,
                        ls="--", label=f"Pr({g0.col_name} plays {g0.col_actions[j]})")
            _ref_lines(ax, hlines, vlines)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Equilibrium mixing probability")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(title or f"Mixed NE vs {param_name}")
            ax.legend(loc="best")
        return fig, ax

    @staticmethod
    def sweep_pure(factory: Callable, param_range, param_name="parameter",
                   title: Optional[str] = None, figsize=(7.0, 3.8),
                   hlines=None, vlines=None):
        """Plot pure-NE structure (which profiles are NE) against a swept parameter.

        ``hlines`` / ``vlines`` draw dotted reference lines at the given y / x values.
        """
        import matplotlib.pyplot as plt

        prange, profiles, is_ne, n_eq = sweep_pure_data(factory, param_range)
        g0 = factory(float(prange[0]))
        active = [p for p in profiles if any(is_ne[p])]
        colors = [C["ne"], C["accent"], C["ce"], C["cce"], C["p1"], C["p2"]]

        with rc_context():
            fig, ax = plt.subplots(figsize=figsize)
            ls_cycle = ["-", "--", ":", "-."]
            for k, p in enumerate(active):
                name = f"({g0.row_actions[p[0]]}, {g0.col_actions[p[1]]})"
                # Fan out coincident indicators (all at y=1) so each legend
                # entry maps to a visible line instead of being over-painted.
                ys = np.asarray(is_ne[p], dtype=float) * (1 - 0.04 * k)
                ax.step(prange, ys, where="mid", lw=2.2,
                        ls=ls_cycle[k % len(ls_cycle)], alpha=0.95 - 0.08 * k,
                        color=colors[k % len(colors)], label=f"{name} is NE")
            ax.step(prange, n_eq, where="mid", lw=2.0, ls="--",
                    color=C["muted"], label="# pure NE")
            _ref_lines(ax, hlines, vlines)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Count / indicator")
            ax.set_yticks(range((max(n_eq) if n_eq else 1) + 1))
            ax.set_title(title or f"Pure NE structure vs {param_name}")
            ax.legend(loc="best")
        return fig, ax

    @staticmethod
    def sweep_ne_regions(factory: Callable, x_range, y_range, *,
                         x_name="$x$", y_name="$y$", title: Optional[str] = None,
                         figsize=(7.2, 5.4), n=121):
        """Heatmap of pure-NE equilibrium regions over a 2-parameter grid.

        ``factory(x, y) -> NormalFormGame``. Returns ``(fig, ax)``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        xs, ys, Z, ne_sets, profiles, profile_names = sweep_regions_data(
            factory, x_range, y_range, n=n)
        nc = len(ne_sets)
        # Saturated palette so each *real* equilibrium region is clearly visible.
        # The neutral C["empty"] gray is reserved for the empty (no-pure-NE) set
        # only -- a non-empty region must never be painted with it. Cycle the
        # palette per non-empty set so each gets a distinct, saturated color.
        pal = [C["p1"], C["ne"], C["cce"], C["ce"], C["p2"], C["accent"],
               C["chance"], C["pareto"]]
        colors, k = [], 0
        for s in ne_sets:
            if not s:
                colors.append(C["empty"])
            else:
                colors.append(pal[k % len(pal)])
                k += 1
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, nc, 1), nc)

        labels = []
        for s in ne_sets:
            if not s:
                labels.append("No pure NE")
            elif len(s) == 1:
                labels.append(profile_names[next(iter(s))])
            else:
                parts = sorted(s, key=lambda p: profiles.index(p))
                labels.append(" & ".join(profile_names[p] for p in parts))

        with rc_context():
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(Z, origin="lower", aspect="auto",
                      extent=[xs[0], xs[-1], ys[0], ys[-1]],
                      cmap=cmap, norm=norm, interpolation="nearest")
            present = sorted(set(int(v) for v in Z.ravel()))
            handles = [Patch(facecolor=colors[k], edgecolor=C["grid"],
                             label=labels[k]) for k in present]
            # Place the legend outside the map so it never covers a region.
            plots.legend_outside(ax, handles=handles, title="Equilibrium type")
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(title or f"Pure NE structure vs {x_name}, {y_name}")
        return fig, ax

    # ── comparison ──────────────────────────────────────────────────────────
    @staticmethod
    def compare(*games: "NormalFormGame") -> None:
        """Render several games' summaries side by side."""
        html.compare_via(games, "summary")

    def __repr__(self) -> str:
        return f"NormalFormGame({self.name!r}, shape={self.shape})"
