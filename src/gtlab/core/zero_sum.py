"""Two-player zero-sum games: minimax value via linear programming."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import C, fmt, fmt_prob_vec, html, plots, rc_context
from ..solvers.zero_sum_extra import (epsilon_security, solve_all_supports,
                                      solve_support, support_enumeration_trace)


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
    row_name: str = "Row"
    col_name: str = "Column"

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

    def best_response_row(self, q: np.ndarray) -> np.ndarray:
        """Row's pure best-response indices against the column mix ``q``."""
        return solvers.best_response_to_mixed(self.A, q)

    def best_response_col(self, p: np.ndarray) -> np.ndarray:
        """Column's pure best-response indices against the row mix ``p``.

        Column minimizes Row's payoff, i.e. minimizes ``(A^T p)``.
        """
        vals = np.asarray(p, dtype=float) @ self.A
        return np.where(vals <= vals.min() + 1e-9)[0]

    def epsilon_security(self, p: np.ndarray, q: np.ndarray) -> dict:
        """Epsilon-security gaps for the strategy pair ``(p, q)``."""
        return epsilon_security(self.A, p, q, self.solve_value()["value"])

    def solve_support(self, I: List[int], J: List[int]) -> Optional[dict]:
        """Solve on a fixed support pair ``(I, J)`` (or ``None`` if infeasible)."""
        return solve_support(self.A, I, J)

    def solve_all_supports(self) -> List[dict]:
        """Enumerate all feasible support pairs, returning distinct equilibria."""
        return solve_all_supports(self.A)

    def as_dataframe(self):
        """Return the bimatrix as a pandas DataFrame of ``(a, -a)`` pairs."""
        import pandas as pd

        m, n = self.shape
        idx = pd.Index(self.row_actions, name=f"{self.row_name} action")
        cols = pd.Index(self.col_actions, name=f"{self.col_name} action")
        return pd.DataFrame(
            [[(self.A[i, j], -self.A[i, j]) for j in range(n)] for i in range(m)],
            index=idx, columns=cols,
        )

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

    def _bimatrix_html(self) -> str:
        """Payoff table showing (a, -a) pairs with player labels."""
        m, n = self.shape
        rows = []
        for i in range(m):
            row = []
            for j in range(n):
                a = self.A[i, j]
                row.append(f'(<span class="gt-row">{fmt(a)}</span>, '
                           f'<span class="gt-col">{fmt(-a)}</span>)')
            rows.append(row)
        return html.table(self.col_actions, rows, row_headers=self.row_actions)

    def summary(self, title: Optional[str] = None) -> None:
        saddle = self.pure_saddle_points()
        sp = ("none" if not saddle else
              ", ".join(f"({self.row_actions[i]}, {self.col_actions[j]})"
                        for i, j in saddle))
        m, n = self.shape
        meta = html.kv([
            ("Game size", f"{m} x {n}"),
            ("Zero-sum", "yes"),
            ("Saddle points", sp),
        ])
        html.show(html.card(title or self.name, self._bimatrix_html() + meta))

    def display(self, title: Optional[str] = None) -> None:
        """Render the bimatrix with ``(a, -a)`` notation."""
        body = self._bimatrix_html()
        html.show(html.card(title or self.name, body))

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

    def plot_convergence(self, T: int = 2000, seed: int = 0, title: Optional[str] = None):
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
                                 ylabel="average payoff",
                                 target_label=f"v = {fmt(s['value'])}")

    # ── detailed walkthroughs ─────────────────────────────────────────────────
    def lp_detail(self, title: Optional[str] = None) -> None:
        """Primal/dual LP formulation with constraint-by-constraint checks."""
        m, n = self.shape
        s = self.solve_value()
        p, q, v = s["p"], s["q"], s["value"]
        Aq, ATp = self.A @ q, self.A.T @ p

        parts = [html.kv([
            ("Primal (Row, maximizer)",
             "max v &nbsp; s.t. A<sup>T</sup>p &ge; v, &nbsp; 1<sup>T</sup>p = 1, &nbsp; p &ge; 0"),
            ("Dual (Column, minimizer)",
             "min v &nbsp; s.t. Aq &le; v, &nbsp; 1<sup>T</sup>q = 1, &nbsp; q &ge; 0"),
        ])]
        parts.append(self._matrix_html())

        sol_rows = [
            [fmt_prob_vec(p)], [fmt_prob_vec(q)], [fmt(v)],
        ]
        parts.append(html.table(["value"], sol_rows,
                                row_headers=["p*", "q*", "v"]))

        # Primal constraints: A^T p >= v.
        prim = []
        for j in range(n):
            lhs = float(ATp[j])
            ok = lhs >= v - 1e-8
            slack = lhs - v
            tag = "tight" if abs(slack) < 1e-8 else f"slack {fmt(slack)}"
            prim.append(f"<b>{self.col_actions[j]}</b>: (A<sup>T</sup>p*)<sub>j</sub> "
                        f"= {fmt(lhs)} {'&ge;' if ok else '&lt;'} v = {fmt(v)} ({tag}) "
                        f"{'&#10003;' if ok else '&#10007;'}")
        # Dual constraints: A q <= v.
        dual = []
        for i in range(m):
            lhs = float(Aq[i])
            ok = lhs <= v + 1e-8
            slack = v - lhs
            tag = "tight" if abs(slack) < 1e-8 else f"slack {fmt(slack)}"
            dual.append(f"<b>{self.row_actions[i]}</b>: (Aq*)<sub>i</sub> "
                        f"= {fmt(lhs)} {'&le;' if ok else '&gt;'} v = {fmt(v)} ({tag}) "
                        f"{'&#10003;' if ok else '&#10007;'}")

        val = float(p @ self.A @ q)
        strong = abs(val - v) < 1e-6
        parts.append(html.steps([
            "<b>Primal constraints A<sup>T</sup>p* &ge; v:</b><br>" + "<br>".join(prim),
            "<b>Dual constraints Aq* &le; v:</b><br>" + "<br>".join(dual),
            f"<b>Simplex:</b> 1<sup>T</sup>p* = {fmt(float(p.sum()))}, "
            f"1<sup>T</sup>q* = {fmt(float(q.sum()))}, p*, q* &ge; 0.",
            f"<b>Strong duality:</b> p*<sup>T</sup>Aq* = {fmt(val)} "
            f"{'=' if strong else '&ne;'} v = {fmt(v)} "
            f"{'&#10003;' if strong else '&#10007;'}",
        ]))
        html.show(html.card(title or f"{self.name} - LP detail", "".join(parts)))

    def support_detail(self, title: Optional[str] = None) -> None:
        """Full support-enumeration walkthrough with indifference equations."""
        m, n = self.shape
        feasible, rejected, n_tried = support_enumeration_trace(self.A)

        items = []
        for k, f in enumerate(feasible, 1):
            I, J = f["support_row"], f["support_col"]
            rs = "{" + ", ".join(self.row_actions[i] for i in I) + "}"
            cs = "{" + ", ".join(self.col_actions[j] for j in J) + "}"
            # Row indifference equations (solving for q).
            row_eqs = []
            for iv in I:
                terms = " + ".join(f"{fmt(self.A[iv, jv])}&middot;q<sub>{self.col_actions[jv]}</sub>"
                                   for jv in J)
                row_eqs.append(f"(Aq)<sub>{self.row_actions[iv]}</sub> = {terms} = v")
            row_eqs.append(" + ".join(f"q<sub>{self.col_actions[jv]}</sub>" for jv in J) + " = 1")
            # Column indifference equations (solving for p).
            col_eqs = []
            for jv in J:
                terms = " + ".join(f"{fmt(self.A[iv, jv])}&middot;p<sub>{self.row_actions[iv]}</sub>"
                                   for iv in I)
                col_eqs.append(f"(A<sup>T</sup>p)<sub>{self.col_actions[jv]}</sub> = {terms} = v")
            col_eqs.append(" + ".join(f"p<sub>{self.row_actions[iv]}</sub>" for iv in I) + " = 1")
            body = (f"<b>Feasible #{k}:</b> I = {rs}, J = {cs}<br>"
                    "<i>Row indifference (find q):</i><br>" + "<br>".join(row_eqs) + "<br>"
                    "<i>Column indifference (find p):</i><br>" + "<br>".join(col_eqs) + "<br>"
                    f"<b>p* = {fmt_prob_vec(f['p'])}, q* = {fmt_prob_vec(f['q'])}, "
                    f"v = {fmt(f['value'])}</b>")
            items.append(body)
        if not items:
            items.append("No feasible support pair found by enumeration.")

        parts = [html.note(f"Enumerated equal-size support pairs |I| = |J|. "
                           f"Tried {n_tried} pair(s), found {len(feasible)} feasible.")]
        parts.append(html.steps(items))

        if feasible:
            srows = []
            for f in feasible:
                rs = "{" + ", ".join(self.row_actions[i] for i in f["support_row"]) + "}"
                cs = "{" + ", ".join(self.col_actions[j] for j in f["support_col"]) + "}"
                srows.append([rs, cs, fmt_prob_vec(f["p"]), fmt_prob_vec(f["q"]),
                              fmt(f["value"])])
            parts.append(html.table(
                ["Row support", "Col support", "p*", "q*", "value"], srows,
                row_headers=[str(i + 1) for i in range(len(feasible))]))

        if rejected:
            rrows = [["{" + ", ".join(self.row_actions[i] for i in I) + "}",
                      "{" + ", ".join(self.col_actions[j] for j in J) + "}",
                      reason] for I, J, reason in rejected]
            parts.append(html.note(f"Rejected {len(rejected)} candidate(s):"))
            parts.append(html.table(["I", "J", "Reason"], rrows))

        html.show(html.card(title or f"{self.name} - support enumeration",
                            "".join(parts)))

    def dominance_detail(self, title: Optional[str] = None) -> None:
        """Strict-dominance check for both players, with mixture dominance."""
        m, n = self.shape
        items = []

        for i in range(m):
            for i2 in range(m):
                if i2 == i:
                    continue
                if (np.all(self.A[i2, :] >= self.A[i, :])
                        and np.any(self.A[i2, :] > self.A[i, :])):
                    comp = ", ".join(f"{fmt(self.A[i, j])} vs {fmt(self.A[i2, j])}"
                                     for j in range(n))
                    items.append(f'<span class="gt-row"><b>{self.row_actions[i]}</b></span> '
                                 f"is strictly dominated by <b>{self.row_actions[i2]}</b> "
                                 f"for {self.row_name}: [{comp}]")
        # Mixture dominance for rows.
        if m >= 3:
            for i in range(m):
                others = [k for k in range(m) if k != i]
                done = False
                for a in range(len(others)):
                    for b in range(a + 1, len(others)):
                        i1, i2 = others[a], others[b]
                        for alpha in (0.25, 0.5, 0.75):
                            mix = alpha * self.A[i1, :] + (1 - alpha) * self.A[i2, :]
                            if np.all(mix >= self.A[i, :] + 1e-10):
                                comp = ", ".join(f"{fmt(self.A[i, j])} vs {fmt(mix[j])}"
                                                 for j in range(n))
                                items.append(
                                    f'<span class="gt-row"><b>{self.row_actions[i]}</b></span> '
                                    f"is dominated by {fmt(alpha)}&middot;{self.row_actions[i1]} "
                                    f"+ {fmt(1 - alpha)}&middot;{self.row_actions[i2]}: [{comp}]")
                                done = True
                                break
                        if done:
                            break
                    if done:
                        break

        for j in range(n):
            for j2 in range(n):
                if j2 == j:
                    continue
                if (np.all(self.A[:, j2] <= self.A[:, j])
                        and np.any(self.A[:, j2] < self.A[:, j])):
                    comp = ", ".join(f"{fmt(self.A[i, j])} vs {fmt(self.A[i, j2])}"
                                     for i in range(m))
                    items.append(f'<span class="gt-col"><b>{self.col_actions[j]}</b></span> '
                                 f"is strictly dominated by <b>{self.col_actions[j2]}</b> "
                                 f"for {self.col_name}: Row payoffs [{comp}]")

        if not items:
            items.append("No strictly dominated pure actions found.")
        html.show(html.card(title or f"{self.name} - dominance",
                            html.steps(items)))

    def security_analysis(self, title: Optional[str] = None) -> None:
        """Epsilon-security at the optimum and under uniform perturbations."""
        m, n = self.shape
        s = self.solve_value()
        p, q, v = s["p"], s["q"], s["value"]
        eps = self.epsilon_security(p, q)

        head = html.kv([
            ("p*", fmt_prob_vec(p)),
            ("q*", fmt_prob_vec(q)),
            ("v", fmt(v)),
            ("&epsilon;<sub>row</sub>", fmt(eps["eps_row"], 8)),
            ("&epsilon;<sub>col</sub>", fmt(eps["eps_col"], 8)),
            ("&epsilon;", fmt(eps["eps_max"], 8) +
             (" (exact equilibrium)" if eps["eps_max"] < 1e-8 else "")),
        ])

        p_unif = np.ones(m) / m
        q_unif = np.ones(n) / n
        rows = []
        for delta in (0.0, 0.05, 0.1, 0.2):
            p_d = (1 - delta) * p + delta * p_unif
            q_d = (1 - delta) * q + delta * q_unif
            e = epsilon_security(self.A, p_d, q_d, v)
            rows.append([fmt(delta), fmt_prob_vec(p_d), fmt_prob_vec(q_d),
                         fmt(e["eps_row"], 6), fmt(e["eps_col"], 6),
                         fmt(e["eps_max"], 6)])
        pert = html.table(
            ["&delta;", "p<sub>&delta;</sub>", "q<sub>&delta;</sub>",
             "&epsilon;<sub>row</sub>", "&epsilon;<sub>col</sub>", "&epsilon;"],
            rows)

        note = html.note("A pair (p, q) is &epsilon;-secure if neither player gains "
                         "more than &epsilon; by best-responding. p<sub>&delta;</sub> "
                         "and q<sub>&delta;</sub> blend the optimum toward uniform; "
                         "&epsilon; grows with &delta;.")
        html.show(html.card(title or f"{self.name} - security analysis",
                            head + note + pert))

    # ── plots ─────────────────────────────────────────────────────────────────
    def _require_2x2(self) -> bool:
        return self.shape == (2, 2)

    def plot_mixed(self, title: Optional[str] = None):
        """Expected-payoff lines for both players (2x2 only). Returns (fig, ax)."""
        if not self._require_2x2():
            raise ValueError("plot_mixed is only available for 2x2 games")
        s = self.solve_value()
        p, q, v = s["p"], s["q"], s["value"]
        import matplotlib.pyplot as plt

        grid = np.linspace(0, 1, 200)
        with rc_context():
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            ax = axes[0]
            for i in range(2):
                pay = self.A[i, 0] * grid + self.A[i, 1] * (1 - grid)
                ax.plot(grid, pay, color=C["p1"] if i == 0 else C["p2"],
                        label=self.row_actions[i])
            ax.axhline(v, ls="--", color=C["ne"], label=f"v = {fmt(v)}")
            ax.axvline(q[0], ls=":", color=C["ne"], alpha=0.6)
            ax.plot(q[0], v, "o", color=C["ne"], zorder=5)
            ax.set_xlabel(f"q = Pr[{self.col_actions[0]}]")
            ax.set_ylabel("Expected payoff to Row")
            ax.set_title("Row's payoff lines")
            ax.legend()

            ax = axes[1]
            for j in range(2):
                pay = -(self.A[0, j] * grid + self.A[1, j] * (1 - grid))
                ax.plot(grid, pay, color=C["p1"] if j == 0 else C["p2"],
                        label=self.col_actions[j])
            ax.axhline(-v, ls="--", color=C["ne"], label=f"-v = {fmt(-v)}")
            ax.axvline(p[0], ls=":", color=C["ne"], alpha=0.6)
            ax.plot(p[0], -v, "o", color=C["ne"], zorder=5)
            ax.set_xlabel(f"p = Pr[{self.row_actions[0]}]")
            ax.set_ylabel("Expected payoff to Column")
            ax.set_title("Column's payoff lines")
            ax.legend()
            fig.suptitle(title or f"{self.name} - expected payoff lines",
                         fontweight="bold")
        return fig, axes

    def plot_value_surface(self, title: Optional[str] = None):
        """Contour of p^T A q over the 2x2 mixing square. Returns (fig, ax)."""
        if not self._require_2x2():
            raise ValueError("plot_value_surface is only available for 2x2 games")
        s = self.solve_value()
        p, q = s["p"], s["q"]
        import matplotlib.pyplot as plt

        rng = np.linspace(0, 1, 100)
        P, Q = np.meshgrid(rng, rng)
        Z = (P * (self.A[0, 0] * Q + self.A[0, 1] * (1 - Q))
             + (1 - P) * (self.A[1, 0] * Q + self.A[1, 1] * (1 - Q)))
        with rc_context():
            fig, ax = plt.subplots(figsize=(6, 5))
            cf = ax.contourf(Q, P, Z, levels=20, cmap="RdBu_r")
            ax.contour(Q, P, Z, levels=20, colors="k", linewidths=0.3, alpha=0.4)
            fig.colorbar(cf, ax=ax, label="p^T A q")
            ax.plot(q[0], p[0], "*", color="white", markersize=15,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=5)
            ax.set_xlabel(f"q = Pr[{self.col_actions[0]}]")
            ax.set_ylabel(f"p = Pr[{self.row_actions[0]}]")
            ax.set_title(title or f"{self.name} - value surface", fontweight="bold")
            ax.grid(False)
        return fig, ax

    def plot_security_levels(self, title: Optional[str] = None):
        """Row maximin vs Column minimax curves (2x2 only). Returns (fig, ax)."""
        if not self._require_2x2():
            raise ValueError("plot_security_levels is only available for 2x2 games")
        v = self.solve_value()["value"]
        import matplotlib.pyplot as plt

        rng = np.linspace(0, 1, 200)
        maximin = np.array([min(pp * self.A[0, 0] + (1 - pp) * self.A[1, 0],
                                pp * self.A[0, 1] + (1 - pp) * self.A[1, 1])
                            for pp in rng])
        minimax = np.array([max(self.A[0, 0] * qq + self.A[0, 1] * (1 - qq),
                                self.A[1, 0] * qq + self.A[1, 1] * (1 - qq))
                            for qq in rng])
        with rc_context():
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            ax = axes[0]
            ax.plot(rng, maximin, color=C["p1"], label="min_q p^T A q")
            ax.axhline(v, ls="--", color=C["ne"], label=f"v = {fmt(v)}")
            bi = int(np.argmax(maximin))
            ax.plot(rng[bi], maximin[bi], "o", color=C["ne"], zorder=5)
            ax.set_xlabel(f"p = Pr[{self.row_actions[0]}]")
            ax.set_ylabel("Guaranteed payoff")
            ax.set_title("Row's maximin (security level)")
            ax.legend(loc="lower center")

            ax = axes[1]
            ax.plot(rng, minimax, color=C["p2"], label="max_p p^T A q")
            ax.axhline(v, ls="--", color=C["ne"], label=f"v = {fmt(v)}")
            bj = int(np.argmin(minimax))
            ax.plot(rng[bj], minimax[bj], "o", color=C["ne"], zorder=5)
            ax.set_xlabel(f"q = Pr[{self.col_actions[0]}]")
            ax.set_ylabel("Worst-case payoff to Row")
            ax.set_title("Column's minimax (holding Row down)")
            ax.legend(loc="upper center")
            fig.suptitle(title or f"{self.name} - security levels", fontweight="bold")
        return fig, axes

    def plot_exploitability(self, title: Optional[str] = None):
        """Exploitability E(q) of Column deviations (2x2 only). Returns (fig, ax)."""
        if not self._require_2x2():
            raise ValueError("plot_exploitability is only available for 2x2 games")
        s = self.solve_value()
        q, v = s["q"], s["value"]
        import matplotlib.pyplot as plt

        rng = np.linspace(0, 1, 200)
        expl = np.array([max(self.A[0, 0] * qq + self.A[0, 1] * (1 - qq),
                             self.A[1, 0] * qq + self.A[1, 1] * (1 - qq)) - v
                         for qq in rng])
        with rc_context():
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(rng, 0, expl, alpha=0.3, color=C["p2"],
                            label="Exploitability E(q)")
            ax.plot(rng, expl, color=C["p2"])
            ax.axvline(q[0], ls="--", color=C["ne"], label=f"q* = {fmt(q[0])}")
            ax.axhline(0, color=C["muted"], linewidth=0.6)
            ax.set_xlabel(f"q = Pr[{self.col_actions[0]}]")
            ax.set_ylabel("Exploitability")
            ax.set_title(title or f"{self.name} - exploitability landscape",
                         fontweight="bold")
            ax.legend()
        return fig, ax

    @staticmethod
    def sweep_value(factory, param_range, param_name: str = "theta",
                    title: Optional[str] = None):
        """Plot value, p*, q* vs a parameter ``theta`` via ``factory(theta)``.

        ``factory`` returns a :class:`ZeroSumGame` for each parameter value.
        Returns ``(fig, axes)``.
        """
        import matplotlib.pyplot as plt

        param_range = list(param_range)
        values, ps, qs = [], [], []
        for theta in param_range:
            g = factory(theta)
            s = g.solve_value()
            values.append(s["value"])
            ps.append(s["p"])
            qs.append(s["q"])
        values = np.array(values)
        ps = np.array(ps)
        qs = np.array(qs)
        g0 = factory(param_range[0])

        with rc_context():
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].plot(param_range, values, color=C["ne"])
            axes[0].set_xlabel(param_name)
            axes[0].set_ylabel("Game value v")
            axes[0].set_title("Value vs parameter")

            for i in range(ps.shape[1]):
                axes[1].plot(param_range, ps[:, i], label=g0.row_actions[i])
            axes[1].set_xlabel(param_name)
            axes[1].set_ylabel("p* components")
            axes[1].set_title("Row optimal strategy")
            axes[1].legend()

            for j in range(qs.shape[1]):
                axes[2].plot(param_range, qs[:, j], label=g0.col_actions[j])
            axes[2].set_xlabel(param_name)
            axes[2].set_ylabel("q* components")
            axes[2].set_title("Column optimal strategy")
            axes[2].legend()
            fig.suptitle(title or f"Parameter sweep over {param_name}",
                         fontweight="bold")
        return fig, axes

    @staticmethod
    def compare(*games: "ZeroSumGame") -> None:
        html.compare_via(games, "solve")

    def __repr__(self) -> str:
        return f"ZeroSumGame({self.name!r}, shape={self.shape})"
