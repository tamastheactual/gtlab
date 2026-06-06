"""Correlated equilibrium, coarse correlated equilibrium, and no-regret learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import C, fmt, html, plots, rc_context
from ..solvers.correlated_extra import (ce_obedience_detail, cce_exante_detail,
                                        sample_payoff_region)


@dataclass
class CorrelatedGame:
    """Two-player general-sum game analyzed through the CE / CCE lens."""

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

    # ── notebook-compatible aliases ──────────────────────────────────────────
    def find_nash(self):
        """Alias for :meth:`nash` (notebook name)."""
        return self.nash()

    def simulate_hedge(self, T: int = 2000, seed: int = 0):
        """Alias for :meth:`hedge` (notebook name)."""
        return self.hedge(T=T, seed=seed)

    def _nash_payoffs(self):
        """List of ``(eu_row, eu_col)`` over all equilibria from :meth:`nash`."""
        out = []
        for p, q in self.nash():
            out.append((float(p @ self.A @ q), float(p @ self.B @ q)))
        return out

    # ── display ──────────────────────────────────────────────────────────────
    def _mu_table(self, mu: np.ndarray) -> str:
        m, n = self.shape
        rows = [[fmt(mu[i, j]) for j in range(n)] for i in range(m)]
        return (html.table(self.col_actions, rows, row_headers=self.row_actions)
                + html.note("cells = P(row, col)"))

    def _mu_sum_guard(self, mu: np.ndarray, tol: float = 1e-9):
        """Normalize ``mu`` to sum 1 if it does not, returning ``(note, mu)``.

        ``note`` is an HTML prefix (empty if no normalization was needed) so
        verification displays warn before reporting on a renormalized table.
        """
        total = float(mu.sum())
        if abs(total - 1.0) > tol and total != 0.0:
            return html.note("mu did not sum to 1; normalized for display"), mu / total
        return "", mu

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
        payoffs = self._nash_payoffs()
        if payoffs:
            # Use the WELFARE-MAXIMIZING NE so the comparison is apples-to-apples
            # with the welfare-maximizing CE / CCE rows (picking an arbitrary NE
            # would fabricate a misleading "correlation improves welfare" gap).
            eu_r, eu_c = max(payoffs, key=lambda pc: pc[0] + pc[1])
            rows.append(["max-welfare NE", fmt(eu_r), fmt(eu_c), fmt(eu_r + eu_c)])
        if ce:
            rows.append(["CE", fmt(ce["eu_row"]), fmt(ce["eu_col"]), fmt(ce["welfare"])])
        if cce:
            rows.append(["CCE", fmt(cce["eu_row"]), fmt(cce["eu_col"]), fmt(cce["welfare"])])
        tbl = html.table(["concept", "E[row]", "E[col]", "welfare"], rows)
        body = tbl + html.note("welfare ordering: NE ≤ CE ≤ CCE (each relaxes the "
                               "deviation constraints of the previous).")
        html.show(html.card(title or f"{self.name} - equilibrium concepts", body))

    def explain(self, title: Optional[str] = None) -> None:
        """Walkthrough: NE ⊆ CE ⊆ CCE, plus the no-regret learning connection."""
        items = [
            "<b>Step 1 - Nash equilibrium.</b> Players randomize independently; "
            "their product distribution must be a mutual best response.",
            "<b>Step 2 - Correlated equilibrium (CE).</b> A trusted device draws a "
            "joint action and privately recommends each player's part; obeying must "
            "be optimal given the conditional belief it induces.",
            "<b>Step 3 - Coarse correlated equilibrium (CCE).</b> Players commit "
            "before seeing the recommendation; only ex-ante deviations are checked, "
            "so CCE ⊇ CE ⊇ NE.",
            "<b>Step 4 - Learning.</b> If both players run a no-regret algorithm "
            "(see <code>plot_regret</code>), the empirical play converges to the CCE "
            "set (Hannan's theorem).",
        ]
        html.show(html.card(title or f"{self.name} - equilibrium concepts",
                            html.steps(items)))

    def verify_ce(self, mu: np.ndarray, title: Optional[str] = None) -> None:
        """Constraint-by-constraint check that ``mu`` is a correlated equilibrium."""
        mu = np.asarray(mu, dtype=float)
        body, mu = self._mu_sum_guard(mu)
        eu_row = float((mu * self.A).sum())
        eu_col = float((mu * self.B).sum())
        detail = ce_obedience_detail(self.A, self.B, mu)

        body += "<b>Input distribution &mu;</b>" + self._mu_table(mu)
        body += html.note(f"E[row] = {fmt(eu_row)}, E[col] = {fmt(eu_col)}, "
                          f"welfare = {fmt(eu_row + eu_col)}")
        rows, classes = [], []
        for d in detail:
            if d["player"] == "row":
                who, told, dev = self.row_name, self.row_actions[d["told"]], self.row_actions[d["deviation"]]
                p_told = float(mu[d["told"], :].sum())
            else:
                who, told, dev = self.col_name, self.col_actions[d["told"]], self.col_actions[d["deviation"]]
                p_told = float(mu[:, d["told"]].sum())
            if abs(p_told) < 1e-9:
                told += " (never recommended, p=0)"
            mark = "ok" if d["ok"] else "x"
            rows.append([who, told, dev, fmt(d["gain"]), mark])
            classes.append(["", "", "", "", "gt-ok" if d["ok"] else "gt-bad"])
        tbl = html.table(["player", "told", "deviate to", "E[gain from obeying]", ""],
                         rows, cell_classes=classes)
        ok = all(d["ok"] for d in detail)
        verdict = "Valid CE: every obedience constraint holds." if ok else \
                  "Not a CE: at least one obedience constraint is violated."
        body += "<b>Obedience constraints</b>" + tbl
        body += html.note("green = obeys / no profitable deviation, red = violation")
        body += html.note(verdict)
        html.show(html.card(title or f"{self.name} - CE verification", body))

    def verify_cce(self, mu: np.ndarray, title: Optional[str] = None) -> None:
        """Constraint-by-constraint check that ``mu`` is a coarse correlated equilibrium."""
        mu = np.asarray(mu, dtype=float)
        body, mu = self._mu_sum_guard(mu)
        rep = cce_exante_detail(self.A, self.B, mu)

        body += "<b>Input distribution &mu;</b>" + self._mu_table(mu)
        body += html.note(f"E[row] = {fmt(rep['eu_row'])}, E[col] = {fmt(rep['eu_col'])}")
        body += html.note("A CCE requires no player gains by committing to a fixed "
                          "action before the signal (ex-ante deviation).")
        rows, classes = [], []
        for d in rep["constraints"]:
            if d["player"] == "row":
                who, dev = self.row_name, self.row_actions[d["deviation"]]
            else:
                who, dev = self.col_name, self.col_actions[d["deviation"]]
            mark = "ok" if d["ok"] else "x"
            rows.append([who, dev, fmt(d["dev_payoff"]), fmt(d["current"]), fmt(d["gain"]), mark])
            classes.append(["", "", "", "", "", "gt-ok" if d["ok"] else "gt-bad"])
        tbl = html.table(["player", "deviate to", "deviation payoff", "current", "gain", ""],
                         rows, cell_classes=classes)
        verdict = "Valid CCE: no profitable ex-ante deviation." if rep["ok"] else \
                  "Not a CCE: a profitable ex-ante deviation exists."
        body += "<b>Ex-ante deviation constraints</b>" + tbl
        body += html.note("green = obeys / no profitable deviation, red = violation")
        body += html.note(verdict)
        html.show(html.card(title or f"{self.name} - CCE verification", body))

    def lp_detail(self, title: Optional[str] = None) -> None:
        """LP formulation of the CE / CCE polytope plus a welfare comparison vs NE."""
        ce = self.find_ce()
        cce = self.find_cce()

        formulation = html.steps([
            "<b>Variables.</b> A joint distribution &mu;<sub>ij</sub> ≥ 0 over action "
            "profiles, with &sum;<sub>ij</sub> &mu;<sub>ij</sub> = 1.",
            "<b>Objective.</b> max<sub>&mu;</sub> &sum;<sub>ij</sub> &mu;<sub>ij</sub> "
            "(A<sub>ij</sub> + B<sub>ij</sub>) (social welfare; row/col variants swap "
            "the objective).",
            "<b>CE - obedience.</b> &sum;<sub>j</sub> &mu;<sub>ij</sub> "
            "(A<sub>ij</sub> - A<sub>i'j</sub>) ≥ 0 for every i, i' (and the column "
            "analogue): obeying a private recommendation is optimal.",
            "<b>CCE - ex-ante.</b> &sum;<sub>ij</sub> &mu;<sub>ij</sub> "
            "(A<sub>ij</sub> - A<sub>i'j</sub>) ≥ 0 for every i' (one constraint per "
            "alternative action): a single up-front deviation is not profitable.",
            "Each is a linear program over a convex polytope; CCE relaxes CE's "
            "per-recommendation constraints into one ex-ante constraint, so CE ⊆ CCE.",
        ])
        body = "<b>Linear program</b>" + formulation

        rows = []
        for eu_r, eu_c in self._nash_payoffs():
            rows.append(["NE", fmt(eu_r), fmt(eu_c), fmt(eu_r + eu_c)])
        if ce:
            rows.append(["max-welfare CE", fmt(ce["eu_row"]), fmt(ce["eu_col"]), fmt(ce["welfare"])])
        if cce:
            rows.append(["max-welfare CCE", fmt(cce["eu_row"]), fmt(cce["eu_col"]), fmt(cce["welfare"])])
        tbl = html.table(["solution concept", "E[row]", "E[col]", "welfare"], rows)
        body += "<b>Welfare comparison</b>" + tbl
        body += html.note("welfare ordering: NE ≤ CE ≤ CCE (each relaxes the "
                          "deviation constraints of the previous).")
        html.show(html.card(title or f"{self.name} - LP for CE / CCE", body))

    def plot_payoff_region(self, n_samples: int = 400, seed: int = 0,
                           title: Optional[str] = None):
        """Scatter the achievable CE and CCE payoff pairs, with NE and pure cells.

        Random linear objectives are optimized over each polytope to trace its
        payoff image. Returns ``(fig, ax)``.
        """
        cce_pts = sample_payoff_region(self.A, self.B, coarse=True,
                                       n_samples=n_samples, seed=seed)
        ce_pts = sample_payoff_region(self.A, self.B, coarse=False,
                                      n_samples=n_samples, seed=seed + 1)
        m, n = self.shape
        with rc_context():
            fig, ax = plots.new_axes(figsize=(6.0, 5.0))
            if len(cce_pts):
                ax.scatter(cce_pts[:, 0], cce_pts[:, 1], s=10, alpha=0.20,
                           color=C["cce"], label="CCE payoffs", zorder=1)
            if len(ce_pts):
                ax.scatter(ce_pts[:, 0], ce_pts[:, 1], s=10, alpha=0.30,
                           color=C["ce"], label="CE payoffs", zorder=2)
            # Pure-strategy outcomes (group cells that share the same payoff point
            # so coincident labels are joined instead of overlapping).
            cell_groups: dict = {}
            for i in range(m):
                for j in range(n):
                    key = (self.A[i, j], self.B[i, j])
                    cell_groups.setdefault(key, []).append(
                        f"({self.row_actions[i]},{self.col_actions[j]})")
            for (ax_val, bx_val), labels in cell_groups.items():
                ax.scatter(ax_val, bx_val, s=40, color=C["muted"],
                           marker="s", alpha=0.8, zorder=3)
                ax.annotate("/".join(labels), (ax_val, bx_val),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7.5, color=C["text"])
            # Nash equilibria.
            for k, (eu_r, eu_c) in enumerate(self._nash_payoffs()):
                ax.scatter(eu_r, eu_c, s=140, color=C["ne"], marker="*",
                           zorder=5, label="NE" if k == 0 else None)
            ax.set_xlabel(f"{self.row_name} payoff")
            ax.set_ylabel(f"{self.col_name} payoff")
            ax.set_title(title or f"{self.name} - payoff region")
            ax.margins(0.15)
            # The NE in CE in CCE relation lives in the (outside) legend title so
            # it never collides with the plot title or data points.
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      title="NE ⊆ CE ⊆ CCE")
        return fig, ax

    @staticmethod
    def plot_welfare_comparison(games, title: Optional[str] = None):
        """Grouped bar chart of max-welfare NE / CE / CCE across several games.

        ``games`` is a mapping ``name -> CorrelatedGame`` (or any iterable of
        games, in which case each game's ``.name`` labels its group). Returns
        ``(fig, ax)``.
        """
        if hasattr(games, "items"):
            items = list(games.items())
        else:
            items = [(getattr(g, "name", f"game {k}"), g) for k, g in enumerate(games)]
        names = [nm for nm, _ in items]
        ne_w, ce_w, cce_w = [], [], []
        for _, g in items:
            payoffs = g._nash_payoffs()
            best_ne = max((r + c for r, c in payoffs), default=float("nan"))
            ne_w.append(best_ne)
            ce = g.find_ce()
            ce_w.append(ce["welfare"] if ce else best_ne)
            cce = g.find_cce()
            cce_w.append(cce["welfare"] if cce else ce_w[-1])

        x = np.arange(len(names))
        width = 0.26
        with rc_context():
            fig, ax = plots.new_axes(figsize=(max(7.0, len(names) * 2.0), 5.0))
            ax.bar(x - width, ne_w, width, label="max-welfare NE", color=C["ne"], alpha=0.85)
            ax.bar(x, ce_w, width, label="max-welfare CE", color=C["ce"], alpha=0.85)
            ax.bar(x + width, cce_w, width, label="max-welfare CCE", color=C["cce"], alpha=0.85)
            ax.set_xticks(x, names, rotation=15, ha="right")
            ax.set_ylabel("social welfare (sum of payoffs)")
            ax.set_title(title or "Welfare comparison: NE subset CE subset CCE")
            ax.legend()
        return fig, ax

    def plot_regret(self, T: int = 2000, seed: int = 0, title: Optional[str] = None):
        """Average regret of both players under Hedge → 0 (Hannan)."""
        res = self.hedge(T=T, seed=seed)
        return plots.convergence(
            {"Row avg regret": res["avg_regret_row"],
             "Col avg regret": res["avg_regret_col"]},
            target=0.0, title=title or f"{self.name} - no-regret learning",
            ylabel="average regret")

    def __repr__(self) -> str:
        return f"CorrelatedGame({self.name!r}, shape={self.shape})"
