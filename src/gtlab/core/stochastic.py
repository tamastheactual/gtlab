"""Zero-sum stochastic (Markov) games solved by value iteration.

Also hosts the general-sum stochastic game (``GeneralSumSG``) and the binary
sender-receiver cheap-talk game (``CheapTalkGame``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import (C, fmt, fmt_prob, fmt_prob_vec, fmt_vec, html, plots,
                   rc_context)
from ..solvers.stochastic_extra import best_response_iteration, pure_values

# Per-state color palette built from the shared theme (no private CSS).
_STATE_COLORS = [C["p1"], C["p2"], C["ne"], C["accent"], C["ce"], C["cce"]]


def _state_color(s: int) -> str:
    return _STATE_COLORS[s % len(_STATE_COLORS)]


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
                                 ylabel="||TV - V|| inf")

    # ── core helpers ──────────────────────────────────────────────────────────
    def stage_game(self, s: int, V: np.ndarray) -> np.ndarray:
        """Stage-game matrix M_s(V) = r(s) + gamma * E_s'[V] for one state."""
        return solvers.stage_game(self.r[s], self.P[s], np.asarray(V, dtype=float),
                                  self.gamma)

    def value_iteration(self, tol: float = 1e-8, max_iter: int = 500) -> dict:
        """Alias for :meth:`solve` matching the notebook API."""
        return self.solve(tol=tol, max_iter=max_iter)

    def q_values(self, result: Optional[dict] = None) -> np.ndarray:
        """Q*(s,a,b) = r(s,a,b) + gamma * sum_s' P(s'|s,a,b) V*(s').

        This is the stage game M_s(V*) stacked across states - the same matrix
        Minimax-Q learns. Shape (nS, nA, nB).
        """
        if result is None:
            result = self.solve()
        V = result["V_star"]
        Q = np.zeros((self.nS, self.nA, self.nB))
        for s in range(self.nS):
            Q[s] = self.r[s] + self.gamma * np.einsum("abs,s->ab", self.P[s], V)
        return Q

    # ── display: value-iteration walkthrough ──────────────────────────────────
    def value_iteration_walkthrough(self, n_show: int = 4, tol: float = 1e-8,
                                    title: Optional[str] = None) -> dict:
        """Step-by-step value iteration showing each stage game and its value."""
        result = self.solve(tol=tol)
        hist = result["history"]
        resids = result["residuals"]
        V_star = result["V_star"]
        n_iter = result["n_iter"]
        parts = [html.note(
            "V(k+1)(s) = val(M_s(V_k)), M_s[a,b] = r(s,a,b) + "
            "gamma sum_s' P(s'|s,a,b) V_k(s').")]
        parts.append(html.kv([
            ("Converged:", f"{n_iter} iterations"),
            ("Final residual:", f"{resids[-1]:.2e}"),
            ("gamma:", fmt(self.gamma)),
            ("V0:", fmt_vec(hist[0])),
            ("V*:", fmt_vec(V_star)),
        ]))
        steps = []
        for k in range(min(n_show, n_iter)):
            V_k = hist[k]
            inner = [f"<b>k = {k}.</b> V_{k} = {fmt_vec(V_k)}, "
                     f"residual = {fmt(resids[k])}."]
            for s in range(self.nS):
                M = self.stage_game(s, V_k)
                sol = solvers.solve_zero_sum(M)
                tbl = html.table(
                    self.col_actions,
                    [[fmt(M[a, b]) for b in range(self.nB)] for a in range(self.nA)],
                    row_headers=self.row_actions)
                inner.append(
                    f'<span style="color:{_state_color(s)}"><b>State '
                    f'{self.state_names[s]}</b></span>: '
                    f'val = {fmt(sol["value"])}, p* = {fmt_prob_vec(sol["p"])}, '
                    f'q* = {fmt_prob_vec(sol["q"])} => V_{k+1}'
                    f'({self.state_names[s]}) = {fmt(sol["value"])}.{tbl}')
            inner.append(f"<b>V_{k+1} = {fmt_vec(hist[k + 1])}.</b>")
            steps.append("<br>".join(inner))
        if n_iter > n_show:
            steps.append(html.note(f"... {n_iter - n_show} more iterations ..."))
        conv = "; ".join(f"V*({self.state_names[s]}) = {fmt(V_star[s])}"
                         for s in range(self.nS))
        body = parts[0] + parts[1] + html.steps(steps) + html.note(
            f"Converged V*: {conv}")
        html.show(html.card(title or f"{self.name} - value iteration walkthrough",
                            body))
        return result

    # ── display: solve / stage games at V* ────────────────────────────────────
    def solve_stage_games(self, result: Optional[dict] = None,
                          title: Optional[str] = None) -> None:
        """Stage games at V* with best-response marking and saddle highlighting."""
        if result is None:
            result = self.solve()
        V_star = result["V_star"]
        policies = result["policies"]
        sections = [html.note(
            "M_s[a,b] = r(s,a,b) + gamma sum_s' P(s'|s,a,b) V*(s'). "
            "Underline = best response, outline = saddle point.")]
        for s in range(self.nS):
            M = self.stage_game(s, V_star)
            p = policies[s]["p"]
            q = policies[s]["q"]
            Mq = M @ q
            row_br = set(np.where(Mq >= Mq.max() - 1e-8)[0])
            saddles = {(a, b) for a in range(self.nA) for b in range(self.nB)
                       if M[a, b] == M[a, :].min() and M[a, b] == M[:, b].max()}
            cells, classes = [], []
            for a in range(self.nA):
                row, cls = [], []
                for b in range(self.nB):
                    txt = fmt(M[a, b])
                    if a in row_br:
                        txt = f'<span class="gt-br">{txt}</span>'
                    row.append(txt)
                    cls.append("gt-ne" if (a, b) in saddles else "")
                row.append(fmt_prob(p[a]))
                cls.append("gt-row")
                cells.append(row)
                classes.append(cls)
            qrow = [fmt_prob(q[b]) for b in range(self.nB)] + [fmt(float(p @ M @ q))]
            cells.append(qrow)
            classes.append(["gt-col"] * self.nB + [""])
            headers = list(self.col_actions) + ["p*"]
            rheads = list(self.row_actions) + ["q*"]
            tbl = html.table(headers, cells, row_headers=rheads,
                             cell_classes=classes)
            sections.append(
                f'<div style="color:{_state_color(s)}"><b>State '
                f'{self.state_names[s]}</b> &nbsp; V* = {fmt(V_star[s])}</div>'
                + tbl)
        html.show(html.card(title or f"{self.name} - stage games at V*",
                            "".join(sections)))

    # ── display: Bellman certificate ──────────────────────────────────────────
    def bellman_certificate(self, result: Optional[dict] = None,
                            title: Optional[str] = None) -> bool:
        """Verify V* satisfies Bellman, complementary slackness, no deviation."""
        if result is None:
            result = self.solve()
        V_star = result["V_star"]
        policies = result["policies"]
        all_pass = True
        steps = []
        for s in range(self.nS):
            M = self.stage_game(s, V_star)
            p = policies[s]["p"]
            q = policies[s]["q"]
            v = V_star[s]
            cs = solvers.complementary_slackness(M, p, q, v)
            v_c = float(p @ M @ q)
            ok1 = abs(v_c - v) < 1e-5
            Mq = M @ q
            pM = p @ M
            rd_ok = float(Mq.max()) <= v + 1e-5
            cd_ok = float(pM.min()) >= v - 1e-5
            passed = ok1 and cs["valid"] and rd_ok and cd_ok
            all_pass = all_pass and passed
            tag = ('<span style="color:%s">PASS</span>' % C["ne"]) if passed \
                else ('<span style="color:%s">FAIL</span>' % C["p2"])
            steps.append(
                f'<span style="color:{_state_color(s)}"><b>State '
                f'{self.state_names[s]}.</b></span> '
                f'Bellman p*Mq* = {fmt(v_c)} vs V* = {fmt(v)}; '
                f'no profitable deviation max(Mq*) = {fmt(float(Mq.max()))} <= V*, '
                f'min(p*M) = {fmt(float(pM.min()))} >= V*; '
                f'complementary slackness {"holds" if cs["valid"] else "fails"}. '
                f'{tag}')
        overall = ('<span style="color:%s"><b>ALL CHECKS PASSED</b></span>'
                   % C["ne"]) if all_pass else \
                  ('<span style="color:%s"><b>SOME CHECKS FAILED</b></span>'
                   % C["p2"])
        body = html.steps(steps) + html.note(
            "Each state: (1) Bellman equation, (2) complementary slackness on "
            "supported actions, (3) no profitable pure deviation.") + \
            f"<div>{overall}</div>"
        html.show(html.card(title or f"{self.name} - Bellman certificate", body))
        return bool(all_pass)

    # ── display + plot: exploitability ────────────────────────────────────────
    def exploitability_analysis(self, result: Optional[dict] = None,
                                n_deltas: int = 9, title: Optional[str] = None):
        """Loss when a player deviates delta toward uniform, opponent best-responds.

        Renders an HTML table and returns ``(fig, ax)``.
        """
        if result is None:
            result = self.solve()
        V_star = result["V_star"]
        policies = result["policies"]
        deltas = np.linspace(0, 1, n_deltas)
        p_unif = np.ones(self.nA) / self.nA
        q_unif = np.ones(self.nB) / self.nB
        row_expl = np.zeros((self.nS, n_deltas))
        col_expl = np.zeros((self.nS, n_deltas))
        for di, delta in enumerate(deltas):
            for s in range(self.nS):
                M = self.stage_game(s, V_star)
                p = policies[s]["p"]
                q = policies[s]["q"]
                v = V_star[s]
                p_d = (1 - delta) * p + delta * p_unif
                q_d = (1 - delta) * q + delta * q_unif
                row_expl[s, di] = v - float((p_d @ M).min())
                col_expl[s, di] = float((M @ q_d).max()) - v
        headers = ["delta"] + [f"Row expl {self.state_names[s]}" for s in range(self.nS)] \
            + [f"Col expl {self.state_names[s]}" for s in range(self.nS)]
        rows = []
        for di, delta in enumerate(deltas):
            row = ["0 (p*,q*)" if di == 0 else fmt(delta)]
            row += [fmt(row_expl[s, di]) for s in range(self.nS)]
            row += [fmt(col_expl[s, di]) for s in range(self.nS)]
            rows.append(row)
        tbl = html.table(headers, rows)
        html.show(html.card(title or f"{self.name} - exploitability",
                            html.note("p_delta = (1-delta) p* + delta * uniform "
                                      "(same for q). Exploitability is the loss "
                                      "when the opponent best-responds.") + tbl))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for s in range(self.nS):
                axes[0].plot(deltas, row_expl[s], color=_state_color(s),
                             marker="o", ms=4, label=self.state_names[s])
                axes[1].plot(deltas, col_expl[s], color=_state_color(s),
                             marker="s", ms=4, label=self.state_names[s])
            axes[0].set_title("Row deviates, Col best-responds")
            axes[1].set_title("Col deviates, Row best-responds")
            for ax in axes:
                ax.set_xlabel("Deviation delta toward uniform")
                ax.set_ylabel("Exploitability (stage-game level)")
                ax.legend()
            fig.suptitle(title or f"{self.name} - exploitability")
        return fig, axes

    # ── simulate + plot trajectories ──────────────────────────────────────────
    def simulate(self, result: Optional[dict] = None, s0: int = 0, T: int = 100,
                 K: int = 20, seed: int = 42) -> dict:
        """Simulate K Monte-Carlo trajectories under (pi*, sigma*).

        Returns ``{"states": (K,T) int, "rewards": (K,T) float}``.
        """
        if result is None:
            result = self.solve()
        policies = result["policies"]
        rng = np.random.default_rng(seed)
        states = np.zeros((K, T), dtype=int)
        rewards = np.zeros((K, T))
        for k in range(K):
            s = s0
            for t in range(T):
                p = policies[s]["p"]
                q = policies[s]["q"]
                a = int(rng.choice(self.nA, p=p))
                b = int(rng.choice(self.nB, p=q))
                rewards[k, t] = self.r[s, a, b]
                s = int(rng.choice(self.nS, p=self.P[s, a, b]))
                states[k, t] = s
        return {"states": states, "rewards": rewards}

    def plot_trajectories(self, result: Optional[dict] = None, s0: int = 0,
                          T: int = 150, K: int = 30, seed: int = 42,
                          title: Optional[str] = None):
        """State visitation + Cesaro average reward convergence. Returns (fig, axes)."""
        if result is None:
            result = self.solve()
        V_star = result["V_star"]
        sim = self.simulate(result=result, s0=s0, T=T, K=K, seed=seed)
        states = sim["states"]
        rewards = sim["rewards"]
        cesaro = np.cumsum(rewards, axis=1) / np.arange(1, T + 1)[np.newaxis, :]
        cem = cesaro.mean(axis=0)
        ces = cesaro.std(axis=0)
        sf = np.array([(states == s).mean(axis=1) for s in range(self.nS)]).T
        with rc_context():
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            ax = axes[0]
            for s in range(self.nS):
                ts = np.where(states[0] == s)[0]
                if len(ts):
                    ax.scatter(ts, [s] * len(ts), color=_state_color(s), s=15,
                               label=self.state_names[s], zorder=3)
            ax.set_yticks(range(self.nS))
            ax.set_yticklabels(self.state_names)
            ax.set_xlabel("Time step t")
            ax.set_title("State sequence (1 trajectory)")
            ax.legend(fontsize=8)
            ax2 = axes[1]
            for k in range(min(K, 6)):
                ax2.plot(cesaro[k], alpha=0.2, color=C["muted"], linewidth=0.8)
            ref = (1 - self.gamma) * V_star[s0]
            ax2.axhline(ref, color=C["p2"], linestyle="--", linewidth=1.5,
                        zorder=1, label=f"(1-gamma) V*(s0) = {fmt(ref)}")
            ax2.fill_between(range(T), cem - ces, cem + ces, alpha=0.2, color=C["ne"])
            ax2.plot(cem, color=C["ne"], linewidth=2, zorder=3,
                     label=f"Mean Cesàro (K={K})")
            lo = min(float((cem - ces).min()), ref)
            hi = max(float((cem + ces).max()), ref)
            pad = max(0.05, 0.1 * (hi - lo))
            ax2.set_ylim(lo - pad, hi + pad)
            ax2.set_xlabel("Time step t")
            ax2.set_ylabel("Cesàro avg reward")
            ax2.set_title(f"Empirical avg reward (K={K})")
            ax2.legend()
            ax3 = axes[2]
            means = sf.mean(axis=0)
            stds = sf.std(axis=0)
            x = np.arange(self.nS)
            ax3.bar(x, means, color=[_state_color(s) for s in range(self.nS)],
                    alpha=0.8)
            ax3.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4)
            ax3.set_xticks(x)
            ax3.set_xticklabels(self.state_names)
            ax3.set_ylabel("Fraction of time")
            ax3.set_title(f"Stationary distribution (K={K})")
            fig.suptitle(title or f"{self.name} - trajectories")
        return fig, axes

    # ── plot: Q heatmap ───────────────────────────────────────────────────────
    def plot_q_heatmap(self, result: Optional[dict] = None,
                       title: Optional[str] = None):
        """Heatmap of Q*(s,a,b) per state. Returns (fig, axes)."""
        Q = self.q_values(result)
        vmin, vmax = float(np.min(Q)), float(np.max(Q))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, self.nS, figsize=(4.5 * self.nS, 4.4),
                                     squeeze=False)
            for s, ax in enumerate(axes[0]):
                im = ax.imshow(Q[s], cmap="coolwarm", vmin=vmin, vmax=vmax,
                               aspect="auto")
                ax.set_xticks(range(self.nB))
                ax.set_xticklabels(self.col_actions)
                ax.set_yticks(range(self.nA))
                ax.set_yticklabels(self.row_actions)
                ax.set_xlabel("Col action")
                ax.set_ylabel("Row action")
                ax.set_title(f"Q*({self.state_names[s]})")
                ax.grid(False)
                for a in range(self.nA):
                    for b in range(self.nB):
                        mid = (vmin + vmax) / 2
                        col = "black" if abs(Q[s, a, b] - mid) < (vmax - vmin) * 0.35 \
                            else "white"
                        ax.text(b, a, f"{Q[s, a, b]:+.2f}", ha="center",
                                va="center", color=col, fontweight="bold")
                fig.colorbar(im, ax=ax, shrink=0.82)
            fig.suptitle(title or f"{self.name} - Q*(s,a,b)")
        return fig, axes

    # ── plot + table: gamma sweep ─────────────────────────────────────────────
    def gamma_sweep(self, gammas: Optional[Sequence[float]] = None,
                    title: Optional[str] = None):
        """Solve V* across discount factors. Renders a table, returns (fig, ax)."""
        if gammas is None:
            gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        V_arr = np.array([
            StochasticGame(self.r, self.P, gamma=g, state_names=self.state_names,
                           row_actions=self.row_actions,
                           col_actions=self.col_actions).solve()["V_star"]
            for g in gammas])
        headers = [f"V*({self.state_names[s]})" for s in range(self.nS)] + ["spread"]
        rows = [[fmt(V_arr[gi, s]) for s in range(self.nS)]
                + [fmt(V_arr[gi].max() - V_arr[gi].min())]
                for gi in range(len(gammas))]
        tbl = html.table(headers, rows, row_headers=[fmt(g) for g in gammas])
        html.show(html.card(title or f"{self.name} - gamma sweep",
                            html.note("V*(s) as the discount factor gamma varies.")
                            + tbl))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for s in range(self.nS):
                ax.plot(gammas, V_arr[:, s], color=_state_color(s), marker="o",
                        label=f"V*({self.state_names[s]})")
            ax.set_xlabel("Discount factor gamma")
            ax.set_ylabel("V*(s)")
            ax.set_title(title or f"{self.name} - gamma sweep")
            ax.legend()
        return fig, ax

    # ── plot + table: perturbation robustness ─────────────────────────────────
    def perturbation_robustness(self, n_trials: int = 50, delta_r: float = 0.5,
                                delta_P: float = 0.05, seed: int = 42,
                                title: Optional[str] = None):
        """Perturb r and P; show the V*(s) distribution. Returns (fig, axes)."""
        rng = np.random.default_rng(seed)
        V_base = self.solve()["V_star"]
        V_pert = np.zeros((n_trials, self.nS))
        for trial in range(n_trials):
            r_p = self.r + rng.uniform(-delta_r, delta_r, self.r.shape)
            P_p = np.clip(self.P + rng.uniform(-delta_P, delta_P, self.P.shape),
                          0, None)
            sums = P_p.sum(axis=3, keepdims=True)
            sums = np.where(sums < 1e-10, 1.0, sums)
            P_p = P_p / sums
            Gp = StochasticGame(r_p, P_p, gamma=self.gamma,
                                state_names=self.state_names,
                                row_actions=self.row_actions,
                                col_actions=self.col_actions)
            V_pert[trial] = Gp.solve(tol=1e-6)["V_star"]
        headers = ["V*(base)", "mean", "std", "min", "max", "95% CI"]
        rows = []
        for s in range(self.nS):
            vs = V_pert[:, s]
            lo, hi = np.percentile(vs, [2.5, 97.5])
            rows.append([fmt(V_base[s]), fmt(vs.mean()), fmt(vs.std()),
                         fmt(vs.min()), fmt(vs.max()), f"[{fmt(lo)}, {fmt(hi)}]"])
        tbl = html.table(headers, rows, row_headers=self.state_names)
        html.show(html.card(title or f"{self.name} - perturbation robustness",
                            html.note(f"Reward noise Uniform(+/-{fmt(delta_r)}), "
                                      f"transition noise Uniform(+/-{fmt(delta_P)}) "
                                      f"renormalised, {n_trials} trials.") + tbl))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, self.nS, figsize=(5 * self.nS, 4.5),
                                     squeeze=False)
            for s, ax in enumerate(axes[0]):
                ax.hist(V_pert[:, s], bins=20, color=_state_color(s), alpha=0.7,
                        edgecolor="white")
                ax.axvline(V_base[s], color="black", linestyle="--", linewidth=2,
                           label=f"V*(base) = {fmt(V_base[s])}")
                ax.set_xlabel(f"V*({self.state_names[s]})")
                ax.set_ylabel("Count")
                ax.set_title(f"State {self.state_names[s]}")
                ax.legend()
            fig.suptitle(title or f"{self.name} - perturbation robustness")
        return fig, axes

    # ── static: compare ───────────────────────────────────────────────────────
    @staticmethod
    def compare(games: Sequence["StochasticGame"],
                labels: Optional[Sequence[str]] = None,
                title: str = "Game comparison") -> list:
        """Side-by-side comparison table of multiple StochasticGame instances."""
        if labels is None:
            labels = [g.name or f"Game {i}" for i, g in enumerate(games)]
        results = [g.solve() for g in games]
        nS = games[0].nS
        snames = games[0].state_names
        rows = [["gamma"] + [fmt(g.gamma) for g in games]]
        for s in range(nS):
            rows.append([f"V*({snames[s]})"]
                        + [fmt(results[i]["V_star"][s]) for i in range(len(games))])
        rows.append(["V* spread"]
                    + [fmt(r["V_star"].max() - r["V_star"].min()) for r in results])
        rows.append(["iterations"] + [str(r["n_iter"]) for r in results])
        rows.append(["final residual"] + [f"{r['residuals'][-1]:.2e}" for r in results])
        for s in range(nS):
            rows.append([f"pi*({snames[s]})"]
                        + [fmt_prob_vec(r["policies"][s]["p"]) for r in results])
            rows.append([f"sigma*({snames[s]})"]
                        + [fmt_prob_vec(r["policies"][s]["q"]) for r in results])
        body = html.table(["metric"] + list(labels), rows)
        html.show(html.card(title, body))
        return results

    def __repr__(self) -> str:
        return f"StochasticGame({self.name!r}, S={self.nS})"


# ── GeneralSumSG ──────────────────────────────────────────────────────────────
@dataclass
class GeneralSumSG:
    """A finite general-sum (non-zero-sum) stochastic game.

    Two reward tensors ``r1`` (row) and ``r2`` (column), both shape (S, A, B),
    plus a transition tensor ``P`` of shape (S, A, B, S). Best-response
    iteration approximates a Markov Perfect Equilibrium when it converges.
    """

    r1: np.ndarray
    r2: np.ndarray
    P: np.ndarray
    gamma: float = 0.9
    state_names: Optional[Sequence[str]] = None
    row_actions: Optional[Sequence[str]] = None
    col_actions: Optional[Sequence[str]] = None
    name: str = "General-sum stochastic game"

    def __post_init__(self) -> None:
        self.r1 = np.asarray(self.r1, dtype=float)
        self.r2 = np.asarray(self.r2, dtype=float)
        self.P = np.asarray(self.P, dtype=float)
        nS, nA, nB = self.r1.shape
        if self.r2.shape != (nS, nA, nB):
            raise ValueError(f"r2 shape {self.r2.shape} != {(nS, nA, nB)}")
        if self.P.shape != (nS, nA, nB, nS):
            raise ValueError(f"P shape {self.P.shape} != {(nS, nA, nB, nS)}")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        self.nS, self.nA, self.nB = nS, nA, nB
        self.state_names = list(self.state_names) if self.state_names else \
            [f"s{i}" for i in range(nS)]
        self.row_actions = list(self.row_actions) if self.row_actions else \
            [f"a{i}" for i in range(nA)]
        self.col_actions = list(self.col_actions) if self.col_actions else \
            [f"b{j}" for j in range(nB)]

    @cached_method
    def best_response_iteration(self, tol: float = 1e-6,
                                max_iter: int = 300) -> dict:
        """Gauss-Seidel best-response dynamics (approximate Markov Perfect Eq)."""
        return best_response_iteration(self.r1, self.r2, self.P, self.gamma,
                                       tol=tol, max_iter=max_iter)

    def pure_values(self, a1_sel: Sequence[int], a2_sel: Sequence[int]):
        """Discounted (V1, V2) of a fixed pure stationary joint policy."""
        return pure_values(self.r1, self.r2, self.P, self.gamma, a1_sel, a2_sel)

    def welfare(self, result: Optional[dict] = None) -> dict:
        """Utilitarian / egalitarian / Nash-product welfare on the joint value."""
        if result is None:
            result = self.best_response_iteration()
        V1, V2 = result["V1"], result["V2"]
        return {
            "utilitarian": V1 + V2,
            "egalitarian": np.minimum(V1, V2),
            "nash_product": V1 * V2,
            "V1": V1, "V2": V2,
        }

    def summary(self, title: Optional[str] = None) -> None:
        result = self.best_response_iteration()
        sections = [html.note(
            "Payoffs are (Row, Col), not necessarily zero-sum. Best-response "
            "iteration approximates a pure Markov Perfect Equilibrium.")]
        for s in range(self.nS):
            cells = [[f'({fmt(self.r1[s, a, b])}, {fmt(self.r2[s, a, b])})'
                      for b in range(self.nB)] for a in range(self.nA)]
            tbl = html.table(self.col_actions, cells, row_headers=self.row_actions)
            sections.append(
                f'<div style="color:{_state_color(s)}"><b>State '
                f'{self.state_names[s]}</b></div>' + tbl)
        headers = ["V1* (Row)", "V2* (Col)"] \
            + [f"pi*({self.row_actions[a]})" for a in range(self.nA)] \
            + [f"sigma*({self.col_actions[b]})" for b in range(self.nB)]
        rows = []
        for s in range(self.nS):
            row = [fmt(result["V1"][s]), fmt(result["V2"][s])]
            row += [fmt_prob(result["pi"][s, a]) for a in range(self.nA)]
            row += [fmt_prob(result["sig"][s, b]) for b in range(self.nB)]
            rows.append(row)
        sections.append("<b>Nash approximation</b>"
                        + html.table(headers, rows, row_headers=self.state_names))
        html.show(html.card(title or self.name, "".join(sections)))

    def explain(self, title: Optional[str] = None) -> None:
        """Explain why the zero-sum contraction argument does not apply here."""
        res = self.best_response_iteration()
        items = [
            "<b>Two reward tensors, two Bellman equations.</b> For each player i, "
            "V_i*(s) = E_(a,b)~(pi*,sigma*)[ r_i(s,a,b) + gamma sum_s' "
            "P(s'|s,a,b) V_i*(s') ].",
            "<b>No contraction.</b> In zero-sum games the Shapley operator is a "
            "gamma-contraction because each minimax reply is uniquely pinned down "
            "by the opponent. With general-sum payoffs that symmetry breaks: the "
            "best-response map need not be a contraction and can cycle.",
            "<b>What best_response_iteration does.</b> It alternates per-state best "
            "responses Gauss-Seidel style until policies stop changing. On "
            "convergence the result is an approximate Markov Perfect Equilibrium; "
            "which one depends on the initial policy.",
            f"<b>This run.</b> "
            f"{'converged' if res['converged'] else 'did not converge'} in "
            f"{res['n_iter']} iterations to V1* = {fmt_vec(res['V1'])}, "
            f"V2* = {fmt_vec(res['V2'])}.",
        ]
        html.show(html.card(title or f"{self.name} - why no contraction",
                            html.steps(items)))

    def __repr__(self) -> str:
        return f"GeneralSumSG({self.name!r}, S={self.nS})"


# ── CheapTalkGame ─────────────────────────────────────────────────────────────
class CheapTalkGame:
    """Binary-type sender-receiver cheap-talk game.

    Sender observes a type theta in {H, L}, sends a costless message, and the
    Receiver chooses Accept or Reject. Solves for pooling / separating /
    babbling equilibria as a function of the prior P(H).
    """

    def __init__(self, u_R_HH: float = 1, u_R_HL: float = -1, u_R_LH: float = -0.5,
                 u_R_LL: float = 1, u_S_H_accept: float = 2, u_S_H_reject: float = 0,
                 u_S_L_accept: float = 0.4, u_S_L_reject: float = 0,
                 name: str = "Cheap-talk game"):
        self.u_R_HH = u_R_HH
        self.u_R_HL = u_R_HL
        self.u_R_LH = u_R_LH
        self.u_R_LL = u_R_LL
        self.u_S_H_accept = u_S_H_accept
        self.u_S_H_reject = u_S_H_reject
        self.u_S_L_accept = u_S_L_accept
        self.u_S_L_reject = u_S_L_reject
        self.name = name
        denom = (u_R_HH - u_R_LH) - (u_R_HL - u_R_LL)
        self.mu_thr = float(np.clip((u_R_LL - u_R_HL) / denom, 0, 1)) \
            if abs(denom) > 1e-10 else 0.5

    def _Ra(self, mu: float) -> float:
        return mu * self.u_R_HH + (1 - mu) * self.u_R_HL

    def _Rr(self, mu: float) -> float:
        return mu * self.u_R_LH + (1 - mu) * self.u_R_LL

    def _act(self, mu: float) -> str:
        return "Accept" if self._Ra(mu) >= self._Rr(mu) else "Reject"

    def _Sp(self, act: str, tp: str) -> float:
        if tp == "H":
            return self.u_S_H_accept if act == "Accept" else self.u_S_H_reject
        return self.u_S_L_accept if act == "Accept" else self.u_S_L_reject

    def equilibrium(self, prior_H: float = 0.5) -> dict:
        """Classify pooling / babbling / separating equilibria at a given prior."""
        act_pool = self._act(prior_H)
        act_off = self._act(0.0)
        H_p = self._Sp(act_pool, "H")
        H_d = self._Sp(act_off, "H")
        L_p = self._Sp(act_pool, "L")
        L_d = self._Sp(act_off, "L")
        no_H = H_d <= H_p
        no_L = L_d <= L_p
        is_pool = no_H and no_L and act_pool == "Accept"
        is_bab = no_H and no_L and act_pool == "Reject"
        as_H = self._act(1.0)
        as_L = self._act(0.0)
        H_on = self._Sp(as_H, "H")
        H_m = self._Sp(as_L, "H")
        L_on = self._Sp(as_L, "L")
        L_m = self._Sp(as_H, "L")
        ic_H = H_m <= H_on
        ic_L = L_m <= L_on
        return {"prior_H": prior_H, "act_pool": act_pool, "act_off": act_off,
                "is_pooling": is_pool, "is_babbling": is_bab,
                "as_H": as_H, "as_L": as_L, "ic_H": ic_H, "ic_L": ic_L,
                "is_separating": ic_H and ic_L,
                "H_pool": H_p, "L_pool": L_p, "H_dev": H_d, "L_dev": L_d,
                "H_on": H_on, "H_mimic": H_m, "L_on": L_on, "L_mimic": L_m}

    def summary(self, title: Optional[str] = None) -> None:
        r_tbl = html.table(["Accept", "Reject"],
                           [[fmt(self.u_R_HH), fmt(self.u_R_LH)],
                            [fmt(self.u_R_HL), fmt(self.u_R_LL)]],
                           row_headers=["H type", "L type"])
        s_tbl = html.table(["Accept", "Reject"],
                           [[fmt(self.u_S_H_accept), fmt(self.u_S_H_reject)],
                            [fmt(self.u_S_L_accept), fmt(self.u_S_L_reject)]],
                           row_headers=["H type", "L type"])
        diff_H = self.u_S_H_accept - self.u_S_H_reject
        diff_L = self.u_S_L_accept - self.u_S_L_reject
        if diff_L > 0 and diff_H > diff_L:
            cond = (f"H type values Accept more than L type (gain H = {fmt(diff_H)}, "
                    f"L = {fmt(diff_L)}); single-crossing may allow separation.")
        elif diff_L <= 0:
            cond = (f"L type prefers Reject (gain = {fmt(diff_L)}); interests "
                    "partially aligned with the Receiver.")
        else:
            cond = "Both types strongly prefer Accept; separation unlikely."
        body = html.note(
            "Sender observes theta in {H, L}, sends a costless message, Receiver "
            "chooses Accept or Reject.") \
            + html.note(f"Acceptance threshold mu* = {fmt(self.mu_thr)} "
                        f"(Receiver Accepts iff P(H|m) >= {fmt(self.mu_thr)}).") \
            + "<b>Receiver payoffs u_R(theta, a)</b>" + r_tbl \
            + "<b>Sender payoffs u_S(theta, a)</b>" + s_tbl \
            + html.note(cond)
        html.show(html.card(title or self.name, body))

    def equilibrium_analysis(self, prior_H: float = 0.5,
                             title: Optional[str] = None) -> dict:
        """HTML walkthrough of pooling and separating equilibrium checks."""
        eq = self.equilibrium(prior_H)

        def tag(ok: bool) -> str:
            color = C["ne"] if ok else C["p2"]
            return f'<span style="color:{color}">{"PASS" if ok else "FAIL"}</span>'

        H_ok = eq["H_dev"] <= eq["H_pool"]
        L_ok = eq["L_dev"] <= eq["L_pool"]
        if eq["is_pooling"]:
            pool_v = f'<span style="color:{C["ne"]}">EXISTS (informative pooling)</span>'
        elif eq["is_babbling"]:
            pool_v = (f'<span style="color:{C["cce"]}">BABBLING (Receiver Rejects '
                      'on prior)</span>')
        else:
            pool_v = f'<span style="color:{C["p2"]}">DOES NOT EXIST</span>'
        pooling = [
            f"Both types send the same message; Receiver posterior = prior = "
            f"{fmt(prior_H)}.",
            f"R(Accept|{fmt(prior_H)}) = {fmt(self._Ra(prior_H))}, "
            f"R(Reject|{fmt(prior_H)}) = {fmt(self._Rr(prior_H))} => Receiver plays "
            f"<b>{eq['act_pool']}</b>.",
            f"Off-path message: mu = 0 => Receiver plays <b>{eq['act_off']}</b>.",
            f"H-type IC: on-path {fmt(eq['H_pool'])} >= dev {fmt(eq['H_dev'])} "
            f"{tag(H_ok)}.",
            f"L-type IC: on-path {fmt(eq['L_pool'])} >= dev {fmt(eq['L_dev'])} "
            f"{tag(L_ok)}.",
            f"<b>Pooling equilibrium: {pool_v}</b>",
        ]
        sep_v = (f'<span style="color:{C["ne"]}">EXISTS</span>' if eq["is_separating"]
                 else f'<span style="color:{C["p2"]}">DOES NOT EXIST</span>')
        separating = [
            f"H sends m_H => mu = 1 => <b>{eq['as_H']}</b>; "
            f"L sends m_L => mu = 0 => <b>{eq['as_L']}</b>.",
            f"H-type IC: on-path {fmt(eq['H_on'])} >= mimic-L {fmt(eq['H_mimic'])} "
            f"{tag(eq['ic_H'])}.",
            f"L-type IC: on-path {fmt(eq['L_on'])} >= mimic-H {fmt(eq['L_mimic'])} "
            f"{tag(eq['ic_L'])}"
            f"{'' if eq['ic_L'] else ' (L mimics H, separating unravels)'}.",
            f"<b>Separating equilibrium: {sep_v}</b>",
        ]

        def mark(b: bool, label: str) -> str:
            return (f'<span style="color:{C["ne"]}">{label}</span>' if b
                    else f'<span style="color:{C["muted"]}">{label}: no</span>')

        body = "<b>Pooling equilibrium</b>" + html.steps(pooling) \
            + "<b>Separating equilibrium</b>" + html.steps(separating) \
            + html.note(f"P(H) = {fmt(prior_H)}: {mark(eq['is_pooling'], 'Pooling')} "
                        f"&middot; {mark(eq['is_babbling'], 'Babbling')} "
                        f"&middot; {mark(eq['is_separating'], 'Separating')}")
        html.show(html.card(
            title or f"{self.name} - equilibrium analysis (P(H) = {fmt(prior_H)})",
            body))
        return eq

    def plot_equilibrium_regions(self, title: Optional[str] = None):
        """Equilibrium existence regions vs the prior P(H). Returns (fig, ax)."""
        priors = np.linspace(0.01, 0.99, 300)
        pool, bab, sep = [], [], []
        for ph in priors:
            eq = self.equilibrium(ph)
            pool.append(int(eq["is_pooling"]))
            bab.append(int(eq["is_babbling"]))
            sep.append(int(eq["is_separating"]))
        with rc_context():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.fill_between(priors, 0, pool, alpha=0.5, color=C["p1"],
                            label="Pooling (Receiver Accepts)")
            ax.fill_between(priors, 0, sep, alpha=0.5, color=C["ne"],
                            label="Separating")
            ax.fill_between(priors, 0, bab, alpha=0.4, color=C["p2"],
                            label="Babbling (Receiver Rejects)")
            ax.axvline(self.mu_thr, color=C["muted"], linestyle="--", linewidth=1.5,
                       label=f"mu* = {fmt(self.mu_thr)}")
            ax.set_xlabel("Prior P(High type)")
            ax.set_ylabel("Equilibrium exists (0/1)")
            ax.set_title(title or f"{self.name} - equilibrium regions vs prior")
            ax.set_ylim(-0.05, 1.5)
            ax.set_yticks([0, 1])
            ax.legend()
        return fig, ax

    def __repr__(self) -> str:
        return f"CheapTalkGame({self.name!r})"
