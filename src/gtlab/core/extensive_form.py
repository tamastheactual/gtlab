"""Extensive-form games: game trees, backward induction, welfare.

Two node schemas are accepted and normalized on construction.

Canonical (compact) schema::

    decision : {"player": 0, "actions": {"L": "n1", "R": "n2"}}
    chance   : {"chance": {"L": (0.5, "n1"), "R": (0.5, "n2")}}
    terminal : {"payoff": (3.0, 1.0)}

Notebook (verbose) schema, useful for information sets and richer trees::

    decision : {"player": 0, "actions": ["L", "R"],
                "children": {"L": "n1", "R": "n2"}}
    chance   : {"player": "chance", "actions": [...], "children": {...},
                "prob": {action: p}}
    terminal : {"is_terminal": True, "payoffs": (3.0, 1.0)}

Internally everything is stored in the verbose form, so both styles share one
analysis/visualisation pipeline. The class wires the pure algorithms in
:mod:`gtlab.solvers` (and :mod:`gtlab.solvers.extensive_form_extra`) into the
shared :mod:`gtlab.viz` layer.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..solvers import extensive_form_extra as efx
from ..viz import C, fmt, fmt_prob, fmt_vec, html
from ..viz import plots
from ..viz.theme import rc_context


@dataclass
class ExtensiveFormGame:
    """A finite extensive-form game over a tree of decision/chance/terminal nodes."""

    tree: Dict[str, Dict[str, Any]]
    root: str = "root"
    players: Optional[List[str]] = None
    name: str = "Extensive-form game"
    info_sets: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.root not in self.tree:
            raise ValueError(f"root node {self.root!r} not in tree")
        self.tree = {nid: self._normalize_node(nid, node)
                     for nid, node in self.tree.items()}
        n_players = self._infer_n_players()
        self.players = list(self.players) if self.players else [
            f"P{i + 1}" for i in range(n_players)]
        self.info_sets = dict(self.info_sets) if self.info_sets else {}
        self._validate()

    # ── schema normalisation ─────────────────────────────────────────────────
    @staticmethod
    def _normalize_node(nid: str, node: Dict[str, Any]) -> Dict[str, Any]:
        """Convert either accepted schema into the internal verbose form."""
        # Already verbose (terminal).
        if node.get("is_terminal", False) or "payoffs" in node:
            payoffs = node.get("payoffs", node.get("payoff"))
            return {"is_terminal": True, "payoffs": tuple(payoffs)}
        # Compact terminal.
        if "payoff" in node:
            return {"is_terminal": True, "payoffs": tuple(node["payoff"])}
        # Compact chance.
        if "chance" in node and "actions" not in node:
            actions = list(node["chance"].keys())
            children = {a: child for a, (_p, child) in node["chance"].items()}
            prob = {a: float(p) for a, (p, _c) in node["chance"].items()}
            return {"player": "chance", "actions": actions,
                    "children": children, "prob": prob, "is_terminal": False}
        # Decision / verbose chance.
        out: Dict[str, Any] = {"is_terminal": False,
                               "player": node.get("player")}
        actions = node["actions"]
        if isinstance(actions, dict):  # compact decision: actions == children map
            out["actions"] = list(actions.keys())
            out["children"] = dict(actions)
        else:  # verbose: separate actions list + children dict
            out["actions"] = list(actions)
            out["children"] = dict(node["children"])
        if node.get("player") == "chance":
            out["prob"] = node.get("prob")
        return out

    def _infer_n_players(self) -> int:
        n = 2
        for node in self.tree.values():
            if node.get("is_terminal", False):
                n = max(n, len(node["payoffs"]))
        return n

    def _validate(self) -> None:
        for nid, node in self.tree.items():
            if node.get("is_terminal", False):
                if len(node["payoffs"]) != len(self.players):
                    raise ValueError(
                        f"terminal {nid!r}: payoffs length != #players")
                continue
            if set(node["actions"]) != set(node["children"].keys()):
                raise ValueError(
                    f"node {nid!r}: actions do not match children keys")
            for child in node["children"].values():
                if child not in self.tree:
                    raise ValueError(f"node {nid!r} -> missing child {child!r}")
        for set_id, nodes in self.info_sets.items():
            if not nodes:
                raise ValueError(f"info set {set_id!r} must have >= 1 node")
            first = self.tree[nodes[0]]
            for n in nodes[1:]:
                if (self.tree[n].get("player") != first.get("player") or
                        set(self.tree[n]["actions"]) != set(first["actions"])):
                    raise ValueError(
                        f"info set {set_id!r}: nodes must share player + actions")
        self._check_cycles()

    def _check_cycles(self) -> None:
        def walk(node: str, ancestors: frozenset) -> None:
            if node in ancestors:
                raise ValueError(f"circular reference at node {node!r}")
            data = self.tree.get(node)
            if data and not data.get("is_terminal", False):
                for child in data["children"].values():
                    walk(child, ancestors | {node})
        walk(self.root, frozenset())

    # ── helpers ──────────────────────────────────────────────────────────────
    def _player_name(self, value) -> str:
        if isinstance(value, int) and 0 <= value < len(self.players):
            return self.players[value]
        return str(value)

    def _node_info_set(self, node: str) -> Optional[str]:
        for set_id, nodes in self.info_sets.items():
            if node in nodes:
                return set_id
        return None

    def _has_imperfect_info(self) -> bool:
        return any(len(v) > 1 for v in self.info_sets.values())

    # ── backward induction ───────────────────────────────────────────────────
    @cached_method
    def backward_induction(self, tol: float = 1e-9) -> Dict[str, Any]:
        """Compute the subgame-perfect equilibrium by backward induction.

        Returns ``{"value": payoff_at_root, "strategy": {node: action},
        "all_actions": {node: [tied actions]}}``.
        """
        if self._has_imperfect_info():
            bad = [k for k, v in self.info_sets.items() if len(v) > 1]
            warnings.warn(
                "backward induction assumes perfect information; non-trivial "
                f"information sets present: {bad}", stacklevel=2)
        strategy: Dict[str, str] = {}
        all_actions: Dict[str, List[str]] = {}

        def value(nid: str) -> np.ndarray:
            node = self.tree[nid]
            if node.get("is_terminal", False):
                return np.asarray(node["payoffs"], dtype=float)
            if node.get("player") == "chance":
                probs = node.get("prob") or {
                    a: 1.0 / len(node["actions"]) for a in node["actions"]}
                out = np.zeros(len(self.players))
                for a, child in node["children"].items():
                    out += probs.get(a, 0.0) * value(child)
                return out
            player = node["player"]
            best_val, best_actions, best_util = None, [], -math.inf
            for action, child in node["children"].items():
                v = value(child)
                u = v[player]
                if u > best_util + tol:
                    best_util, best_val, best_actions = u, v, [action]
                elif abs(u - best_util) <= tol:
                    best_actions.append(action)
            strategy[nid] = best_actions[0]
            all_actions[nid] = best_actions
            return best_val

        root_value = value(self.root)
        return {"value": root_value, "strategy": strategy,
                "all_actions": all_actions}

    def simulate(self, strategy: Optional[Dict[str, str]] = None,
                 rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Play one path from the root and return the realised outcome.

        ``strategy`` maps decision-node ids to chosen actions (defaults to the
        SPE strategy). Chance nodes are sampled with their probabilities.
        """
        if strategy is None:
            strategy = self.backward_induction()["strategy"]
        rng = rng or np.random.default_rng()
        path: List[Tuple[str, str]] = []
        node = self.root
        while not self.tree[node].get("is_terminal", False):
            data = self.tree[node]
            if data.get("player") == "chance":
                probs = data.get("prob") or {
                    a: 1.0 / len(data["actions"]) for a in data["actions"]}
                acts = list(data["actions"])
                weights = np.array([probs.get(a, 0.0) for a in acts], dtype=float)
                weights = weights / weights.sum()
                action = acts[int(rng.choice(len(acts), p=weights))]
            else:
                action = strategy.get(node, data["actions"][0])
            path.append((node, action))
            node = data["children"][action]
        return {"path": path, "terminal": node,
                "payoffs": np.asarray(self.tree[node]["payoffs"], dtype=float)}

    # ── terminal payoffs / welfare ───────────────────────────────────────────
    def terminal_payoffs(self) -> np.ndarray:
        """All terminal payoff vectors as an ``(N, players)`` array."""
        out = []

        def walk(node: str) -> None:
            data = self.tree[node]
            if data.get("is_terminal", False):
                out.append(data["payoffs"])
            else:
                for child in data["children"].values():
                    walk(child)
        walk(self.root)
        return np.array(out, dtype=float)

    def joint_payoffs(self) -> np.ndarray:
        return self.terminal_payoffs()

    def pareto_frontier(self) -> np.ndarray:
        return solvers.pareto_frontier(self.terminal_payoffs())

    def pareto_outcomes(self) -> List[Tuple[float, ...]]:
        return efx.pareto_outcomes(self.terminal_payoffs())

    def pareto_frontier_vertices(self) -> List[Tuple[float, float]]:
        return efx.pareto_frontier_vertices(self.terminal_payoffs())

    def social_welfare(self, objective: str = "utilitarian") -> Tuple[np.ndarray, float]:
        """Best terminal outcome and its welfare score for the given objective."""
        outcomes = self.terminal_payoffs()
        idx = solvers.best_outcome(outcomes, objective)
        score = {"utilitarian": solvers.utilitarian,
                 "egalitarian": solvers.egalitarian,
                 "nash": solvers.nash_welfare}[objective](outcomes[idx])
        return outcomes[idx], float(score)

    def price_of_anarchy(self) -> float:
        """Optimal utilitarian welfare divided by the SPE's total payoff."""
        _, opt = self.social_welfare("utilitarian")
        eq_w = float(np.sum(self.backward_induction()["value"]))
        if abs(eq_w) < 1e-12:
            return math.inf if opt > 0 else 1.0
        return opt / eq_w

    # ── induced normal-form analysis ─────────────────────────────────────────
    @cached_method
    def convert_to_normal(self) -> Tuple[np.ndarray, np.ndarray]:
        """Induce the two-player bimatrix ``(A, B)`` from the tree."""
        return efx.convert_to_normal(self.tree, self.info_sets, len(self.players))

    def _nf(self) -> Optional[Dict[str, Any]]:
        if len(self.players) != 2:
            return None
        return efx.normal_form_labels(self.tree, self.info_sets, len(self.players))

    def as_dataframe(self):
        """Return the induced normal-form payoff matrix as a pandas DataFrame."""
        import pandas as pd
        nf = self._nf()
        A, B = nf["A"], nf["B"]
        idx = pd.Index(nf["row_labels"], name=f"{self.players[0]} strategy")
        cols = pd.Index(nf["col_labels"], name=f"{self.players[1]} strategy")
        return pd.DataFrame(
            [[(A[i, j], B[i, j]) for j in range(A.shape[1])]
             for i in range(A.shape[0])], index=idx, columns=cols)

    def is_zero_sum(self, tol: float = 1e-9) -> bool:
        if len(self.players) != 2:
            return False
        A, B = self.convert_to_normal()
        return bool(np.all(np.abs(A + B) <= tol))

    def is_symmetric(self, tol: float = 1e-9) -> bool:
        if len(self.players) != 2:
            return False
        A, B = self.convert_to_normal()
        if A.shape[0] != A.shape[1]:
            return False
        return bool(np.all(np.abs(A - B.T) <= tol))

    def best_responses(self):
        """``(br_row, br_col)`` boolean masks over the induced bimatrix."""
        A, B = self.convert_to_normal()
        return solvers.br_masks(A, B)

    def best_responses_row(self, j: int) -> List[int]:
        A, _ = self.convert_to_normal()
        return efx.best_responses_row(A, j)

    def best_responses_col(self, i: int) -> List[int]:
        _, B = self.convert_to_normal()
        return efx.best_responses_col(B, i)

    def strictly_dominated_rows(self) -> Dict[int, List[int]]:
        A, _ = self.convert_to_normal()
        return efx.strictly_dominated_rows(A)

    def strictly_dominated_cols(self) -> Dict[int, List[int]]:
        _, B = self.convert_to_normal()
        return efx.strictly_dominated_cols(B)

    def pure_nash_nf(self) -> List[Tuple[int, int]]:
        A, B = self.convert_to_normal()
        return efx.pure_nash(A, B)

    def pareto_optimal_nf(self) -> List[Tuple[int, int]]:
        A, B = self.convert_to_normal()
        return efx.pareto_optimal(A, B)

    def iesds_strict(self, max_rounds: int = 50):
        """Run IESDS; returns ``(A, B, row_labels, col_labels, rounds)``."""
        nf = self._nf()
        A, B, rl, cl, rounds = efx.iesds_log(
            nf["A"], nf["B"], nf["row_labels"], nf["col_labels"], max_rounds)
        return A, B, rl, cl, rounds

    def mixed_equilibria(self):
        A, B = self.convert_to_normal()
        return efx.mixed_equilibria(A, B)

    # ── display: summary / explain / solve ───────────────────────────────────
    def _decision_rows(self):
        rows = []
        for nid, data in self.tree.items():
            if data.get("is_terminal", False):
                continue
            rows.append([
                nid,
                self._player_name(data.get("player", "-")),
                ", ".join(data["actions"]),
                self._node_info_set(nid) or "-",
            ])
        return rows

    def _terminal_rows(self):
        rows = []
        for nid, data in self.tree.items():
            if data.get("is_terminal", False):
                rows.append([nid, fmt_vec(data["payoffs"])])
        return rows

    def _summary_html(self) -> str:
        n_decision = sum(1 for d in self.tree.values()
                         if not d.get("is_terminal", False))
        n_terminal = len(self.tree) - n_decision
        info = "Imperfect" if self._has_imperfect_info() else "Perfect"
        props = [f"Players: {', '.join(self.players)}",
                 f"Decision nodes: {n_decision}",
                 f"Terminal nodes: {n_terminal}",
                 f"Information: {info}"]
        if len(self.players) == 2:
            if self.is_zero_sum():
                props.append("zero-sum")
            if self.is_symmetric():
                props.append("symmetric")
        body = html.legend(*props)
        body += html.table(["node", "player", "actions", "info set"],
                           self._decision_rows())
        body += html.table(["terminal", "payoffs"], self._terminal_rows())
        pm = self._matrix_html(show_ne=True)
        if pm:
            body += pm
        return body

    def summary(self, title: Optional[str] = None) -> None:
        """Compact overview: nodes, terminals, induced matrix with NE marked."""
        html.show(html.card(title or self.name, self._summary_html()))

    def explain(self, title: Optional[str] = None) -> None:
        """Step-by-step backward-induction walkthrough."""
        res = self.backward_induction()
        items = [
            "<b>Step 1 - Start at the leaves.</b> Terminal nodes already carry "
            "payoffs.",
            "<b>Step 2 - Fold the tree.</b> At each decision node the acting "
            "player picks the action leading to the child with the highest "
            "payoff <i>for them</i>; that child's payoff vector propagates up.",
            f"<b>Step 3 - Read the root.</b> The subgame-perfect equilibrium "
            f"yields payoff {fmt_vec(res['value'])} (chosen actions tabulated "
            "above).",
        ]
        body = self._solution_html() + html.steps(items)
        html.show(html.card(title or f"{self.name} - backward induction", body))

    def _solution_html(self) -> str:
        res = self.backward_induction()
        rows = []
        for nid, acts in res["all_actions"].items():
            player = self._player_name(self.tree[nid].get("player"))
            chosen = acts[0] if len(acts) == 1 else f"{', '.join(acts)} (tied)"
            rows.append([nid, player, chosen])
        tbl = html.table(["node", "player", "chosen action"], rows)
        return html.kv([("SPE payoff", fmt_vec(res["value"]))]) + tbl

    def solve(self, title: Optional[str] = None) -> None:
        """Show the backward-induction (SPE) solution."""
        html.show(html.card(title or f"{self.name} - backward induction",
                            self._solution_html()))

    # ── induced normal-form rich rendering (NormalFormGame flag set) ─────────
    def _matrix_html(self, show_br: bool = False, show_ne: bool = False,
                     show_pareto: bool = False, show_dominated: bool = False) -> str:
        """Annotated induced-bimatrix table; empty string if not two-player."""
        nf = self._nf()
        if nf is None:
            return ""
        A, B = nf["A"], nf["B"]
        if A.shape[0] > 8 or A.shape[1] > 8:
            return ""
        row_labels, col_labels = nf["row_labels"], nf["col_labels"]
        m, n = A.shape
        br_row, br_col = (solvers.br_masks(A, B) if show_br else (None, None))
        ne = set(efx.pure_nash(A, B)) if show_ne else set()
        pareto = set(efx.pareto_optimal(A, B)) if show_pareto else set()
        dom_rows = set(efx.strictly_dominated_rows(A)) if show_dominated else set()
        dom_cols = set(efx.strictly_dominated_cols(B)) if show_dominated else set()
        rows, classes = [], []
        for i in range(m):
            r, c = [], []
            for j in range(n):
                a_cls = "gt-row"
                if show_br and br_row[i, j]:
                    a_cls += " gt-br"
                if i in dom_rows:
                    a_cls += " gt-dom"
                b_cls = "gt-col"
                if show_br and br_col[i, j]:
                    b_cls += " gt-br"
                if j in dom_cols:
                    b_cls += " gt-dom"
                r.append(f'<span class="{a_cls}">{fmt(A[i, j])}</span>, '
                         f'<span class="{b_cls}">{fmt(B[i, j])}</span>')
                cell = []
                if (i, j) in ne:
                    cell.append("gt-ne")
                if (i, j) in pareto:
                    cell.append("gt-pareto")
                c.append(" ".join(cell))
            rows.append(r)
            classes.append(c)
        return html.table(col_labels, rows, row_headers=row_labels,
                          cell_classes=classes)

    def show_payoff_matrix(self, title: Optional[str] = None) -> None:
        """Render the induced normal-form payoff matrix."""
        body = self._matrix_html()
        if not body:
            body = html.note("Matrix unavailable (not two-player or too large).")
        html.show(html.card(title or f"{self.name} - normal form", body))

    def solve_nf(self, title: Optional[str] = None, show_br: bool = True,
                 show_ne: bool = True, show_pareto: bool = True,
                 show_dominated: bool = True, show_mixed: bool = True) -> None:
        """Rich normal-form analysis of the induced bimatrix.

        Mirrors :meth:`NormalFormGame.solve`'s flag set: best responses,
        pure NE, Pareto cells, dominated rows/cols and (optionally) mixed NE.
        """
        nf = self._nf()
        if nf is None:
            html.show(html.card(title or f"{self.name} - normal form",
                                html.note("Requires exactly two players.")))
            return
        body = self._matrix_html(show_br=show_br, show_ne=show_ne,
                                 show_pareto=show_pareto,
                                 show_dominated=show_dominated)
        legend = ["underline = best response",
                  "green outline = Nash equilibrium",
                  "star = Pareto optimal",
                  "strikethrough = dominated"]
        body += html.legend(*legend)
        if show_mixed:
            A, B = nf["A"], nf["B"]
            mixed = [(p, q) for p, q in efx.mixed_equilibria(A, B)
                     if np.max(p) < 1 - 1e-9 or np.max(q) < 1 - 1e-9]
            if mixed:
                p, q = mixed[0]
                rb = ", ".join(f"{nf['row_labels'][i]}={fmt_prob(p[i])}"
                               for i in range(len(p)) if p[i] > 1e-9)
                cb = ", ".join(f"{nf['col_labels'][j]}={fmt_prob(q[j])}"
                               for j in range(len(q)) if q[j] > 1e-9)
                body += html.note(f"Mixed NE: {self.players[0]} ({rb}) vs "
                                  f"{self.players[1]} ({cb})")
        html.show(html.card(title or f"{self.name} - normal-form analysis", body))

    def iesds_explain(self, title: Optional[str] = None) -> None:
        """Walk through iterated elimination of strictly dominated strategies."""
        nf = self._nf()
        if nf is None:
            html.show(html.card(title or f"{self.name} - IESDS",
                                html.note("Requires exactly two players.")))
            return
        A, B, rl, cl, rounds = self.iesds_strict()
        if not rounds:
            body = html.note("No strictly dominated strategies - IESDS does "
                             "nothing.")
            html.show(html.card(title or f"{self.name} - IESDS", body))
            return
        items = []
        for k, rec in enumerate(rounds, start=1):
            bits = [f"<b>Round {k}.</b>"]
            for lbl, by in rec["rows"]:
                bits.append(f"{self.players[0]}: {lbl} dominated by "
                            f"{', '.join(by)} - eliminate.")
            for lbl, by in rec["cols"]:
                bits.append(f"{self.players[1]}: {lbl} dominated by "
                            f"{', '.join(by)} - eliminate.")
            items.append(" ".join(bits))
        if A.shape == (1, 1):
            items.append(f"<b>Result.</b> Unique outcome ({rl[0]}, {cl[0]}) "
                         f"with payoffs ({fmt(A[0, 0])}, {fmt(B[0, 0])}).")
        else:
            items.append(f"<b>Reduced game.</b> {A.shape[0]}x{A.shape[1]} "
                         f"remains: {self.players[0]} in {{{', '.join(rl)}}}, "
                         f"{self.players[1]} in {{{', '.join(cl)}}}.")
        html.show(html.card(title or f"{self.name} - IESDS", html.steps(items)))

    # ── plots ────────────────────────────────────────────────────────────────
    def plot_tree(self, title: Optional[str] = None, figsize=None):
        """Draw the game tree (decision/chance/terminal colouring, info sets)."""
        import matplotlib.patches as patches
        import networkx as nx

        G = nx.DiGraph()

        def label_of(node: str) -> str:
            data = self.tree[node]
            if data.get("is_terminal", False):
                return f"{node}\n{fmt_vec(data['payoffs'])}"
            player = self._player_name(data.get("player"))
            acts = data["actions"]
            atext = ", ".join(acts) if len(acts) <= 2 else ",\n".join(acts)
            return f"{node}\n{player}\n{atext}"

        levels: Dict[str, int] = {}

        def build(node: str, parent=None, action=None, level=0):
            G.add_node(node, label=label_of(node))
            levels[node] = level
            if parent is not None:
                G.add_edge(parent, node, label=action)
            data = self.tree[node]
            if not data.get("is_terminal", False):
                for act, child in data["children"].items():
                    build(child, node, act, level + 1)
        build(self.root)

        by_level: Dict[int, List[str]] = {}
        for node, lv in levels.items():
            by_level.setdefault(lv, []).append(node)
        pos = {}
        for lv, nodes in by_level.items():
            xs = np.linspace(-(len(nodes) - 1) / 2, (len(nodes) - 1) / 2,
                             len(nodes)) * 3.5
            for x, node in zip(xs, nodes):
                pos[node] = (x, -lv * 2.8)

        max_width = max(len(v) for v in by_level.values())
        depth = max(levels.values()) + 1
        if figsize is None:
            figsize = (max(10, 3 * max_width), max(6, 2.4 * depth))
        s = (figsize[0] * figsize[1] / 60.0) ** 0.5

        with rc_context({"axes.grid": False}):
            import matplotlib.pyplot as plt
            with plt.rc_context({"figure.autolayout": False}):
                fig, ax = plt.subplots(figsize=figsize)
            nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>",
                                   arrowsize=18 * s, width=1.2 * s,
                                   edge_color=C["muted"])
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=nx.get_edge_attributes(G, "label"),
                font_size=10 * s, rotate=True,
                bbox=dict(alpha=0.0, color="none"), ax=ax)
            for node, (x, y) in pos.items():
                data = self.tree[node]
                if data.get("is_terminal", False):
                    fc, ec = C["p1_light"], C["terminal"]
                elif data.get("player") == "chance":
                    fc, ec = "#fde7c8", C["chance"]
                elif data.get("player") == 0:
                    fc, ec = C["p1_light"], C["p1"]
                else:
                    fc, ec = C["p2_light"], C["p2"]
                ax.text(x, y, G.nodes[node]["label"], ha="center", va="center",
                        fontsize=10 * s,
                        bbox=dict(boxstyle="round,pad=0.45", facecolor=fc,
                                  edgecolor=ec, linewidth=0.8 * s, alpha=0.95))
            for set_id, nodes in self.info_sets.items():
                if len(nodes) <= 1:
                    continue
                xs = [pos[n][0] for n in nodes]
                ys = [pos[n][1] for n in nodes]
                w = (max(xs) - min(xs)) + 4.0
                h = (max(ys) - min(ys)) + 2.3
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                ax.add_patch(patches.Ellipse(
                    (cx, cy), width=w, height=h, fill=False, linestyle="--",
                    linewidth=1.3 * s, edgecolor=C["ce"]))
                ax.text(cx - w / 2, cy + h / 2 + 0.35, set_id, color=C["ce"],
                        fontsize=10 * s, ha="left", va="bottom",
                        fontweight="bold")
            ax.set_title(title or f"{self.name} - game tree",
                         fontsize=13 * s, fontweight="bold")
            ax.axis("off")
        return fig, ax

    def plot_frontier(self, title: Optional[str] = None, figsize=(7, 6)):
        """Feasible set (convex hull) + Pareto frontier + SPE point."""
        pts = self.terminal_payoffs()
        with rc_context():
            fig, ax = plots.new_axes(figsize)
            if len(pts) < 2 or pts.shape[1] != 2:
                ax.set_title(title or f"{self.name} - feasible set")
                return fig, ax
            ax.scatter(pts[:, 0], pts[:, 1], s=60, color=C["accent"],
                       edgecolors="white", linewidths=1.0, zorder=4,
                       label="terminals")
            if len(pts) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(pts)
                    ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1],
                            alpha=0.08, color=C["accent"])
                    for simplex in hull.simplices:
                        ax.plot(pts[simplex, 0], pts[simplex, 1],
                                color=C["grid"], lw=0.8)
                except Exception:
                    pass
            elif len(pts) == 2:
                ax.plot(pts[:, 0], pts[:, 1], color=C["grid"], lw=0.8)
            front = np.array(self.pareto_frontier_vertices(), dtype=float)
            if len(front) >= 2:
                ax.plot(front[:, 0], front[:, 1], lw=2.5, marker="o", ms=8,
                        color=C["pareto"], markeredgecolor="white",
                        markeredgewidth=1.5, zorder=5, label="Pareto frontier")
            elif len(front) == 1:
                ax.plot(front[0, 0], front[0, 1], marker="o", ms=8,
                        color=C["pareto"], markeredgecolor="white",
                        markeredgewidth=1.5, zorder=5, label="Pareto frontier")
            if not self._has_imperfect_info():
                spe = self.backward_induction()["value"]
                ax.plot(spe[0], spe[1], "o", ms=10, color=C["p2"],
                        markeredgecolor="white", markeredgewidth=2.0, zorder=6,
                        label="SPE")
            ax.set_xlabel(self.players[0])
            ax.set_ylabel(self.players[1])
            ax.set_title(title or f"{self.name} - feasible set & frontier")
            ax.legend(loc="best")
            ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def plot_br_map(self, title: Optional[str] = None):
        """Best-response heatmap over the induced bimatrix."""
        if len(self.players) != 2:
            raise ValueError("plot_br_map requires two players")
        A, B = self.convert_to_normal()
        nf = self._nf()
        br_row, br_col = solvers.br_masks(A, B)
        return plots.br_heatmap(br_row, br_col, solvers.ne_mask(A, B),
                                nf["row_labels"], nf["col_labels"],
                                title=title or f"{self.name} - best responses")

    def plot_mixed(self, title: Optional[str] = None, figsize=(6.0, 4.0)):
        """2x2 induced game: expected-payoff indifference lines."""
        if len(self.players) != 2:
            raise ValueError("plot_mixed requires two players")
        A, B = self.convert_to_normal()
        nf = self._nf()
        rl, cl = nf["row_labels"], nf["col_labels"]
        if A.shape != (2, 2):
            raise ValueError("plot_mixed supports 2x2 induced games only")
        p = np.linspace(0, 1, 201)
        with rc_context():
            import matplotlib.pyplot as plt
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(figsize[0] * 2,
                                                          figsize[1]))
            for i in range(2):
                ax0.plot(p, p * A[i, 0] + (1 - p) * A[i, 1],
                         color=C["p1"] if i == 0 else C["accent"], lw=2.2,
                         label=f"{self.players[0]} plays {rl[i]}")
            ax0.set_xlabel(f"Pr({self.players[1]} plays {cl[0]})")
            ax0.set_ylabel(f"expected payoff to {self.players[0]}")
            ax0.set_title(f"{self.players[0]}'s indifference")
            ax0.legend(loc="best")
            for j in range(2):
                ax1.plot(p, p * B[0, j] + (1 - p) * B[1, j],
                         color=C["p2"] if j == 0 else C["chance"], lw=2.2,
                         label=f"{self.players[1]} plays {cl[j]}")
            ax1.set_xlabel(f"Pr({self.players[0]} plays {rl[0]})")
            ax1.set_ylabel(f"expected payoff to {self.players[1]}")
            ax1.set_title(f"{self.players[1]}'s indifference")
            ax1.legend(loc="best")
            fig.suptitle(title or f"{self.name} - mixed-strategy indifference",
                         fontweight="bold")
        return fig, (ax0, ax1)

    # ── parameter sweeps (static, over a game-factory + range) ───────────────
    @staticmethod
    def sweep_mixed(factory, param_range, param_name: str = "parameter",
                    title: Optional[str] = None, figsize=(7.0, 4.0)):
        """Track equilibrium mixing probabilities vs a scalar parameter."""
        param_range = np.asarray(param_range, dtype=float)
        g0 = factory(float(param_range[0]))
        nf0 = g0._nf()
        if nf0 is None:
            raise ValueError("sweep_mixed requires two-player games")
        rl, cl = nf0["row_labels"], nf0["col_labels"]
        r, c = len(rl), len(cl)
        row_probs = [[] for _ in range(r)]
        col_probs = [[] for _ in range(c)]
        for v in param_range:
            g = factory(float(v))
            A, B = g.convert_to_normal()
            eqs = efx.mixed_equilibria(A, B)
            if eqs:
                sr, sc = eqs[0]
                for i in range(r):
                    row_probs[i].append(float(sr[i]))
                for j in range(c):
                    col_probs[j].append(float(sc[j]))
            else:
                for i in range(r):
                    row_probs[i].append(np.nan)
                for j in range(c):
                    col_probs[j].append(np.nan)
        with rc_context():
            fig, ax = plots.new_axes(figsize)
            for i in range(r - 1):
                ax.plot(param_range, row_probs[i], lw=2.2,
                        label=f"Pr({g0.players[0]} plays {rl[i]})")
            for j in range(c - 1):
                ax.plot(param_range, col_probs[j], lw=2.2, ls="--",
                        label=f"Pr({g0.players[1]} plays {cl[j]})")
            ax.set_xlabel(param_name)
            ax.set_ylabel("equilibrium mixing probability")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(title or f"Mixed NE vs {param_name}")
            ax.legend(loc="best")
        return fig, ax

    @staticmethod
    def sweep_pure(factory, param_range, param_name: str = "parameter",
                   title: Optional[str] = None, figsize=(7.0, 3.8)):
        """Track which profiles are pure NE (and the NE count) vs a parameter."""
        param_range = np.asarray(param_range, dtype=float)
        g0 = factory(float(param_range[0]))
        nf0 = g0._nf()
        if nf0 is None:
            raise ValueError("sweep_pure requires two-player games")
        rl, cl = nf0["row_labels"], nf0["col_labels"]
        profiles = [(i, j) for i in range(len(rl)) for j in range(len(cl))]
        is_ne = {p: [] for p in profiles}
        n_eq = []
        for v in param_range:
            nes = set(factory(float(v)).pure_nash_nf())
            n_eq.append(len(nes))
            for p in profiles:
                is_ne[p].append(int(p in nes))
        active = [p for p in profiles if any(is_ne[p])]
        with rc_context():
            fig, ax = plots.new_axes(figsize)
            for p in active:
                ax.step(param_range, is_ne[p], where="mid", lw=2.2,
                        label=f"({rl[p[0]]}, {cl[p[1]]}) is NE")
            ax.step(param_range, n_eq, where="mid", lw=2.2, ls="--",
                    color=C["muted"], label="# pure NE")
            ax.set_xlabel(param_name)
            ax.set_ylabel("count / indicator")
            ax.set_yticks(range((max(n_eq) if n_eq else 1) + 1))
            ax.set_title(title or f"Pure NE structure vs {param_name}")
            ax.legend(loc="best")
        return fig, ax

    @staticmethod
    def sweep_ne_regions(factory, x_range, y_range, x_name: str = "x",
                         y_name: str = "y", title: Optional[str] = None,
                         figsize=(7.2, 5.4), n: int = 61):
        """2-D map colouring (x, y) regions by their pure-NE set."""
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        xs = np.linspace(*x_range, n)
        ys = np.linspace(*y_range, n)
        g0 = factory(float(xs[0]), float(ys[0]))
        nf0 = g0._nf()
        if nf0 is None:
            raise ValueError("sweep_ne_regions requires two-player games")
        rl, cl = nf0["row_labels"], nf0["col_labels"]
        profiles = [(i, j) for i in range(len(rl)) for j in range(len(cl))]
        names = {p: f"({rl[p[0]]}, {cl[p[1]]})" for p in profiles}

        ne_sets: List[Any] = []
        Z_raw = [[None] * n for _ in range(n)]
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                ne = frozenset(factory(float(xv), float(yv)).pure_nash_nf())
                Z_raw[iy][ix] = ne
                if ne not in ne_sets:
                    ne_sets.append(ne)
        code = {s: k for k, s in enumerate(ne_sets)}
        Z = np.array([[code[Z_raw[iy][ix]] for ix in range(n)]
                      for iy in range(n)], dtype=int)
        nc = len(ne_sets)
        pal = [C["grid"], C["p1"], C["accent"], C["chance"], C["p2"],
               C["ne"], C["ce"], C["pareto"]]
        colors = [pal[k % len(pal)] for k in range(nc)]
        labels = []
        for s in ne_sets:
            if not s:
                labels.append("no pure NE")
            else:
                labels.append(" & ".join(names[p] for p in
                                         sorted(s, key=profiles.index)))
        with rc_context({"axes.grid": False}):
            fig, ax = plots.new_axes(figsize)
            ax.imshow(Z, origin="lower", aspect="auto",
                      extent=[xs[0], xs[-1], ys[0], ys[-1]],
                      cmap=ListedColormap(colors),
                      norm=BoundaryNorm(np.arange(-0.5, nc, 1), nc),
                      interpolation="nearest")
            handles = [Patch(facecolor=colors[k], edgecolor="#666",
                             label=labels[k]) for k in sorted(set(Z.ravel()))]
            ax.legend(handles=handles, loc="upper left", title="NE type")
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(title or "Pure NE regions")
        return fig, ax

    @staticmethod
    def sweep_spe_regions(factory, x_range, y_range, x_name: str = "x",
                          y_name: str = "y", title: Optional[str] = None,
                          decision_node: str = "incumbent",
                          figsize=(7.5, 5.5), n: int = 41):
        """2-D map of the SPE action at ``decision_node`` + PoA contours."""
        from matplotlib.colors import ListedColormap

        xs = np.linspace(*x_range, n)
        ys = np.linspace(*y_range, n)
        g0 = factory(float(xs[0]), float(ys[0]))
        acts = sorted(set(g0.tree[decision_node]["actions"]))
        act_code = {a: k for k, a in enumerate(acts)}
        spe_type = np.zeros((n, n))
        poa = np.zeros((n, n))
        for iy, yv in enumerate(ys):
            for ix, xv in enumerate(xs):
                g = factory(float(xv), float(yv))
                action = g.backward_induction()["strategy"].get(
                    decision_node, acts[0])
                spe_type[iy, ix] = act_code.get(action, 0)
                poa[iy, ix] = g.price_of_anarchy()
        with rc_context({"axes.grid": False}):
            fig, ax = plots.new_axes(figsize)
            pal = [C["p1_light"], C["p2_light"], C["pareto"], C["ne"]]
            cmap = ListedColormap(pal[:max(2, len(acts))])
            im = ax.imshow(spe_type, cmap=cmap,
                           extent=(x_range[0], x_range[1], y_range[0],
                                   y_range[1]), origin="lower", aspect="auto",
                           vmin=0, vmax=len(acts) - 1 if len(acts) > 1 else 1)
            cbar = fig.colorbar(im, ax=ax,
                                ticks=list(range(len(acts))))
            cbar.ax.set_yticklabels(acts)
            cbar.set_label(f"SPE action at {decision_node}")
            finite = np.isfinite(poa)
            if finite.any():
                cs = ax.contour(xs, ys, np.where(finite, poa, np.nan),
                                levels=5, colors="white", linewidths=1.0)
                ax.clabel(cs, inline=True, fmt="PoA %.1f")
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(title or "SPE regions & price of anarchy")
        return fig, ax

    # ── comparison ───────────────────────────────────────────────────────────
    @staticmethod
    def compare(*games: "ExtensiveFormGame") -> None:
        """Render several games' summaries side by side."""
        html.compare_via(games, "summary")

    def __repr__(self) -> str:
        n_decision = sum(1 for d in self.tree.values()
                         if not d.get("is_terminal", False))
        n_terminal = len(self.tree) - n_decision
        return (f"ExtensiveFormGame({self.name!r}, {n_decision} decision + "
                f"{n_terminal} terminal nodes, players={self.players!r})")
