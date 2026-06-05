"""Extensive-form games: game trees, backward induction, welfare.

The tree is a dict keyed by node id. Each node is either a decision node::

    {"player": 0, "actions": {"L": "n1", "R": "n2"}}

a chance node::

    {"chance": {"L": (0.5, "n1"), "R": (0.5, "n2")}}

or a terminal node::

    {"payoff": (3.0, 1.0)}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import solvers
from .._memo import cached_method
from ..viz import fmt_vec, html


@dataclass
class ExtensiveFormGame:
    """A finite extensive-form game over a tree of decision/chance/terminal nodes."""

    tree: Dict[str, Dict[str, Any]]
    root: str = "root"
    players: Optional[List[str]] = None
    name: str = "Extensive-form game"

    def __post_init__(self) -> None:
        if self.root not in self.tree:
            raise ValueError(f"root node {self.root!r} not in tree")
        n_players = self._infer_n_players()
        self.players = list(self.players) if self.players else [f"P{i+1}" for i in range(n_players)]
        self._validate()

    def _infer_n_players(self) -> int:
        n = 2
        for node in self.tree.values():
            if "payoff" in node:
                n = max(n, len(node["payoff"]))
        return n

    def _validate(self) -> None:
        for nid, node in self.tree.items():
            kinds = sum(k in node for k in ("actions", "chance", "payoff"))
            if kinds != 1:
                raise ValueError(f"node {nid!r} must be exactly one of "
                                 "decision/chance/terminal")
            if "actions" in node:
                for child in node["actions"].values():
                    if child not in self.tree:
                        raise ValueError(f"node {nid!r} → missing child {child!r}")

    # ── backward induction ───────────────────────────────────────────────────
    @cached_method
    def backward_induction(self, tol: float = 1e-9) -> Dict[str, Any]:
        """Compute the subgame-perfect equilibrium by backward induction.

        Returns ``{"value": payoff_at_root, "strategy": {node: action}}``.
        """
        strategy: Dict[str, str] = {}

        def value(nid: str) -> np.ndarray:
            node = self.tree[nid]
            if "payoff" in node:
                return np.asarray(node["payoff"], dtype=float)
            if "chance" in node:
                out = np.zeros(len(self.players))
                for prob, child in node["chance"].values():
                    out += prob * value(child)
                return out
            player = node["player"]
            best_action, best_val = None, None
            for action, child in node["actions"].items():
                v = value(child)
                if best_val is None or v[player] > best_val[player] + tol:
                    best_action, best_val = action, v
            strategy[nid] = best_action
            return best_val

        root_value = value(self.root)
        return {"value": root_value, "strategy": strategy}

    def terminal_payoffs(self) -> np.ndarray:
        """All terminal payoff vectors as an ``(N, players)`` array."""
        return np.array([node["payoff"] for node in self.tree.values()
                         if "payoff" in node], dtype=float)

    def pareto_frontier(self) -> np.ndarray:
        return solvers.pareto_frontier(self.terminal_payoffs())

    def social_welfare(self, objective: str = "utilitarian") -> Tuple[np.ndarray, float]:
        """Best terminal outcome and its welfare score for the given objective."""
        outcomes = self.terminal_payoffs()
        idx = solvers.best_outcome(outcomes, objective)
        score = {"utilitarian": solvers.utilitarian,
                 "egalitarian": solvers.egalitarian,
                 "nash": solvers.nash_welfare}[objective](outcomes[idx])
        return outcomes[idx], float(score)

    # ── display ──────────────────────────────────────────────────────────────
    def _solution_html(self) -> str:
        res = self.backward_induction()
        rows = [[nid, action] for nid, action in res["strategy"].items()]
        tbl = html.table(["node", "chosen action"], rows)
        return html.kv([("SPE payoff", fmt_vec(res["value"]))]) + tbl

    def solve(self, title: Optional[str] = None) -> None:
        html.show(html.card(title or f"{self.name} - backward induction",
                            self._solution_html()))

    def explain(self, title: Optional[str] = None) -> None:
        res = self.backward_induction()
        items = [
            "<b>Step 1 - Start at the leaves.</b> Terminal nodes already carry payoffs.",
            "<b>Step 2 - Fold the tree.</b> At each decision node the acting player "
            "picks the action leading to the child with the highest payoff <i>for "
            "them</i>; that child's payoff vector propagates up.",
            f"<b>Step 3 - Read the root.</b> The resulting subgame-perfect equilibrium "
            f"yields payoff {fmt_vec(res['value'])} (chosen actions tabulated above).",
        ]
        body = self._solution_html() + html.steps(items)
        html.show(html.card(title or f"{self.name} - backward induction", body))

    def __repr__(self) -> str:
        return f"ExtensiveFormGame({self.name!r}, nodes={len(self.tree)})"
