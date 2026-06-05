"""Pure algorithms for extensive-form analysis ported from the EFG notebook.

Everything here is numpy/array-in, plain-data-out and free of display concerns.
The core :class:`ExtensiveFormGame` wires these into the shared viz layer.

The functions operate on a normalized tree representation::

    decision : {"player": int, "actions": [..], "children": {action: child}}
    chance   : {"player": "chance", "actions": [..], "children": {...},
                "prob": {action: p}}
    terminal : {"is_terminal": True, "payoffs": tuple}

plus an ``info_sets`` mapping ``{set_id: [node, ...]}``.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import numpy as np


# ── normal-form conversion ──────────────────────────────────────────────────
def _node_to_info_set(info_sets: Dict[str, List[str]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for set_id, nodes in info_sets.items():
        for node in nodes:
            mapping[node] = set_id
    return mapping


def decision_points(tree: Dict[str, Dict[str, Any]],
                    info_sets: Dict[str, List[str]],
                    n_players: int):
    """Return ``(points_by_player, actions_by_point)`` for the two-player case.

    A decision point is an information set (if the node belongs to one) or a
    single node, identified by a stable string id.
    """
    n2i = _node_to_info_set(info_sets)
    points: Dict[int, List[str]] = {p: [] for p in range(n_players)}
    actions: Dict[str, List[str]] = {}
    for node, data in tree.items():
        if data.get("is_terminal", False):
            continue
        player = data.get("player")
        if not isinstance(player, int):
            continue
        dp_id = f"INFO::{n2i[node]}" if node in n2i else f"NODE::{node}"
        if dp_id not in points[player]:
            points[player].append(dp_id)
            actions[dp_id] = list(data["actions"])
    return points, actions


def strategy_profiles(points: List[str], actions: Dict[str, List[str]]):
    """Enumerate pure strategies for one player as ``(profiles, labels)``."""
    if not points:
        return [dict()], ["-"]
    action_lists = [actions[dp] for dp in points]
    profiles, labels = [], []
    for combo in itertools.product(*action_lists):
        profiles.append({dp: a for dp, a in zip(points, combo)})
        if len(points) == 1:
            labels.append(combo[0])
        else:
            parts = []
            for dp, a in zip(points, combo):
                short = dp.replace("INFO::", "").replace("NODE::", "")
                parts.append(f"{short}: {a}")
            labels.append(" | ".join(parts))
    return profiles, labels


def convert_to_normal(tree: Dict[str, Dict[str, Any]],
                      info_sets: Dict[str, List[str]],
                      n_players: int) -> Tuple[np.ndarray, np.ndarray]:
    """Induce the bimatrix ``(A, B)`` of a two-player extensive-form game."""
    if n_players != 2:
        raise ValueError("normal-form conversion assumes 2 players")
    n2i = _node_to_info_set(info_sets)
    points, actions = decision_points(tree, info_sets, n_players)
    strat0, _ = strategy_profiles(points[0], actions)
    strat1, _ = strategy_profiles(points[1], actions)

    def act(node: str, strat: Dict[str, str]) -> str:
        dp_id = f"INFO::{n2i[node]}" if node in n2i else f"NODE::{node}"
        return strat[dp_id]

    def simulate(node: str, s0: Dict[str, str], s1: Dict[str, str]):
        data = tree[node]
        if data.get("is_terminal", False):
            return np.asarray(data["payoffs"], dtype=float)
        if data.get("player") == "chance":
            probs = data.get("prob") or {a: 1.0 / len(data["actions"])
                                         for a in data["actions"]}
            ev = np.zeros(n_players)
            for a, child in data["children"].items():
                ev = ev + probs.get(a, 0.0) * simulate(child, s0, s1)
            return ev
        player = data["player"]
        a = act(node, s0) if player == 0 else act(node, s1)
        return simulate(data["children"][a], s0, s1)

    A = np.zeros((len(strat0), len(strat1)))
    B = np.zeros((len(strat0), len(strat1)))
    for i, s0 in enumerate(strat0):
        for j, s1 in enumerate(strat1):
            payoffs = simulate("root", s0, s1)
            A[i, j], B[i, j] = payoffs[0], payoffs[1]
    return A, B


def normal_form_labels(tree, info_sets, n_players) -> Dict[str, Any]:
    """``{row_labels, col_labels, A, B}`` for the induced two-player game."""
    points, actions = decision_points(tree, info_sets, n_players)
    _, row_labels = strategy_profiles(points[0], actions)
    _, col_labels = strategy_profiles(points[1], actions)
    A, B = convert_to_normal(tree, info_sets, n_players)
    return {"row_labels": row_labels, "col_labels": col_labels, "A": A, "B": B}


# ── best responses / dominance / IESDS on the induced matrix ────────────────
def best_responses_row(A: np.ndarray, j: int, tol: float = 1e-12) -> List[int]:
    col = A[:, j]
    m = float(np.max(col))
    return [i for i, v in enumerate(col) if abs(float(v) - m) <= tol]


def best_responses_col(B: np.ndarray, i: int, tol: float = 1e-12) -> List[int]:
    row = B[i, :]
    m = float(np.max(row))
    return [j for j, v in enumerate(row) if abs(float(v) - m) <= tol]


def strictly_dominated_rows(A: np.ndarray) -> Dict[int, List[int]]:
    dom: Dict[int, List[int]] = {}
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i != j and np.all(A[i] <= A[j]) and np.any(A[i] < A[j]):
                dom.setdefault(i, []).append(j)
                break
    return dom


def strictly_dominated_cols(B: np.ndarray) -> Dict[int, List[int]]:
    dom: Dict[int, List[int]] = {}
    for i in range(B.shape[1]):
        for j in range(B.shape[1]):
            if i != j and np.all(B[:, i] <= B[:, j]) and np.any(B[:, i] < B[:, j]):
                dom.setdefault(i, []).append(j)
                break
    return dom


def pure_nash(A: np.ndarray, B: np.ndarray) -> List[Tuple[int, int]]:
    r, c = A.shape
    out = []
    for i in range(r):
        for j in range(c):
            if i in best_responses_row(A, j) and j in best_responses_col(B, i):
                out.append((i, j))
    return out


def pareto_optimal(A: np.ndarray, B: np.ndarray) -> List[Tuple[int, int]]:
    r, c = A.shape
    cells = [(i, j) for i in range(r) for j in range(c)]
    out = []
    for i, j in cells:
        dominated = any(
            (i2, j2) != (i, j)
            and A[i2, j2] >= A[i, j] and B[i2, j2] >= B[i, j]
            and (A[i2, j2] > A[i, j] or B[i2, j2] > B[i, j])
            for i2, j2 in cells
        )
        if not dominated:
            out.append((i, j))
    return out


def iesds_log(A: np.ndarray, B: np.ndarray,
              row_labels: List[str], col_labels: List[str],
              max_rounds: int = 50):
    """Iterated elimination of strictly dominated strategies.

    Returns ``(A, B, row_labels, col_labels, rounds)`` where ``rounds`` is a
    list of per-round dicts describing the eliminations (for explanations).
    """
    A, B = A.copy(), B.copy()
    row_labels, col_labels = list(row_labels), list(col_labels)
    rounds: List[Dict[str, Any]] = []
    for _ in range(max_rounds):
        row_dom = strictly_dominated_rows(A)
        col_dom = strictly_dominated_cols(B)
        if not row_dom and not col_dom:
            break
        rec = {
            "rows": [(row_labels[i], [row_labels[d] for d in doms])
                     for i, doms in sorted(row_dom.items())],
            "cols": [(col_labels[j], [col_labels[d] for d in doms])
                     for j, doms in sorted(col_dom.items())],
        }
        rounds.append(rec)
        keep_r = [i for i in range(A.shape[0]) if i not in row_dom]
        keep_c = [j for j in range(A.shape[1]) if j not in col_dom]
        A = A[np.ix_(keep_r, keep_c)]
        B = B[np.ix_(keep_r, keep_c)]
        row_labels = [row_labels[i] for i in keep_r]
        col_labels = [col_labels[j] for j in keep_c]
    return A, B, row_labels, col_labels, rounds


# ── feasible set / Pareto frontier of terminal payoffs ──────────────────────
def pareto_outcomes(points: np.ndarray) -> List[Tuple[float, ...]]:
    pts = [tuple(p) for p in np.asarray(points, dtype=float)]
    out = []
    for u in pts:
        dominated = any(
            v != u and all(vi >= ui for vi, ui in zip(v, u))
            and any(vi > ui for vi, ui in zip(v, u))
            for v in pts
        )
        if not dominated:
            out.append(u)
    return out


def pareto_frontier_vertices(points: np.ndarray) -> List[Tuple[float, float]]:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return []
    pts = np.unique(pts, axis=0)
    if len(pts) == 1:
        return [tuple(pts[0])]

    def _non_dominated(arr):
        frontier = []
        for p in arr:
            dominated = any(
                (not np.array_equal(p, q)) and np.all(q >= p) and np.any(q > p)
                for q in arr
            )
            if not dominated:
                frontier.append(tuple(p))
        return sorted(set(frontier), key=lambda x: x[0])

    if len(pts) == 2 or pts.shape[1] != 2:
        return _non_dominated(pts)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return _non_dominated(pts[hull.vertices])
    except Exception:
        return _non_dominated(pts)


# ── mixed equilibria via support enumeration on the induced bimatrix ────────
def mixed_equilibria(A: np.ndarray, B: np.ndarray):
    """All mixed equilibria ``(p, q)`` via :mod:`gtlab.solvers` support enum.

    Lazy import to avoid a circular dependency with the solvers package init.
    """
    from . import all_equilibria
    return all_equilibria(A, B)
