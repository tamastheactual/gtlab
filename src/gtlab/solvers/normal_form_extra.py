"""Extra pure algorithms for normal-form games (ported from the course notebook).

These are display-free helpers used by :class:`gtlab.core.NormalFormGame` for the
mixed-strategy walkthrough, comparative-statics sweeps, and best-response curve
data. They take plain numpy arrays / callables and return numpy / Python data.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .nash import all_equilibria, pure_nash

EPS = 1e-9


# ── 2x2 mixed-equilibrium indifference data ─────────────────────────────────

def mixed_2x2_indifference(A: np.ndarray, B: np.ndarray):
    """Return ``(p_star, q_star)`` for a 2x2 game's interior mixed equilibrium.

    ``q_star`` = Pr(column plays its first action) that makes the row player
    indifferent; ``p_star`` = Pr(row plays its first action) that makes the
    column player indifferent. Either may be ``None`` when degenerate.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    denom_q = float(A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1])
    denom_p = float(B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1])
    q_star = None if abs(denom_q) < EPS else (A[1, 1] - A[0, 1]) / denom_q
    p_star = None if abs(denom_p) < EPS else (B[1, 1] - B[1, 0]) / denom_p
    return p_star, q_star


def mixed_equilibria(A: np.ndarray, B: np.ndarray):
    """All non-pure (mixed) equilibria as ``(p, q)`` arrays."""
    out = []
    for p, q in all_equilibria(A, B):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        if np.max(p) > 1 - EPS and np.max(q) > 1 - EPS:
            continue  # pure
        out.append((p, q))
    return out


def expected_payoffs(A: np.ndarray, B: np.ndarray, p: np.ndarray, q: np.ndarray):
    """Expected ``(row, col)`` payoffs of mixed profile ``(p, q)``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(p @ A @ q), float(p @ B @ q)


# ── Indifference lines (for plot_mixed) ─────────────────────────────────────

def row_indifference_lines(A: np.ndarray, p_grid: np.ndarray):
    """Expected payoff to the row player for each pure row action.

    ``p`` is Pr(column plays its first action). Returns one array per row action.
    """
    A = np.asarray(A, dtype=float)
    p = np.asarray(p_grid, dtype=float)
    m = A.shape[0]
    return [p * A[i, 0] + (1 - p) * A[i, 1] for i in range(m)]


def envelope_crossings(lines, p_grid):
    """Upper-envelope crossing points ``(p*, u*)`` of a set of payoff lines."""
    p = np.asarray(p_grid, dtype=float)
    n = len(lines)
    found = []
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            diff = lines[i] - lines[j]
            idxs = np.where(np.diff(np.sign(diff)))[0]
            for idx in idxs:
                d0, d1 = diff[idx], diff[idx + 1]
                if abs(d1 - d0) < EPS:
                    continue
                ps = p[idx] - d0 * (p[idx + 1] - p[idx]) / (d1 - d0)
                us = lines[i][idx] + (lines[i][idx + 1] - lines[i][idx]) * \
                    ((ps - p[idx]) / (p[idx + 1] - p[idx]) if p[idx + 1] != p[idx] else 0)
                vals_at_ps = [float(np.interp(ps, p, line)) for line in lines]
                grid_tol = (p[-1] - p[0]) / max(len(p) - 1, 1)
                if us >= max(vals_at_ps) - grid_tol:
                    key = (round(float(ps), 6), round(float(us), 6))
                    if key not in seen:
                        seen.add(key)
                        found.append((float(ps), float(us)))
    return found


# ── Comparative statics: data collectors ────────────────────────────────────

def sweep_mixed_data(factory: Callable, param_range) -> Tuple:
    """Collect first-equilibrium mixing probabilities across a 1-D sweep.

    Returns ``(param_range, row_probs, col_probs)`` where ``row_probs[i]`` is a
    list over the sweep of Pr(row plays action i) (NaN where no equilibrium).
    """
    param_range = np.asarray(param_range, dtype=float)
    g0 = factory(float(param_range[0]))
    r, c = g0.shape
    row_probs = [[] for _ in range(r)]
    col_probs = [[] for _ in range(c)]
    for v in param_range:
        g = factory(float(v))
        eqs = all_equilibria(g.A, g.B)
        if eqs:
            sr, sc = eqs[0]
            sr = np.asarray(sr, dtype=float)
            sc = np.asarray(sc, dtype=float)
            for i in range(r):
                row_probs[i].append(float(sr[i]))
            for j in range(c):
                col_probs[j].append(float(sc[j]))
        else:
            for i in range(r):
                row_probs[i].append(np.nan)
            for j in range(c):
                col_probs[j].append(np.nan)
    return param_range, row_probs, col_probs


def sweep_pure_data(factory: Callable, param_range) -> Tuple:
    """Collect pure-NE structure across a 1-D sweep.

    Returns ``(param_range, profiles, is_ne, n_eq)`` where ``is_ne[(i, j)]`` is a
    list of 0/1 indicators and ``n_eq`` is the count of pure NE per param value.
    """
    param_range = np.asarray(param_range, dtype=float)
    g0 = factory(float(param_range[0]))
    r, c = g0.shape
    profiles = [(i, j) for i in range(r) for j in range(c)]
    is_ne = {p: [] for p in profiles}
    n_eq: List[int] = []
    for v in param_range:
        g = factory(float(v))
        nes = set(pure_nash(g.A, g.B))
        n_eq.append(len(nes))
        for p in profiles:
            is_ne[p].append(int(p in nes))
    return param_range, profiles, is_ne, n_eq


def sweep_regions_data(factory: Callable, x_range, y_range, n: int = 151) -> Tuple:
    """Compute pure-NE equilibrium-set codes over a 2-parameter grid.

    Returns ``(xs, ys, Z, ne_sets, profiles, profile_names)`` where ``Z`` is an
    integer grid indexing ``ne_sets`` (a sorted list of frozensets of profiles).
    """
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    g0 = factory(float(xs[0]), float(ys[0]))
    r, c = g0.shape
    profiles = [(i, j) for i in range(r) for j in range(c)]
    profile_names = {
        p: f"({g0.row_actions[p[0]]}, {g0.col_actions[p[1]]})" for p in profiles
    }

    ne_sets: List[frozenset] = []
    raw = [[None] * n for _ in range(n)]
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            g = factory(float(xv), float(yv))
            ne = frozenset(pure_nash(g.A, g.B))
            raw[iy][ix] = ne
            if ne not in ne_sets:
                ne_sets.append(ne)

    def _sort_key(s):
        if not s:
            return (0,)
        if len(s) == 1:
            return (1, profiles.index(next(iter(s))))
        return (2, min(profiles.index(p) for p in s))
    ne_sets.sort(key=_sort_key)

    code_of = {s: k for k, s in enumerate(ne_sets)}
    nc = len(ne_sets)
    Z = np.array([[code_of[raw[iy][ix]] for ix in range(n)] for iy in range(n)],
                 dtype=int)

    # Clean measure-zero boundary stripes by majority-vote of 4-neighbours.
    from collections import Counter
    threshold = max(2, int(0.015 * Z.size))
    for _ in range(3):
        counts = np.bincount(Z.ravel(), minlength=nc)
        changed = False
        for code in range(nc):
            if 0 < counts[code] <= threshold:
                ys_i, xs_i = np.where(Z == code)
                for py, px in zip(ys_i, xs_i):
                    nbrs = []
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < n and 0 <= nx < n and Z[ny, nx] != code:
                            nbrs.append(Z[ny, nx])
                    if nbrs:
                        Z[py, px] = Counter(nbrs).most_common(1)[0][0]
                        changed = True
        if not changed:
            break

    return xs, ys, Z, ne_sets, profiles, profile_names
