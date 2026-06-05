"""Reusable matplotlib helpers shared by the game classes.

These are the plotting primitives that recurred across notebooks: best-response
heatmaps, convergence curves, and simplex/probability sweeps. Each is a thin,
game-agnostic function operating on plain arrays so any class can call them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .theme import C, rc_context


def new_axes(figsize=(6.0, 4.0)):
    """Create a styled figure/axes pair under the shared rc context."""
    import matplotlib.pyplot as plt

    with rc_context():
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def ref_lines(ax, hlines=None, vlines=None) -> None:
    """Draw optional dotted reference lines at the given y / x values.

    Used by the parameter-sweep plots so callers can mark a probability level
    or a parameter of interest (matches the notebook ``hlines``/``vlines`` API).
    """
    for y in (hlines or []):
        ax.axhline(float(y), ls=":", lw=1.0, color=C["muted"], alpha=0.8, zorder=1)
    for x in (vlines or []):
        ax.axvline(float(x), ls=":", lw=1.0, color=C["muted"], alpha=0.8, zorder=1)


def br_heatmap(
    br_row: np.ndarray,
    br_col: np.ndarray,
    ne_mask: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str = "Best-response map",
    figsize=(6.4, 4.5),
    row_player: str = "Row",
    col_player: str = "Column",
):
    """Heatmap coloring cells by who best-responds (row / col / both = NE).

    A patch legend documents the tint encoding and the axes are labelled with
    the two players' names so orientation is unambiguous.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    m, n = br_row.shape
    grid = np.zeros((m, n))
    grid[br_row] += 1
    grid[br_col] += 2  # 1=row, 2=col, 3=both
    cmap = ListedColormap(["#ffffff", C["p1_light"], C["p2_light"], C["ne"]])
    with rc_context():
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
        ax.set_xticks(range(n), col_labels, rotation=45, ha="right")
        ax.set_yticks(range(m), row_labels)
        ax.set_xlabel(col_player)
        ax.set_ylabel(row_player)
        for i in range(m):
            for j in range(n):
                if ne_mask[i, j]:
                    ax.text(j, i, "NE", ha="center", va="center",
                            color="white", fontweight="bold")
        handles = [
            Patch(facecolor=C["p1_light"], edgecolor=C["grid"],
                  label=f"{row_player} best response"),
            Patch(facecolor=C["p2_light"], edgecolor=C["grid"],
                  label=f"{col_player} best response"),
            Patch(facecolor=C["ne"], edgecolor=C["grid"], label="Nash equilibrium"),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  frameon=False, fontsize=9)
        ax.set_title(title)
        ax.grid(False)
    return fig, ax


# distinct line styles cycled so coincident series stay visible
_LS_CYCLE = ["-", "--", ":", "-."]


def convergence(
    series: dict[str, np.ndarray],
    target: float | None = None,
    title: str = "Convergence",
    xlabel: str = "iteration",
    ylabel: str = "value",
    logy: bool = False,
    figsize=(6.5, 4.0),
    target_label: str | None = None,
):
    """Plot one or more convergence curves, optionally with a target line.

    Series cycle line style/width so two coincident trajectories (e.g. the two
    players' regret in a symmetric game) remain distinguishable instead of one
    over-painting the other. ``target_label`` names the reference line.
    """
    import matplotlib.pyplot as plt

    with rc_context():
        fig, ax = plt.subplots(figsize=figsize)
        for k, (label, ys) in enumerate(series.items()):
            ax.plot(np.arange(len(ys)), ys, label=label,
                    ls=_LS_CYCLE[k % len(_LS_CYCLE)],
                    lw=2.2 - 0.5 * k, alpha=0.95, zorder=3 + k)
        if target is not None:
            ax.axhline(target, ls=(0, (6, 4)), color=C["muted"], lw=1.3,
                       zorder=2, label=target_label or "target")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    return fig, ax


def simplex2_lines(
    payoff_lines: dict[str, np.ndarray],
    p_grid: np.ndarray,
    title: str = "Expected payoff vs. mixing probability",
    xlabel: str = "p (probability of first action)",
    figsize=(6.0, 4.0),
):
    """Plot expected-payoff lines over a 2x2 mixing probability sweep."""
    import matplotlib.pyplot as plt

    with rc_context():
        fig, ax = plt.subplots(figsize=figsize)
        for label, ys in payoff_lines.items():
            ax.plot(p_grid, ys, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("expected payoff")
        ax.set_title(title)
        ax.legend()
    return fig, ax
