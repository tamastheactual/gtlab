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


def br_heatmap(
    br_row: np.ndarray,
    br_col: np.ndarray,
    ne_mask: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str = "Best-response map",
    figsize=(5.5, 4.5),
):
    """Heatmap coloring cells by who best-responds (row / col / both = NE)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

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
        for i in range(m):
            for j in range(n):
                if ne_mask[i, j]:
                    ax.text(j, i, "NE", ha="center", va="center",
                            color="white", fontweight="bold")
        ax.set_title(title)
        ax.grid(False)
    return fig, ax


def convergence(
    series: dict[str, np.ndarray],
    target: float | None = None,
    title: str = "Convergence",
    xlabel: str = "iteration",
    ylabel: str = "value",
    logy: bool = False,
    figsize=(6.5, 4.0),
):
    """Plot one or more convergence curves, optionally with a target line."""
    import matplotlib.pyplot as plt

    with rc_context():
        fig, ax = plt.subplots(figsize=figsize)
        for label, ys in series.items():
            ax.plot(np.arange(len(ys)), ys, label=label)
        if target is not None:
            ax.axhline(target, ls="--", color=C["muted"], label="target")
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
