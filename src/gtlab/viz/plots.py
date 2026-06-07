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


def _ref_value_label(item):
    """Unpack a reference-line spec into ``(value, label)``.

    Accepts a bare number, or a ``(value, label)`` / ``(value,)`` tuple/list.
    """
    if isinstance(item, (tuple, list)):
        value = float(item[0])
        label = str(item[1]) if len(item) > 1 and item[1] is not None else None
        return value, label
    return float(item), None


def ref_lines(ax, hlines=None, vlines=None) -> None:
    """Draw optional dotted reference lines at the given y / x positions.

    Each entry is either a number or a ``(value, label)`` tuple (matching the
    notebook ``hlines``/``vlines`` API); when a label is given it is annotated
    next to the line, placed just inside the axes so it is never clipped.
    """
    style = dict(ls=":", lw=1.2, color=C["text"], alpha=0.55, zorder=1)
    lbl = dict(fontsize=8.5, color=C["text"], alpha=0.8, clip_on=False,
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    for item in (hlines or []):
        y, label = _ref_value_label(item)
        ax.axhline(y, **style)
        if label:
            ax.annotate(label, xy=(0.012, y), xycoords=("axes fraction", "data"),
                        va="bottom", ha="left", **lbl)
    for item in (vlines or []):
        x, label = _ref_value_label(item)
        ax.axvline(x, **style)
        if label:
            ax.annotate(label, xy=(x, 0.03), xycoords=("data", "axes fraction"),
                        va="bottom", ha="left", rotation=90, **lbl)


def legend_outside(ax, **kwargs):
    """Place the legend just outside the axes (upper-left of the right margin).

    The shared convention for 2-D map / imshow / region plots so the legend
    never overlaps the data. ``savefig.bbox='tight'`` keeps it in the PNG.
    """
    opts = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                fontsize=9)
    opts.update(kwargs)
    return ax.legend(**opts)


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
        # Only legend the categories that actually appear in this grid, so a
        # game with (say) no row-only-best cells does not show a dangling swatch.
        present = set(np.unique(grid).astype(int))
        catalog = [
            (1, C["p1_light"], f"{row_player} best response"),
            (2, C["p2_light"], f"{col_player} best response"),
            (3, C["ne"], "Nash equilibrium"),
        ]
        handles = [Patch(facecolor=col, edgecolor=C["grid"], label=lbl)
                   for code, col, lbl in catalog if code in present]
        if handles:
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
