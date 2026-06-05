"""Single source of truth for colors, CSS, and matplotlib styling.

Every notebook used to carry its own copy of a ``C`` color dict, a
``_XXX_CSS`` string, and a ``plt.rcParams`` block. They are consolidated
here so a theme change happens once instead of six times.
"""
from __future__ import annotations

from contextlib import contextmanager

# ── Color palette ────────────────────────────────────────────────────────
# Shared across HTML tables and matplotlib plots so the two always agree.
C = {
    "p1": "#2563eb",       # Row / player 1
    "p2": "#dc2626",       # Column / player 2
    "p1_light": "#dbeafe",
    "p2_light": "#fee2e2",
    "ne": "#16a34a",       # Nash equilibrium highlight
    "ce": "#9333ea",       # Correlated equilibrium
    "cce": "#f59e0b",      # Coarse correlated equilibrium
    "chance": "#f97316",   # chance nodes (extensive form)
    "terminal": "#16a34a",
    "pareto": "#eab308",
    "grid": "#94a3b8",
    "cell": "#f8fafc",
    "cell_alt": "#eef2f7",
    "text": "#0f172a",
    "muted": "#64748b",
    "accent": "#0ea5e9",
}

# matplotlib rcParams shared by every plot helper.
RC_PARAMS = {
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "figure.autolayout": True,
}

# ── CSS ──────────────────────────────────────────────────────────────────
# One stylesheet for all HTML output. It is THEME-AGNOSTIC: instead of picking
# light/dark via @media (prefers-color-scheme) -- which tracks the OS, not
# Colab's own light/dark toggle and so produced dark cards on a light Colab
# page -- it inherits the page's text color and uses translucent grey overlays
# for backgrounds/borders. That blends correctly into light OR dark notebooks.
# Player accent colors (blue/red) and highlights are chosen to read on both.
# Class names are prefixed ``gt-`` so they never collide with notebook styles.
CSS = """
<style>
.gt-wrap { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
           color: inherit; margin: 0.4em 0; }
.gt-wrap table { border-collapse: collapse; margin: 0.4em 0; }
.gt-wrap th, .gt-wrap td { border: 1px solid rgba(128,128,128,0.35);
           padding: 6px 10px; text-align: center;
           font-variant-numeric: tabular-nums; }
.gt-wrap th { background: rgba(128,128,128,0.16); font-weight: 600; }
.gt-card { border: 1px solid rgba(128,128,128,0.3); border-radius: 10px;
           padding: 12px 16px; margin: 8px 0; background: rgba(128,128,128,0.06); }
.gt-title { font-weight: 700; font-size: 1.05em; margin-bottom: 6px; }
.gt-muted { opacity: 0.7; font-size: 0.9em; }
.gt-br { text-decoration: underline; text-decoration-thickness: 2px; }
.gt-ne { outline: 2px solid #16a34a; outline-offset: -2px; }
.gt-pareto::after { content: " \\2605"; color: #eab308; }
.gt-dom { text-decoration: line-through; opacity: 0.55; }
.gt-row { color: #3b82f6; }
.gt-col { color: #ef4444; }
.gt-flex { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.gt-steps { margin: 6px 0 4px 0; padding-left: 1.3em; }
.gt-steps li { margin: 4px 0; line-height: 1.45; }
</style>
"""


def apply_rc(rc: dict | None = None) -> None:
    """Apply the shared matplotlib style globally (call once per notebook)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)
    if rc:
        plt.rcParams.update(rc)


@contextmanager
def rc_context(rc: dict | None = None):
    """Scoped matplotlib styling - preferred over mutating global rcParams."""
    import matplotlib.pyplot as plt

    params = dict(RC_PARAMS)
    if rc:
        params.update(rc)
    with plt.rc_context(params):
        yield
