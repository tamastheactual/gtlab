"""Single source of truth for colors, CSS, and matplotlib styling.

Every notebook used to carry its own copy of a ``C`` color dict, a
``_XXX_CSS`` string, and a ``plt.rcParams`` block. They are consolidated
here so a theme change happens once instead of six times.
"""
from __future__ import annotations

from contextlib import contextmanager

# ── Color palette ────────────────────────────────────────────────────────
# Based on the seaborn "deep" palette: a restrained, mid-tone, colorblind-
# friendly scheme that reads on both light and dark notebook backgrounds.
# Shared across HTML tables and matplotlib plots so the two always agree.
C = {
    "p1": "#4c72b0",       # Row / player 1   (deep blue)
    "p2": "#c44e52",       # Column / player 2 (muted red)
    # 8-digit hex (#RRGGBBAA) so the tints are valid in BOTH matplotlib and CSS.
    # Tints are ~30% alpha so region fills / BR-map cells read on white.
    "p1_light": "#4c72b04d",
    "p2_light": "#c44e524d",
    "ne": "#55a868",       # Nash equilibrium  (muted green)
    "ne_light": "#55a8684d",
    "ce": "#8172b3",       # Correlated equilibrium (muted violet)
    "cce": "#dd8452",      # Coarse correlated equilibrium (muted orange)
    "chance": "#dd8452",   # chance nodes (extensive form)
    "terminal": "#55a868",
    "terminal_light": "#55a8684d",  # terminal-node fill (extensive form)
    "pareto": "#c9a227",   # Pareto star (muted gold, not neon yellow)
    "grid": "#9aa0a6",
    "empty": "#e6e8eb",    # neutral fill for an EMPTY / N-A region (never for data)
    "cell": "#8080800f",
    "cell_alt": "#8080801f",
    "text": "#1f2937",     # plot annotations (figures render on white)
    "muted": "#8a93a0",
    "accent": "#64b5cd",   # muted cyan
}

# Ordered cycle for multi-series plots (seaborn "deep").
CYCLE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
         "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]

# matplotlib rcParams shared by every plot helper. Tuned for clean, slide-ready
# figures: soft off-white axes, grid behind data, muted spines/ticks, and a
# light rounded legend frame for readability when a legend sits near data.
RC_PARAMS = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfbfc",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 10.0,
    "axes.titlecolor": "#1f2937",
    "axes.labelsize": 11,
    "axes.labelcolor": "#374151",
    "axes.labelpad": 6.0,
    "axes.edgecolor": "#b8bcc2",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.color": "#9aa0a6",
    "grid.linewidth": 0.7,
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.2,
    "lines.solid_capstyle": "round",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#e2e4e8",
    "legend.fancybox": True,
    "legend.borderpad": 0.5,
    "legend.fontsize": 9.5,
    "figure.autolayout": True,
}


def _rc_with_cycle(extra: dict | None = None) -> dict:
    """RC_PARAMS plus the professional color cycle (cycler imported lazily)."""
    from cycler import cycler

    params = dict(RC_PARAMS)
    params["axes.prop_cycle"] = cycler(color=CYCLE)
    if extra:
        params.update(extra)
    return params

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
.gt-ne { outline: 2px solid #55a868; outline-offset: -2px; }
.gt-pareto::after { content: " \\2605"; color: #c9a227; }
.gt-dom { text-decoration: line-through; opacity: 0.7; }
.gt-row { color: #4c72b0; }
.gt-col { color: #c44e52; }
.gt-flex { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }
.gt-steps { margin: 6px 0 4px 0; padding-left: 1.3em; }
.gt-steps li { margin: 4px 0; line-height: 1.45; }
.gt-ok { color: #2e8b57; font-weight: 600; }
.gt-bad { color: #b3434b; font-weight: 600; }
</style>
"""


def apply_rc(rc: dict | None = None) -> None:
    """Apply the shared matplotlib style globally (call once per notebook)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(_rc_with_cycle(rc))


@contextmanager
def rc_context(rc: dict | None = None):
    """Scoped matplotlib styling - preferred over mutating global rcParams."""
    import matplotlib.pyplot as plt

    with plt.rc_context(_rc_with_cycle(rc)):
        yield
