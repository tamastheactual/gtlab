"""HTML building blocks and the ``compare()`` capture mechanism.

Every notebook reimplemented table/card construction by hand and used a
fragile ``_display`` monkey-patch to capture output for side-by-side
comparison. Both live here once.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, List, Sequence

from .theme import CSS


def _ipython_display():
    """Return IPython's display fn, or a no-op fallback outside Jupyter."""
    try:
        from IPython.display import display  # type: ignore
        return display
    except Exception:  # pragma: no cover - non-notebook environments
        return lambda *a, **k: None


def _ipython_html(payload: str):
    try:
        from IPython.display import HTML  # type: ignore
        return HTML(payload)
    except Exception:  # pragma: no cover
        return payload


def show(html_body: str) -> None:
    """Render an HTML fragment in a Jupyter cell (CSS injected once per call).

    Routes through :func:`emit` so output is captured by :func:`capture`.
    """
    emit(html_body)


def card(title: str, body: str) -> str:
    """A titled card container."""
    return f'<div class="gt-card"><div class="gt-title">{title}</div>{body}</div>'


def note(text: str) -> str:
    """Muted helper/footnote text."""
    return f'<div class="gt-muted">{text}</div>'


def legend(*entries: str) -> str:
    """A muted, dot-separated legend line, e.g. ``legend("a = x", "b = y")``."""
    return note(" · ".join(entries))


def steps(items: Sequence[str]) -> str:
    """Render an ordered list of walkthrough steps.

    Each item is HTML; a leading ``<b>Title.</b>`` convention reads best.
    """
    lis = "".join(f"<li>{it}</li>" for it in items)
    return f'<ol class="gt-steps">{lis}</ol>'


def kv(pairs: Sequence[tuple[str, str]]) -> str:
    """A compact key/value paragraph block."""
    return "".join(f'<p><b>{k}</b> {v}</p>' for k, v in pairs)


def table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    row_headers: Sequence[str] | None = None,
    cell_classes: Sequence[Sequence[str]] | None = None,
) -> str:
    """Build an HTML table.

    Parameters
    ----------
    headers       : column header labels (top row).
    rows          : 2D sequence of pre-formatted cell strings.
    row_headers   : optional left-column labels, one per row.
    cell_classes  : optional CSS classes per cell, matching ``rows`` shape.
    """
    out: List[str] = ["<table>"]
    if headers:
        out.append("<tr>")
        if row_headers is not None:
            out.append("<th></th>")
        out.extend(f"<th>{h}</th>" for h in headers)
        out.append("</tr>")
    for i, row in enumerate(rows):
        out.append("<tr>")
        if row_headers is not None:
            out.append(f"<th>{row_headers[i]}</th>")
        for j, cell in enumerate(row):
            cls = ""
            if cell_classes is not None and cell_classes[i][j]:
                cls = f' class="{cell_classes[i][j]}"'
            out.append(f"<td{cls}>{cell}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


# ── compare() capture ──────────────────────────────────────────────────────
class CaptureSink:
    """Collects HTML emitted via :func:`capture` instead of rendering it."""

    def __init__(self) -> None:
        self.parts: List[str] = []

    def __call__(self, payload: Any) -> None:
        # Mirror IPython.display(HTML(...)): pull out the .data if present.
        self.parts.append(getattr(payload, "data", str(payload)))


# Module-level display hook used by emit(); swapped during capture().
_DISPLAY: Callable[[Any], None] | None = None


@contextmanager
def capture():
    """Temporarily redirect :func:`show` into a list of HTML strings.

    Usage::

        with capture() as sink:
            game_a.summary()
            game_b.summary()
        # sink.parts now holds the two HTML fragments
    """
    sink = CaptureSink()
    global _DISPLAY
    prev = _DISPLAY
    _DISPLAY = sink
    try:
        yield sink
    finally:
        _DISPLAY = prev


def emit(html_body: str) -> None:
    """Internal: route an HTML fragment to the active sink or to Jupyter."""
    payload = CSS + f'<div class="gt-wrap">{html_body}</div>'
    if _DISPLAY is not None:
        _DISPLAY(_ipython_html(payload))
    else:
        _ipython_display()(_ipython_html(payload))


def compare(items: Sequence[tuple[str, str]]) -> None:
    """Render ``(label, html_fragment)`` pairs side by side in a flex row."""
    cols = "".join(
        f'<div class="gt-card"><div class="gt-title">{label}</div>{body}</div>'
        for label, body in items
    )
    emit(f'<div class="gt-flex">{cols}</div>')


def compare_via(games: Sequence[Any], method: str = "summary") -> None:
    """Render several games side by side by capturing one display ``method`` each.

    Factors the capture/loop boilerplate that every game class repeated. Each
    game's ``.name`` labels its column.
    """
    items = []
    for g in games:
        with capture() as sink:
            getattr(g, method)()
        items.append((getattr(g, "name", type(g).__name__), "".join(sink.parts)))
    compare(items)
