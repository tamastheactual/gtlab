"""Shared display layer: theme, formatters, HTML builders, plot helpers."""
from .format import fmt, fmt_money, fmt_pct, fmt_prob, fmt_prob_vec, fmt_vec
from .html import (capture, card, compare, compare_via, emit, kv, legend, note,
                   show, steps, table)
from .theme import C, CSS, RC_PARAMS, apply_rc, rc_context

__all__ = [
    "fmt", "fmt_vec", "fmt_prob", "fmt_prob_vec", "fmt_money", "fmt_pct",
    "show", "emit", "card", "table", "compare", "compare_via", "capture",
    "note", "legend", "steps", "kv",
    "C", "CSS", "RC_PARAMS", "apply_rc", "rc_context",
]
