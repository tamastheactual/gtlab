"""Number / probability / money formatters shared across the package.

Replaces the six independent copies of ``_fmt`` / ``_fmtv`` / ``_fmt_prob``
that had already drifted apart (e.g. fraction denominator caps of 20 vs 100).
This is the one canonical behavior.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable

# Canonical denominator cap when rendering probabilities as fractions.
PROB_DENOM_LIMIT = 64
EPS = 1e-9


def fmt(x: float, prec: int = 4) -> str:
    """Format a scalar: integer when whole, else trimmed to ``prec`` decimals."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(xf - round(xf)) < EPS:
        return str(int(round(xf)))
    s = f"{xf:.{prec}f}".rstrip("0").rstrip(".")
    # Snap a value that rounds to zero at display precision to "0" (avoids "-0").
    if s in ("-0", "0", "-0.", "0."):
        return "0"
    return s


def fmt_vec(arr: Iterable[float], prec: int = 4) -> str:
    """Format a vector as ``(a, b, c)``."""
    return "(" + ", ".join(fmt(v, prec) for v in arr) + ")"


def fmt_prob_vec(arr: Iterable[float], denom_limit: int = PROB_DENOM_LIMIT) -> str:
    """Format a probability vector as ``(p1, p2, ...)`` using fraction-aware probs."""
    return "(" + ", ".join(fmt_prob(v, denom_limit) for v in arr) + ")"


def fmt_prob(x: float, denom_limit: int = PROB_DENOM_LIMIT, prec: int = 3) -> str:
    """Format a probability as a reduced fraction when clean, else a decimal."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(xf) < EPS:
        return "0"
    if abs(xf - 1.0) < EPS:
        return "1"
    frac = Fraction(xf).limit_denominator(denom_limit)
    if abs(float(frac) - xf) < 1e-6 and frac.denominator <= denom_limit:
        return f"{frac.numerator}/{frac.denominator}"
    return f"{xf:.{prec}f}".rstrip("0").rstrip(".")


def fmt_money(x: float, symbol: str = "$") -> str:
    """Format a monetary amount, e.g. ``$1,250``."""
    xf = float(x)
    if abs(xf - round(xf)) < EPS:
        return f"{symbol}{int(round(xf)):,}"
    return f"{symbol}{xf:,.2f}"


def fmt_pct(x: float, prec: int = 1) -> str:
    """Format a fraction as a percentage, e.g. ``42.5%``."""
    return f"{float(x) * 100:.{prec}f}%"
