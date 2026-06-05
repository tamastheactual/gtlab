"""A tiny per-instance memoization decorator for game-class analysis methods.

Game classes are dataclasses wrapping numpy arrays, so they are unhashable and
``functools.lru_cache`` does not apply. The payoff data is treated as immutable
after construction, so caching results on the instance is safe and lets repeated
display calls (and ``compare_via``) reuse expensive solves instead of redoing
them. Mutating a payoff matrix in place after the first call is unsupported
(call :func:`clear_cache` if you must).
"""
from __future__ import annotations

import functools
from typing import Callable


def cached_method(func: Callable) -> Callable:
    """Memoize a method's result on the instance, keyed by ``(name, args)``."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cache = self.__dict__.setdefault("_cache", {})
        key = (func.__name__, args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
        return cache[key]

    return wrapper


def clear_cache(instance) -> None:
    """Drop all memoized results on ``instance`` (use after mutating payoffs)."""
    instance.__dict__.pop("_cache", None)
