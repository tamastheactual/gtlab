"""Regression tests for memoization and the vectorized CE/CCE constraints."""
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import numpy as np

import gtlab.solvers as S
from gtlab import CorrelatedGame
from gtlab._memo import clear_cache
from gtlab.games import battle_of_the_sexes, prisoners_dilemma, rock_paper_scissors
from gtlab.viz import capture


def _count(fname, action):
    orig = getattr(S, fname)
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    with mock.patch.object(S, fname, wrapped), capture():
        action()
    return calls["n"]


def test_normal_form_memoizes_across_displays():
    g = prisoners_dilemma()
    n = _count("pure_nash", lambda: (g.summary(), g.solve(), g.explain()))
    assert n == 1


def test_correlated_memoizes_lp():
    bos = battle_of_the_sexes()
    cg = CorrelatedGame(bos.A, bos.B, name="BoS")
    n = _count("find_ce", lambda: (cg.summary(), cg.compare_equilibria()))
    assert n == 1


def test_zero_sum_memoizes_value():
    z = rock_paper_scissors()
    n = _count("solve_zero_sum", lambda: (z.solve(), z.explain(), z.verify()))
    assert n == 1


def test_cache_keyed_by_args():
    bos = battle_of_the_sexes()
    cg = CorrelatedGame(bos.A, bos.B)
    r = cg.find_ce("row")
    w = cg.find_ce("welfare")
    assert r["eu_row"] >= w["eu_row"] - 1e-6           # row-opt favors the row player
    assert cg.find_ce("row")["eu_row"] == r["eu_row"]  # repeat is a cache hit


def test_clear_cache():
    z = rock_paper_scissors()
    z.solve_value()
    assert "_cache" in z.__dict__
    clear_cache(z)
    assert "_cache" not in z.__dict__


def _constraints_loop(A, B, coarse):
    """Reference O(mn) loop implementation, kept only for the equivalence test."""
    m, n = A.shape
    N = m * n
    rows = []
    if coarse:
        for ip in range(m):
            r = np.zeros(N)
            for i in range(m):
                for j in range(n):
                    r[i * n + j] = A[i, j] - A[ip, j]
            rows.append(-r)
        for jp in range(n):
            r = np.zeros(N)
            for i in range(m):
                for j in range(n):
                    r[i * n + j] = B[i, j] - B[i, jp]
            rows.append(-r)
    else:
        for i in range(m):
            for ip in range(m):
                if ip == i:
                    continue
                r = np.zeros(N)
                for j in range(n):
                    r[i * n + j] = A[i, j] - A[ip, j]
                rows.append(-r)
        for j in range(n):
            for jp in range(n):
                if jp == j:
                    continue
                r = np.zeros(N)
                for i in range(m):
                    r[i * n + j] = B[i, j] - B[i, jp]
                rows.append(-r)
    return np.array(rows)


def test_vectorized_constraints_match_loop():
    from gtlab.solvers import correlated as ce
    rng = np.random.default_rng(0)
    for (m, n) in [(2, 2), (3, 2), (2, 3), (3, 4), (5, 3)]:
        A = rng.normal(size=(m, n))
        B = rng.normal(size=(m, n))
        for coarse in (False, True):
            ref = _constraints_loop(A, B, coarse)
            vec = ce._constraints(A, B, coarse)
            assert vec.shape == ref.shape
            assert np.allclose(vec, ref)
