"""Golden-value tests for the solver layer (the behavior-preserving core)."""
import numpy as np
import pytest

from gtlab import solvers
from gtlab.games import prisoners_dilemma, matching_pennies, rock_paper_scissors


def test_pure_nash_prisoners_dilemma():
    g = prisoners_dilemma()
    # Defect/Defect is the unique pure NE (index (1, 1)).
    assert g.pure_nash() == [(1, 1)]


def test_no_pure_nash_matching_pennies():
    g = matching_pennies()
    assert solvers.pure_nash(g.A, -g.A) == []


def test_best_response_masks_shape():
    g = prisoners_dilemma()
    br_row, br_col = solvers.br_masks(g.A, g.B)
    assert br_row.shape == g.shape
    assert br_col.shape == g.shape


def test_dominance_pd():
    g = prisoners_dilemma()
    # Cooperate is strictly dominated by Defect for the row player.
    dr = solvers.strictly_dominated_rows(g.A)
    assert dr == {0: 1}


def test_iesds_reduces_pd():
    g = prisoners_dilemma()
    A, B, rows, cols, log = solvers.iesds(g.A, g.B, g.row_actions, g.col_actions)
    assert A.shape == (1, 1)
    assert rows == ["Defect"] and cols == ["Defect"]
    assert len(log) == 2


def test_pareto_pd():
    g = prisoners_dilemma()
    pareto = set(solvers.pareto_optimal_cells(g.A, g.B))
    # Defect/Defect (1,1) is the only NON-Pareto cell.
    assert (1, 1) not in pareto
    assert (0, 0) in pareto


def test_welfare_summary():
    w = solvers.welfare_summary(np.array([3.0, 1.0]))
    assert w["utilitarian"] == 4.0
    assert w["egalitarian"] == 1.0
    assert w["nash"] == 3.0


@pytest.mark.skipif(
    pytest.importorskip("scipy", reason="scipy required") is None, reason="scipy"
)
def test_zero_sum_value_matching_pennies():
    g = matching_pennies()
    sol = solvers.solve_zero_sum(g.A)
    assert abs(sol["value"]) < 1e-6
    assert np.allclose(sol["p"], [0.5, 0.5], atol=1e-6)


def test_rps_value_is_zero():
    pytest.importorskip("scipy")
    g = rock_paper_scissors()
    sol = solvers.solve_zero_sum(g.A)
    assert abs(sol["value"]) < 1e-6
    assert np.allclose(sol["p"], [1/3, 1/3, 1/3], atol=1e-6)
