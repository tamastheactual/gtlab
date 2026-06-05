"""Deduplicated game-theory algorithms shared by every game class.

Each function is pure (numpy in, numpy/dict out) and free of display concerns,
so it can be used standalone, tested in isolation, or composed by the core
game classes.
"""
from . import correlated
from .best_response import (best_response_to_mixed, best_responses_to_col,
                            best_responses_to_row, br_masks, exploitability)
from .correlated import check_ce, check_cce, find_ce, find_cce
from .dominance import (iesds, strictly_dominated_cols,
                        strictly_dominated_rows)
from .learning import fictitious_play, hedge
from .linprog import complementary_slackness, solve_zero_sum
from .nash import all_equilibria, ne_mask, pure_nash
from .pareto import pareto_frontier, pareto_optimal_cells
from .value_iteration import shapley_operator, stage_game, value_iteration
from .welfare import (best_outcome, egalitarian, nash_welfare, utilitarian,
                      welfare_summary)

__all__ = [
    "best_responses_to_col", "best_responses_to_row", "br_masks",
    "best_response_to_mixed", "exploitability",
    "pure_nash", "ne_mask", "all_equilibria",
    "strictly_dominated_rows", "strictly_dominated_cols", "iesds",
    "pareto_optimal_cells", "pareto_frontier",
    "solve_zero_sum", "complementary_slackness",
    "stage_game", "shapley_operator", "value_iteration",
    "utilitarian", "egalitarian", "nash_welfare", "welfare_summary",
    "best_outcome",
    "hedge", "fictitious_play",
    "find_ce", "find_cce", "check_ce", "check_cce", "correlated",
]
