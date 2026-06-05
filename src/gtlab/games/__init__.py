"""Library of ready-made games. Extend by adding a factory and registering it.

    from gtlab.games import prisoners_dilemma, REGISTRY
    g = prisoners_dilemma()
    g2 = REGISTRY["chicken"]()
"""
from .applied import cournot_duopoly, entry_deterrence
from .bayesian import (cautious_entrant, costly_diploma, honest_auctioneer,
                       public_project, reverse_procurement, shaded_bid_auction,
                       suspicious_buyer, vcg_assignment)
from .classic import (battle_of_the_sexes, chicken, matching_pennies,
                      prisoners_dilemma, rock_paper_scissors, stag_hunt)
from .correlated import (battle_of_sexes_ce, chicken_ce, prisoners_dilemma_ce,
                        rps_ce, stag_hunt_ce, traffic_coordination)
from .extensive_form import (battle_of_the_sexes_game, chain_store_game,
                            chicken_game, parametric_bos, stag_hunt_game)
from .normal_form import (cooperation_job, driver_police_game,
                         pick_a_number_game, privilege_game,
                         transform_payoffs, waste_game, worker_checking_game)
from .stochastic import market_entry_general_sum, patrol_game, security_game
from .zero_sum import \
    rock_paper_scissors as rock_paper_scissors_zs  # noqa: E501  (avoid name clash with classic NF version)
from .zero_sum import (micro_check_2x2, penalty_kick, security_audit,
                      weighted_rps)

# Name → factory. New games belong here so they are discoverable.
REGISTRY = {
    "prisoners_dilemma": prisoners_dilemma,
    "stag_hunt": stag_hunt,
    "chicken": chicken,
    "battle_of_the_sexes": battle_of_the_sexes,
    "matching_pennies": matching_pennies,
    "rock_paper_scissors": rock_paper_scissors,
    "cournot_duopoly": cournot_duopoly,
    "entry_deterrence": entry_deterrence,
    # extensive form
    "chain_store": chain_store_game,
    "battle_of_the_sexes_efg": battle_of_the_sexes_game,
    "parametric_bos_efg": parametric_bos,
    "stag_hunt_efg": stag_hunt_game,
    "chicken_efg": chicken_game,
    # stochastic
    "security_game": security_game,
    "patrol_game": patrol_game,
    "market_entry_general_sum": market_entry_general_sum,
    # bayesian
    "suspicious_buyer": suspicious_buyer,
    "cautious_entrant": cautious_entrant,
    "shaded_bid_auction": shaded_bid_auction,
    "honest_auctioneer": honest_auctioneer,
    "costly_diploma": costly_diploma,
    "reverse_procurement": reverse_procurement,
    "vcg_assignment": vcg_assignment,
    "public_project": public_project,
    # normal form (applied)
    "worker_checking_game": worker_checking_game,
    "driver_police_game": driver_police_game,
    "cooperation_job": cooperation_job,
    "privilege_game": privilege_game,
    "waste_game": waste_game,
    "pick_a_number_game": pick_a_number_game,
    # zero sum
    "micro_check_2x2": micro_check_2x2,
    "rock_paper_scissors_zs": rock_paper_scissors_zs,
    "weighted_rps": weighted_rps,
    "penalty_kick": penalty_kick,
    "security_audit": security_audit,
    # correlated
    "chicken_ce": chicken_ce,
    "battle_of_sexes_ce": battle_of_sexes_ce,
    "stag_hunt_ce": stag_hunt_ce,
    "prisoners_dilemma_ce": prisoners_dilemma_ce,
    "traffic_coordination": traffic_coordination,
    "rps_ce": rps_ce,
}

__all__ = [
    "prisoners_dilemma", "stag_hunt", "chicken", "battle_of_the_sexes",
    "matching_pennies", "rock_paper_scissors", "cournot_duopoly",
    "entry_deterrence",
    # extensive form
    "chain_store_game", "battle_of_the_sexes_game", "parametric_bos",
    "stag_hunt_game", "chicken_game",
    # stochastic
    "security_game", "patrol_game", "market_entry_general_sum",
    # bayesian
    "suspicious_buyer", "cautious_entrant", "shaded_bid_auction",
    "honest_auctioneer", "costly_diploma", "reverse_procurement",
    "vcg_assignment", "public_project",
    # normal form (applied)
    "worker_checking_game", "driver_police_game", "cooperation_job",
    "privilege_game", "waste_game", "pick_a_number_game", "transform_payoffs",
    # zero sum
    "micro_check_2x2", "rock_paper_scissors_zs", "weighted_rps",
    "penalty_kick", "security_audit",
    # correlated
    "chicken_ce", "battle_of_sexes_ce", "stag_hunt_ce",
    "prisoners_dilemma_ce", "traffic_coordination", "rps_ce",
    "REGISTRY",
]
