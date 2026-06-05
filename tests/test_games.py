"""Smoke tests: every registered game builds and exposes its core API."""
import numpy as np

from gtlab.core import ExtensiveFormGame
from gtlab.games import REGISTRY, entry_deterrence


def test_registry_builds_all():
    for name, factory in REGISTRY.items():
        g = factory()
        assert g is not None
        assert hasattr(g, "name")


def test_normal_form_games_have_shapes():
    for name in ["prisoners_dilemma", "chicken", "cournot_duopoly"]:
        g = REGISTRY[name]()
        assert g.shape[0] >= 2 and g.shape[1] >= 2


def test_backward_induction_entry_deterrence():
    g = entry_deterrence()
    res = g.backward_induction()
    # Incumbent accommodates (fighting is not credible) → entrant enters.
    assert res["strategy"]["incumbent"] == "Accommodate"
    assert res["strategy"]["root"] == "Enter"
    assert np.allclose(res["value"], [1, 2])


def test_extensive_form_validation():
    bad = {"root": {"player": 0, "actions": {"L": "missing"}}}
    try:
        ExtensiveFormGame(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for dangling child")
