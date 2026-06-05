"""Every display method must emit uniform, gt-wrapped HTML and be capturable."""
import matplotlib

matplotlib.use("Agg")

import numpy as np

from gtlab import (CorrelatedGame, NormalFormGame, Procurement, PublicProject,
                   SpenceSignaling, VCGAssignment)
from gtlab.games import (battle_of_the_sexes, chicken, entry_deterrence,
                         prisoners_dilemma, rock_paper_scissors)
from gtlab.viz import capture


def _grab(fn):
    with capture() as sink:
        fn()
    out = "".join(sink.parts)
    assert "gt-wrap" in out and len(out) > 50, fn
    return out


def test_normal_form_displays():
    g = prisoners_dilemma()
    for fn in (g.summary, g.solve, g.explain):
        _grab(fn)
    assert "gt-steps" in _grab(g.explain)


def test_compare_renders_flex():
    out = _grab(lambda: NormalFormGame.compare(prisoners_dilemma(), chicken()))
    assert "gt-flex" in out


def test_zero_sum_displays():
    g = rock_paper_scissors()
    for fn in (g.summary, g.solve, g.explain):
        _grab(fn)


def test_correlated_displays():
    bos = battle_of_the_sexes()
    g = CorrelatedGame(bos.A, bos.B, name="BoS")
    for fn in (g.summary, g.compare_equilibria, g.explain):
        _grab(fn)


def test_extensive_form_displays():
    g = entry_deterrence()
    for fn in (g.solve, g.explain):
        _grab(fn)


def test_mechanism_displays():
    mechs = [SpenceSignaling(10, 20, 2, 1),
             VCGAssignment(np.array([[10, 4], [7, 9]], dtype=float)),
             PublicProject([6, 5, 2], 10),
             Procurement([2, 5], [0.5, 0.5], 2)]
    for m in mechs:
        _grab(m.summary)
