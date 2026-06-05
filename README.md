# gtlab — Game Theory Lab

[![CI](https://github.com/tamastheactual/gtlab/actions/workflows/ci.yml/badge.svg)](https://github.com/tamastheactual/gtlab/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gtlab.svg)](https://pypi.org/project/gtlab/)
[![Python](https://img.shields.io/pypi/pyversions/gtlab.svg)](https://pypi.org/project/gtlab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A teaching toolkit for the ELTE Game Theory course. It consolidates the solvers,
example games, and Jupyter visualizations that previously lived (duplicated) in
six separate Colab notebooks into one installable, extensible package.

## Install

```bash
pip install -e ".[full]"     # full = nashpy + pulp extras (recommended in Colab)
pip install -e .             # core only (numpy / scipy / matplotlib / pandas)
```

## Quick start

```python
import gtlab
gtlab.apply_rc()                         # shared matplotlib styling (call once)

from gtlab.games import prisoners_dilemma
prisoners_dilemma().solve()              # annotated bimatrix in a Jupyter cell
```

Build your own game:

```python
import numpy as np
from gtlab import NormalFormGame

g = NormalFormGame(
    A=np.array([[3, 0], [5, 1]]),
    B=np.array([[3, 5], [0, 1]]),
    row_actions=["Cooperate", "Defect"],
    col_actions=["Cooperate", "Defect"],
    name="My Game",
)
g.explain()
```

## Architecture

The design separates the three concerns that were tangled together in the
notebooks:

```text
gtlab/
├── core/      game classes — hold data, expose a thin API
│   ├── normal_form.py      NormalFormGame
│   ├── zero_sum.py         ZeroSumGame
│   ├── correlated.py       CorrelatedGame
│   ├── stochastic.py       StochasticGame
│   ├── extensive_form.py   ExtensiveFormGame
│   └── bayesian.py         PostedPrice, FirstPriceAuction, SecondPriceAuction
├── solvers/   pure algorithms — numpy in, numpy/dict out, no display
│   ├── best_response.py    nash.py          dominance.py
│   ├── pareto.py           linprog.py       value_iteration.py
│   ├── welfare.py          learning.py      correlated.py
├── viz/       display layer — ONE theme, formatters, HTML, plots
│   ├── theme.py  format.py  html.py  plots.py
└── games/     ready-made example games + REGISTRY
```

Rule of thumb: **math goes in `solvers/`, rendering goes in `viz/`, and the
classes in `core/` just wire them together.** A theme tweak is one edit in
`viz/theme.py`; a new algorithm is one file in `solvers/`; a new example is one
factory in `games/`.

## Extending

| To add… | …do this |
|---|---|
| a new example game | write a factory in `games/`, add it to `REGISTRY` |
| a new algorithm | add a pure function in `solvers/`, export it |
| a new game type | add a class in `core/` that calls `solvers` + `viz` |
| restyle output | edit `viz/theme.py` (colors / CSS / rcParams) |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The solver layer has golden-value tests so refactors stay behavior-preserving.

## Notebook migration

Each lecture notebook drops its ~2,000-line engine class and imports from
`gtlab` instead, keeping only narrative and example-specific calls. The methods
the notebooks relied on (`.summary()`, `.solve()`, `.explain()`, `.plot_*()`)
are preserved on the core classes.

```python
# old: 1,800 lines of NormalFormGame defined inline
# new:
from gtlab.games import prisoners_dilemma
prisoners_dilemma().solve(show_br=True, show_ne=True)
```
