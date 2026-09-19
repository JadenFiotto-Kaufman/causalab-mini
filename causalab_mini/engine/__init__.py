"""Engines: how a plan reaches tensors.

`base.py` is the contract — three classmethods, no implementation. `steps.py`
is everything a plan means that an engine has no opinion about: the walk, the
fit loop, the metrics. `nnterp.py` is the one engine that exists, and it is
nnsight traces in one session, local or on NDIF.

A second engine (torch hooks, say) writes `open` and `forward` and gets the
rest for free. That the engine-specific part is this small is the finding, not
the design.
"""

from .base import Engine
from .nnterp import NNterpEngine

__all__ = ["Engine", "NNterpEngine"]
