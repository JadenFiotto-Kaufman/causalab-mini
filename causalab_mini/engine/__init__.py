"""Engines: how a plan reaches tensors.

`base.py` is the contract — seven members, no implementation. `steps.py` is
everything a plan means that an engine has no opinion about: the walk over
steps, the fit loop, the metrics. `engines/` holds one directory per runtime;
today that is `engines/nnterp`, nnsight traces in one session, local or on
NDIF.

A second engine (torch hooks, say) writes those seven members and gets the rest
for free. That the engine-specific part is this small is the finding, not the
design.
"""

from .base import Engine
from .engines.nnterp import NNterpEngine

__all__ = ["Engine", "NNterpEngine"]
