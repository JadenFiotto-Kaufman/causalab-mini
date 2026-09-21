"""Engines: how a plan reaches tensors.

`base.py` is the contract — seven members, no implementation. `steps.py` is
everything a plan means that an engine has no opinion about: the walk over
steps, the fit loop, the metrics. `engines/` holds one directory per runtime:

    engines/nnterp   nnsight traces in one session, local or on NDIF
    engines/hooks    a plain HuggingFace model and torch forward hooks, here

The second engine writes those seven members and gets the rest for free — the
walk, the fit loop, the metrics and the write algebra are untouched by it, and
the two produce bit-identical numbers on the same document. That the
engine-specific part is this small is the finding, not the design; what the
hooks engine had to supply that nnterp was quietly providing is FINDINGS §6.
"""

from .base import Engine, EngineError
from .engines.hooks import HooksEngine
from .engines.nnterp import NNterpEngine

__all__ = ["Engine", "EngineError", "HooksEngine", "NNterpEngine"]
