"""Engines: how a plan reaches tensors.

`base.py` is the contract — nine members, no implementation. `steps.py` is
everything a plan means that an engine has no opinion about: the walk over
steps, the fit loop, the metrics. `engines/` holds one directory per runtime:

    engines/nnterp   nnsight traces in one session, local or on NDIF
    engines/hooks    a plain HuggingFace model and torch forward hooks, here

The second engine writes those nine members and gets the rest for free — the
walk, the fit loop, the metrics and the write algebra are untouched by it, and
the two produce bit-identical numbers on the same document. That the
engine-specific part is this small is the finding, not the design; what the
hooks engine had to supply that nnterp was quietly providing is FINDINGS §6.
"""

from .base import Engine, EngineError
# nnterp before the hooks engine's transformers imports: nnterp imported after
# a `transformers.models` module can segfault the process.
from .engines.nnterp import NNterpEngine
from .engines.hooks import HooksEngine  # noqa: I001

__all__ = ["Engine", "EngineError", "HooksEngine", "NNterpEngine"]
