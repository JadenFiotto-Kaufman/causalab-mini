"""Agnostic operations: tensors in, tensors out.

Nothing here knows what a model is, what a layer is or where a tensor came
from. `intervene.py` is the seam — gather, scatter, and the one write that all
mechanisms and all featurizers go through; `featurizer.py` is the rotation DAS
fits, which is a `Featurizer` and nothing more; `metrics.py` turns logits into
one number per row.

The seam's surface is re-exported here, so a caller writes `ops.apply_write`
and `ops.gather` without caring which file they live in.
"""

from .intervene import (
    FEATURIZERS,
    MECHANISMS,
    Featurizer,
    Identity,
    Mechanism,
    apply_write,
    gather,
    scatter,
    swap,
)

__all__ = [
    "FEATURIZERS",
    "MECHANISMS",
    "Featurizer",
    "Identity",
    "Mechanism",
    "apply_write",
    "gather",
    "scatter",
    "swap",
]
