"""Load the model through nnterp. The handle everything else uses.

nnterp's StandardizedTransformer is the only model surface in the project: it
gives `layers` and `lm_head` the same names on every architecture, and it owns
the tokenizer. What it does *not* give us is written down in FINDINGS.md.
"""

from __future__ import annotations

import torch
from nnterp import StandardizedTransformer

from ..plan.document import ModelSpec

# `model.dtype` is part of the experiment's identity, not of the run: the same
# document at bf16 and at fp32 is two different experiments.
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load(spec: ModelSpec, device_map: str = "auto") -> StandardizedTransformer:
    return StandardizedTransformer(
        spec.key,
        revision=spec.revision,
        dtype=DTYPES[spec.dtype],
        device_map=device_map,
    )
