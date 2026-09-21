"""Load the model through nnterp. The handle everything else uses.

nnterp's StandardizedTransformer is the only model surface in the project: it
gives `layers` and `lm_head` the same names on every architecture, and it owns
the tokenizer. What it does *not* give us is written down in FINDINGS.md.
"""

from __future__ import annotations

from typing import Any

import torch
from nnterp import StandardizedTransformer

# `model.dtype` is part of the experiment's identity, not of the run: the same
# document at bf16 and at fp32 is two different experiments.
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load(spec: Any, device_map: str = "auto", weights: bool = True) -> StandardizedTransformer:
    """`spec` is a model block from either authoring format: both carry
    `key`, `revision` and `dtype`, which is all a loader needs.

    `weights=False` is nnsight's `dispatch=False`: the module tree on the
    meta device, the config and the tokenizer, and nothing downloaded but
    those. Everything the compiler asks is answered from it — including an
    interior's `.source`, which is the forward's *code*, not its weights.
    """
    if not weights:
        return StandardizedTransformer(
            spec.key, revision=spec.revision, dtype=DTYPES[spec.dtype],
            dispatch=False, device_map=None,
        )
    return StandardizedTransformer(
        spec.key,
        revision=spec.revision,
        dtype=DTYPES[spec.dtype],
        device_map=device_map,
    )
