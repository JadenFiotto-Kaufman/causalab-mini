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


def load(spec: Any, **options: Any) -> StandardizedTransformer:
    """`spec` is a model block from either authoring format: both carry
    `key`, `revision` and `dtype`, which is all a loader needs. Everything
    else is nnterp's and passes through — `device_map`, and `dispatch=False`
    for a meta-device shell: the module tree, the config and the tokenizer,
    nothing downloaded but those. Everything the compiler asks is answered
    from it, including an interior's `.source`, which is the forward's
    *code*. The same shell is what runs on NDIF.
    """
    if options.get("dispatch") is False:
        options.setdefault("device_map", None)
    if getattr(spec, "attn_implementation", None) is not None:
        # the document's, because it decides which tensors exist (the
        # attention pattern is only ever materialized under "eager")
        options.setdefault("attn_implementation", spec.attn_implementation)
    return StandardizedTransformer(
        spec.key, revision=spec.revision, dtype=DTYPES[spec.dtype], **options
    )
