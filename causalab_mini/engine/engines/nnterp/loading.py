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
    """`spec` is a document's model block: `key`, `revision` and `dtype`,
    which is all a loader needs. Everything
    else is nnterp's and passes through — `device_map`, and `dispatch=False`
    for a meta-device shell: the module tree, the config and the tokenizer,
    nothing downloaded but those. Everything the compiler asks is answered
    from it, where every place is included. The same shell is what runs on
    NDIF.
    """
    if options.get("dispatch") is False:
        options.setdefault("device_map", None)
    else:
        # nnsight loads weights lazily, on the first trace, by *replacing* the
        # module — which would discard the freeze below along with the meta
        # shell it was applied to. Load them now, so what is frozen is what runs.
        options["dispatch"] = True
    if getattr(spec, "attn_implementation", None) is not None:
        # the document's, because it decides which tensors exist (the
        # attention pattern is only ever materialized under "eager")
        options.setdefault("attn_implementation", spec.attn_implementation)
    if options.get("attn_implementation") == "eager":
        # nnterp disables its attention-pattern row unless asked, and a
        # document says "eager" for exactly one reason: that tensor. Asking
        # also puts nnterp's own shape/sum/causality check behind it.
        options.setdefault("enable_attention_probs", True)
    model = StandardizedTransformer(
        spec.key, revision=spec.revision, dtype=DTYPES[spec.dtype], **options
    )
    # The model is an instrument, not a parameter. Without this a fit's
    # backward reaches every weight the patched forward touched and leaves a
    # `.grad` the size of the model behind (FINDINGS §1.15): nothing reads it,
    # because the optimizer holds only the featurizers, but the memory is real.
    # `eval()` is the other half — dropout in a measurement is noise. (On NDIF
    # the served model is the server's to freeze; this is the local one.)
    model._module.eval()
    model._module.requires_grad_(False)
    return model
