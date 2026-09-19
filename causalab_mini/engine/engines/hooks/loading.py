"""Load a plain HuggingFace model, and give it nnterp's names by hand.

The nnterp engine's `loading.py` is one call because `StandardizedTransformer`
answers everything downstream asks: the tokenizer, the layer count, the widths,
and `layers`/`lm_head`/`attentions` under the same spellings on every family.
This file is what that costs when nothing has been standardized, and the answer
is `standardized` below — the whole translation layer, and the measurement this
engine exists to produce (FINDINGS §6).

Everything this engine knows about the shape of a raw HuggingFace tree is in
this file. `engine.py` beside it knows only addresses and hooks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ....plan.document import ModelSpec


class HooksEngineError(ValueError):
    pass


#: The same table as the nnterp loader's: a dtype is part of the experiment's
#: identity, and the document spells it in its own vocabulary.
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load(spec: ModelSpec, device_map: str = "cpu") -> tuple[Any, Any]:
    """The model and the tokenizer — two loads, because nothing pairs them."""
    model = AutoModelForCausalLM.from_pretrained(
        spec.key, revision=spec.revision, dtype=DTYPES[spec.dtype], device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.key, revision=spec.revision)
    # nnsight forces this on every model it loads (`modeling/transformers.py`:
    # `self.tokenizer.padding_side = "left"`), and the numbers depend on it: a
    # right-padded batch gives the real tokens different position ids, so the
    # same document scores differently. Matching it is what makes the two
    # engines comparable at all. FINDINGS §6.3.
    tokenizer.padding_side = "left"
    return model, tokenizer


def standardized(model: Any) -> Any:
    """nnterp's names, against a raw HuggingFace tree.

    `Address.path` is written in nnterp's spellings — `layers.{L}`, `lm_head` —
    and a raw Llama has `model.layers` while a raw GPT-2 has `transformer.h`.
    This returns an object carrying those spellings, so `Address.resolve` walks
    it unchanged and no other line of this engine has to know the difference.

    Neither lookup needed a family table, which is the finding: transformers
    publishes the decoder under `base_model_prefix` and the head under
    `get_output_embeddings`. What it does not publish is the layer stack, so
    that one is "the decoder's only `ModuleList`" — true of every decoder-only
    family here, and exactly the guess nnterp replaces with a maintained list
    of names. `attentions` is absent because this engine refuses the one
    component that would reach through it; see `engine.locate`.
    """
    stacks = [child for child in model.base_model.children() if isinstance(child, nn.ModuleList)]
    if len(stacks) != 1:
        raise HooksEngineError(
            f"{type(model).__name__}: its decoder has {len(stacks)} ModuleList children, "
            "so which one is the layer stack is not decidable here. nnterp's rename "
            "table names it per family; this engine guesses."
        )
    return SimpleNamespace(layers=stacks[0], lm_head=model.get_output_embeddings())
