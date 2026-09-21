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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer



class HooksEngineError(ValueError):
    pass


#: The same table as the nnterp loader's: a dtype is part of the experiment's
#: identity, and the document spells it in its own vocabulary.
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def load(spec: Any, device_map: str = "cpu", dispatch: bool = True) -> tuple[Any, Any]:
    """The model and the tokenizer — two loads, because nothing pairs them.

    `dispatch=False`, spelled as nnsight spells it, builds the module tree
    from the config on the meta device: the shape of the model and nothing
    else, which is all the compiler needs. Unlike nnsight's, this shell can
    never run — there is no server for hooks — so the engine remembers.
    """
    if dispatch:
        model = AutoModelForCausalLM.from_pretrained(
            spec.key, revision=spec.revision, dtype=DTYPES[spec.dtype], device_map=device_map
        )
    else:
        config = AutoConfig.from_pretrained(spec.key, revision=spec.revision)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=DTYPES[spec.dtype])
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

    Three of the six are published by transformers and needed no family
    knowledge at all — the decoder under `base_model_prefix`, the head under
    `get_output_embeddings`, the embeddings under `get_input_embeddings`. The
    other three are guesses, and that split is the measurement (FINDINGS §6):

    * `layers` — the decoder's only `nn.ModuleList`.
    * `attentions` / `mlps` — the block's child whose class name ends in
      `Attention` / `MLP`. True of `LlamaAttention`/`LlamaMLP` and
      `GPT2Attention`/`GPT2MLP`; a convention, not a contract.
    * `ln_final` — the decoder's only normalization child.

    Each guess raises rather than picking a first match, because a wrong
    module here reads a real tensor from the wrong place and nothing
    downstream could tell.
    """
    decoder = model.base_model
    stacks = [child for child in decoder.children() if isinstance(child, nn.ModuleList)]
    if len(stacks) != 1:
        raise HooksEngineError(
            f"{type(model).__name__}: its decoder has {len(stacks)} ModuleList children, "
            "so which one is the layer stack is not decidable here. nnterp's rename "
            "table names it per family; this engine guesses."
        )
    layers = stacks[0]
    return SimpleNamespace(
        layers=layers,
        attentions=[_by_class(block, "Attention", "attentions") for block in layers],
        mlps=[_by_class(block, "MLP", "mlps") for block in layers],
        ln_final=_norm(decoder),
        embed_tokens=model.get_input_embeddings(),
        lm_head=model.get_output_embeddings(),
    )


def _by_class(block: nn.Module, suffix: str, name: str) -> nn.Module:
    """The block's one child whose class name ends in `suffix`."""
    found = [child for child in block.children() if type(child).__name__.endswith(suffix)]
    if len(found) != 1:
        raise HooksEngineError(
            f"{type(block).__name__}: {len(found)} children have a class name ending "
            f"in {suffix!r}, so {name!r} is not decidable here — transformers "
            "publishes no accessor for it and this engine matches on the naming "
            f"convention. Children: {[type(c).__name__ for c in block.children()]}"
        )
    return found[0]


def _norm(decoder: nn.Module) -> nn.Module:
    """The decoder's one normalization child, outside the layer stack."""
    found = [
        child
        for name, child in decoder.named_children()
        if not isinstance(child, nn.ModuleList) and "norm" in type(child).__name__.lower()
    ]
    if len(found) != 1:
        raise HooksEngineError(
            f"{type(decoder).__name__}: {len(found)} normalization children outside the "
            "layer stack, so 'ln_final' is not decidable here. Children: "
            f"{[type(c).__name__ for c in decoder.children()]}"
        )
    return found[0]
