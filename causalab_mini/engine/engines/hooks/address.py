"""How the hooks engine answers the address questions.

A forward hook reaches a module's boundary and nothing inside its forward,
so this engine's vocabulary is the part of mini's names that are one: ten
places, each a module `loading.standardized` finds in the raw tree and the
side of it the tensor is on. That table is the translation, and it is kept
this small on purpose — a component outside it is refused by name, not
rederived.

Everything else — whether the checkpoint has the place, its axes, every
width — is asked of a weightless nnterp shell of the same checkpoint, so the
two engines size a plan alike.
"""

from __future__ import annotations

from typing import Any

from ....address import Address, AddressError
from ..nnterp import address as nnterp_address

#: The places a forward hook reaches: one of the modules `standardized` finds
#: — a per-layer list, or one whole-model module — and which side of it.
PLACES = {
    "input_ids": ("embed_tokens", "input"),
    "embeddings": ("embed_tokens", "output"),
    "block_input": ("layers", "input"),
    "attention_output": ("attentions", "output"),
    "mlp_input": ("mlps", "input"),
    "mlp_output": ("mlps", "output"),
    "block_output": ("layers", "output"),
    "ln_final": ("ln_final", "output"),
    "lm_head": ("lm_head", "output"),
    # the model's own output, whose `logits` field is the tensor
    "logits": ("model", "output"),
}


def locate(shell: Any, component: str, layer: int | None) -> Address:
    """The address of `(component, layer)`, or a refusal at compile time.

    A hook fires at a module boundary, and nnsight reaches inside by
    recompiling the forward (`.source`), which hooks have no equivalent of.
    Building one would be a different engine — patch the module's `forward`
    to expose the tensor, or hook `q_proj` and reimplement the reshape and
    the rotary embedding the query has had by the time it is the query.
    """
    if component not in PLACES:
        raise AddressError(
            f"component {component!r} is not one of the module boundaries a forward hook "
            f"reaches here, {sorted(PLACES)}. A hook sees a module's boundary and nothing "
            "inside its forward; nnsight reaches inside through `.source`, which has no "
            "hooks equivalent; the nnterp engine reaches it."
        )
    return nnterp_address.locate(shell, component, layer)


def resolve(address: Address, names: Any) -> Any:
    """The raw module an address is hooked on, in the tree `standardized`
    translated: a per-layer place's module at its layer."""
    found = getattr(names, PLACES[address.component][0])
    return found if address.layer is None else found[address.layer]
