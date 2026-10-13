"""WHERE: mini's names for the places a tensor can be read, what is true of
each place on any runtime, and what an address of one says.

A component is a name — `block_output`, `attention_query`, `mlp_activation`
— and what the place IS, whatever runs it: whether it is one per layer, its
axes, and whether it may be written. That is the whole table. Everything
about reaching the place — which module, which value off it, where it comes
in the forward, how deep the runtime has to instrument to get at it, how wide
it is on this checkpoint — is that engine's, answered live, in its
own `engines/<name>/address.py`, asked through `Engine.locate`, `width` and
`heads`. nnterp is the reference engine, and its `address.py` is the fullest
mapping: every name here is one of nnterp 2.0's standard values or a module
boundary beside one. Every other engine maps the same names onto its own
runtime, and how much mapping that takes is a measurement of what the
standardization is worth.

The axes are in nnterp's axis names, where an axis that is a model fact is
named as the size attribute that holds it (`hidden_size`, `num_heads`). A
test holds each row's axes to nnterp's own layout for the value it names.

An `Address` is pure data — the document's `(component, layer)` plus the
layout the locating engine declares it will serve there — so it pickles,
prints and travels in a plan, and nothing on it reads this table again. It
holds no routing and no order: the engine that runs the plan finds each
place again on the model it holds, and puts the taps in whatever order its
runtime needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The three axes of the residual stream, which is what most places carry.
#: Axis names are nnterp's (`nnterp.components` exports them as constants);
#: they are spelled here as literals so nothing here imports nnterp, and a
#: test holds the two spellings together.
_STREAM = ("batch", "seq", "hidden_size")


@dataclass(frozen=True)
class Component:
    """What one component is, on any runtime."""

    #: The tensor's axes, in nnterp's axis names.
    dims: tuple[str, ...]
    #: Whether this place is one per layer. A document gives a layer for one
    #: and may not for the others.
    per_layer: bool = True
    #: A tensor that may be read and never written: the token ids are the
    #: model's input, and swapping a float activation into them means nothing.
    read_only: bool = False


COMPONENTS = {
    # --- inside the attention: the call that hands the heads to the attention
    # implementation, and its result.
    # projected, reshaped to heads and rotated
    "attention_query": Component(("batch", "num_heads", "seq", "qk_head_dim")),
    # the key heads' keys, before a grouped model repeats them: narrower than
    # the query by the grouping ratio, at the same per-head width
    "attention_key": Component(("batch", "num_kv_heads", "seq", "qk_head_dim")),
    # the softmax's argument: QKᵀ·scale with the causal and padding mask
    # already added, so a masked key is -inf here and 0 after
    "attention_scores": Component(("batch", "num_heads", "query", "key")),
    # The attention pattern: the dropout call's output rather than the
    # softmax's, which is the pattern the values are actually mixed with on
    # every family (after the cast, and after an attention sink has been
    # dropped) and the identity in eval mode. A write here is "make this head
    # attend there".
    "attention_probs": Component(("batch", "num_heads", "query", "key")),
    # each head's mix of the values, before the heads are merged for the
    # output projection
    "attention_z": Component(("batch", "seq", "num_heads", "head_dim")),
    "input_ids": Component(("batch", "seq"), per_layer=False, read_only=True),
    # The embedding table's output, before layer 0: `block_input` at layer 0
    # is the same tensor on a model with nothing between them (GPT-2 adds
    # position embeddings), and this is the one address that names no layer.
    "embeddings": Component(_STREAM, per_layer=False),
    # --- the block. block_mid = block_input + attention_output, block_output =
    # block_mid + mlp_output, mlp_input_norm = mlp_input on a pre-norm block.
    "block_input": Component(_STREAM),
    "attention_input_norm": Component(_STREAM),
    # every head's result side by side, as the output projection takes it:
    # heads · head_dim wide, which is not hidden_size on Qwen3 or Gemma
    "attention_premix": Component(("batch", "seq", "num_heads", "head_dim")),
    "attention_output": Component(_STREAM),
    "block_mid": Component(_STREAM),
    "mlp_input_norm": Component(_STREAM),
    "mlp_input": Component(_STREAM),
    "mlp_activation": Component(("batch", "seq", "intermediate_size")),
    # act(gate)·up on a gated MLP, the activation itself on GPT-2, which has
    # no gate — the same place, a different tensor
    "mlp_neuron_output": Component(("batch", "seq", "intermediate_size")),
    "mlp_output": Component(_STREAM),
    "block_output": Component(_STREAM),
    "ln_final": Component(_STREAM, per_layer=False),
    "lm_head": Component(("batch", "seq", "vocab_size"), per_layer=False),
    # The model's own output, which is not always the head's: Gemma-2 caps it
    # with `final_logit_softcapping`, so a metric scored on `lm_head` there is
    # scored on numbers the model does not predict from.
    "logits": Component(("batch", "seq", "vocab_size"), per_layer=False),
}


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """One tap: the document's words, `(component, layer)`, and one fact the
    locating engine resolved — the layout it declares it will serve there.

    That one is stamped rather than read off the table because an engine
    may serve another layout than the standard one — a vLLM engine serves
    flat `[tokens, hidden_size]` rows with no batch axis — and the compiler
    sizes, slices and checks a site against what will actually arrive.
    `Component.dims` is the language's statement; on the nnterp engine a
    test holds the two equal.

    Nothing else is resolved into it: which module, which value off it, and
    in what order a forward reaches it are the engine's, found again at
    execute from the model it holds. A plan is mini's language end to end."""

    component: str
    layer: int | None
    #: The tensor's axes, in nnterp's names (`("batch", "num_heads", "seq",
    #: "qk_head_dim")`), as the locating engine will serve them.
    dims: tuple[str, ...]

    @property
    def key_axis(self) -> bool:
        """Whether this tensor's last axis is the keys of the padded batch:
        true of the pattern and the scores under it. A value read there only
        means something beside a prompt laid out the same way."""
        return self.dims[-1] == "key"

    @property
    def heads_kind(self) -> str | None:
        """"q" or "kv": which heads this tensor is per, if it is per head.
        Such a tensor is handed on flat, `(rows, w, heads · per_head)`, whether
        the model holds it flat (the output projection's input) or as two axes
        (the query) — so a site may name `heads`, and a featurizer sees one
        width."""
        return _heads(self.dims)

    @property
    def seq_axis(self) -> int:
        """Which axis of the tensor the sequence runs along — the pattern's
        queries, on the pattern."""
        for name in ("seq", "query"):
            if name in self.dims:
                return self.dims.index(name)
        raise AddressError(f"component {self.component!r} has no sequence axis: its axes are {self.dims}")


def _heads(axes: tuple[str, ...]) -> str | None:
    return "q" if "num_heads" in axes else "kv" if "num_kv_heads" in axes else None


def layered(component: str, layer: int | None) -> str | None:
    """Why this component may not be addressed at this layer, or `None`.

    A whole-model place takes none and a per-layer one takes exactly one,
    and which a component is, is this table's to say — so the document asks
    here rather than keeping a list. A name the table does not have is not
    refused for a layer either way: the engine's `locate` decides at the
    layer given.
    """
    entry = COMPONENTS.get(component)
    if entry is None:
        return None
    if not entry.per_layer and layer is not None:
        return f"{component} takes no layers"
    if entry.per_layer and not isinstance(layer, int):
        return f"{component} is addressed at one layer"
    return None


def describe() -> dict[str, dict[str, Any]]:
    """The component vocabulary, as data an agent can read: for each name,
    whether it takes a layer, its axes, whether it may be written and
    whether it is per head. This is the table, not a description of it, so
    it cannot drift. Whether a checkpoint has the place, and where it is
    there, is an engine's to say — `causalab-mini model` asks it."""
    return {
        name: {
            "layered": entry.per_layer,
            "dims": list(entry.dims),
            "read_only": entry.read_only,
            "heads": _heads(entry.dims) is not None,
        }
        for name, entry in COMPONENTS.items()
    }
