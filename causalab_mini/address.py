"""WHERE: the standard set of places a tensor can be read, and what an
address of one says.

The language is nnterp's. nnterp gives every family one envoy tree —
`model.layers[i].self_attn`, `.mlp`, `model.norm`, `model.lm_head` — and on
those envoys the standard values: `layer_output`, `attention_queries`,
`attention_probabilities`, `logits`. A component is one of those values,
named by the envoy that serves it and the value's name, so `block_output` is
the layer's `layer_output` and `attention_query` is
`self_attn.attention_queries`. nnterp is the reference engine; every other
engine resolves the same strings by translating them to its own runtime, and
how much translating that takes is a measurement of what the standardization
is worth.

This file is engine-neutral: the table of components and the `Address` an
engine hands back. Resolving a component on a loaded model — whether the
checkpoint has it, which module, where it comes in the forward, how wide it
is — is a runtime's question, asked through `Engine.locate`, `width` and
`heads`, and each engine answers it in its own `engines/<name>/address.py`.

Nine components are module boundaries nnterp 2.0 has no value for — a norm's
output, the output projection's input, the MLP's activation — and for those
the row says what nnterp cannot: the module (a list of spellings where the
families disagree and nnterp names none), the axes, and which of nnterp's
values the place sits beside in the forward. See `Component.boundary`.

The table is a floor, not a fence: an engine may accept a name it does not
list — nnterp's does, for any value nnterp serves by its dotted `support()`
name.

An `Address` is pure data, stamped whole by `locate` — the document's
`(component, layer)` plus everything resolved against the checkpoint — so it
pickles, sorts, prints and travels in a plan, and means the same wherever it
lands: nothing on it reads this table again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The three axes of the residual stream, which is what most boundaries carry.
#: Axis names are nnterp's (`nnterp.components` exports them as constants);
#: they are spelled here as literals so a plan loads without importing nnterp,
#: and a test holds the two spellings together.
_STREAM = ("batch", "seq", "hidden_size")


@dataclass(frozen=True)
class Component:
    """What one component is: the envoy that serves it and the value read off
    it, and what an experiment needs to know beyond that."""

    #: The envoy, as a dotted path from the layer for a per-layer place and
    #: from the model for a whole-model one ("" is that envoy itself). More
    #: than one is the spellings families use for a module nnterp gives no
    #: standard name, tried in order.
    hosts: tuple[str, ...]
    #: Read off the envoy: one of nnterp's standard values, or `input` /
    #: `output`, a module boundary nnsight serves as it is.
    value: str
    #: Whether this place is one per layer. A document gives a layer for one
    #: and may not for the others.
    per_layer: bool = True
    #: A module boundary's `(axes, beside, step)`: its axes in nnterp's axis
    #: names, and its place in the forward — the nnterp value it sits beside,
    #: at the same layer, and how far before (negative) or after it. nnterp's
    #: own values carry their axes and are ranked by nnterp, so a row that
    #: names one leaves this out; nnterp ranks only its values, so a
    #: boundary's place is stated against one.
    boundary: tuple[tuple[str, ...], str, int] | None = None
    #: A tensor that may be read and never written: the token ids are the
    #: model's input, and swapping a float activation into them means nothing.
    read_only: bool = False


COMPONENTS = {
    # --- inside the attention: the call that hands the heads to the attention
    # implementation, and its result. nnterp serves them under eager attention
    # only.
    # projected, reshaped to heads and rotated
    "attention_query": Component(("self_attn",), "attention_queries"),
    # the key heads' keys, before a grouped model repeats them: narrower than
    # the query by the grouping ratio, at the same per-head width
    "attention_key": Component(("self_attn",), "attention_keys"),
    # the softmax's argument: QKᵀ·scale with the causal and padding mask
    # already added, so a masked key is -inf here and 0 after
    "attention_scores": Component(("self_attn",), "attention_scores"),
    # The attention pattern: the dropout call's output rather than the
    # softmax's, which is the pattern the values are actually mixed with on
    # every family (after the cast, and after an attention sink has been
    # dropped) and the identity in eval mode. A write here is "make this head
    # attend there".
    "attention_probs": Component(("self_attn",), "attention_probabilities"),
    # each head's mix of the values, before the heads are merged for the
    # output projection
    "attention_z": Component(("self_attn",), "attention_head_outputs"),
    "input_ids": Component(("",), "input_ids", per_layer=False, read_only=True),
    # The embedding table's output, before layer 0: `block_input` at layer 0
    # is the same tensor on a model with nothing between them (GPT-2 adds
    # position embeddings), and this is the one address that names no layer.
    "embeddings": Component(("",), "token_embeddings", per_layer=False),
    # --- the block. block_mid = block_input + attention_output, block_output =
    # block_mid + mlp_output, mlp_input_norm = mlp_input on a pre-norm block.
    "block_input": Component(("",), "layer_input"),
    "attention_input_norm": Component(("input_layernorm",), "output", boundary=(_STREAM, "layer_input", 1)),
    # every head's result side by side, as the output projection takes it:
    # heads · head_dim wide, which is not hidden_size on Qwen3 or Gemma
    "attention_premix": Component(
        ("self_attn.o_proj", "self_attn.c_proj", "self_attn.dense", "self_attn.out_proj"), "input",
        boundary=(("batch", "seq", "num_heads", "head_dim"), "self_attn.attention_output", -1),
    ),
    "attention_output": Component(("self_attn",), "attention_output"),
    "block_mid": Component(("post_attention_layernorm",), "input", boundary=(_STREAM, "self_attn.attention_output", 1)),
    "mlp_input_norm": Component(
        ("post_attention_layernorm",), "output", boundary=(_STREAM, "self_attn.attention_output", 2)
    ),
    "mlp_input": Component(("mlp",), "input", boundary=(_STREAM, "self_attn.attention_output", 3)),
    "mlp_activation": Component(
        ("mlp.act_fn", "mlp.act", "mlp.activation_fn"), "output",
        boundary=(("batch", "seq", "intermediate_size"), "mlp.mlp_output", -2),
    ),
    "mlp_neuron_output": Component(
        # act(gate)·up on a gated MLP, the activation itself on GPT-2, which
        # has no gate — the same place, a different tensor
        ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.dense_4h_to_h", "mlp.fc_out", "mlp.w2"), "input",
        boundary=(("batch", "seq", "intermediate_size"), "mlp.mlp_output", -1),
    ),
    "mlp_output": Component(("mlp",), "mlp_output"),
    "block_output": Component(("",), "layer_output"),
    "ln_final": Component(("norm",), "output", per_layer=False, boundary=(_STREAM, "logits", -2)),
    "lm_head": Component(
        ("lm_head",), "output", per_layer=False, boundary=(("batch", "seq", "vocab_size"), "logits", -1)
    ),
    # The model's own output, which is not always the head's: Gemma-2 caps it
    # with `final_logit_softcapping`, so a metric scored on `lm_head` there is
    # scored on numbers the model does not predict from.
    "logits": Component(("",), "logits", per_layer=False),
}


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """One tap, as the document named it, and what `locate` resolved for it
    on the checkpoint. Every field is filled in by the engine that located
    it; there is no half-made address."""

    component: str
    layer: int | None
    #: The envoy that serves it, as a dotted path in nnterp's standard names
    #: from the model: `layers.3.self_attn`, `layers.3` for the layer itself,
    #: `lm_head` for a whole-model place. Which of a row's spellings this
    #: checkpoint has is a fact about it.
    path: str
    #: What is read off the envoy: a standard value, or `input`/`output`.
    value: str
    #: Where this place is in the model's forward pass, `(side, rank, step)`,
    #: and the sort key that puts taps in forward order — nnsight requires
    #: it: reading layer 8 after layer 2 raises, and a write has to go in
    #: above the read that observes it. `side` and `rank` are nnterp's `rank`
    #: (`side` the layer, or -1 / `num_layers` for a whole-model place before
    #: / after the blocks); `step` is a module boundary's distance from the
    #: value it sits beside, and 0 for nnterp's own.
    rank: tuple[int, int, int]
    #: Whether the place is an operation inside a module's forward rather
    #: than a module boundary — the query, the scores, the pattern.
    inside: bool
    #: The tensor's axes, in nnterp's names (`("batch", "heads", "seq",
    #: "qk_head_dim")`), off the value's layout on this checkpoint.
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
    where nnterp serves it and the tensor's axes there. This is the table,
    not a description of it, so it cannot drift. The axes are the standard
    layout, before any model is loaded — a row that names one of nnterp's
    values takes its axes from nnterp's standard class, the reference
    engine's own statement of the layout. Whether a checkpoint has the
    place, and which module it is, is an engine's to say per checkpoint —
    `causalab-mini model` asks it."""
    from nnterp import Attention, Layer, Mlp, StandardizedTransformer

    standard = {"self_attn": Attention, "mlp": Mlp}
    found = {}
    for name, entry in COMPONENTS.items():
        if entry.boundary is None:
            owner = standard.get(entry.hosts[0]) or (Layer if entry.per_layer else StandardizedTransformer)
            dims = getattr(owner, entry.value).dims
        else:
            dims = entry.boundary[0]
        found[name] = {
            "value": " | ".join(f"{host}.{entry.value}" if host else entry.value for host in entry.hosts),
            "layered": entry.per_layer,
            "dims": list(dims),
            "read_only": entry.read_only,
            "heads": _heads(dims) is not None,
        }
    return found
