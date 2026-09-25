"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about a model's
*internals*, and where a tensor is, it asks nnterp. Every component is one of
nnterp's accessors — `block_output` is `layers_output`, `attention_query` is
`attention_queries` — and the address is that accessor and a layer, nothing
else. Which child module a family spells a place with, which side of it, which
operation inside a forward, where in the value the tensor sits, whether the
family has the place at all and where it comes in the forward pass: all of it
is nnterp's row, per family, asserted there on 26 of them.

So is how the tensor is laid out — which axis the sequence runs along (none,
for a recurrent state), how wide it is, how many heads it is per, whether its
last axis is the keys, which attention implementation it exists under — and
`locate` stamps what the row says into the address. What is left here is the
alias map from mini's names to nnterp's, and the one thing nnterp's rows do not
say: that the token ids may be read and never written.

The table is a floor, not a fence: a component name mini has never heard of
but `model.internals` has is addressable, because `RenameConfig(addresses=
{...})` is how a user adds a place — or moves one nnterp has wrong for their
model — and a document should be able to name it, with the width and sequence
axis its row states.

An `Address` stays pure data — the document's `(component, layer)`, plus what
`locate` resolved on the checkpoint (the module path, the side, the rank, the
layout) — so it pickles, sorts, prints and travels in a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Mini's name for each place, and the nnterp accessor it is.
COMPONENTS = {
    # --- inside the attention: the call that hands the heads to whichever
    # attention implementation the checkpoint runs
    "attention_query": "attention_queries",  # projected, split into heads and rotated
    # the key heads' keys, before a grouped model repeats them: narrower than
    # the query by the grouping ratio, at the same per-head width
    "attention_key": "attention_keys",
    # the softmax's argument: QKᵀ·scale with the causal and padding mask
    # already added, so a masked key is -inf here and 0 after
    "attention_scores": "attention_scores",
    # The attention pattern: the dropout call's output rather than the
    # softmax's, which is the pattern the values are actually mixed with on
    # every family (after the cast, and after an attention sink has been
    # dropped) and the identity in eval mode. A write here is "make this head
    # attend there".
    "attention_probs": "attention_probabilities",
    # the same call's result: each head's mix of the values, before the heads
    # are merged for the output projection
    "attention_z": "attention_head_outputs",
    "input_ids": "embeddings_input",
    # The embedding table's output, before layer 0: `block_input` at layer 0 is
    # the same tensor on a model with nothing between them (GPT-2 adds position
    # embeddings), and this is the one address that names no layer.
    "embeddings": "embeddings_output",
    # --- the block, as nnterp addresses it. Definitions (nnterp's, asserted
    # there on 26 families): block_mid = block_input + attention_output,
    # block_output = block_mid + mlp_output, mlp_input_norm = mlp_input.
    "block_input": "layers_input",
    "attention_input_norm": "attentions_norm_output",
    # every head's result side by side: heads * head_dim wide, which is not
    # hidden_size on Qwen3 or Gemma
    "attention_premix": "attentions_premix",
    "attention_output": "attentions_output",
    "block_mid": "layers_mid",
    "mlp_input_norm": "mlps_norm_output",
    "mlp_input": "mlps_input",
    "mlp_activation": "mlps_activation",
    # act(gate)·up on a gated MLP, the activation itself on GPT-2, which has no
    # gate — the same place, a different tensor
    "mlp_neuron_output": "mlps_neurons",
    "mlp_output": "mlps_output",
    "block_output": "layers_output",
    # --- a Gated DeltaNet layer of a hybrid (Qwen3-Next, Qwen3.5), in the
    # attention's place. The state is the one the layer hands on: a (key,
    # value) matrix per head with no sequence axis, which is the state *after
    # the forward's last token* — `{"index": -1}` after the prompt, and the
    # generated frame's step k after the token that step processed. The state
    # a forward starts from is not a place of its own here: over a prompt it
    # is nothing, and at step k it is the state after step k - 1.
    "linear_attention_state": "linear_attentions_state_output",
    # the delta rule's arguments, per value head, and what it returns per head
    "linear_attention_query": "linear_attention_queries",
    "linear_attention_key": "linear_attention_keys",
    "linear_attention_value": "linear_attention_values",
    "linear_attention_decay": "linear_attention_decays",
    "linear_attention_beta": "linear_attention_betas",
    "linear_attention_z": "linear_attention_head_outputs",
    "linear_attention_output": "linear_attentions_output",
    "ln_final": "ln_final_output",
    "lm_head": "lm_head_output",
    # The model's own output, which is not always the head's: Gemma-2 caps it
    # with `final_logit_softcapping`, so a metric scored on `lm_head` there is
    # scored on numbers the model does not predict from. nnterp carries the
    # two as separate rows and this is the second one.
    "logits": "logits",
}

#: A tensor that may be read and never written: the token ids are the model's
#: input, and swapping a float activation into them means nothing.
READ_ONLY = frozenset({"input_ids"})


def _row(component: str) -> Any:
    """nnterp's row for a component, as its table states it before any model is
    loaded — which is what a document is checked against. `None` for a name
    only a user's `RenameConfig` knows. A family's row moves the module or the
    operation, not the layout or whether the place is one per layer, and what
    `locate` stamps is the loaded model's row either way."""
    # imported where it is asked, so that a plan, which holds addresses, loads
    # without nnterp and the torch behind it
    from nnterp.rename_utils import DEFAULT_ADDRESSES, STRUCTURAL_ADDRESSES

    accessor = COMPONENTS.get(component, component)
    if accessor in STRUCTURAL_ADDRESSES:
        return STRUCTURAL_ADDRESSES[accessor][0]
    return DEFAULT_ADDRESSES.get(accessor)


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """One tap, as the document named it — plus what `locate` resolved
    against the model."""

    component: str
    layer: int | None = None
    #: The module this checkpoint spells it with — relative to the layer for
    #: a layered one (`post_attention_layernorm`, `self_attn`, "" for the
    #: layer itself), to the model for a whole-model one (`lm_head`). Filled
    #: by `locate`; a plan holds the string, so an engine without nnterp's
    #: accessors can still walk to it.
    module: str | None = None
    #: Which side of that module, as nnterp knows it.
    io: str | None = None
    #: Where this place is in the model's forward pass, `(layer, order)` —
    #: nnterp's `internals.rank`, filled by `locate`, because it is a fact
    #: about the checkpoint's family: nnterp overrides the order per family
    #: where a block is built differently.
    rank: tuple[int, int] | None = None
    #: Whether the place is an operation inside a module's forward rather
    #: than a module boundary — the query, the scores, the pattern. An engine
    #: that only sees module boundaries has to know.
    inside: bool = False
    #: The layout, as the row says, stamped by `locate`. Which axis of the
    #: tensor the sequence runs along: 1 at a module boundary, 2 inside the
    #: attention, where a tensor is (batch, head, seq, head_dim), and None for
    #: a tensor with no sequence axis — a recurrent state, which is one per
    #: forward: the state after its last token.
    seq_axis: int | None = 1
    #: The model attribute that counts the heads this tensor is per, or None.
    #: Such a tensor is handed on flat, `(rows, w, heads · per_head)`, whether
    #: the model holds it flat (`o_proj`'s input) or as two axes (the query) —
    #: so a site may name `heads`, and a featurizer sees one width.
    heads: str | None = None
    #: Its last axis is the *keys* of the padded batch, not a feature width:
    #: true of the attention pattern and the scores under it. A value read
    #: here only means something beside a prompt laid out the same way.
    keys: bool = False

    @property
    def accessor(self) -> str:
        """The nnterp accessor this component is: its alias's, or the
        component's own name, for one only nnterp knows."""
        return COMPONENTS.get(self.component, self.component)

    @property
    def path(self) -> str:
        """The module, as a dotted path against the nnterp handle, in nnterp's
        standardized names — known only once `locate` has asked the
        checkpoint."""
        if self.module is None:
            raise AddressError(
                f"component {self.component!r}: its module is a fact about the checkpoint; "
                "build the address with engine.locate(...)"
            )
        if not self.per_layer:
            return self.module
        return f"layers.{self.layer}" + (f".{self.module}" if self.module else "")

    @property
    def per_layer(self) -> bool:
        """Whether this place is one per layer, as nnterp's table says; a name
        only a user's row knows is per layer unless that row says otherwise,
        which `locate` finds out."""
        row = _row(self.component)
        return True if row is None else row.per_layer

    @property
    def key(self) -> tuple[int, int]:
        """Sort key putting addresses in forward order: the embeddings, then
        the layer stack by depth and by where inside the block, then the final
        norm and the head. nnsight requires the order: reading layer 8 after
        layer 2 raises, and a write has to go in above the read that observes
        it.

        It is `internals.rank`'s answer, stamped by `locate` — nnterp moves a
        place's order per family (DBRX puts the attention's norm above the
        mixer) and mini keeping its own numbering beside it is how the two
        drift apart without anything saying so."""
        if self.rank is None:
            raise AddressError(
                f"component {self.component!r}: where it sits in the forward pass is a "
                "fact about the family; build the address with engine.locate(...)"
            )
        return self.rank

    @property
    def read_only(self) -> bool:
        return self.component in READ_ONLY

    def resolve(self, root: Any) -> Any:
        """Walk this address's path against `root`, by getattr, taking a
        numeric segment as an index: "layers.0" is `root.layers[0]`.

        The path is written in nnterp's standardized names. An engine that
        does not have those names translates first — how much translating that
        takes is a measurement of what the standardization is worth.
        """
        target = root
        try:
            for segment in self.path.split("."):
                target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            raise AddressError(
                f"component {self.component!r}: {self.path!r} does not exist on this model"
            ) from error
        if target is None:
            raise AddressError(f"component {self.component!r}: {self.path!r} is None on this model")
        return target


def locate(model: Any, component: str, layer: int | None) -> Address:
    """The address of `(component, layer)` on a loaded nnterp model, or a
    refusal: a place nnterp has no accessor for on this family, or the
    pattern under an attention implementation that never forms it.

    What nnterp resolved is written into the address — the module's
    spelling on this checkpoint, which side of it, whether it is inside the
    forward, **where it sits in the forward pass**, and how the tensor there
    is laid out — so the plan says where and how, any engine can walk there,
    and mini keeps no second copy of any of it to drift.

    A name mini's alias map does not have but `model.internals` does is
    accepted: `RenameConfig(addresses={...})` is nnterp's extension point, and
    a document should be able to name what a user added.
    """
    if component not in COMPONENTS and component not in getattr(model, "internals", {}):
        raise AddressError(
            f"component {component!r} has no address here, and this model's internals "
            f"have no accessor of that name either. Mini's are {sorted(COMPONENTS)}; "
            "add one to a model with RenameConfig(addresses={...})"
        )
    accessor = model.internals[COMPONENTS.get(component, component)]
    row = accessor.address
    have = getattr(model.config, "_attn_implementation", None)
    if row.needs is not None and have != row.needs:
        raise AddressError(
            f"component {component!r} exists only under {row.needs!r} attention, "
            f"and this model runs {have!r}: say \"attn_implementation\": \"{row.needs}\" "
            "in the document's model block"
        )
    reason = accessor.unavailable_on(layer if accessor.per_layer else None)
    if reason is not None:  # a place this family, or this layer, lacks — nnterp says why
        raise AddressError(f"component {component!r}: {reason}")
    return Address(
        component,
        layer,
        module=row.module,
        io=accessor.io_type.value,
        rank=model.internals.rank(accessor.name, layer if accessor.per_layer else None),
        inside=bool(row.op),
        seq_axis=row.seq_axis,
        heads=row.heads,
        keys=row.keys,
    )


def head_count(model: Any, address: Address) -> int:
    """How many heads the tensor at `address` has, off what nnterp publishes."""
    count = model.internals[address.accessor].num_heads
    if count is None:
        raise AddressError(f"component {address.component!r} is not a per-head tensor")
    return int(count)


def width(model: Any, address: Address) -> int:
    """The feature width at `address`, off the row and what nnterp publishes.
    A per-head tensor is handed on flat: every head, side by side."""
    accessor = model.internals[address.accessor]
    if accessor.address.width is None:
        raise AddressError(
            f"the width of {address.component!r} is not a fact about the model: "
            "it depends on the batch (the attention pattern's key axis) or is not a tensor's"
        )
    value = accessor.width
    if value is None:
        raise AddressError(f"nnterp could not find {accessor.address.width} for this model")
    return int(value)


def layered(component: str, layer: int | None) -> str | None:
    """Why this component may not be addressed at this layer, or `None`.

    A whole-model place takes none and a per-layer one takes exactly one,
    and which a component is, is nnterp's table's to say — so the document
    asks here rather than keeping a list. A name only a user's row knows is
    not refused for a layer either way; `locate` finds out.
    """
    row = _row(component)
    if row is None:
        return None
    if not row.per_layer and layer is not None:
        return f"{component} takes no layers"
    if row.per_layer and not isinstance(layer, int):
        return f"{component} is addressed at one layer"
    return None


def describe() -> dict[str, dict[str, Any]]:
    """The component vocabulary, as data an agent can read: for each name,
    the nnterp accessor it is and what kind of place its row says it is. It
    is read off the rows, not a description of them, so it cannot drift.
    Whether a place is inside a forward, which module it is, and whether a
    given checkpoint has it at all are nnterp's to say per checkpoint —
    `causalab-mini model` asks it."""
    described = {}
    for name, accessor in COMPONENTS.items():
        row = _row(name)
        described[name] = {
            "accessor": accessor,
            "layered": row.per_layer,
            "seq_axis": row.seq_axis,
            # per head where the place is: the attributes it multiplies out of
            "width": None if row.width is None else " * ".join(row.width) or "1",
            "read_only": name in READ_ONLY,
            "heads": row.heads is not None,
            "needs": row.needs,
        }
    return described
