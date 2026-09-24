"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about a model's
*internals*, and where a tensor is, it asks nnterp. Every component is one of
nnterp's accessors — `block_output` is `layers_output`, `attention_query` is
`attention_queries` — and the address is that accessor and a layer, nothing
else. Which child module a family spells a place with, which side of it, which
operation inside a forward, where in the value the tensor sits, whether the
family has the place at all and where it comes in the forward pass: all of it
is nnterp's row, per family, asserted there on 26 of them. That includes the
four places inside the attention — the query, the key, the scores and each
head's output — which are operations of the attention's forward, reached
through nnsight's `.source`, and which mini used to find by matching a call's
source line itself.

What a row here keeps is what the tensor *means* to an experiment, which no
accessor says: its width, whether it is per head, whether its last axis is the
keys, which axis the sequence runs along, whether it may be written.

The table is a floor, not a fence: a component name mini has never heard of
but `model.internals` has is addressable, because `RenameConfig(addresses=
{...})` is how a user adds a place — or moves one nnterp has wrong for their
model — and a document should be able to name it.

An `Address` stays pure data — the document's `(component, layer)`, plus what
`locate` resolved on the checkpoint (the module path, the side, the rank) — so
it pickles, sorts, prints and travels in a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Component:
    """What one component means to an experiment, and the nnterp accessor
    that says where it is."""

    #: The nnterp accessor this component is. `None` only for the row a
    #: component that only nnterp knows gets, whose accessor is its own name.
    accessor: str | None = None
    #: Whether this place is one per layer. A document gives a layer for one
    #: and may not for the others; `locate` checks the answer against
    #: nnterp's, so the two cannot drift.
    per_layer: bool = True
    #: The attention implementation this place exists under. sdpa and flash
    #: never materialize the pattern, or the scores under it, so there is no
    #: tensor to address.
    needs: str | None = None
    #: The tensor is head-major: "q" per query head, "kv" per key/value head
    #: (fewer under grouped-query attention). Such a tensor is handed on flat,
    #: `(rows, w, heads · per_head)`, whether the model holds it flat
    #: (`o_proj`'s input) or as two axes (the query) — so a site may name
    #: `heads`, and a featurizer sees one width.
    heads: str | None = None
    #: Its last axis is the *keys* of the padded batch, not a feature width:
    #: true of the attention pattern and the scores under it. A value read
    #: here only means something beside a prompt laid out the same way.
    keys: bool = False
    seq_axis: int = 1  # which axis of the tensor the sequence runs along
    #: The model attribute nnterp publishes this tap's width as; "head_dim"
    #: and "qk_head_dim" are per head and multiplied out.
    width: str | None = None
    #: A tensor that may be read and never written: the token ids are the
    #: model's input, and swapping a float activation into them means nothing.
    read_only: bool = False


# TODO: the layout facts each row carries beside its accessor name — seq_axis,
# the width attribute, the heads kind, keys, needs — are facts about the tensor
# at that place, the same on every model of the family, and may belong on
# nnterp's Address rows. If they moved there, this table would be an alias map
# from mini's names to nnterp's, and a row a user adds to nnterp would get a
# width and a sequence axis for free.
_COMPONENTS = {
    # --- inside the attention: the call that hands the heads to whichever
    # attention implementation the checkpoint runs. (batch, head, seq, dim),
    # so the sequence is axis 2.
    "attention_query": _Component(
        # projected, reshaped to heads and rotated
        accessor="attention_queries", seq_axis=2, heads="q", width="qk_head_dim",
    ),
    "attention_key": _Component(
        # the key heads' keys, before a grouped model repeats them: narrower
        # than the query by the grouping ratio, at the same per-head width
        accessor="attention_keys", seq_axis=2, heads="kv", width="qk_head_dim",
    ),
    "attention_scores": _Component(
        # the softmax's argument: QKᵀ·scale with the causal and padding mask
        # already added, so a masked key is -inf here and 0 after
        accessor="attention_scores", seq_axis=2, needs="eager", heads="q", keys=True,
    ),
    "attention_probs": _Component(
        # The attention pattern: the dropout call's output rather than the
        # softmax's, which is the pattern the values are actually mixed with
        # on every family (after the cast, and after an attention sink has
        # been dropped) and the identity in eval mode. A write here is "make
        # this head attend there".
        accessor="attention_probabilities", seq_axis=2, needs="eager", heads="q", keys=True,
    ),
    "attention_z": _Component(
        # the same call's result: each head's mix of the values, before the
        # heads are merged for the output projection. (batch, seq, head, dim)
        accessor="attention_head_outputs", heads="q", width="head_dim",
    ),
    "input_ids": _Component(accessor="embeddings_input", per_layer=False, read_only=True),
    "embeddings": _Component(
        # The embedding table's output, before layer 0: `block_input` at layer 0
        # is the same tensor on a model with nothing between them (GPT-2 adds
        # position embeddings), and this is the one address that names no layer.
        accessor="embeddings_output", per_layer=False, width="hidden_size",
    ),
    # --- the block, as nnterp addresses it. Definitions (nnterp's, asserted
    # there on 26 families): block_mid = block_input + attention_output,
    # block_output = block_mid + mlp_output, mlp_input_norm = mlp_input.
    "block_input": _Component(accessor="layers_input", width="hidden_size"),
    "attention_input_norm": _Component(accessor="attentions_norm_output", width="hidden_size"),
    "attention_premix": _Component(
        # every head's result side by side: heads * head_dim wide, which is
        # not hidden_size on Qwen3 or Gemma
        accessor="attentions_premix", width="head_dim", heads="q",
    ),
    "attention_output": _Component(accessor="attentions_output", width="hidden_size"),
    "block_mid": _Component(accessor="layers_mid", width="hidden_size"),
    "mlp_input_norm": _Component(accessor="mlps_norm_output", width="hidden_size"),
    "mlp_input": _Component(accessor="mlps_input", width="hidden_size"),
    "mlp_activation": _Component(accessor="mlps_activation", width="intermediate_size"),
    "mlp_neuron_output": _Component(
        # act(gate)·up on a gated MLP, the activation itself on GPT-2, which
        # has no gate — the same place, a different tensor
        accessor="mlps_neurons", width="intermediate_size",
    ),
    "mlp_output": _Component(accessor="mlps_output", width="hidden_size"),
    "block_output": _Component(accessor="layers_output", width="hidden_size"),
    "ln_final": _Component(accessor="ln_final_output", per_layer=False, width="hidden_size"),
    "lm_head": _Component(accessor="lm_head_output", per_layer=False, width="vocab_size"),
    # The model's own output, which is not always the head's: Gemma-2 caps it
    # with `final_logit_softcapping`, so a metric scored on `lm_head` there is
    # scored on numbers the model does not predict from. nnterp carries the
    # two as separate rows and this is the second one.
    "logits": _Component(accessor="logits", per_layer=False, width="vocab_size"),
}


#: The row a component only nnterp knows gets: the place is an accessor of
#: that name and nothing else is claimed about it. A featurizer there is
#: refused (no width), and it is not per head — which is what
#: `RenameConfig(addresses={...})` adds.
_PASS_THROUGH = _Component()


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

    @property
    def _entry(self) -> _Component:
        """This component's row, or the default one for a name that is
        nnterp's alone — a user's `RenameConfig(addresses=…)` accessor,
        which a document may name and which mini knows nothing else about."""
        return _COMPONENTS.get(self.component, _PASS_THROUGH)

    @property
    def accessor(self) -> str:
        """The nnterp accessor this component is: its row's, or the
        component's own name, for one only nnterp knows."""
        return self._entry.accessor or self.component

    @property
    def width_attribute(self) -> str | None:
        """The model attribute nnterp publishes this tap's width as, if it has
        one. Names an attribute rather than reading it, because reading is the
        engine's job."""
        return self._entry.width

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
        """Whether this place is one per layer. `locate` checks mini's answer
        against nnterp's, so a row that moved cannot go unnoticed."""
        return self._entry.per_layer

    @property
    def needs(self) -> str | None:
        """The attention implementation this place exists under, if any."""
        return self._entry.needs

    @property
    def key_axis(self) -> bool:
        """Whether this tensor's last axis is the keys of the padded batch."""
        return self._entry.keys

    @property
    def heads_kind(self) -> str | None:
        """"q" or "kv": which heads this tensor is per, if it is per head."""
        return self._entry.heads

    @property
    def seq_axis(self) -> int:
        """Which axis of the tensor at this address the sequence runs along."""
        return self._entry.seq_axis

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
        return self._entry.read_only

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
    forward, and **where it sits in the forward pass** — so the plan says
    where, any engine can walk there, and mini keeps no second numbering to
    drift.

    A name mini's table does not have but `model.internals` does is accepted:
    `RenameConfig(addresses={...})` is nnterp's extension point, and a
    document should be able to name what a user added. Mini claims nothing
    else about such a place — no width, so a featurizer there is refused.
    """
    known = component in _COMPONENTS
    if not known and component not in getattr(model, "internals", {}):
        raise AddressError(
            f"component {component!r} has no address here, and this model's internals "
            f"have no accessor of that name either. Mini's are {sorted(_COMPONENTS)}; "
            "add one to a model with RenameConfig(addresses={...})"
        )
    address = Address(component, layer)
    have = getattr(model.config, "_attn_implementation", None)
    if address.needs is not None and have != address.needs:
        raise AddressError(
            f"component {component!r} exists only under {address.needs!r} attention, "
            f"and this model runs {have!r}: say \"attn_implementation\": \"{address.needs}\" "
            "in the document's model block"
        )
    accessor = model.internals[address.accessor]
    if known and accessor.per_layer != address.per_layer:
        raise AddressError(
            f"component {component!r}: nnterp says per_layer={accessor.per_layer} and this "
            f"table says {address.per_layer}. One of them moved; the table is mini's to fix"
        )
    reason = accessor.unavailable_on(layer if accessor.per_layer else None)
    if reason is not None:  # a place this family, or this layer, lacks — nnterp says why
        raise AddressError(f"component {component!r}: {reason}")
    return Address(
        component,
        layer,
        module=accessor.address.module,
        io=accessor.io_type.value,
        rank=model.internals.rank(address.accessor, layer if accessor.per_layer else None),
        inside=bool(accessor.address.op),
    )


def head_count(model: Any, address: Address) -> int:
    """How many heads the tensor at `address` has, off what nnterp publishes."""
    kind = address.heads_kind
    if kind is None:
        raise AddressError(f"component {address.component!r} is not a per-head tensor")
    return int(model.num_heads if kind == "q" else model.num_kv_heads)


def width(model: Any, address: Address) -> int:
    """The feature width at `address`, off what nnterp publishes. A per-head
    tensor is handed on flat: every head, side by side."""
    attribute = address.width_attribute
    if attribute is None:
        raise AddressError(
            f"the width of {address.component!r} is not a fact about the model: "
            "it depends on the batch (the attention pattern's key axis) or is not a tensor's"
        )
    value = getattr(model, attribute)
    if value is None:
        raise AddressError(f"nnterp could not find {attribute!r} for this model")
    if attribute in ("head_dim", "qk_head_dim"):
        return head_count(model, address) * int(value)
    return int(value)


def layered(component: str, layer: int | None) -> str | None:
    """Why this component may not be addressed at this layer, or `None`.

    A whole-model place takes none and a per-layer one takes exactly one,
    and which a component is, is this table's to say — so the document asks
    here rather than keeping a list. A name only nnterp
    knows is per layer unless nnterp says otherwise, which `locate` finds
    out; here it is not refused for a layer either way.
    """
    entry = _COMPONENTS.get(component)
    if entry is None:
        return None
    if not entry.per_layer and layer is not None:
        return f"{component} takes no layers"
    if entry.per_layer and not isinstance(layer, int):
        return f"{component} is addressed at one layer"
    return None


def describe() -> dict[str, dict[str, Any]]:
    """The component vocabulary, as data an agent can read: for each name,
    the nnterp accessor it is and what kind of place that is. This is the
    table, not a description of it, so it cannot drift. Whether a place is
    inside a forward, and which module it is, is nnterp's to say per
    checkpoint — `causalab-mini model` asks it."""
    return {
        name: {
            "accessor": entry.accessor,
            "layered": entry.per_layer,
            "seq_axis": entry.seq_axis,
            "width": entry.width,
            "read_only": entry.read_only,
            "heads": entry.heads is not None,
            "needs": entry.needs,
        }
        for name, entry in _COMPONENTS.items()
    }
