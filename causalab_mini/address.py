"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about a model's
*internals*, and where a tensor is, it asks nnterp. nnterp gives every family
one envoy tree — `model.layers[i].self_attn`, `.mlp`, `model.norm`,
`model.lm_head` — and on those envoys the standard values: `layer_output`,
`attention_queries`, `attention_probabilities`, `logits`. A component is one of
those values, named by the envoy that serves it and the value's name, so
`block_output` is the layer's `layer_output` and `attention_query` is
`self_attn.attention_queries`. Which operation inside a forward that is, where
in the call's arguments the tensor sits, whether this checkpoint has it at all
— the attention's interior needs eager attention, a hybrid's linear blocks
have no `self_attn` — and **where it comes in the forward pass** are nnterp's
(`support`, `rank`); what the tensor's axes are is the value's layout
(`dims`), in names that are the same on every family.

Six components are module boundaries nnterp 2.0 has no value for — a norm's
output, the output projection's input, the MLP's activation — and for those
the row says what nnterp cannot: the module (a list of spellings where the
families disagree and nnterp names none), the axes, and which of nnterp's
values the place sits beside in the forward. See `_Component.beside`.

A width is never written here, only which size an axis is — `hidden` is the
model's `hidden_size`, `head_dim` the attention's — and the number is asked of
the model (`_SIZES`).

The table is a floor, not a fence: a component mini has never heard of but
nnterp serves — a value a user installed with `envoys=` on a family's
attention, say, by its dotted `support()` name — is addressable, and gets its
width and its sequence axis from its own layout.

An `Address` stays pure data — the document's `(component, layer)`, plus what
`locate` resolved on the checkpoint (the envoy, the rank, the axes) — so it
pickles, sorts, prints and travels in a plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The three axes of the residual stream, which is what most boundaries carry.
_STREAM = ("batch", "seq", "hidden")


@dataclass(frozen=True)
class _Component:
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
    #: A module boundary's axes, in nnterp's axis names. nnterp's own values
    #: carry theirs, so a row that names one leaves this out.
    dims: tuple[str, ...] | None = None
    #: A module boundary's place in the forward: the nnterp value it sits
    #: beside, at the same layer, and how far before (negative) or after it.
    #: nnterp ranks only its values, so the place is stated against one.
    beside: tuple[str, int] | None = None
    #: A tensor that may be read and never written: the token ids are the
    #: model's input, and swapping a float activation into them means nothing.
    read_only: bool = False


_COMPONENTS = {
    # --- inside the attention: the call that hands the heads to the attention
    # implementation, and its result. nnterp serves them under eager attention
    # only.
    # projected, reshaped to heads and rotated
    "attention_query": _Component(("self_attn",), "attention_queries"),
    # the key heads' keys, before a grouped model repeats them: narrower than
    # the query by the grouping ratio, at the same per-head width
    "attention_key": _Component(("self_attn",), "attention_keys"),
    # the softmax's argument: QKᵀ·scale with the causal and padding mask
    # already added, so a masked key is -inf here and 0 after
    "attention_scores": _Component(("self_attn",), "attention_scores"),
    # The attention pattern: the dropout call's output rather than the
    # softmax's, which is the pattern the values are actually mixed with on
    # every family (after the cast, and after an attention sink has been
    # dropped) and the identity in eval mode. A write here is "make this head
    # attend there".
    "attention_probs": _Component(("self_attn",), "attention_probabilities"),
    # each head's mix of the values, before the heads are merged for the
    # output projection
    "attention_z": _Component(("self_attn",), "attention_head_outputs"),
    "input_ids": _Component(("",), "input_ids", per_layer=False, read_only=True),
    # The embedding table's output, before layer 0: `block_input` at layer 0
    # is the same tensor on a model with nothing between them (GPT-2 adds
    # position embeddings), and this is the one address that names no layer.
    "embeddings": _Component(("",), "token_embeddings", per_layer=False),
    # --- the block. block_mid = block_input + attention_output, block_output =
    # block_mid + mlp_output, mlp_input_norm = mlp_input on a pre-norm block.
    "block_input": _Component(("",), "layer_input"),
    "attention_input_norm": _Component(("input_layernorm",), "output", dims=_STREAM, beside=("layer_input", 1)),
    # every head's result side by side, as the output projection takes it:
    # heads · head_dim wide, which is not hidden_size on Qwen3 or Gemma
    "attention_premix": _Component(
        ("self_attn.o_proj", "self_attn.c_proj", "self_attn.dense", "self_attn.out_proj"), "input",
        dims=("batch", "seq", "heads", "head_dim"), beside=("self_attn.attention_output", -1),
    ),
    "attention_output": _Component(("self_attn",), "attention_output"),
    "block_mid": _Component(("post_attention_layernorm",), "input", dims=_STREAM, beside=("self_attn.attention_output", 1)),
    "mlp_input_norm": _Component(
        ("post_attention_layernorm",), "output", dims=_STREAM, beside=("self_attn.attention_output", 2)
    ),
    "mlp_input": _Component(("mlp",), "input", dims=_STREAM, beside=("self_attn.attention_output", 3)),
    "mlp_activation": _Component(
        ("mlp.act_fn", "mlp.act", "mlp.activation_fn"), "output",
        dims=("batch", "seq", "intermediate"), beside=("mlp.mlp_output", -2),
    ),
    "mlp_neuron_output": _Component(
        # act(gate)·up on a gated MLP, the activation itself on GPT-2, which
        # has no gate — the same place, a different tensor
        ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.dense_4h_to_h", "mlp.fc_out", "mlp.w2"), "input",
        dims=("batch", "seq", "intermediate"), beside=("mlp.mlp_output", -1),
    ),
    "mlp_output": _Component(("mlp",), "mlp_output"),
    "block_output": _Component(("",), "layer_output"),
    "ln_final": _Component(("norm",), "output", per_layer=False, dims=_STREAM, beside=("logits", -2)),
    "lm_head": _Component(
        ("lm_head",), "output", per_layer=False, dims=("batch", "seq", "vocab"), beside=("logits", -1)
    ),
    # The model's own output, which is not always the head's: Gemma-2 caps it
    # with `final_logit_softcapping`, so a metric scored on `lm_head` there is
    # scored on numbers the model does not predict from.
    "logits": _Component(("",), "logits", per_layer=False),
}

#: Which size each axis is: the envoy that holds the number, from the layer
#: ("" is the model itself), and the attribute. The attention's and the MLP's
#: are the block's own, read off its module, so they hold on a model whose
#: blocks differ. An axis not here — the keys of the pattern, the tokens —
#: has no width that is a fact about the model.
_SIZES = {
    "hidden": ("", "hidden_size"),
    "vocab": ("", "vocab_size"),
    "heads": ("self_attn", "num_heads"),
    "kv_heads": ("self_attn", "num_kv_heads"),
    "head_dim": ("self_attn", "head_dim"),
    "qk_head_dim": ("self_attn", "qk_head_dim"),
    "intermediate": ("mlp", "intermediate_size"),
}


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """One tap, as the document named it — plus what `locate` resolved
    against the model."""

    component: str
    layer: int | None = None
    #: The envoy that serves it, in nnterp's standard names — relative to the
    #: layer for a layered place (`self_attn`, `post_attention_layernorm`, ""
    #: for the layer itself), to the model for a whole-model one (`lm_head`).
    #: Filled by `locate`, because where a row names several spellings which
    #: one this checkpoint has is a fact about it.
    module: str | None = None
    #: Where this place is in the model's forward pass, `(side, rank, step)`:
    #: nnterp's `rank` — `side` the layer, or -1 / `num_layers` for a
    #: whole-model place before / after the blocks — and `step`, a module
    #: boundary's distance from the value it sits beside (0 for nnterp's own).
    rank: tuple[int, int, int] | None = None
    #: Whether the place is an operation inside a module's forward rather
    #: than a module boundary — the query, the scores, the pattern.
    inside: bool = False
    #: The tensor's axes, in nnterp's names (`("batch", "heads", "seq",
    #: "qk_head_dim")`), off the value's layout on this checkpoint.
    dims: tuple[str, ...] | None = None

    @property
    def _entry(self) -> _Component:
        """This component's row, or the one a name only nnterp knows gets: a
        dotted value name is the envoy and the value."""
        known = _COMPONENTS.get(self.component)
        if known is not None:
            return known
        host, _, value = self.component.rpartition(".")
        return _Component((host,), value, per_layer=self.layer is not None)

    @property
    def value(self) -> str:
        """What is read off the envoy: a standard value, or `input`/`output`."""
        return self._entry.value

    @property
    def path(self) -> str:
        """The envoy, as a dotted path against the nnterp handle — known only
        once `locate` has asked the checkpoint."""
        if self.module is None:
            raise AddressError(
                f"component {self.component!r}: its module is a fact about the checkpoint; "
                "build the address with engine.locate(...)"
            )
        if self.layer is None:
            return self.module
        return f"layers.{self.layer}" + (f".{self.module}" if self.module else "")

    @property
    def _axes(self) -> tuple[str, ...]:
        if self.dims is None:
            raise AddressError(
                f"component {self.component!r}: its axes are the layout of the value on this "
                "checkpoint; build the address with engine.locate(...)"
            )
        return self.dims

    @property
    def key_axis(self) -> bool:
        """Whether this tensor's last axis is the keys of the padded batch:
        true of the pattern and the scores under it. A value read there only
        means something beside a prompt laid out the same way."""
        return self._axes[-1] == "key"

    @property
    def heads_kind(self) -> str | None:
        """"q" or "kv": which heads this tensor is per, if it is per head.
        Such a tensor is handed on flat, `(rows, w, heads · per_head)`, whether
        the model holds it flat (the output projection's input) or as two axes
        (the query) — so a site may name `heads`, and a featurizer sees one
        width."""
        return _heads(self._axes)

    @property
    def seq_axis(self) -> int:
        """Which axis of the tensor the sequence runs along — the pattern's
        queries, on the pattern."""
        axes = self._axes
        for name in ("seq", "query"):
            if name in axes:
                return axes.index(name)
        raise AddressError(f"component {self.component!r} has no sequence axis: its axes are {axes}")

    @property
    def key(self) -> tuple[int, int, int]:
        """Sort key putting addresses in forward order: the embeddings, then
        the layer stack by depth and by where inside the block, then the final
        norm and the head. nnsight requires the order: reading layer 8 after
        layer 2 raises, and a write has to go in above the read that observes
        it.

        It is nnterp's `rank`, stamped by `locate` — nnterp measures the order
        on the checkpoint, so a family that builds its block differently is
        ordered as it runs, and mini keeps no numbering of its own beside it."""
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
        numeric segment as an index: "layers.0" is `root.layers[0]`, and the
        empty path is `root`.

        The path is written in nnterp's standardized names. An engine that
        does not have those names translates first — how much translating that
        takes is a measurement of what the standardization is worth.
        """
        return _walk(root, self.path, self.component)


def _walk(root: Any, path: str, component: str) -> Any:
    target = root
    try:
        for segment in filter(None, path.split(".")):
            target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise AddressError(f"component {component!r}: {path!r} does not exist on this model") from error
    if target is None:
        raise AddressError(f"component {component!r}: {path!r} is None on this model")
    return target


def _heads(axes: tuple[str, ...]) -> str | None:
    return "q" if "heads" in axes else "kv" if "kv_heads" in axes else None


def locate(model: Any, component: str, layer: int | None) -> Address:
    """The address of `(component, layer)` on a loaded nnterp model, or a
    refusal: a place this family, or this block, does not have — nnterp's
    reason, which for the attention's interior under sdpa is that it is read
    inside the eager attention forward.

    What nnterp resolved is written into the address — the envoy that serves
    it, where it sits in the forward pass and the tensor's axes — so the plan
    says where, and mini keeps no second numbering to drift.

    A name mini's table does not have but nnterp serves — a dotted
    `support()` name at a layer, a root value without — is accepted, and its
    axes are its value's own.
    """
    from nnterp import Unavailable

    known = component in _COMPONENTS
    entry = Address(component, layer)._entry
    try:
        root = model if layer is None else model.layers[layer]
        module, host = _host(root, entry.hosts, component)
        if entry.beside is None:
            name = f"{module}.{entry.value}" if module else entry.value
            side, rank = model.rank(name, layer)
            served = getattr(type(host), entry.value)
            dims, inside, step = served.dims, served.inside_forward(), 0
        else:
            anchor, step = entry.beside
            side, rank = model.rank(anchor, layer)
            dims, inside = entry.dims, False
    except Unavailable as reason:  # a place this family, or this block, lacks
        raise AddressError(f"component {component!r}: {reason}") from None
    except (AddressError, KeyError, IndexError) as error:
        if known:
            raise AddressError(f"component {component!r}: {error.args[0]}") from None
        raise AddressError(
            f"component {component!r} has no address here, and nnterp serves no value of that "
            f"name {'at a layer' if layer is not None else 'on the model'} either. Mini's are "
            f"{sorted(_COMPONENTS)}; a value a family lacks is added with `envoys=` at load"
        ) from None
    return Address(component, layer, module=module, rank=(side, rank, step), inside=inside, dims=dims)


def _host(root: Any, hosts: tuple[str, ...], component: str) -> tuple[str, Any]:
    """The first of a row's envoy paths this checkpoint has, and the envoy."""
    for path in hosts:
        try:
            return path, _walk(root, path, component)
        except AddressError:
            continue
    raise AddressError(f"this block has no {' or '.join(hosts)} module")


def _size(model: Any, axis: str, layer: int | None) -> int:
    """The number an axis is, asked of the envoy `_SIZES` says holds it — the
    model's own for a whole-model place."""
    host, attribute = _SIZES[axis]
    owner = model if layer is None or not host else _walk(model.layers[layer], host, axis)
    value = getattr(owner, attribute, None)
    if value is None:
        raise AddressError(f"nnterp could not find {attribute!r} for this model")
    return int(value)


def head_count(model: Any, address: Address) -> int:
    """How many heads the tensor at `address` has, off what nnterp publishes."""
    kind = address.heads_kind
    if kind is None:
        raise AddressError(f"component {address.component!r} is not a per-head tensor")
    return _size(model, "heads" if kind == "q" else "kv_heads", address.layer)


def width(model: Any, address: Address) -> int:
    """The feature width at `address`, off what nnterp publishes. A per-head
    tensor is handed on flat: every head, side by side."""
    last = address._axes[-1]
    if last not in _SIZES or last in ("heads", "kv_heads"):
        raise AddressError(
            f"the width of {address.component!r} is not a fact about the model: its last axis is "
            f"{last!r}, which depends on the batch (the attention pattern's keys) or is not a feature axis"
        )
    size = _size(model, last, address.layer)
    if last in ("head_dim", "qk_head_dim"):
        return head_count(model, address) * size
    return size


def layered(component: str, layer: int | None) -> str | None:
    """Why this component may not be addressed at this layer, or `None`.

    A whole-model place takes none and a per-layer one takes exactly one,
    and which a component is, is this table's to say — so the document asks
    here rather than keeping a list. A name only nnterp knows is not refused
    for a layer either way: `locate` asks nnterp at the layer given.
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
    where nnterp serves it and the tensor's axes there. This is the table,
    not a description of it, so it cannot drift. The axes are the standard
    layout, before any model is loaded; whether a checkpoint has the place,
    and which module it is, is nnterp's to say per checkpoint —
    `causalab-mini model` asks it."""
    from nnterp import Attention, Layer, Mlp, StandardizedTransformer

    standard = {"self_attn": Attention, "mlp": Mlp}
    found = {}
    for name, entry in _COMPONENTS.items():
        if entry.dims is None:
            owner = standard.get(entry.hosts[0]) or (Layer if entry.per_layer else StandardizedTransformer)
            dims = getattr(owner, entry.value).dims
        else:
            dims = entry.dims
        found[name] = {
            "value": " | ".join(f"{host}.{entry.value}" if host else entry.value for host in entry.hosts),
            "layered": entry.per_layer,
            "dims": list(dims),
            "read_only": entry.read_only,
            "heads": _heads(dims) is not None,
        }
    return found
