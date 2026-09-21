"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about a model's
*internals*. Every model fact we had to encode ourselves is in the table
below, and each one is an entry in FINDINGS.md.

An address says **where**, in terms that are true of the architecture: which
module, which side, which argument of which operation, which axis the sequence
runs along. It does not say how to reach there, because that is a property of
the runtime and not of the model — `engine/engines/nnterp/` reads an address
with nnsight envoys, and another engine reads the same address differently.

An `Address` stays pure data — the document's `(component, layer)`, plus, for an
interior, the name of one `.source` operation — so it pickles, sorts, prints and
travels in a plan. Everything a model is needed for (which module path, which
side, where the sequence axis is, where in the forward pass it sits) is a lookup
in `_COMPONENTS`, and the model is passed in, never held.

Two kinds of address live here:

* a **module boundary** — `block_output`, `lm_head`. A path is a string walked
  by getattr against the nnterp handle, with a numeric segment taken as an
  index: "layers.0" is `model.layers[0]`. The plan holds the string; the envoy
  is never pickled.
* an **interior** — `attention_query`. The tensor never crosses a module
  boundary, so the address is a module *plus one operation inside its forward*,
  reached through nnsight's `.source`. The operation is named by the call site
  it appears on, and that name is resolved against the loaded model by
  `Address.locate`, because an occurrence suffix is a property of the
  transformers version, not of the document.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Lens:
    """The escape hatch for a boundary value a path cannot describe: two
    module-level functions, because a write needs the way back as much as a
    read needs the way in. `get(value)` is the activation; `put(value,
    tensor)` is `value` with the activation replaced."""

    get: Callable[[Any], Any]
    put: Callable[[Any, Any], Any]


#: How the activation sits inside what a module boundary hands over.
#:   None     decided from the value: a tuple's first element, or the tensor
#:            itself — which of the two a block returns is a property of the
#:            transformers version, so the default must not pin it
#:   a path   indices and keys, walked in: `(1,)`, `("hidden_states",)`
#:   a Lens   anything else
Select = None | tuple[int | str, ...] | Lens


@dataclass(frozen=True)
class _Component:
    """Everything about one component that is a fact about models."""

    #: A dotted path template against the nnterp handle. `a|b` is a choice:
    #: the one alternative that exists on this checkpoint — `input_layernorm`
    #: on Llama, `ln_1` on GPT-2. nnterp standardizes the block, the mixer
    #: and the MLP; it does not name their children, and the checkpoint can
    #: say which it has without a family column here.
    path: str
    side: str  # "output" or "input"
    stage: int  # order within one block, in forward order
    #: Where this tap sits relative to the layer stack: 0 before it, 1 inside
    #: it, 2 after it. The embeddings are the reason this exists — every other
    #: layerless tap is downstream of every layer, and that one is upstream of
    #: all of them.
    band: int = 1
    op: str | None = None  # the call site of the `.source` operation, for an interior
    #: For an interior, which handle of that call carries the tensor: one of
    #: its positional arguments ("inputs") or its return ("output").
    handle: str = "inputs"
    #: Which positional argument, or which element of the return; None when
    #: the return is the tensor itself rather than a tuple holding it.
    arg: int | None = 0
    #: For an operation *inside* the operation: its name in the outer call's
    #: own `.source`. nnsight can only open that inside a trace — the callee
    #: is whatever the call dispatches to at run time — so unlike `op` this is
    #: a name, resolved where the run runs, and not a needle resolved here.
    inner: str | None = None
    #: The attention implementation this place exists under. sdpa and flash
    #: never materialize the pattern, so there is no tensor to address.
    needs: str | None = None
    #: The config attribute counting the heads of a head-major tensor. Such a
    #: tensor is handed on flat, `(rows, w, heads · per_head)`, whether the
    #: model holds it flat (`o_proj`'s input) or as two axes (the query) —
    #: so a site may name `heads`, and a featurizer sees one width.
    heads: str | None = None
    seq_axis: int = 1  # which axis of the tensor the sequence runs along
    width: str | None = None  # the nnterp handle attribute holding this tap's width
    #: A tensor that may be read and never written: the token ids are the
    #: model's input, and swapping a float activation into them means nothing.
    read_only: bool = False
    #: At a module boundary: where in the boundary's value the activation is.
    select: Select = None


#: The exceptions, and only those: `(config.model_type, component)` -> the
#: fields of the row that differ on that family. The `a|b` paths cover a
#: child that is *named* differently, because the checkpoint can say which
#: it has; this covers the same name handing over a different *structure*,
#: which nothing about existence can distinguish. Empty until a model needs
#: it — e.g. a block returning `(router_logits, hidden)` would be
#:
#:     ("some_moe", "block_output"): {"select": (1,)},
_OVERRIDES: dict[tuple[str, str], dict[str, Any]] = {}


_COMPONENTS = {
    "attention_query": _Component(
        # The attention module, and inside it the call that hands the query to
        # whichever attention implementation the checkpoint is running. The
        # interface takes (module, query, key, value, attention_mask, ...), so
        # the query is positional argument 1, and it arrives already projected,
        # reshaped to (batch, head, seq, head_dim) and rotated — the sequence is
        # axis 2, not axis 1.
        path="attentions.{layer}",
        side="input",
        stage=2,
        op="attention_interface(",
        arg=1,
        seq_axis=2,
        heads="num_attention_heads",
        width="head_dim",
    ),
    "attention_key": _Component(
        # The same call as the query, argument 2: the keys *before* GQA's
        # repeat_kv, so this is key-head space and narrower than the query by
        # the grouping ratio on a model that groups. No width: nnterp
        # publishes nothing for it, which is what refuses a featurizer here.
        path="attentions.{layer}",
        side="input",
        stage=2,
        op="attention_interface(",
        arg=2,
        seq_axis=2,
        heads="num_key_value_heads",
        width="head_dim",
    ),
    "attention_scores": _Component(
        # Inside the eager attention function the call dispatches to: the
        # softmax's argument — QKᵀ·scale with the causal and padding mask
        # already added, so a masked key is -inf here and 0 after.
        # (batch, head, query, key): the sequence axis is the *query's*, and
        # the last axis is the keys, whose length is the padded batch's.
        path="attentions.{layer}",
        side="input",
        stage=3,
        op="attention_interface(",
        inner="nn_functional_softmax_0",
        arg=0,
        seq_axis=2,
        needs="eager",
        heads="num_attention_heads",
    ),
    "attention_probs": _Component(
        # The same softmax's return: the attention pattern, before dropout
        # (a no-op in eval) and before it meets the values. A write here is
        # "make this head attend there".
        path="attentions.{layer}",
        side="input",
        stage=4,
        op="attention_interface(",
        inner="nn_functional_softmax_0",
        handle="output",
        arg=None,
        seq_axis=2,
        needs="eager",
        heads="num_attention_heads",
    ),
    "attention_z": _Component(
        # The same call's *return*, element 0: the mixer's per-head result
        # before the output projection. The first tap whose handle is the
        # call's output rather than one of its arguments, which is the whole
        # reason `handle` exists.
        path="attentions.{layer}",
        side="input",
        stage=5,
        op="attention_interface(",
        handle="output",
        arg=0,
        seq_axis=1,
        heads="num_attention_heads",
        width="head_dim",
    ),
    "input_ids": _Component(
        # The model's input: integer token ids, (batch, seq), no width axis.
        path="embed_tokens", side="input", stage=0, band=0, read_only=True
    ),
    "embeddings": _Component(
        # The vector the token ids look up. Band 0: it is the one tap upstream
        # of the whole stack.
        path="embed_tokens",
        side="output",
        stage=1,
        band=0,
        width="hidden_size",
    ),
    "block_input": _Component(
        path="layers.{layer}", side="input", stage=0, width="hidden_size"
    ),
    "attention_input_norm": _Component(
        # The first norm's output — what the mixer actually consumes.
        path="layers.{layer}.input_layernorm|layers.{layer}.ln_1",
        side="output", stage=1, width="hidden_size",
    ),
    "attention_premix": _Component(
        # The output projection's *input*: the heads' results, merged
        # head-major and not yet mixed — where a per-head edit belongs.
        path="attentions.{layer}.o_proj|attentions.{layer}.c_proj",
        side="input", stage=6, width="hidden_size", heads="num_attention_heads",
    ),
    "attention_output": _Component(
        # The mixer's contribution to the residual stream, before it is added:
        # block_mid = block_input + attention_output.
        path="attentions.{layer}", side="output", stage=7, width="hidden_size"
    ),
    "block_mid": _Component(
        # The residual stream after the mixer is added: the second norm's input.
        path="layers.{layer}.post_attention_layernorm|layers.{layer}.ln_2",
        side="input", stage=8, width="hidden_size",
    ),
    "mlp_input_norm": _Component(
        path="layers.{layer}.post_attention_layernorm|layers.{layer}.ln_2",
        side="output", stage=9, width="hidden_size",
    ),
    "mlp_input": _Component(
        path="mlps.{layer}", side="input", stage=10, width="hidden_size"
    ),
    "mlp_activation": _Component(
        # The activation function's output. No width: nnterp publishes no
        # intermediate size, which refuses a featurizer here.
        path="mlps.{layer}.act_fn|mlps.{layer}.act", side="output", stage=11, width="intermediate_size",
    ),
    "mlp_neuron_output": _Component(
        # The down-projection's input: act(gate)·up on a gated MLP, and the
        # activation itself on GPT-2, which has no gate — the same *place*,
        # a different tensor, and the table says so rather than hiding it.
        path="mlps.{layer}.down_proj|mlps.{layer}.c_proj", side="input", stage=12, width="intermediate_size",
    ),
    "mlp_output": _Component(
        # block_output = block_mid + mlp_output.
        path="mlps.{layer}", side="output", stage=13, width="hidden_size"
    ),
    "block_output": _Component(
        path="layers.{layer}", side="output", stage=14, width="hidden_size"
    ),
    "ln_final": _Component(
        path="ln_final", side="output", stage=0, band=2, width="hidden_size"
    ),
    "lm_head": _Component(
        path="lm_head", side="output", stage=1, band=2, width="vocab_size"
    ),
}


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """One tap, as the document named it — plus, for an interior, the operation
    `locate` resolved against the model."""

    component: str
    layer: int | None = None
    op: str | None = None
    #: `config.model_type`, when the engine that located this knows it. A
    #: string, so the plan stays data; it selects a row's overrides.
    family: str | None = None

    def __post_init__(self) -> None:
        if self.component not in _COMPONENTS:
            raise AddressError(f"component {self.component!r} has no address here")
        if self.op is not None and self._entry.op is None:
            raise AddressError(f"component {self.component!r} is a module boundary, not an operation")

    @property
    def _entry(self) -> _Component:
        entry = _COMPONENTS[self.component]
        differs = _OVERRIDES.get((self.family or "", self.component))
        return replace(entry, **differs) if differs else entry

    @property
    def where(self) -> tuple[str, int | None, str | None]:
        """The place, without the family it was located on. Equal across
        families whenever the table needed no exception for either."""
        return (self.component, self.layer, self.op)

    def get(self, value: Any) -> Any:
        """The activation inside a boundary's value."""
        select = self._entry.select
        if select is None:
            return value[0] if isinstance(value, tuple) else value
        found = select.get(value) if isinstance(select, Lens) else _walk(value, select)
        if not hasattr(found, "shape"):
            raise AddressError(
                f"component {self.component!r} on {self.family!r}: select {select!r} "
                f"reaches a {type(found).__name__}, not a tensor"
            )
        return found

    def put(self, value: Any, tensor: Any) -> Any:
        """`value` with the activation replaced — what a write hands back."""
        select = self._entry.select
        if select is None:
            return (tensor, *value[1:]) if isinstance(value, tuple) else tensor
        if isinstance(select, Lens):
            return select.put(value, tensor)
        return _rebuild(value, select, tensor)

    @property
    def call_site(self) -> str | None:
        """For an interior: the source text an engine matches to find the
        operation this address is about. `None` at a module boundary."""
        return self._entry.op

    @property
    def width_attribute(self) -> str | None:
        """The nnterp handle attribute holding this tap's width, if it has
        one. Names an attribute rather than reading it, because reading is the
        engine's job."""
        return self._entry.width

    @property
    def path(self) -> str:
        """The module, as a dotted path against the nnterp handle."""
        return self._entry.path.format(layer=self.layer)

    @property
    def side(self) -> str:
        """`input` or `output` — of the module, or of the operation."""
        return self._entry.side

    @property
    def interior(self) -> bool:
        """Whether this address is inside a forward rather than at a module
        boundary. The two are reached differently by every engine."""
        return self._entry.op is not None

    @property
    def arg(self) -> int | None:
        """For an interior: which positional argument of the call the tensor
        is, or which element of its return — None when the return is the
        tensor."""
        return self._entry.arg

    @property
    def inner(self) -> str | None:
        """For an operation inside the operation: its name there."""
        return self._entry.inner

    @property
    def needs(self) -> str | None:
        """The attention implementation this place exists under, if any."""
        return self._entry.needs

    @property
    def heads_attribute(self) -> str | None:
        """The config attribute counting this tensor's heads, if it has any."""
        return self._entry.heads

    @property
    def handle(self) -> str:
        """For an interior: whether the tensor is one of the call's arguments
        (`"inputs"`) or part of its return (`"output"`)."""
        return self._entry.handle

    @property
    def seq_axis(self) -> int:
        """Which axis of the tensor at this address the sequence runs along."""
        return self._entry.seq_axis

    @property
    def key(self) -> tuple[int, int, int]:
        """Sort key putting addresses in forward order: the embeddings, then
        the layer stack by depth and by where inside the block, then the final
        norm and the head. nnsight requires the order: reading layer 8 after
        layer 2 raises, and a write has to go in above the read that observes
        it."""
        return (self._entry.band, self.layer or 0, self._entry.stage)

    @property
    def read_only(self) -> bool:
        return self._entry.read_only

    def resolve(self, root: Any) -> Any:
        """Walk this address's path against `root`, by getattr, taking a
        numeric segment as an index: "layers.0" is `root.layers[0]`.

        The path is written in nnterp's standardized names. An engine that
        does not have those names translates first — how much translating that
        takes is a measurement of what the standardization is worth. Where
        the path offers alternatives, exactly one must exist here.
        """
        found = []
        for candidate in self.path.split("|"):
            target = root
            try:
                for segment in candidate.split("."):
                    target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            found.append(target)
        if len(found) != 1:
            raise AddressError(
                f"component {self.component!r}: {len(found)} of {self.path.split('|')} "
                "exist on this model; an address needs exactly one"
            )
        return found[0]


def head_count(config: Any, address: Address) -> int:
    """How many heads the tensor at `address` has. Key-head space is
    narrower than the query's on a model that groups, and a config that
    does not group does not say so."""
    attribute = address.heads_attribute
    if attribute is None:
        raise AddressError(f"component {address.component!r} is not a per-head tensor")
    return int(getattr(config, attribute, None) or config.num_attention_heads)


def width(config: Any, address: Address) -> int:
    """The feature width at `address`, off the config — the one place that
    knows how each family spells it."""
    attribute = address.width_attribute
    if attribute is None:
        raise AddressError(
            f"the width of {address.component!r} is not derivable from a config: "
            "it depends on the batch (the attention pattern's key axis) or is not a tensor's"
        )
    if attribute == "head_dim":
        # a per-head tensor is handed on flat: every head, side by side
        return head_count(config, address) * head_dim(config)
    if attribute == "intermediate_size":
        # GPT-2 calls it n_inner, and leaves it None to mean four times
        # hidden. Asked first, because a GPT-2 config can also carry a stray
        # `intermediate_size` its modules never read — the tiny test
        # checkpoint says 37 there and builds 128-wide MLPs.
        if hasattr(config, "n_inner"):
            return int(config.n_inner or 4 * config.hidden_size)
        return int(config.intermediate_size)
    return int(getattr(config, attribute))


def head_dim(config: Any) -> int:
    """One head's width: the config's own `head_dim` where it has one (it is
    not always hidden/heads), else the quotient."""
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


def check(config: Any, address: Address) -> None:
    """Refuse a place this checkpoint, as loaded, does not have."""
    have = getattr(config, "_attn_implementation", None)
    if address.needs is not None and have != address.needs:
        raise AddressError(
            f"component {address.component!r} exists only under {address.needs!r} attention, "
            f"and this model runs {have!r}: say \"attn_implementation\": \"{address.needs}\" "
            "in the document's model block"
        )


def _walk(value: Any, path: tuple[int | str, ...]) -> Any:
    for step in path:
        try:
            value = value[step]
        except (IndexError, KeyError, TypeError) as error:
            raise AddressError(f"select {path!r}: no {step!r} in a {type(value).__name__}") from error
    return value


def _rebuild(value: Any, path: tuple[int | str, ...], tensor: Any) -> Any:
    """`value` with the thing at `path` replaced, containers rebuilt on the
    way out: a tuple is immutable, and the caller's value is not ours to
    edit in place."""
    if not path:
        return tensor
    step, rest = path[0], path[1:]
    inner = _rebuild(_walk(value, (step,)), rest, tensor)
    if isinstance(value, tuple):
        items = [*value[:step], inner, *value[step + 1 :]]  # type: ignore[index, operator]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, (list, MutableMapping)):
        copied = copy.copy(value)
        copied[step] = inner  # type: ignore[index]
        return copied
    raise AddressError(f"select {path!r}: cannot rebuild a {type(value).__name__}")


def describe() -> dict[str, dict[str, Any]]:
    """The component vocabulary, as data an agent can read: for each name,
    where it is and what kind of place that is. This is the table, not a
    description of it, so it cannot drift."""
    return {
        name: {
            "path": entry.path,
            "side": entry.side,
            "interior": entry.op is not None,
            "layered": "{layer}" in entry.path,
            "seq_axis": entry.seq_axis,
            "width": entry.width,
            "read_only": entry.read_only,
            "heads": entry.heads is not None,
            "needs": entry.needs,
            # the families this row has an exception for, and what differs
            "overrides": {
                family: {key: repr(value) for key, value in differs.items()}
                for (family, component), differs in _OVERRIDES.items()
                if component == name
            },
        }
        for name, entry in _COMPONENTS.items()
    }
