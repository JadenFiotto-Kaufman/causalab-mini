"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about a model's
*internals* — `loading.py` beside it knows how to build the handle and
nothing else — and `Address` is why that is structural rather than a
convention: a tap holds an `Address`, and an `Address` is the only thing with a
`read`/`write` that takes a model. Every model fact we had to encode
ourselves is in the table below, and each one is an entry in FINDINGS.md.

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

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class _Component:
    """Everything about one component that is a fact about models."""

    path: str  # a dotted path template against the nnterp handle
    side: str  # "output" or "input"
    stage: int  # order within one block: the attention interior precedes its block's output
    op: str | None = None  # the call site of the `.source` operation, for an interior
    arg: int = 0  # which positional argument of that call the tensor is
    seq_axis: int = 1  # which axis of the tensor the sequence runs along
    width: str | None = None  # the nnterp handle attribute holding this tap's width


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
        stage=0,
        op="attention_interface(",
        arg=1,
        seq_axis=2,
    ),
    "block_output": _Component(
        path="layers.{layer}", side="output", stage=1, width="hidden_size"
    ),
    "lm_head": _Component(path="lm_head", side="output", stage=0, width="vocab_size"),
}


class AddressError(ValueError):
    pass


def find_op(source: Any, call_site: str) -> str:
    """The single operation of a module's `.source` whose call site contains
    `call_site`, or a refusal naming everything the forward does have.

    Matching the *source line* rather than the operation's name is the whole
    point. nnsight names an operation `{callable}_{occurrence}` and gives
    assignments the same namespace as calls, so on transformers 5.17 the
    attention forward has both `attention_interface_0` (the assignment
    `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)`) and
    `attention_interface_1` (the call). A name match on "attention_interface"
    hits both; the needle `"attention_interface("` is call-shaped and hits one.
    """
    hits = [op.name for op in source if call_site in op.text.split("\n")[op.line - 1]]
    if len(hits) != 1:
        raise AddressError(
            f"{call_site!r} matches {len(hits)} operations {hits} of this forward; "
            f"an address serves exactly one. The forward's operations are: "
            f"{list(source.names)}"
        )
    return hits[0]


@dataclass(frozen=True)
class Address:
    """One tap, as the document named it — plus, for an interior, the operation
    `locate` resolved against the model."""

    component: str
    layer: int | None = None
    op: str | None = None

    def __post_init__(self) -> None:
        if self.component not in _COMPONENTS:
            raise AddressError(f"component {self.component!r} has no address here")
        if self.op is not None and self._entry.op is None:
            raise AddressError(f"component {self.component!r} is a module boundary, not an operation")

    @property
    def _entry(self) -> _Component:
        return _COMPONENTS[self.component]

    @property
    def path(self) -> str:
        """The module, as a dotted path against the nnterp handle."""
        return self._entry.path.format(layer=self.layer)

    @property
    def side(self) -> str:
        """`input` or `output` — of the module, or of the operation."""
        return self._entry.side

    @property
    def seq_axis(self) -> int:
        """Which axis of the tensor at this address the sequence runs along."""
        return self._entry.seq_axis

    @property
    def key(self) -> tuple[int, int, int]:
        """Sort key putting addresses in forward order: everything inside the
        layer stack, by depth and then by where inside the block, before
        everything after it. nnsight requires the order: reading layer 8 after
        layer 2 raises, and a write has to go in above the read that observes
        it."""
        return (0 if self.layer is not None else 1, self.layer or 0, self._entry.stage)

    def width(self, model: Any) -> int:
        """The size of the tap's last axis — the `d` a featurizer's `k` is a
        subspace of. It is derived from (model, site) and may never be authored,
        which is exactly why it is asked of an address and not of a document."""
        attribute = self._entry.width
        if attribute is None:
            raise AddressError(
                f"the width of {self.component!r} is not derivable here; nnterp "
                "publishes hidden_size and vocab_size on the handle and nothing "
                "for an attention interior"
            )
        return int(getattr(model, attribute))

    @classmethod
    def locate(cls, model: Any, component: str, layer: int | None = None) -> "Address":
        """The address of `(component, layer)` against a loaded model.

        For a module boundary that is just the pair. For an interior it also
        resolves the `.source` operation, here on the client, so the plan
        carries a name and a refusal happens before anything runs.
        """
        one = cls(component, layer)
        if one._entry.op is None:
            return one
        return replace(one, op=find_op(one.envoy(model).source, one._entry.op))

    def envoy(self, model: Any) -> Any:
        """The envoy this address names, by getattr walking. Numeric segments
        index."""
        target = model
        for segment in self.path.split("."):
            target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
        return target

    def _operation(self, model: Any) -> Any:
        if self.op is None:
            raise AddressError(
                f"component {self.component!r} is an interior; build its address "
                "with Address.locate(model, ...) so the operation is resolved"
            )
        return getattr(self.envoy(model).source, self.op)

    def read(self, model: Any) -> Any:
        """The tensor at this address.

        At a module boundary the output may be a bare tensor or a tuple whose
        first element is the hidden state, and which one it is depends on the
        transformers version, not on anything we can see in the document — so it
        is decided from the value. Inside a forward the value is one argument of
        one call, and nothing is ambiguous.
        """
        if self._entry.op is None:
            value = getattr(self.envoy(model), self.side)
            return value[0] if isinstance(value, tuple) else value
        args, _ = self._operation(model).inputs
        return args[self._entry.arg]

    def write(self, model: Any, tensor: Any) -> None:
        """Put a tensor back: rebuilding the tuple if there was one, or
        rebuilding the call's arguments around the new one."""
        if self._entry.op is None:
            envoy = self.envoy(model)
            current = getattr(envoy, self.side)
            setattr(
                envoy,
                self.side,
                (tensor, *current[1:]) if isinstance(current, tuple) else tensor,
            )
            return
        operation = self._operation(model)
        args, kwargs = operation.inputs
        index = self._entry.arg
        operation.inputs = ((*args[:index], tensor, *args[index + 1 :]), kwargs)
