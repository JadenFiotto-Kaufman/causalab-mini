"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about a model's
*internals* — `loading.py` beside it knows how to build the handle and
nothing else. Every model fact we had to encode ourselves is in the table
below, and each one is an entry in FINDINGS.md.

An address says **where**, in terms that are true of the architecture: which
module, which side, which argument of which operation, which axis the sequence
runs along. It does not say how to reach there, because that is a property of
the runtime and not of the model — `engine/nnterp.py` reads an address with
nnsight envoys, and another engine would read the same address differently.

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
    def arg(self) -> int:
        """For an interior: which positional argument of the call the tensor
        is."""
        return self._entry.arg

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

    def resolve(self, root: Any) -> Any:
        """Walk this address's path against `root`, by getattr, taking a
        numeric segment as an index: "layers.0" is `root.layers[0]`.

        The path is written in nnterp's standardized names. An engine that
        does not have those names translates first — how much translating that
        takes is a measurement of what the standardization is worth.
        """
        target = root
        for segment in self.path.split("."):
            target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
        return target
