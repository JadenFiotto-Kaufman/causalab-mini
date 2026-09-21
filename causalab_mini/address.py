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

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class _Component:
    """Everything about one component that is a fact about models."""

    path: str  # a dotted path template against the nnterp handle
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
    arg: int = 0  # which positional argument, or which element of the return
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
        stage=1,
        op="attention_interface(",
        arg=1,
        seq_axis=2,
    ),
    "attention_key": _Component(
        # The same call as the query, argument 2: the keys *before* GQA's
        # repeat_kv, so this is key-head space and narrower than the query by
        # the grouping ratio on a model that groups. No width: nnterp
        # publishes nothing for it, which is what refuses a featurizer here.
        path="attentions.{layer}",
        side="input",
        stage=1,
        op="attention_interface(",
        arg=2,
        seq_axis=2,
    ),
    "attention_z": _Component(
        # The same call's *return*, element 0: the mixer's per-head result
        # before the output projection. The first tap whose handle is the
        # call's output rather than one of its arguments, which is the whole
        # reason `handle` exists.
        path="attentions.{layer}",
        side="input",
        stage=2,
        op="attention_interface(",
        handle="output",
        arg=0,
        seq_axis=1,
    ),
    "embeddings": _Component(
        # The vector the token ids look up. Band 0: it is the one tap upstream
        # of the whole stack.
        path="embed_tokens",
        side="output",
        stage=0,
        band=0,
        width="hidden_size",
    ),
    "block_input": _Component(
        path="layers.{layer}", side="input", stage=0, width="hidden_size"
    ),
    "attention_output": _Component(
        # The mixer's contribution to the residual stream, before it is added:
        # block_mid = block_input + attention_output.
        path="attentions.{layer}", side="output", stage=3, width="hidden_size"
    ),
    "mlp_input": _Component(
        path="mlps.{layer}", side="input", stage=4, width="hidden_size"
    ),
    "mlp_output": _Component(
        # block_output = block_mid + mlp_output.
        path="mlps.{layer}", side="output", stage=5, width="hidden_size"
    ),
    "block_output": _Component(
        path="layers.{layer}", side="output", stage=6, width="hidden_size"
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
        is, or which element of its return."""
        return self._entry.arg

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
        }
        for name, entry in _COMPONENTS.items()
    }
