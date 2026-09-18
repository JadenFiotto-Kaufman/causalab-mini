"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about models, and
`Address` is why that is structural rather than a convention: a tap holds an
`Address`, and an `Address` is the only thing with a `read`/`write` that takes a
model. Every model fact we had to encode ourselves is in the table below, and
each one is an entry in FINDINGS.md.

An `Address` stays pure data — the document's `(component, layer)` and nothing
else, so it pickles, sorts, prints and travels in a plan. Everything a model is
needed for (which module path, which side, where the sequence axis is, where in
the forward pass it sits) is a lookup, and the model is passed in, never held.

A path is a string walked by getattr against the nnterp handle, with a numeric
segment taken as an index: "layers.0" is `model.layers[0]`, "lm_head" is
`model.lm_head`. The plan holds the string; the envoy is never pickled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# component -> (path template, side). `lm_head` is layer-less.
_PATHS = {
    "block_output": ("layers.{layer}", "output"),
    "lm_head": ("lm_head", "output"),
}

# Where a component sits in the forward pass, relative to the layer stack. A
# block_output is inside the stack at its layer; the head runs after all of it.
_STAGE = {"block_output": 0, "lm_head": 1}


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """One tap, as the document named it."""

    component: str
    layer: int | None = None

    def __post_init__(self) -> None:
        if self.component not in _PATHS:
            raise AddressError(f"component {self.component!r} has no address here")

    @property
    def path(self) -> str:
        """The module, as a dotted path against the nnterp handle."""
        return _PATHS[self.component][0].format(layer=self.layer)

    @property
    def side(self) -> str:
        """`input` or `output` of that module."""
        return _PATHS[self.component][1]

    @property
    def key(self) -> tuple[int, int]:
        """Sort key putting addresses in forward order. nnsight requires it:
        reading layer 8 after layer 2 raises, and a write has to go in above the
        read that observes it."""
        return (_STAGE[self.component], self.layer or 0)

    def envoy(self, model: Any) -> Any:
        """The envoy this address names, by getattr walking. Numeric segments
        index."""
        target = model
        for segment in self.path.split("."):
            target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
        return target

    def read(self, model: Any) -> Any:
        """The tensor at this address.

        A module's output may be a bare tensor or a tuple whose first element is
        the hidden state, and which one it is depends on the transformers
        version, not on anything we can see in the document — so it is decided
        from the value.
        """
        value = getattr(self.envoy(model), self.side)
        return value[0] if isinstance(value, tuple) else value

    def write(self, model: Any, tensor: Any) -> None:
        """Put a tensor back, rebuilding the tuple if there was one."""
        envoy = self.envoy(model)
        current = getattr(envoy, self.side)
        setattr(
            envoy,
            self.side,
            (tensor, *current[1:]) if isinstance(current, tuple) else tensor,
        )
