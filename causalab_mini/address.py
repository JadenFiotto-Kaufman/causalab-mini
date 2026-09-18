"""ENGINE: a component -> where that tensor lives.

This is the only file in the project that knows anything about models. Given a
component and a layer it answers one question — which module path, which side,
and what to do with what is found there. Every model fact we had to encode
ourselves is here, and each one is an entry in FINDINGS.md.

A path is a string walked by getattr against the nnterp handle, with a numeric
segment taken as an index: "layers.0" is `model.layers[0]`, "lm_head" is
`model.lm_head`. The plan holds the string; the envoy is never pickled.
"""

from __future__ import annotations

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


def locate(component: str, layer: int | None) -> tuple[str, str]:
    if component not in _PATHS:
        raise AddressError(f"component {component!r} has no address here")
    template, side = _PATHS[component]
    return template.format(layer=layer), side


def order(component: str, layer: int | None) -> tuple[int, int]:
    """Sort key putting addresses in forward order. nnsight requires it: reading
    layer 8 after layer 2 raises, and a write has to go in above the read that
    observes it."""
    return (_STAGE[component], layer if layer is not None else 0)


def resolve(model, path: str):
    """The envoy at `path`, by getattr walking. Numeric segments index."""
    target = model
    for segment in path.split("."):
        target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
    return target


def read(model, path: str, side: str):
    """The tensor at an address.

    A module's output may be a bare tensor or a tuple whose first element is the
    hidden state, and which one it is depends on the transformers version, not
    on anything we can see in the document — so it is decided from the value.
    """
    value = getattr(resolve(model, path), side)
    return value[0] if isinstance(value, tuple) else value


def write(model, path: str, side: str, tensor) -> None:
    """Put a tensor back at an address, rebuilding the tuple if there was one."""
    envoy = resolve(model, path)
    current = getattr(envoy, side)
    setattr(envoy, side, (tensor, *current[1:]) if isinstance(current, tuple) else tensor)
