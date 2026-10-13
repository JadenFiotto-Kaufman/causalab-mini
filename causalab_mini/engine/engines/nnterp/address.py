"""How a loaded nnterp model answers the address questions.

The standard set is `causalab_mini/address.py`, written in nnterp's own
names, so here resolving is asking: whether the checkpoint has a place and
why not (`support`, `nnterp.Unavailable` — the attention's interior needs
eager attention, a hybrid's linear blocks have no `self_attn`), **where it
comes in the forward pass** (`model.rank`), the tensor's axes (the served
value's `dims`) and whether it is an operation inside a forward.

A width is never written here. nnterp names a layout axis that is a model
fact exactly as the size attribute that holds it — `hidden_size`,
`num_heads`, `intermediate_size` — so `width` and `head_count` hand the axis
name to `nnterp.axis_size`, which resolves it on the serving envoy or the
first envoy above it that declares it. An axis no envoy declares — the batch,
the sequence, the pattern's keys — is not a fact about the model, and that is
the refusal.
"""

from __future__ import annotations

from typing import Any

from nnterp import Unavailable, axis_size

from ....address import COMPONENTS, Address, AddressError, Component


def locate(model: Any, component: str, layer: int | None) -> Address:
    """The address of `(component, layer)` on a loaded nnterp model, or a
    refusal: a place this family, or this block, does not have — nnterp's
    reason, which for the attention's interior under sdpa is that it is read
    inside the eager attention forward.

    A name the table does not have but nnterp serves — a dotted `support()`
    name at a layer, a root value without, such as a value a user installed
    with `envoys=` — is accepted: its envoy and value are the name's two
    halves, and its axes are its value's own.
    """
    entry = COMPONENTS.get(component)
    if entry is None:
        host, _, value = component.rpartition(".")
        entry = Component((host,), value)
    try:
        root = model if layer is None else model.layers[layer]
        module, host = _host(root, entry.hosts, component)
        if entry.boundary is None:
            side, rank = model.rank(f"{module}.{entry.value}" if module else entry.value, layer)
            served = getattr(type(host), entry.value)
            dims, inside, step = served.dims, served.inside_forward(), 0
        else:
            dims, anchor, step = entry.boundary
            side, rank = model.rank(anchor, layer)
            inside = False
    except Unavailable as reason:  # a place this family, or this block, lacks
        raise AddressError(f"component {component!r}: {reason}") from None
    except (AddressError, KeyError, IndexError) as error:
        if component in COMPONENTS:
            raise AddressError(f"component {component!r}: {error.args[0]}") from None
        raise AddressError(
            f"component {component!r} has no address here, and nnterp serves no value of that "
            f"name {'at a layer' if layer is not None else 'on the model'} either. Mini's are "
            f"{sorted(COMPONENTS)}; a value a family lacks is added with `envoys=` at load"
        ) from None
    path = module if layer is None else f"layers.{layer}" + (f".{module}" if module else "")
    return Address(component, layer, path, entry.value, (side, rank, step), inside, dims)


def envoy(model: Any, address: Address) -> Any:
    """The envoy `address` is read off: its path walked from the model."""
    return _walk(model, address.path, address.component)


def _walk(root: Any, path: str, component: str) -> Any:
    """`path` from `root`, by getattr, a numeric segment as an index:
    "layers.0" is `root.layers[0]`, and the empty path is `root`."""
    target = root
    try:
        for segment in filter(None, path.split(".")):
            target = target[int(segment)] if segment.isdigit() else getattr(target, segment)
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise AddressError(f"component {component!r}: {path!r} does not exist on this model") from error
    if target is None:
        raise AddressError(f"component {component!r}: {path!r} is None on this model")
    return target


def _host(root: Any, hosts: tuple[str, ...], component: str) -> tuple[str, Any]:
    """The first of a row's envoy paths this checkpoint has, and the envoy."""
    for path in hosts:
        try:
            return path, _walk(root, path, component)
        except AddressError:
            continue
    raise AddressError(f"this block has no {' or '.join(hosts)} module")


def head_count(model: Any, address: Address) -> int:
    """How many heads the tensor at `address` has, off what nnterp publishes:
    the serving envoy's own count, so it holds on a model whose blocks differ."""
    kind = address.heads_kind
    if kind is None:
        raise AddressError(f"component {address.component!r} is not a per-head tensor")
    return int(axis_size(envoy(model, address), "num_heads" if kind == "q" else "num_kv_heads"))


def width(model: Any, address: Address) -> int:
    """The feature width at `address`, off what nnterp publishes. An axis is
    named as the size that holds it, so the last axis is handed to nnterp's
    `axis_size`, which asks the serving envoy and the envoys above it. A
    per-head tensor is handed on flat: every head, side by side."""
    last = address.dims[-1]
    try:
        size = int(axis_size(envoy(model, address), last))
    except ValueError as reason:
        raise AddressError(f"the width of {address.component!r} is not a fact about the model: {reason}") from None
    if last in ("head_dim", "qk_head_dim"):
        return head_count(model, address) * size
    return size
