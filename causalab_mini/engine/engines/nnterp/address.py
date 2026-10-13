"""How a loaded nnterp model answers the address questions.

nnterp is the reference engine, so this is the fullest mapping of mini's
names: every component is one of nnterp 2.0's standard values on an envoy,
or a module boundary beside one. `ROUTES` says which, and resolving is then
asking nnterp: whether the checkpoint has the place and why not (`support`,
`nnterp.Unavailable` — the attention's interior needs eager attention, a
hybrid's linear blocks have no `self_attn`), **where it comes in the forward
pass** (`model.rank`), the tensor's axes as this checkpoint lays them out
(the served value's `dims`) and whether it is an operation inside a forward.

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

from ....address import COMPONENTS, Address, AddressError

#: Each of mini's names, as nnterp reaches it: `(hosts, value, beside)`.
#:
#: `hosts` is the envoy, a dotted path from the layer for a per-layer place
#: and from the model for a whole-model one ("" is that envoy itself); more
#: than one is the spellings families use for a module nnterp gives no
#: standard name, tried in order. `value` is read off it: one of nnterp's
#: standard values, or `input` / `output`, a module boundary nnsight serves
#: as it is. `beside` is set for a module boundary nnterp 2.0 has no value
#: for: nnterp ranks only its values, so the boundary's place in the forward
#: is the value it sits beside at the same layer, and how far before
#: (negative) or after it.
ROUTES: dict[str, tuple[tuple[str, ...], str, tuple[str, int] | None]] = {
    # the attention's interior: nnterp serves these under eager attention only
    "attention_query": (("self_attn",), "attention_queries", None),
    "attention_key": (("self_attn",), "attention_keys", None),
    "attention_scores": (("self_attn",), "attention_scores", None),
    "attention_probs": (("self_attn",), "attention_probabilities", None),
    "attention_z": (("self_attn",), "attention_head_outputs", None),
    "input_ids": (("",), "input_ids", None),
    "embeddings": (("",), "token_embeddings", None),
    "block_input": (("",), "layer_input", None),
    "attention_input_norm": (("input_layernorm",), "output", ("layer_input", 1)),
    "attention_premix": (
        ("self_attn.o_proj", "self_attn.c_proj", "self_attn.dense", "self_attn.out_proj"), "input",
        ("self_attn.attention_output", -1),
    ),
    "attention_output": (("self_attn",), "attention_output", None),
    "block_mid": (("post_attention_layernorm",), "input", ("self_attn.attention_output", 1)),
    "mlp_input_norm": (("post_attention_layernorm",), "output", ("self_attn.attention_output", 2)),
    "mlp_input": (("mlp",), "input", ("self_attn.attention_output", 3)),
    "mlp_activation": (("mlp.act_fn", "mlp.act", "mlp.activation_fn"), "output", ("mlp.mlp_output", -2)),
    "mlp_neuron_output": (
        ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.dense_4h_to_h", "mlp.fc_out", "mlp.w2"), "input",
        ("mlp.mlp_output", -1),
    ),
    "mlp_output": (("mlp",), "mlp_output", None),
    "block_output": (("",), "layer_output", None),
    "ln_final": (("norm",), "output", ("logits", -2)),
    "lm_head": (("lm_head",), "output", ("logits", -1)),
    "logits": (("",), "logits", None),
}


def resolve(model: Any, component: str, layer: int | None) -> tuple[Any, str, str, tuple[str, int] | None]:
    """Where `(component, layer)` is on this model: `(envoy, module, value,
    beside)` — the envoy that serves it and its path from the layer (or the
    model), the value read off it, and the `ROUTES` anchor of a boundary.

    The one routing step, shared by `locate` at compile time and `read` /
    `write` in the trace, so the two cannot come to disagree. An address
    carries no routing; the engine that holds the model finds it again.
    A name `ROUTES` does not have is a dotted `support()` name: its envoy
    and value are the name's two halves."""
    if component in ROUTES:
        hosts, value, beside = ROUTES[component]
    else:
        host, _, value = component.rpartition(".")
        hosts, beside = (host,), None
    module, envoy = _host(model if layer is None else model.layers[layer], hosts, component)
    return envoy, module, value, beside


def locate(model: Any, component: str, layer: int | None) -> Address:
    """The address of `(component, layer)` on a loaded nnterp model, or a
    refusal: a place this family, or this block, does not have — nnterp's
    reason, which for the attention's interior under sdpa is that it is read
    inside the eager attention forward.

    A name mini does not have but nnterp serves — a dotted `support()` name
    at a layer, a root value without, such as a value a user installed with
    `envoys=` — is accepted, and its axes are its value's own.
    """
    try:
        envoy, module, value, beside = resolve(model, component, layer)
        if beside is None:
            side, rank = model.rank(f"{module}.{value}" if module else value, layer)
            served = getattr(type(envoy), value)
            dims, inside, step = served.dims, served.inside_forward(), 0
        else:
            anchor, step = beside
            side, rank = model.rank(anchor, layer)
            dims, inside = COMPONENTS[component].dims, False
    except Unavailable as reason:  # a place this family, or this block, lacks
        raise AddressError(f"component {component!r}: {reason}") from None
    except (AddressError, KeyError, IndexError) as error:
        if component in ROUTES:
            raise AddressError(f"component {component!r}: {error.args[0]}") from None
        raise AddressError(
            f"component {component!r} has no address here, and nnterp serves no value of that "
            f"name {'at a layer' if layer is not None else 'on the model'} either. Mini's are "
            f"{sorted(ROUTES)}; a value a family lacks is added with `envoys=` at load"
        ) from None
    return Address(component, layer, (side, rank, step), inside, dims)


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
    return int(axis_size(resolve(model, address.component, address.layer)[0], "num_heads" if kind == "q" else "num_kv_heads"))


def width(model: Any, address: Address) -> int:
    """The feature width at `address`, off what nnterp publishes. An axis is
    named as the size that holds it, so the last axis is handed to nnterp's
    `axis_size`, which asks the serving envoy and the envoys above it. A
    per-head tensor is handed on flat: every head, side by side."""
    last = address.dims[-1]
    try:
        size = int(axis_size(resolve(model, address.component, address.layer)[0], last))
    except ValueError as reason:
        raise AddressError(f"the width of {address.component!r} is not a fact about the model: {reason}") from None
    if last in ("head_dim", "qk_head_dim"):
        return head_count(model, address) * size
    return size
