"""AGNOSTIC: gather, scatter, and the do mechanisms. Tensors in, tensors out.

Nothing here knows what a layer is, what a model is, or what a document said.
Positions arrive as integers and activations arrive as tensors.

The shape of a write is the seam the rest of the project is built around:

    write = inverse(do(featurize(x)), err, x)

With the identity featurizer that collapses to "replace the tensor", which is
activation patching. Swap in a rotation for `featurize`/`inverse` and the same
line is DAS — the error term and the unselected directions come from `x`, the
pre-write value, which is exactly what makes a subspace swap leave the
orthogonal complement alone. That is why `err` and `x` are threaded through
`inverse` even though the identity ignores both.

**A featurizer is an object and a mechanism is a function**, and that asymmetry
is forced rather than chosen: a trained rotation *is state*, so `featurize` and
`inverse` have to be two methods over one parameter, while `swap` has nothing to
remember (a coefficient, when a mechanism needs one, comes from the document as
an argument). Both get a named protocol so the shared signature is written down
once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Protocol, runtime_checkable

import torch

from ..shapes import Positions, Selection


@runtime_checkable
class Featurizer(Protocol):
    """A pair of maps between an activation and the feature space a write acts
    in. `featurize` may throw information away; `err` is what it threw away, and
    `inverse` gets it back together with the pre-write activation `x`."""

    def featurize(self, x: Any) -> tuple[Any, Any]:
        """x -> (f, err)."""
        ...

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        """(f, err, x) -> an activation of x's shape."""
        ...


class Identity:
    """The featurizer a read or write with no `featurizer` key gets: the feature
    space is the activation itself and nothing is left over."""

    def featurize(self, x: Any) -> tuple[Any, None]:
        return x, None

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        return f


class Mechanism(Protocol):
    """A `do`: the feature value in, the feature value out. Stateless. A
    mechanism that needs a number — a scale, a seed — takes it by keyword,
    from the write's `params`."""

    def __call__(self, f: Any, operand: Any, **params: Any) -> Any: ...


def swap(f: Any, operand: Any) -> Any:
    """The absolute class: it replaces, it does not add."""
    return operand


def add_scaled(f: Any, operand: Any, scale: float = 1.0) -> Any:
    """The additive class: `f + scale·operand`. Steering is this with a
    direction as the operand; a scaled mean is this with the mean."""
    return f + scale * operand


def lerp(f: Any, operand: Any, t: float) -> Any:
    """`(1−t)·f + t·operand`: swap at t=1, nothing at t=0, and every partial
    interchange between."""
    return (1.0 - t) * f + t * operand


def gaussian(f: Any, operand: Any, seed: int, scale: float = 1.0) -> Any:
    """`f + scale·ε`, ε ~ N(0, 1) drawn from `seed` — noise ablation. The
    draw is on the CPU generator and moved, so the same seed is the same
    noise on any device. Takes no operand."""
    noise = torch.randn(f.shape, generator=torch.Generator().manual_seed(seed))
    return f + scale * noise.to(f)


def clamp(f: Any, operand: Any, lo: float | None = None, hi: float | None = None) -> Any:
    """Bound the feature value: `min(max(f, lo), hi)`, either side optional.
    Takes no operand. With a gate or a site of `units` it is "cap these
    neurons"; alone it is activation clipping."""
    return f.clamp(min=lo, max=hi)


def renormalize(f: Any, operand: Any) -> Any:
    """`f · ‖f₀‖ / ‖f‖` — put the norm back after other writes moved it.

    Its operand is not authored: it is `f₀`, the feature value here *before
    any write of this forward touched it*, which the seam supplies. That is
    also why it must be the last write at its site — run first, `f` is `f₀`
    and this is the identity — and why a document that orders it otherwise
    is refused rather than quietly doing nothing. Steering by `add_scaled`
    and then renormalizing is "change the direction, keep the magnitude"."""
    target = operand.norm(dim=-1, keepdim=True)
    return f * (target / f.norm(dim=-1, keepdim=True).clamp_min(1e-12))


#: Mechanisms whose operand is the pre-write feature value, supplied by the
#: write seam rather than named by the document.
PRE_WRITE = frozenset({"renormalize"})


def applies(frame: int | str | None, step: int | None) -> bool:
    """Whether a tap in `frame` acts at decode `step` (None: a plain forward).
    The prompt frame is the prefill, step 0; `"all"` is every step."""
    if frame is None:
        return step in (None, 0)
    if frame == "all":
        return step is not None
    return step == frame


def at_step(at: Selection, tensor: Any, seq_axis: int, frame: int | str | None, step: int | None) -> Selection:
    """Where a tap's positions land in *this* forward's tensor.

    In the prompt frame at the prefill the plan's positions are right. Past
    the prefill a forward sees one position, and so does the head at the
    prefill under generation (transformers keeps only the last row of
    logits) — both are the last index of whatever the sequence axis has,
    which is what a step tap means anyway, and what a prompt-frame `-1` at
    the head meant. A tensor with one position is one position; the plan's
    index into the prompt cannot apply to it.
    """
    length = tensor.shape[seq_axis]
    if frame is not None or (step is not None and length == 1):
        last = length - 1
        moved = tuple((last,) if window else () for window in at.positions)
        return replace(at, positions=moved, flat=at.flat or is_ragged(moved))
    return at


def rows(tensor: Any, positions: Positions | None, start: int, stop: int) -> Any:
    """Rows `start:stop` of a gathered value. A rectangle is sliced; a ragged
    value is flat, so its rows are wherever their windows' lengths put them.
    `positions` None is a value with no row axis — a mean, a basis — which
    every window of rows shares whole."""
    if positions is None:
        return tensor
    if is_ragged(positions):
        before = sum(len(window) for window in positions[:start])
        return tensor[before : before + sum(len(window) for window in positions[start:stop])]
    return tensor[start:stop]


def resolve_operand(values: dict[str, Any], operand: Any) -> Any:
    """What a write acts with: a named value read earlier, a literal number
    (zero ablation is `0.0`), or nothing for a mechanism that takes none."""
    if isinstance(operand, str):
        return values[operand]
    if operand is None:
        return None
    return torch.tensor(float(operand))


# The two closed vocabularies, by the name a document spells.
#
# `FEATURIZERS` holds the ones that are *stateless*, so one instance per process
# is the same as one per run. A trained featurizer is not one of those, and that
# is the single thing DAS changed about this file: a rotation's parameter is run
# state that an optimizer steps, so it cannot live in a module-level table, and
# `apply_write` therefore takes either a registered name or the object itself.
# See `featurizer.KINDS` for the constructors a document's `featurizers` section
# names.
FEATURIZERS: dict[str, Featurizer] = {"identity": Identity()}
#: Typed loosely on purpose: each mechanism names the numbers it takes as
#: keywords, and `Mechanism` above documents the shape rather than checking it.
MECHANISMS: dict[str, Callable[..., Any]] = {
    "swap": swap,
    "add_scaled": add_scaled,
    "lerp": lerp,
    "gaussian": gaussian,
    "clamp": clamp,
    "renormalize": renormalize,
}


def is_ragged(positions: Positions) -> bool:
    """Whether the rows' windows differ in width. A plan says so; a tensor
    cannot, which is why the shape below is keyed on the positions."""
    return len({len(window) for window in positions}) > 1


def _flat(positions: Positions, device: Any) -> tuple[Any, Any]:
    """Every (row, position) pair in row order, for a ragged window. An
    empty window — an excluded row — contributes nothing, and a window that
    is empty on *every* row is an empty gather rather than an error: the
    dtype is stated because an empty Python list would be floats, which is
    not a thing a tensor can be indexed by."""
    rows = [row for row, window in enumerate(positions) for _ in window]
    index = [position for window in positions for position in window]
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(index, dtype=torch.long, device=device),
    )


def _selection(at: Selection | Positions) -> Selection:
    """A bare `Positions` is every feature at those positions, flat if they
    are ragged."""
    return at if isinstance(at, Selection) else Selection(at, flat=is_ragged(at))


def gather(tensor: Any, at: Selection | Positions, seq_axis: int = 1) -> Any:
    """A window per row, and of it the features the selection names.

    Uniform windows keep the rectangle: (batch, seq, width) -> (batch, w,
    width), with the unit window w=1 that a metric squeezes. Ragged windows
    cannot, and come back **flat**: (total, width), every row's positions
    in row order, the plan's own `positions` saying where each row begins
    and ends. Which shape it is is decided by the positions, never by the
    tensor.

    `seq_axis` is which axis the sequence runs along — 1 at a module
    boundary, 2 inside attention, where a tensor is (batch, head, seq,
    head_dim). Which one it is is a fact about the address, not about the
    tensor, so it is passed in.
    """
    at = _selection(at)
    window = _window(tensor, at, seq_axis)
    if at.groups is None:
        return window
    split = window.reshape(*_lead(window, at), at.groups, -1)
    if at.take is not None:
        split = split[..., list(at.take), :]
    return split.flatten(-2)


def _window(tensor: Any, at: Selection, seq_axis: int) -> Any:
    positions = at.positions
    moved = tensor.movedim(seq_axis, 1)
    if _is_flat(at):
        rows, index = _flat(positions, tensor.device)
        return moved[rows, index]
    rows = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    # the dtype is stated for the same reason as in `_flat`: a rectangle of
    # empty windows is a list of empty lists, and that is a float tensor
    index = torch.as_tensor(positions, dtype=torch.long, device=tensor.device)  # (batch, w)
    return moved[rows, index]


def _is_flat(at: Selection) -> bool:
    """Ragged positions can only come back flat; `flat` says so for a window
    of a ragged pass whose own rows happen to line up."""
    return at.flat or is_ragged(at.positions)


def _lead(window: Any, at: Selection) -> tuple[int, ...]:
    """The axes of a gathered window that are not features: `(total,)` for a
    ragged one, `(rows, w)` for a rectangle."""
    return tuple(window.shape[: 1 if _is_flat(at) else 2])


def scatter(tensor: Any, at: Selection | Positions, values: Any, seq_axis: int = 1) -> Any:
    """A copy of `tensor` with the selection replaced by `values`: the shape
    `gather` would return for it, or anything that broadcasts to that — a
    published (w, width) or (width,) mean, say. Features the selection does
    not name are left as they were."""
    at = _selection(at)
    positions = at.positions
    if at.groups is not None:
        window = _window(tensor, at, seq_axis)
        lead = _lead(window, at)
        split = window.reshape(*lead, at.groups, -1).clone()
        take = tuple(range(at.groups)) if at.take is None else at.take
        piece = values.to(split).expand(*lead, len(take) * split.shape[-1])
        split[..., list(take), :] = piece.reshape(*lead, len(take), -1)
        values = split.reshape(window.shape)
    out = tensor.clone()
    moved = out.movedim(seq_axis, 1)  # a view of `out`
    if _is_flat(at):
        rows, index = _flat(positions, tensor.device)
        moved[rows, index] = values.to(out)
        return out
    rows = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    index = torch.as_tensor(positions, device=tensor.device)
    moved[rows, index] = values.to(out)
    return out


def apply_write(
    tensor: Any,
    at: Selection | Positions,
    operand: Any,
    mechanism: str = "swap",
    featurizer: str | Featurizer = "identity",
    seq_axis: int = 1,
    params: dict[str, Any] | None = None,
    original: Any = None,
    features: tuple[int, ...] | None = None,
) -> Any:
    """`original` is the tensor at this address before any write of this
    forward: what a `PRE_WRITE` mechanism measures against. `features` are
    the coordinates of the *featurizer's* space the mechanism acts on — one
    SAE latent, three directions of a rotation — the rest passing through."""
    featurize = FEATURIZERS[featurizer] if isinstance(featurizer, str) else featurizer
    x = gather(tensor, at, seq_axis)
    with exact(x):
        f, err = featurize.featurize(x)
        if mechanism in PRE_WRITE:
            operand = featurize.featurize(gather(tensor if original is None else original, at, seq_axis))[0]
        if features is None:
            f = MECHANISMS[mechanism](f, operand, **(params or {}))
        else:
            index = list(features)
            # an operand in the same feature space is cut the same way; a
            # number, or one already that narrow, is used as it is
            if hasattr(operand, "shape") and operand.shape[-1:] == f.shape[-1:]:
                operand = operand[..., index]
            f = f.clone()
            acted = MECHANISMS[mechanism](f[..., index], operand, **(params or {}))
            f[..., index] = acted.to(f) if hasattr(acted, "to") else acted  # a literal is a number
        written = featurize.inverse(f, err, x)
    return scatter(tensor, at, written, seq_axis)


def softcap(f: Any, cap: float | None) -> Any:
    """`tanh(f / cap) · cap`, or `f` where the family does not cap.

    The logit lens pushes a residual through the final norm and head by hand,
    and on a family whose logits are capped — Gemma-2's
    `final_logit_softcapping` — the head's output is not what the model
    predicts from. This is the last step the model would have taken.
    """
    return f if cap is None else torch.tanh(f / cap) * cap


def exact(x: Any) -> Any:
    """Featurizer math runs outside autocast.

    A server may wrap a whole request in `torch.autocast` (NDIF does, at the
    served dtype). Inside it a matmul of two fp32 tensors comes back bf16
    while a subtraction stays fp32 — so half of a Cayley solve is downcast
    and `linalg.solve` refuses the pair; and a rotation that *did* run would
    be orthonormal only to bf16. A featurizer declares its own dtype, fp32,
    and this is where that declaration is kept. Found on the first real NDIF
    run; no local run autocasts. FINDINGS §19."""
    return torch.autocast(device_type=x.device.type, enabled=False)
