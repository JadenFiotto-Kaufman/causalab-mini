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

from typing import Any, Callable, Protocol, runtime_checkable

import torch

from ..shapes import Positions


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
}


def is_ragged(positions: Positions) -> bool:
    """Whether the rows' windows differ in width. A plan says so; a tensor
    cannot, which is why the shape below is keyed on the positions."""
    return len({len(window) for window in positions}) > 1


def _flat(positions: Positions, device: Any) -> tuple[Any, Any]:
    """Every (row, position) pair in row order, for a ragged window. An
    empty window — an excluded row — contributes nothing."""
    rows = [row for row, window in enumerate(positions) for _ in window]
    index = [position for window in positions for position in window]
    return torch.as_tensor(rows, device=device), torch.as_tensor(index, device=device)


def gather(tensor: Any, positions: Positions, seq_axis: int = 1) -> Any:
    """A window per row.

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
    moved = tensor.movedim(seq_axis, 1)
    if is_ragged(positions):
        rows, index = _flat(positions, tensor.device)
        return moved[rows, index]
    rows = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    index = torch.as_tensor(positions, device=tensor.device)  # (batch, w)
    return moved[rows, index]


def scatter(tensor: Any, positions: Positions, values: Any, seq_axis: int = 1) -> Any:
    """A copy of `tensor` with each row's window replaced by `values`: the
    shape `gather` would return for these positions, or anything that
    broadcasts to it — a published (w, width) or (width,) mean, say."""
    out = tensor.clone()
    moved = out.movedim(seq_axis, 1)  # a view of `out`
    if is_ragged(positions):
        rows, index = _flat(positions, tensor.device)
        moved[rows, index] = values.to(out.dtype)
        return out
    rows = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
    index = torch.as_tensor(positions, device=tensor.device)
    moved[rows, index] = values.to(out.dtype)
    return out


def apply_write(
    tensor: Any,
    positions: Positions,
    operand: Any,
    mechanism: str = "swap",
    featurizer: str | Featurizer = "identity",
    seq_axis: int = 1,
    params: dict[str, Any] | None = None,
) -> Any:
    featurize = FEATURIZERS[featurizer] if isinstance(featurizer, str) else featurizer
    x = gather(tensor, positions, seq_axis)
    f, err = featurize.featurize(x)
    f = MECHANISMS[mechanism](f, operand, **(params or {}))
    return scatter(tensor, positions, featurize.inverse(f, err, x), seq_axis)
