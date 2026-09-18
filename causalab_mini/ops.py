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

from typing import Any, Protocol, runtime_checkable

import torch

from .shapes import Positions


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
    """A `do`: the feature value in, the feature value out. Stateless."""

    def __call__(self, f: Any, operand: Any) -> Any: ...


def swap(f: Any, operand: Any) -> Any:
    """The absolute class: it replaces, it does not add."""
    return operand


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
MECHANISMS: dict[str, Mechanism] = {"swap": swap}


def gather(tensor: Any, positions: Positions, seq_axis: int = 1) -> Any:
    """One position per row, with the sequence axis dropped: (batch, seq, width)
    -> (batch, width). `seq_axis` is which axis the sequence runs along — 1 at a
    module boundary, 2 inside attention, where a tensor is (batch, head, seq,
    head_dim). Which one it is is a fact about the address, not about the
    tensor, so it is passed in."""
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    index = torch.as_tensor(positions, device=tensor.device)
    return tensor.movedim(seq_axis, 1)[rows, index]


def scatter(tensor: Any, positions: Positions, values: Any, seq_axis: int = 1) -> Any:
    """A copy of `tensor` with one position per row replaced by `values`."""
    rows = torch.arange(tensor.shape[0], device=tensor.device)
    index = torch.as_tensor(positions, device=tensor.device)
    out = tensor.clone()
    out.movedim(seq_axis, 1)[rows, index] = values.to(out.dtype)  # a view of `out`
    return out


def apply_write(
    tensor: Any,
    positions: Positions,
    operand: Any,
    mechanism: str = "swap",
    featurizer: str | Featurizer = "identity",
    seq_axis: int = 1,
) -> Any:
    featurize = FEATURIZERS[featurizer] if isinstance(featurizer, str) else featurizer
    x = gather(tensor, positions, seq_axis)
    f, err = featurize.featurize(x)
    f = MECHANISMS[mechanism](f, operand)
    return scatter(tensor, positions, featurize.inverse(f, err, x), seq_axis)
