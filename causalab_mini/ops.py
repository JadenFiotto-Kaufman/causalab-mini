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
"""

from __future__ import annotations

import torch


def gather(tensor, positions):
    """(batch, seq, width) at one position per row -> (batch, width)."""
    index = torch.as_tensor(positions, device=tensor.device)
    return tensor[torch.arange(tensor.shape[0], device=tensor.device), index]


def scatter(tensor, positions, values):
    """A copy of `tensor` with one position per row replaced by `values`."""
    index = torch.as_tensor(positions, device=tensor.device)
    out = tensor.clone()
    out[torch.arange(tensor.shape[0], device=tensor.device), index] = values.to(out.dtype)
    return out


def _identity_featurize(x):
    return x, None


def _identity_inverse(f, err, x):
    return f


# name -> (featurize, inverse). The only entry today; `featurizer.py` will add
# `subspace` here and change nothing else.
FEATURIZERS = {"identity": (_identity_featurize, _identity_inverse)}


def _swap(f, operand):
    return operand


# name -> mechanism. `swap` is the absolute class: it replaces, it does not add.
MECHANISMS = {"swap": _swap}


def apply_write(tensor, positions, operand, mechanism="swap", featurizer="identity"):
    featurize, inverse = FEATURIZERS[featurizer]
    x = gather(tensor, positions)
    f, err = featurize(x)
    f = MECHANISMS[mechanism](f, operand)
    return scatter(tensor, positions, inverse(f, err, x))
