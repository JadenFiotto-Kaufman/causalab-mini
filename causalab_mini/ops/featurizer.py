"""AGNOSTIC: the rotation DAS fits. Tensors in, tensors out.

A `subspace` featurizer is one orthonormal `(d, k)` basis `Q`:

    featurize(x) = (Qᵀx, 0)
    inverse(f, 0, x) = Q f + (x − Q Qᵀ x)

so a write replaces the activation's component **in the subspace** and leaves
the orthogonal complement alone. That sentence is what DAS is, and it is the
whole of this file's contribution to the write seam: `ops.apply_write` is
already `inverse(do(featurize(x)), err, x)`, and the complement it hands back
comes from `x`, the pre-write value, exactly as the error-term contract says.

`Q` is never stored. The stored parameter is `weight`, a `(d, k)` tensor, and
`Q` is the Cayley transform of the skew matrix that parameter defines, applied
to the first `k` columns of the identity — "the Cayley transform from the start
basis, rank `k`":

    A = W Eᵀ − E Wᵀ          skew by construction, rank at most 2k
    Q = (I − A)⁻¹ (I + A) E  orthonormal by construction

`(I − A)⁻¹(I + A)` is orthogonal whenever `A` is skew, and `E` has orthonormal
columns, so `QᵀQ = I` for **every** value of the parameter. That is the reason
to parametrize rather than to constrain: the optimizer is an ordinary AdamW over
an unconstrained `(d, k)` tensor, there is no retraction step, no penalty term
and no projection after the update, and the rotation cannot drift off the
manifold however long the fit runs.
"""

from __future__ import annotations

from typing import Any

import torch


def cayley(weight: torch.Tensor) -> torch.Tensor:
    """The `(d, k)` orthonormal basis a `(d, k)` parameter names."""
    d, k = weight.shape
    eye = torch.eye(d, dtype=weight.dtype, device=weight.device)
    start = eye[:, :k]
    skew = weight @ start.T - start @ weight.T
    return torch.linalg.solve(eye - skew, (eye + skew) @ start)


def start_weight(d: int, k: int, seed: int) -> torch.Tensor:
    """The parameter a fit starts from: a rank-`k` skew matrix drawn from
    `seed`, which makes an *untrained* subspace a reproducible random rank-`k`
    basis rather than the first `k` coordinate axes (which is what the zero
    parameter would give)."""
    if not 0 < k <= d:
        raise ValueError(f"k must be in 1..{d} for a {d}-wide activation, got {k}")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(d, k, generator=generator, dtype=torch.float32)


class Subspace:
    """One rotation, as a `Featurizer`.

    The parameter is held, the basis is not: `basis` recomputes the Cayley
    transform on every access, because the optimizer steps `weight` between
    accesses and a cached `Q` would be a stale one. `d` is 16 here and the
    transform is one `k`-column solve, so the honest thing is also the cheap
    one; on a real residual stream this is the place to cache per forward.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    @property
    def basis(self) -> torch.Tensor:
        return cayley(self.weight)

    def featurize(self, x: Any) -> tuple[Any, None]:
        """x -> (Qᵀx, 0). `err` is always zero for a subspace: nothing is
        thrown away that `inverse` could not recover from `x` itself."""
        return x.to(self.weight.dtype) @ self.basis, None

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        """(f, 0, x) -> Q f + (x − Q Qᵀ x).

        The second term is the orthogonal complement of the **pre-write**
        activation, which is why `x` has to reach here: with `f` swapped for a
        counterfactual's features this is "take the subspace from there, keep
        everything else from here".
        """
        basis = self.basis
        x = x.to(basis.dtype)
        return f @ basis.T + (x - (x @ basis) @ basis.T)


class Basis:
    """A fixed orthonormal `(d, k)` basis — loaded, never trained. The same
    `featurize`/`inverse` as a `Subspace`, with `Q` stored rather than
    parametrized: what a PCA of harvested activations gives you, and the
    untrained control a DAS fit is compared against.

    `weight` is the basis itself, so a `Weights` step can publish it and a
    save can stamp it exactly as it would a rotation's parameter.
    """

    def __init__(self, basis: torch.Tensor) -> None:
        self.weight = basis

    @property
    def basis(self) -> torch.Tensor:
        return self.weight

    def featurize(self, x: Any) -> tuple[Any, None]:
        return x.to(self.weight.dtype) @ self.basis, None

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        basis = self.basis
        x = x.to(basis.dtype)
        return f @ basis.T + (x - (x @ basis) @ basis.T)


def pca(rows: torch.Tensor, k: int) -> torch.Tensor:
    """The top-`k` principal directions of `rows`, `(n, d) -> (d, k)`,
    orthonormal by construction: the right singular vectors of the centered
    rows. A harvest reduced this way is the basis a `pca` featurizer loads."""
    flat = rows.reshape(-1, rows.shape[-1]).to(torch.float32)
    if not 0 < k <= min(flat.shape):
        raise ValueError(f"k={k} principal directions of {tuple(flat.shape)} rows is not a basis")
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    return vt[:k].T.contiguous()


#: The featurizer kinds a document may declare. Unlike `ops.FEATURIZERS` this is
#: a table of *constructors*, not of instances: a subspace carries a trained
#: parameter, so one exists per run and not one per process. Each takes the
#: one tensor it is made of — a Cayley parameter, a basis.
KINDS = {"subspace": Subspace, "pca": Basis}
