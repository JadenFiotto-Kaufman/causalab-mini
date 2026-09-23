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


def _on(parameter: torch.Tensor, x: Any) -> torch.Tensor:
    """`parameter`, where `x` is. A featurizer computes on the device of the
    activation it is handed — which it cannot know ahead of time: one GPU, a
    model spread over several by `device_map="auto"`, a server's. The
    parameter itself stays a single leaf wherever it was built, so the
    optimizer holds one tensor whatever the model's placement, and the
    gradient comes home through this `.to`. (A `(d, k)` copy per access; the
    Cayley solve then runs beside the activation, which is the part that
    would have been slow.)"""
    return parameter.to(x.device)


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
    # Scaled by 1/√d so the skew matrix it names has singular values near 1
    # at any width. Unscaled they grow like √d, and the Cayley transform
    # saturates: (I − A)⁻¹(I + A) → −I as A grows, its derivative falling off
    # like 1/σ², so at d = 2048 a unit-variance start is a rotation no
    # gradient can move (measured: loss 3.23 → 3.29 over 60 updates, held-out
    # IIA 0.00; with this scale 3.22 → 0.11 and IIA 1.00). At d = 16, where
    # this was first written, the difference is invisible. FINDINGS §19.
    return torch.randn(d, k, generator=generator, dtype=torch.float32) / d**0.5


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
        return x.to(self.weight.dtype) @ cayley(_on(self.weight, x)), None

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        """(f, 0, x) -> Q f + (x − Q Qᵀ x).

        The second term is the orthogonal complement of the **pre-write**
        activation, which is why `x` has to reach here: with `f` swapped for a
        counterfactual's features this is "take the subspace from there, keep
        everything else from here".
        """
        basis = cayley(_on(self.weight, x))
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

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    @property
    def basis(self) -> torch.Tensor:
        return self.weight

    def featurize(self, x: Any) -> tuple[Any, None]:
        return x.to(self.weight.dtype) @ _on(self.basis, x), None

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        basis = _on(self.basis, x)
        x = x.to(basis.dtype)
        return f @ basis.T + (x - (x @ basis) @ basis.T)


class Encoder:
    """A loaded encoder/decoder pair — a sparse autoencoder, or any fixed
    linear map into a feature space. The one featurizer with an **error
    term**, which is what `err` in the write seam has been for all along:

        featurize(x) = (f, x − decode(f))     f = act((x − b_dec) W_enc + b_enc)
        inverse(f′, err, x) = decode(f′) + err

    so a write changes what the dictionary explains and leaves what it does
    not explain exactly as it was — with `f′ = f` the activation comes back
    to an ulp, however bad the SAE's reconstruction is (measured: 1.49e-08
    in fp32, because `decode(f) + (x − decode(f))` rounds twice). Without the error
    term, every SAE intervention would also be "replace the activation with
    its reconstruction", and the two effects could not be told apart.

    The tensors are SAELens's names and orientations: `W_enc (d, k)`,
    `W_dec (k, d)`, optional `b_enc (k)` and `b_dec (d)`. `linear` is the
    same object with no nonlinearity, and a missing `W_dec` means tied
    weights, `W_encᵀ`. Loaded, never trained.
    """

    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor | None = None,
                 b_enc: torch.Tensor | None = None, b_dec: torch.Tensor | None = None,
                 activation: str = "relu") -> None:
        d, k = W_enc.shape
        self.W_enc = W_enc
        self.W_dec = W_enc.T.contiguous() if W_dec is None else W_dec
        self.b_enc = torch.zeros(k) if b_enc is None else b_enc
        self.b_dec = torch.zeros(d) if b_dec is None else b_dec
        self.activation = activation

    def decode(self, f: Any) -> Any:
        return f @ _on(self.W_dec, f) + _on(self.b_dec, f)

    def featurize(self, x: Any) -> tuple[Any, Any]:
        x = x.to(self.W_enc.dtype)
        f = (x - _on(self.b_dec, x)) @ _on(self.W_enc, x) + _on(self.b_enc, x)
        if self.activation == "relu":
            f = torch.relu(f)
        return f, x - self.decode(f)

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        return self.decode(f) + err


class Gate:
    """A learned binary mask over a site's units — the DBM featurizer.

        featurize(x) = (x, 0)
        inverse(f, 0, x) = m ⊙ f + (1 − m) ⊙ x

    so a swap through a gate takes the masked units from the counterfactual
    and leaves the rest alone: the same sentence as a subspace, with "units"
    for "directions". The features *are* the activation, which is why a gate
    needs no `k`.

    The mask has two readings of one parameter `θ`, and which one is in force
    is the one piece of mode in this library:

        training   m = σ(θ / T)     soft, so a gradient reaches θ
        otherwise  m = [θ > 0]      hard, so the score is of a real mask

    `training` is a plain attribute the fit loop sets around an update and
    clears around its eval pass; `temperature` is `T`, which the fit anneals
    toward zero so the soft mask the optimizer sees approaches the hard one
    the score uses. A gate nobody is fitting is always hard — including one
    loaded from a file, which is a mask and nothing else.
    """

    training = False

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.temperature = 1.0

    @property
    def mask(self) -> torch.Tensor:
        if self.training:
            return torch.sigmoid(self.weight / self.temperature)
        return (self.weight > 0).to(self.weight.dtype)

    def featurize(self, x: Any) -> tuple[Any, None]:
        return x.to(self.weight.dtype), None

    def inverse(self, f: Any, err: Any, x: Any) -> Any:
        mask = _on(self.mask, x)
        return mask * f + (1 - mask) * x.to(mask.dtype)


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
KINDS: dict[str, Any] = {
    "subspace": Subspace,
    "pca": Basis,
    "gate": Gate,
    "sae": Encoder,
    "linear": lambda **tensors: Encoder(**tensors, activation="none"),
}

#: The tensors a bundle of each kind holds: (required, optional).
TENSORS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "subspace": (("weight",), ()),
    "pca": (("weight",), ()),
    "gate": (("weight",), ()),
    "sae": (("W_enc", "W_dec"), ("b_enc", "b_dec")),
    "linear": (("W_enc",), ("W_dec", "b_enc", "b_dec")),
}


def start(kind: str, d: int, k: int, seed: int) -> torch.Tensor:
    """The parameter a featurizer nobody loaded starts from. A gate starts at
    θ = 0: every unit at σ(0) = ½ while training, and — because the hard mask
    is θ > 0 — every unit *off* until a fit says otherwise."""
    if kind == "gate":
        return torch.zeros(d, dtype=torch.float32)
    return start_weight(d, k, seed)
