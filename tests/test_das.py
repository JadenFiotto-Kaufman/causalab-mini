"""DAS: the rotation, and the fit that turns it.

The first half is arithmetic — the defining property of a subspace swap, checked
on tensors with no model anywhere near them. The second half runs
`documents/das_cpu_reduction.json` end to end.
"""

import torch

from causalab_mini import featurizer, ops


def rotation(d=16, k=8, seed=0):
    return featurizer.Subspace(featurizer.start_weight(d, k, seed))


# --------------------------------------------------------------------- #
# the rotation is orthonormal by construction
# --------------------------------------------------------------------- #


def test_the_cayley_transform_is_orthonormal_for_any_parameter():
    """Not "after training", not "within tolerance of the initialization": for
    every value the optimizer could possibly produce, including a wild one."""
    for k in (1, 8, 16):
        for scale in (0.0, 1e-3, 1.0, 50.0):
            basis = featurizer.cayley(featurizer.start_weight(16, k, seed=3) * scale)
            assert basis.shape == (16, k)
            deviation = (basis.T @ basis - torch.eye(k)).abs().max()
            assert deviation < 1e-5, f"k={k} scale={scale}: |QᵀQ − I| = {deviation}"


def test_the_start_basis_is_the_seeds_and_nothing_elses():
    assert torch.equal(featurizer.start_weight(16, 8, 0), featurizer.start_weight(16, 8, 0))
    assert not torch.equal(featurizer.start_weight(16, 8, 0), featurizer.start_weight(16, 8, 1))


def test_a_subspace_featurizer_satisfies_the_protocol_unchanged():
    assert isinstance(rotation(), ops.Featurizer)


# --------------------------------------------------------------------- #
# the defining property: the complement is untouched
# --------------------------------------------------------------------- #


def test_a_subspace_swap_takes_the_subspace_and_keeps_the_complement():
    """The whole method in one assertion, on the write seam as it already is.

    After the swap the activation's coordinates *in* the rotation are the
    counterfactual's, and its component *orthogonal* to the rotation is still
    the base's. Neither half is bit-identical, because both are reconstructed by
    a projection rather than copied — see FINDINGS §5.1.
    """
    subspace = rotation(k=4)
    basis = subspace.basis
    base = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1))
    counterfactual = torch.randn(3, 16, generator=torch.Generator().manual_seed(2))

    out = ops.apply_write(base, (4, 4, 4), counterfactual @ basis, featurizer=subspace)
    written, before = ops.gather(out, (4, 4, 4)), ops.gather(base, (4, 4, 4))

    # in the subspace: the counterfactual's coordinates, not the base's.
    assert torch.allclose(written @ basis, counterfactual @ basis, atol=1e-5)
    assert not torch.allclose(written @ basis, before @ basis, atol=1e-3)
    # orthogonal to it: the base's component, still.
    complement = lambda x: x - (x @ basis) @ basis.T
    assert torch.allclose(complement(written), complement(before), atol=1e-5)
    # and every other position of the tensor is untouched, bit for bit.
    assert torch.equal(ops.gather(out, (0, 0, 0)), ops.gather(base, (0, 0, 0)))


def test_a_full_width_subspace_swap_is_a_full_swap():
    """k = d is the bridge between the two methods: the complement is empty, so
    `inverse` returns the operand and DAS's write *is* activation patching's."""
    subspace = rotation(k=16)
    base = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1))
    counterfactual = torch.randn(3, 16, generator=torch.Generator().manual_seed(2))

    rotated = ops.apply_write(
        base, (4, 4, 4), counterfactual @ subspace.basis, featurizer=subspace
    )
    plain = ops.apply_write(base, (4, 4, 4), counterfactual)
    assert torch.allclose(ops.gather(rotated, (4, 4, 4)), ops.gather(plain, (4, 4, 4)), atol=1e-5)


def test_the_read_and_the_write_are_one_parameter_set():
    """`featurize` and `inverse` are two methods over one tensor, so a gradient
    arriving through either reaches the same parameter — which is what "the same
    featurizer name at a read and at a write is one rotation" means."""
    subspace = rotation(k=4)
    subspace.weight.requires_grad_(True)
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(7))

    read, _ = subspace.featurize(x)
    read.sum().backward()
    assert subspace.weight.grad is not None
    from_the_read = subspace.weight.grad.clone()
    subspace.weight.grad = None

    subspace.inverse(torch.zeros(2, 4), None, x).sum().backward()
    assert subspace.weight.grad is not None
    assert not torch.equal(subspace.weight.grad, from_the_read)  # two paths, one tensor
