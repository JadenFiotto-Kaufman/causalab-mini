"""The write seam: `inverse(do(featurize(x)), err, x)`.

DAS is not built yet. What is checked here is that the seam it will be built in
has the shape it needs: a featurizer is an object with two methods, and the
pre-write activation and the error term reach `inverse`.
"""

import torch

from causalab_mini import ops


class FirstColumn:
    """A stand-in for a trained rotation: the feature space is column 0, the
    error term is the rest of the activation, and `inverse` puts them back."""

    def featurize(self, x):
        return x[..., :1], x[..., 1:]

    def inverse(self, f, err, x):
        return torch.cat([f, err], dim=-1)


def test_the_identity_featurizer_satisfies_the_protocol():
    assert isinstance(ops.Identity(), ops.Featurizer)
    assert isinstance(FirstColumn(), ops.Featurizer)


def test_a_featurizer_with_state_only_writes_its_own_feature_space():
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    last, first = ((2,), (2,)), ((0,), (0,))  # a unit window per row
    ops.FEATURIZERS["first_column"] = FirstColumn()
    try:
        out = ops.apply_write(tensor, last, torch.tensor([[[-1.0]], [[-2.0]]]), featurizer="first_column")
    finally:
        del ops.FEATURIZERS["first_column"]

    written = ops.gather(out, last)  # (rows, 1, width)
    # The feature space took the operand; the complement came from the
    # pre-write value, which is the whole point of threading `err` and `x`.
    assert written[..., 0].flatten().tolist() == [-1.0, -2.0]
    assert torch.equal(written[..., 1:], ops.gather(tensor, last)[..., 1:])
    # And every other position is untouched.
    assert torch.equal(ops.gather(out, first), ops.gather(tensor, first))


def test_the_identity_write_replaces_the_whole_activation():
    tensor = torch.zeros(2, 3, 4)
    operand = torch.ones(2, 1, 4)
    out = ops.apply_write(tensor, ((1,), (1,)), operand)
    assert torch.equal(ops.gather(out, ((1,), (1,))), operand)
    assert torch.equal(ops.gather(out, ((0,), (0,))), torch.zeros(2, 1, 4))
