"""The `do` vocabulary: what a write does with its operand.

Four mechanisms behind one protocol — `(f, operand, **params) -> f` — and
three kinds of operand: a name, a number, or nothing. The write seam
`inverse(do(featurize(x)), err, x)` does not change; only the middle does.
"""

import json
import pathlib

import pytest
import torch
from conftest import same_numbers
from pydantic import ValidationError

from causalab_mini import ops, plan
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
ZERO = REPO / "documents" / "v2" / "zero_ablation.json"
AT = ((2,), (2,))


@pytest.fixture
def zero_raw():
    return json.loads(ZERO.read_text())


# --------------------------------------------------------------------- #
# the mechanisms, as arithmetic
# --------------------------------------------------------------------- #


def test_add_scaled_adds_and_lerp_interpolates():
    tensor = torch.ones(2, 4, 3)
    direction = torch.full((2, 1, 3), 2.0)

    added = ops.apply_write(tensor, AT, direction, "add_scaled", params={"scale": 0.5})
    assert torch.equal(ops.gather(added, AT), torch.full((2, 1, 3), 2.0))  # 1 + 0.5·2

    half = ops.apply_write(tensor, AT, direction, "lerp", params={"t": 0.5})
    assert torch.equal(ops.gather(half, AT), torch.full((2, 1, 3), 1.5))
    # the endpoints: t=1 is a swap, t=0 is nothing
    assert torch.equal(ops.apply_write(tensor, AT, direction, "lerp", params={"t": 1.0}),
                       ops.apply_write(tensor, AT, direction, "swap"))
    assert torch.equal(ops.apply_write(tensor, AT, direction, "lerp", params={"t": 0.0}), tensor)


def test_gaussian_is_the_same_noise_for_the_same_seed_and_takes_no_operand():
    tensor = torch.zeros(2, 4, 3)
    once = ops.apply_write(tensor, AT, None, "gaussian", params={"seed": 7, "scale": 0.1})
    again = ops.apply_write(tensor, AT, None, "gaussian", params={"seed": 7, "scale": 0.1})
    other = ops.apply_write(tensor, AT, None, "gaussian", params={"seed": 8, "scale": 0.1})
    assert torch.equal(once, again) and not torch.equal(once, other)
    assert torch.equal(once[:, :2], tensor[:, :2]), "only the window moved"


def test_an_operand_is_a_name_a_number_or_nothing():
    values = {"v": torch.ones(2, 1, 3)}
    assert ops.resolve_operand(values, "v") is values["v"]
    assert ops.resolve_operand(values, 0.0).item() == 0.0
    assert ops.resolve_operand(values, None) is None


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


def test_zero_ablation_runs_and_moves_the_logits(zero_raw, data_root, model_engine):
    executed = model_engine.execute(plan.build_request(zero_raw, data_root, model_engine))
    clean = executed.step("clean", plan.Observe).results["logit_diff"]
    zeroed = executed.step("zeroed", plan.Observe).results["logit_diff"]
    assert not torch.equal(clean, zeroed)
    p = executed.step("zeroed", plan.Observe).results["p_answer"]
    assert ((p > 0) & (p < 1)).all()


def test_the_two_engines_agree_on_zero_ablation(zero_raw, data_root, model_engine):
    hooks = HooksEngine.load(Spec.model_validate(zero_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(zero_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(zero_raw, data_root, hooks))
    for name in ("logit_diff", "p_answer"):
        assert same_numbers(
            traced.step("zeroed", plan.Observe).results[name],
            hooked.step("zeroed", plan.Observe).results[name],
        ), name


def test_a_literal_operand_orders_no_forward(zero_raw, data_root, model_engine):
    """A write whose operand is a number depends on no read, so its model is
    scheduled like an un-intervened one: one forward, no source pass."""
    built = plan.build_request(zero_raw, data_root, model_engine)
    forwards = built.step("zeroed", plan.Observe).forwards
    assert [f.name for f in forwards] == ["zeroed"]
    write = forwards[0].taps[0].writes[0]
    assert write.operand == 0.0 and write.mechanism == "swap"


# --------------------------------------------------------------------- #
# what the document may not say
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda w: w.update(mechanism="lerp"), "needs params \\['t'\\]"),
        (lambda w: w.update(mechanism="gaussian", params={"seed": 1}), "takes no operand"),
        (lambda w: w.update(mechanism="gaussian", operand=None), "needs params \\['seed'\\]"),
        (lambda w: w.update(operand=None), "needs an operand"),
        (lambda w: w.update(params={"scale": 2.0}), "takes no params \\['scale'\\]"),
    ],
    ids=["lerp without t", "gaussian with an operand", "gaussian without a seed",
         "swap without an operand", "swap with a scale"],
)
def test_a_mechanism_and_its_numbers_must_agree(zero_raw, edit, message):
    edit(zero_raw["interventions"]["zeroed"]["writes"]["zero"])
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(zero_raw)


def test_a_fit_may_early_stop_on_a_minimized_metric(data_root, model_engine):
    """`mode: min` on the cross-entropy: the objective and the early stop
    agree on direction, which the protocol allows and mini refused."""
    raw = json.loads((REPO / "documents" / "v2" / "das.json").read_text())
    raw["steps"]["fit"]["early_stop"] = {"metric": "ce", "mode": "min", "patience": 3}
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    curve = executed.step("fit", plan.Fit).results["train/eval"].squeeze(-1)
    assert curve.shape[0] >= 1
