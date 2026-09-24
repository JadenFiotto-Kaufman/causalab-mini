"""The `do` vocabulary: what a write does with its operand.

Four mechanisms behind one protocol — `(f, operand, **params) -> f` — and
three kinds of operand: a name, a number, or nothing. The write seam
`inverse(do(featurize(x)), err, x)` does not change; only the middle does.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError
from conftest import same_numbers

from causalab_mini import ops, plan
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.spec_v2 import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
ZERO = REPO / "tests" / "fixtures" / "v2_old" / "zero_ablation.json"
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
    clean = executed.result("clean.logit_diff")
    zeroed = executed.result("zeroed.logit_diff")
    assert not torch.equal(clean, zeroed)
    p = executed.result("zeroed.p_answer")
    assert ((p > 0) & (p < 1)).all()


def test_the_two_engines_agree_on_zero_ablation(zero_raw, data_root, model_engine):
    hooks = HooksEngine.load(Spec.model_validate(zero_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(zero_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(zero_raw, data_root, hooks))
    for name in ("logit_diff", "p_answer"):
        assert same_numbers(
            traced.result(f"zeroed.{name}"),
            hooked.result(f"zeroed.{name}"),
        ), name


def test_a_literal_operand_orders_no_forward(zero_raw, data_root, model_engine):
    """A write whose operand is a number depends on no read, so its model is
    scheduled like an un-intervened one: one forward, no source pass."""
    built = plan.build_request(zero_raw, data_root, model_engine)
    zeroed = built.step("zeroed.zeroed", plan.Forward)
    write = zeroed.taps[0].writes[0]
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
    raw = json.loads((REPO / "tests" / "fixtures" / "v2_old" / "das.json").read_text())
    raw["steps"]["fit"]["early_stop"] = {"metric": "ce", "mode": "min", "patience": 3}
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    curve = executed.step("fit", plan.Fit).results["train/eval"].squeeze(-1)
    assert curve.shape[0] >= 1


# --------------------------------------------------------------------- #
# clamp and renormalize
# --------------------------------------------------------------------- #

STEER = REPO / "tests" / "fixtures" / "v2_old" / "steer_renormalize.json"


def test_clamp_bounds_either_side_and_takes_no_operand():
    tensor = torch.arange(-6.0, 6.0).reshape(1, 4, 3)
    at = ((1, 2),)
    both = ops.apply_write(tensor, at, None, "clamp", params={"lo": -1.0, "hi": 1.0})
    assert both[0, 1:3].min() == -1 and both[0, 1:3].max() == 1
    assert torch.equal(both[0, [0, 3]], tensor[0, [0, 3]]), "only the window"
    top = ops.apply_write(tensor, at, None, "clamp", params={"hi": 0.0})
    assert top[0, 1:3].max() == 0 and top[0, 1].min() == tensor[0, 1].min()


def test_renormalize_restores_the_pre_write_norm_and_keeps_the_new_direction():
    original = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(0))
    steered = ops.apply_write(original, AT, torch.ones(2, 1, 8), "add_scaled", params={"scale": 5.0})
    restored = ops.apply_write(steered, AT, None, "renormalize", original=original)

    before, moved, after = (ops.gather(t, AT) for t in (original, steered, restored))
    assert torch.allclose(after.norm(dim=-1), before.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(moved.norm(dim=-1), before.norm(dim=-1))
    cosine = torch.nn.functional.cosine_similarity
    assert torch.allclose(cosine(after, moved, dim=-1), torch.ones(2, 1), atol=1e-6), "direction is the steered one"
    # alone it is the identity, which is why a document may not say so
    assert torch.allclose(ops.apply_write(original, AT, None, "renormalize", original=original), original, atol=1e-6)


def test_clamping_to_zero_is_zero_ablation(zero_raw, data_root, model_engine):
    """The new mechanism checked against an old one: `lo = hi = 0` is the
    literal-zero swap, bit for bit."""
    zeroed = model_engine.execute(plan.build_request(zero_raw, data_root, model_engine))
    write = zero_raw["interventions"]["zeroed"]["writes"]["zero"]
    write.update(mechanism="clamp", params={"lo": 0.0, "hi": 0.0})
    del write["operand"]
    clamped = model_engine.execute(plan.build_request(zero_raw, data_root, model_engine))
    for name in ("logit_diff", "p_answer"):
        assert torch.equal(
            zeroed.result(f"zeroed.{name}"),
            clamped.result(f"zeroed.{name}"),
        ), name


@pytest.mark.parametrize("engine_name", ["nnterp", "hooks"])
def test_steering_then_renormalizing_on_the_model(engine_name, data_root, model_engine):
    """The read at the site sees the model's writes, so the document can
    measure its own claim: the steered activation is longer than the
    original, the renormalized one is exactly as long, and the two models
    answer differently."""
    raw = json.loads(STEER.read_text())
    engine = model_engine if engine_name == "nnterp" else HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    result = engine.execute(plan.build_request(raw, data_root, engine)).result

    before, after, raw_after = (result(name).norm(dim=-1) for name in ("steer.before", "steer.after", "steer.after_raw"))
    assert torch.allclose(after, before, rtol=1e-5)
    assert (raw_after > before).all()
    assert not torch.equal(result("steer.logit_diff"), result("steer.logit_diff_raw"))


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda one: one["models"]["steered_renormed"].update(writes=["restore", "steer"]), "last among them"),
        (lambda one: one["models"]["steered_renormed"].update(writes=["restore"]), "it is the identity"),
        (lambda one: one["writes"]["restore"].update(operand=0.0), "takes no operand"),
        (lambda one: one["writes"]["steer"].update(mechanism="clamp", params={}), "takes no operand"),
        (lambda one: one["writes"]["restore"].update(mechanism="clamp"), "needs a bound"),
    ],
    ids=["renormalize first", "renormalize alone", "renormalize with an operand", "clamp with an operand", "clamp with no bound"],
)
def test_what_clamp_and_renormalize_may_not_say(edit, message):
    raw = json.loads(STEER.read_text())
    edit(raw["interventions"]["steer"])
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(raw)
