"""Three documents ported from causalab's own corpus, each for one reason.

`documents/minimal_cpu.json` and the DAS pair exercise one write, one model
and one fit. These three exercise what that leaves out: several writes in one
model, several intervened models feeding each other, and an experiment that
is three experiments with nothing trained at all.

They are retargets, not copies — the tiny CPU Llama has 2 layers where the
originals name layer 13 of an 8B — and each says so in its own
`header.description`.
"""

import copy
import json
import pathlib

import pytest
import torch

from causalab_mini import plan
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan import document

REPO = pathlib.Path(__file__).resolve().parents[1]
MULTI = REPO / "documents" / "multi_position_patch_cpu.json"
HYDRA = REPO / "documents" / "hydra_effect_cpu.json"
CONTROL = REPO / "documents" / "random_subspace_cpu.json"


@pytest.fixture
def multi_raw():
    return json.loads(MULTI.read_text())


@pytest.fixture
def hydra_raw():
    return json.loads(HYDRA.read_text())


@pytest.fixture
def control_raw():
    return json.loads(CONTROL.read_text())


# --------------------------------------------------------------------- #
# several writes, one model, disjoint positions
# --------------------------------------------------------------------- #


def test_three_writes_share_one_address_and_one_tap(multi_raw, data_root, model_engine):
    """Rule 8 is about *overlapping* positions, so three disjoint indices at
    one site are three legal absolute writes. They compile to one tap,
    because a tap is an address and these share one."""
    built = plan.build_request(multi_raw, data_root, model_engine)
    source, patched = built.step("observe", plan.Observe).forwards

    (tap,) = [one for one in patched.taps if one.writes]
    assert [write.name for write in tap.writes] == ["at_m4", "at_m3", "at_m2"]
    positions = [write.positions for write in tap.writes]
    assert len({tuple(one) for one in positions}) == 3, "the three writes must be disjoint"
    # and the three reads they take their operands from share the source tap
    assert [read.name for read in source.taps[0].reads] == ["v_m4", "v_m3", "v_m2"]


def test_the_joint_patch_is_not_any_of_its_single_positions(multi_raw, data_root, model_engine):
    """The reason the document exists: an effect carried jointly by several
    positions can read as nothing when each is patched alone."""
    joint = model_engine.execute(plan.build_request(multi_raw, data_root, model_engine))

    alone = {}
    for keep in ("at_m4", "at_m3", "at_m2"):
        one = copy.deepcopy(multi_raw)
        one["method"]["intervened_models"]["patched"]["writes"] = [keep]
        one["method"]["writes"] = {keep: one["method"]["writes"][keep]}
        operand = one["method"]["writes"][keep]["do"]["swap"]
        one["method"]["reads"] = {
            name: spec
            for name, spec in one["method"]["reads"].items()
            if name in (operand, "logits")
        }
        alone[keep] = model_engine.execute(
            plan.build_request(one, data_root, model_engine)
        ).result("logit_diff")

    for keep, scored in alone.items():
        assert not torch.equal(joint.result("logit_diff"), scored), keep


def test_the_multi_position_document_agrees_across_engines(multi_raw, data_root, model_engine):
    """The one document where the two engines are NOT bit-identical, and the
    tolerance is the measurement rather than a shrug.

    Measured 2026-09-19: `logit_diff` differs by 1.49e-08 on one of four rows
    — one ulp at fp32 — and only here. Every single-write document in this
    suite still asserts `torch.equal`, so this is specific to installing
    several writes at one address.

    What was ruled out, each by a probe: the operands (bit-identical), the
    activation the three writes compute (bit-identical), the tensor actually
    installed at the address including its strides and contiguity
    (bit-identical), the weights, the attention implementation (`sdpa` both
    sides), and the KV cache (forcing `use_cache=False` changes nothing). An
    un-intervened forward through both is bit-identical on this same batch.
    So it enters after the replacement is installed, somewhere in how nnsight
    continues the forward — which is nnsight's internals, not mini's.
    """
    hooks = HooksEngine.load(document.Document.from_json(multi_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(multi_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(multi_raw, data_root, hooks))

    a, b = traced.result("logit_diff"), hooked.result("logit_diff")
    assert torch.allclose(a, b, rtol=0, atol=1e-7)
    assert (a - b).abs().max() < 2e-8, (a - b).abs().max()


# --------------------------------------------------------------------- #
# five intervened models, and one feeding another
# --------------------------------------------------------------------- #


def test_a_read_inside_one_intervened_model_is_another_models_operand(
    hydra_raw, data_root, model_engine
):
    """The only cross-model operand chain in causalab's corpus: `a1_abl` is
    read inside `ablated`, and `with_inj1_abl` writes it. The schedule is
    what has to notice — `ablated` must run before `with_inj1_abl`, and
    nothing but the read graph says so."""
    built = plan.build_request(hydra_raw, data_root, model_engine)
    order = [forward.name for forward in built.step("observe", plan.Observe).forwards]

    # Six, not five: `original` runs twice, once per input role, because the
    # resample operand is read off the counterfactual and everything else off
    # the base. A (model, input) pair is the unit, not a model.
    assert len(order) == 6 and order.count("original") == 2
    assert order.index("ablated") < order.index("with_inj1_abl")
    assert order.index("original") < order.index("with_inj0_clean")


def test_the_hydra_document_runs_and_the_ablation_moves_the_measured_logit(
    hydra_raw, data_root, model_engine, tmp_path
):
    executed = model_engine.execute(plan.build_request(hydra_raw, data_root, model_engine))

    assert sorted(executed.all_results()) == [
        "de0_c", "de1_a", "de1_c", "te_abl", "te_clean"
    ]
    # resample-ablating the mixer moves the token's logit, and the direct
    # effect of the downstream layer moves with it
    assert not torch.equal(executed.result("te_clean"), executed.result("te_abl"))
    assert not torch.equal(executed.result("de1_c"), executed.result("de1_a"))

    written = executed.write(tmp_path)
    assert len(written) == 5
    rows = json.loads((tmp_path / "te_clean.json").read_text())
    assert {row["unit"] for row in rows} == {"logit"}
    assert {row["estimand_version"] for row in rows} == {"token_logit/v1"}


# --------------------------------------------------------------------- #
# three experiments, nothing trained
# --------------------------------------------------------------------- #


def test_the_control_is_three_untrained_rotations(control_raw, data_root, model_engine):
    """The matched-k control: a subspace is a complete object without a fit,
    and the swept coordinate is a featurizer parameter rather than a position
    or a layer."""
    built = plan.build_request(control_raw, data_root, model_engine)

    assert list(built.steps) == ["seed=0", "seed=1", "seed=2"]
    for label, point in built.steps.items():
        assert isinstance(point, plan.Plan)
        assert "fit" not in point.steps, "the control trains nothing"
        (spec,) = point.step("featurizers", plan.Featurizers).specs
        assert spec.trained is False
        assert spec.seed == int(label.removeprefix("seed=")), label


def test_the_three_draws_give_three_different_answers(control_raw, data_root, model_engine):
    """Which is the point of a control: the spread across draws is the null
    a trained rotation has to beat."""
    executed = model_engine.execute(
        plan.build_request(control_raw, data_root, model_engine)
    )
    scored = [point.result("logit_diff") for point in executed.steps.values()]
    assert not torch.equal(scored[0], scored[1])
    assert not torch.equal(scored[1], scored[2])


def test_an_authored_seed_is_reproducible_and_beats_the_fit_default(
    control_raw, data_root, model_engine
):
    """The seed is data, so the same document twice is the same rotation —
    and an authored seed overrides the document-level default of 0."""
    one = plan.build_request(control_raw, data_root, model_engine)
    two = plan.build_request(control_raw, data_root, model_engine)
    assert one == two

    control_raw["method"]["featurizers"]["rot"]["seed"] = 7
    single = plan.build_request(control_raw, data_root, model_engine)
    (spec,) = single.step("featurizers", plan.Featurizers).specs
    assert spec.seed == 7
