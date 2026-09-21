"""The plan-shaped authoring format.

`document.py` reads the protocol's JSON: one flat experiment, execution
order implied, one `save` list. `spec.py` reads a document shaped like the
plan instead — its `steps` are the plan's steps, and a save sits on the step
whose result it names.

Two claims are tested here. That the two formats compile to the *same* plan,
because they share every helper below the front end; and that the new shape
can say the thing the old one cannot — save a fit's held-out score, which in
the protocol's format has no name a document can write down.
"""

import json
import pathlib

import pytest
import torch
from conftest import same_numbers
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.plan import document
from causalab_mini.plan import spec as spec_module
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2_DAS = REPO / "documents" / "v2" / "das.json"
V2_PATCHING = REPO / "documents" / "v2" / "patching.json"
V2_MEAN = REPO / "documents" / "v2" / "mean_ablation.json"


@pytest.fixture
def das_spec_raw():
    return json.loads(V2_DAS.read_text())


@pytest.fixture
def patching_spec_raw():
    return json.loads(V2_PATCHING.read_text())


@pytest.fixture
def mean_raw():
    return json.loads(V2_MEAN.read_text())


# --------------------------------------------------------------------- #
# the shape
# --------------------------------------------------------------------- #


def test_the_documents_steps_are_the_plans_steps(das_spec_raw, data_root, model_engine):
    """The whole point of the format: what you read in the file is what runs,
    in that order, with `featurizers` the one step the compiler adds because
    declaring a parameter set is what builds it."""
    spec = Spec.model_validate(das_spec_raw)
    built = plan.build_spec(spec, data_root, model_engine)

    assert list(spec.steps) == ["fit", "score", "weights"]
    assert list(built.steps) == ["featurizers", "fit", "score", "weights"]


def test_a_save_sits_on_the_step_that_produces_it(das_spec_raw, data_root, model_engine):
    """No prefixes, no search, no ambiguity rule. Two steps both produce
    `iia` and each saves its own."""
    built = plan.build_spec(Spec.model_validate(das_spec_raw), data_root, model_engine)

    assert [s.file_path for s in built.step("score", plan.Observe).saves] == [
        "iia.json",
        "ce.json",
    ]
    fit = built.step("fit", plan.Fit)
    assert [s.file_path for s in fit.evaluation.saves] == ["held_out_iia.json"]
    assert [s.value for s in fit.evaluation.saves] == ["iia"]  # the same name, elsewhere


def test_the_held_out_score_reaches_disk(das_spec_raw, data_root, model_engine, tmp_path):
    """What the format was changed for. In the protocol's shape this number
    is computed and unreachable: a save naming `iia` resolves to the scored
    pass, and the eval pass has no name of its own."""
    executed = model_engine.execute(
        plan.build_spec(Spec.model_validate(das_spec_raw), data_root, model_engine)
    )
    written = {path.name for path in executed.write(tmp_path)}
    assert written == {
        "iia.json", "ce.json", "held_out_iia.json", "rot.safetensors",
        "document.json", "run.json",
    }

    on_disk = [row["value"] for row in json.loads((tmp_path / "held_out_iia.json").read_text())]
    in_memory = executed.step("fit", plan.Fit).evaluation.results["iia"].tolist()
    assert on_disk == pytest.approx(in_memory)

    scored = [row["value"] for row in json.loads((tmp_path / "iia.json").read_text())]
    assert scored != on_disk, "the two passes are different rows and different numbers"


# --------------------------------------------------------------------- #
# the two front ends meet
# --------------------------------------------------------------------- #


def test_both_formats_compile_to_the_same_run(das_spec_raw, das_raw, data_root, model_engine):
    """`documents/v2/das.json` and `documents/das_cpu_reduction.json` are the
    same experiment written two ways. They share every helper below the front
    end, so they cannot drift into different numbers — and this is what says
    so."""
    from_protocol = model_engine.execute(plan.build_request(das_raw, data_root, model_engine))
    from_spec = model_engine.execute(
        plan.build_spec(Spec.model_validate(das_spec_raw), data_root, model_engine)
    )

    assert torch.equal(
        from_protocol.step("observe", plan.Observe).results["iia"],
        from_spec.step("score", plan.Observe).results["iia"],
    )
    assert torch.equal(from_protocol.result("rot"), from_spec.result("rot"))


def test_the_patching_document_matches_its_protocol_twin(
    patching_spec_raw, minimal_raw, data_root, model_engine
):
    from_protocol = model_engine.execute(plan.build_request(minimal_raw, data_root, model_engine))
    from_spec = model_engine.execute(
        plan.build_spec(Spec.model_validate(patching_spec_raw), data_root, model_engine)
    )
    for name in ("iia", "logit_diff"):
        assert torch.equal(from_protocol.result(name), from_spec.result(name)), name


# --------------------------------------------------------------------- #
# what pydantic buys
# --------------------------------------------------------------------- #


def test_an_unknown_key_anywhere_is_refused_with_its_path(patching_spec_raw):
    """`extra="forbid"` is the catch-all `document.py` needed four silent
    bugs to learn it wanted, and the error says where."""
    patching_spec_raw["interventions"]["patching"]["reads"]["v_cf"]["shuffle"] = {"seed": 1}
    with pytest.raises(ValidationError) as refusal:
        Spec.model_validate(patching_spec_raw)
    assert "interventions.patching.reads.v_cf.shuffle" in str(refusal.value)
    assert "Extra inputs are not permitted" in str(refusal.value)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["interventions"]["patching"]["reads"]["v_cf"].update(site="nope"), "undeclared site"),
        (lambda raw: raw["interventions"]["patching"]["writes"]["patch"].update(operand="nope"), "is not a read of this intervention"),
        (lambda raw: raw["interventions"]["patching"]["metrics"]["iia"].update(of="nope"), "must be a read name"),
        (lambda raw: raw["steps"]["score"]["saves"].append({"value": "nope", "file_path": "x.json"}), "does not produce"),
        (lambda raw: raw["steps"]["score"]["rows"].pop("counterfactual"), "no rows for role"),
    ],
    ids=["undeclared site", "operand is not a read", "metric of nothing",
         "saving what it does not produce", "a step with no rows"],
)
def test_the_cross_checks_refuse_by_name(patching_spec_raw, edit, message):
    edit(patching_spec_raw)
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(patching_spec_raw)


def test_the_format_has_a_machine_readable_schema():
    """Which the protocol does not — 372 KB of authoritative prose and a
    Python implementation (NOTES §1). Here the schema is the model."""
    schema = Spec.model_json_schema()
    assert schema["required"] == ["model", "roles", "sites", "interventions", "steps"]
    assert "Fit" in schema["$defs"] and "Observe" in schema["$defs"]


# --------------------------------------------------------------------- #
# order, sweeps, and the write's two fields
# --------------------------------------------------------------------- #


def test_a_step_may_not_use_a_featurizer_before_it_is_trained(das_spec_raw):
    """The silent failure this closes: `score` before `fit` scored the
    untrained rotation and nothing complained. Now it is refused with the
    fix in the message."""
    das_spec_raw["steps"] = {k: das_spec_raw["steps"][k] for k in ("score", "fit", "weights")}
    with pytest.raises(ValidationError, match="uses featurizer 'rot' before step 'fit' trains it"):
        Spec.model_validate(das_spec_raw)


def test_weights_may_not_be_published_before_they_are_trained(das_spec_raw):
    das_spec_raw["steps"] = {k: das_spec_raw["steps"][k] for k in ("weights", "fit", "score")}
    with pytest.raises(ValidationError, match="step 'weights' uses featurizer 'rot' before"):
        Spec.model_validate(das_spec_raw)


def test_an_untrained_rotation_may_be_scored_in_any_order(das_spec_raw):
    """The rule is about training, not about featurizers: with no fit, the
    rotation is the seeded random basis throughout and any order is fine.
    (Scoring an untrained *and* a trained rotation in one document needs
    two interventions — REVIEW §2.C — which is why the refusal above points
    at a second featurizer rather than at reordering.)"""
    del das_spec_raw["steps"]["fit"], das_spec_raw["steps"]["weights"]
    Spec.model_validate(das_spec_raw)


def test_a_declared_featurizer_nothing_uses_is_refused(das_spec_raw):
    das_spec_raw["featurizers"]["spare"] = {"kind": "subspace", "k": 4, "parametrization": "cayley"}
    with pytest.raises(ValidationError, match="featurizer 'spare' is used at \\[\\]"):
        Spec.model_validate(das_spec_raw)


def test_a_wrapper_left_in_a_document_is_refused_with_the_fix(das_spec_raw):
    das_spec_raw["featurizers"]["rot"]["seed"] = {"sweep": [0, 1]}
    with pytest.raises(ValidationError, match="lowered before a document is validated"):
        Spec.model_validate(das_spec_raw)


def test_a_swept_v2_document_is_one_plan_per_point(das_spec_raw, data_root, model_engine, tmp_path):
    """The lowering works on raw JSON, so the plan-shaped format got sweeps
    for free — and a swept featurizer seed is the random-subspace control in
    this format too."""
    das_spec_raw["featurizers"]["rot"]["seed"] = {"sweep": [0, 1, 2]}
    del das_spec_raw["steps"]["fit"], das_spec_raw["steps"]["weights"]
    root = plan.build_request(das_spec_raw, data_root, model_engine)

    assert list(root.steps) == ["seed=0", "seed=1", "seed=2"]
    for label in root.steps:
        (spec,) = root.step(label, plan.Plan).step("featurizers", plan.Featurizers).specs
        assert spec.seed == int(label.removeprefix("seed=")) and spec.trained is False

    executed = model_engine.execute(root)
    scored = [point.result("iia") for point in executed.steps.values()]
    assert not torch.equal(scored[0], scored[1])
    written = {str(p.relative_to(tmp_path)) for p in executed.write(tmp_path)}
    assert "seed=1/iia.json" in written and "seed=1/document.json" in written


def test_a_write_is_two_fields_a_schema_can_enumerate():
    """`{"swap": "v_cf"}` used the mechanism as a key, which JSON Schema
    cannot enumerate. `mechanism` is a literal now, and the schema says so."""
    schema = Spec.model_json_schema()
    write = schema["$defs"]["Write"]
    # `operand` is optional now: `gaussian` draws its own and takes none
    assert write["required"] == ["site", "pos", "mechanism"]
    from causalab_mini.ops import intervene

    # the schema's list and the table of functions are the same list
    assert write["properties"]["mechanism"]["enum"] == list(intervene.MECHANISMS)


# --------------------------------------------------------------------- #
# several interventions, and what one step hands the next
# --------------------------------------------------------------------- #


def test_a_step_names_its_intervention_or_there_is_only_one(das_spec_raw, mean_raw):
    """One intervention: steps need not say. Several: they must, and an
    unknown name is refused."""
    assert Spec.model_validate(das_spec_raw).intervention_of(Spec.model_validate(das_spec_raw).steps["score"])
    unnamed = json.loads(json.dumps(mean_raw))
    del unnamed["steps"]["clean"]["intervention"]
    with pytest.raises(ValidationError, match="must name its intervention when the document declares 3"):
        Spec.model_validate(unnamed)
    wrong = json.loads(json.dumps(mean_raw))
    wrong["steps"]["clean"]["intervention"] = "nope"
    with pytest.raises(ValidationError, match="undeclared intervention 'nope'"):
        Spec.model_validate(wrong)


def test_mean_ablation_is_three_steps_and_the_mean_never_needs_a_file(
    mean_raw, data_root, model_engine, tmp_path
):
    """The document the protocol's format cannot write. `harvest` publishes a
    mean; `ablated` swaps it in. The mean crosses the steps through the walk's
    state — never a file, never the plan — and reaches disk only because a
    save on `harvest` asks."""
    executed = model_engine.execute(plan.build_request(mean_raw, data_root, model_engine))

    harvest = executed.step("harvest", plan.Observe)
    assert [o.name for o in harvest.outputs] == ["mean"]
    assert tuple(harvest.results["mean"].shape) == (1, 16)  # rows averaged away, the window kept

    clean = executed.step("clean", plan.Observe).results["logit_diff"]
    ablated = executed.step("ablated", plan.Observe).results["logit_diff"]
    assert not torch.equal(clean, ablated), "the swap landed"

    written = {p.name for p in executed.write(tmp_path)}
    assert {"mean.safetensors", "clean.json", "ablated.json"} <= written


def test_the_two_engines_agree_on_mean_ablation(mean_raw, data_root, model_engine):
    """A published output is seeded into `values` before the forward, on both
    engines, by the shared walk — so the engine cannot tell an operand that
    came from an earlier step from one read in this pass."""
    from causalab_mini.engine.engines.hooks import HooksEngine

    hooks = HooksEngine.load(Spec.model_validate(mean_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(mean_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(mean_raw, data_root, hooks))
    assert same_numbers(traced.result("mean"), hooked.result("mean"))
    assert same_numbers(
        traced.step("ablated", plan.Observe).results["logit_diff"],
        hooked.step("ablated", plan.Observe).results["logit_diff"],
    )


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["steps"].update({k: raw["steps"].pop(k) for k in ("harvest",)}),
         "references 'mean', which no earlier step outputs"),
        (lambda raw: raw["interventions"]["ablated"]["writes"]["ablate"].update(operand={"ref": "nope"}),
         "references 'nope', which no earlier step outputs"),
        (lambda raw: raw["steps"]["harvest"]["outputs"].update(logits={"read": "acts"}),
         "shares its name with a read"),
        (lambda raw: raw["steps"]["harvest"]["outputs"].update(mean={"read": "nope"}),
         "keeps read 'nope', which its intervention does not have"),
        (lambda raw: raw["steps"]["harvest"]["saves"].__setitem__(0, {"value": "acts", "file_path": "x.json"}),
         "saves 'acts', which it does not produce"),
    ],
    ids=["consumer before producer", "unknown output", "output named like a read",
         "output of a read it lacks", "saving a read that is not an output"],
)
def test_outputs_and_references_are_checked(mean_raw, edit, message):
    edit(mean_raw)
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(mean_raw)


def test_an_unreduced_output_must_match_the_rows_it_is_swapped_over(mean_raw, data_root, model_engine):
    """`(rows, width)` swapped over a different number of rows is a shape error
    the block would find; the compiler knows both counts and says so first."""
    mean_raw["steps"]["harvest"]["outputs"]["mean"] = {"read": "acts"}  # unreduced
    mean_raw["steps"]["harvest"]["saves"] = []
    mean_raw["steps"]["ablated"]["rows"] = {"base": "weekdays/data#test"}  # 2 rows, not 4
    with pytest.raises(plan.PlanError, match="has 4 rows, over 2 rows"):
        plan.build_request(mean_raw, data_root, model_engine)


def test_an_output_saves_as_a_tensor_not_a_table(mean_raw, data_root, model_engine):
    mean_raw["steps"]["harvest"]["saves"] = [{"value": "mean", "file_path": "mean.json"}]
    with pytest.raises(plan.PlanError, match="a tensor, not a table"):
        plan.build_request(mean_raw, data_root, model_engine)


def test_a_reads_shorthand_keeps_it_unreduced(mean_raw):
    mean_raw["steps"]["harvest"]["outputs"] = {"acts_kept": "acts"}
    mean_raw["steps"]["harvest"]["saves"] = []
    mean_raw["interventions"]["ablated"]["writes"]["ablate"]["operand"] = {"ref": "acts_kept"}
    spec = Spec.model_validate(mean_raw)
    harvest = spec.steps["harvest"]
    assert isinstance(harvest, spec_module.Observe)
    assert harvest.outputs["acts_kept"].reduce == "none"
