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
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.plan import document
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2_DAS = REPO / "documents" / "v2" / "das.json"
V2_PATCHING = REPO / "documents" / "v2" / "patching.json"


@pytest.fixture
def das_spec_raw():
    return json.loads(V2_DAS.read_text())


@pytest.fixture
def patching_spec_raw():
    return json.loads(V2_PATCHING.read_text())


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
    assert written == {"iia.json", "ce.json", "held_out_iia.json", "rot.safetensors"}

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
    patching_spec_raw["intervention"]["reads"]["v_cf"]["shuffle"] = {"seed": 1}
    with pytest.raises(ValidationError) as refusal:
        Spec.model_validate(patching_spec_raw)
    assert "intervention.reads.v_cf.shuffle" in str(refusal.value)
    assert "Extra inputs are not permitted" in str(refusal.value)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["intervention"]["reads"]["v_cf"].update(site="nope"), "undeclared site"),
        (lambda raw: raw["intervention"]["writes"]["patch"]["do"].update(swap="nope"), "must be a read name"),
        (lambda raw: raw["intervention"]["metrics"]["iia"].update(of="nope"), "must be a read name"),
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
    assert schema["required"] == ["model", "roles", "sites", "intervention", "steps"]
    assert "Fit" in schema["$defs"] and "Observe" in schema["$defs"]
