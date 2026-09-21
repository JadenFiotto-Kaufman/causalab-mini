"""The artifact round trip: a fit writes a bundle, a later document loads it.

A rotation is a grid of numbers; nothing in the numbers says which model or
layer it came from. The bundle's header does, and loading checks it key by
key — a rotation fitted at one layer loaded at another would run, and would
be nonsense. `pca` is the same machinery for a basis nobody trained.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from causalab_mini import ops, plan
from causalab_mini.ops import featurizer
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
DAS = REPO / "documents" / "v2" / "das.json"
APPLY = REPO / "documents" / "v2" / "das_apply.json"
HARVEST = REPO / "documents" / "v2" / "pca_harvest.json"
CONTROL = REPO / "documents" / "v2" / "pca_control.json"


def _fit_into(tmp_path, data_root, engine):
    """Run the fit and return (its executed plan, the bundle it wrote)."""
    executed = engine.execute(plan.build_request(json.loads(DAS.read_text()), data_root, engine))
    executed.write(tmp_path)
    return executed, tmp_path / "rot.safetensors"


def _apply_from(bundle):
    raw = json.loads(APPLY.read_text())
    raw["featurizers"]["rot"]["file_path"] = str(bundle)
    return raw


# --------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------- #


def test_a_loaded_rotation_scores_exactly_what_the_fit_scored_on_the_same_rows(
    tmp_path, data_root, model_engine
):
    """The strongest check there is: the fit's own held-out pass and a later
    document loading the bundle onto the same rows are the same rotation on
    the same data, and agree to the bit."""
    fitted, bundle = _fit_into(tmp_path, data_root, model_engine)
    applied = model_engine.execute(plan.build_request(_apply_from(bundle), data_root, model_engine))

    held_out = fitted.step("fit", plan.Fit).evaluation.results
    scored = applied.step("apply", plan.Observe).results
    assert torch.equal(held_out["iia"], scored["iia"]) and torch.equal(held_out["ce"], scored["ce"])
    # and the plan carried the weights as plain floats, so it is still data
    (spec,) = applied.step("featurizers", plan.Featurizers).specs
    assert spec.source == str(bundle) and len(spec.weight) == 16 and len(spec.weight[0]) == 8
    assert isinstance(spec.weight[0][0], float)


def test_the_committed_fixtures_load_and_run(data_root, model_engine, monkeypatch):
    """`documents/artifacts/` holds what das.json and pca_harvest.json wrote,
    so the shipped apply and control documents run as they are."""
    monkeypatch.chdir(REPO)
    for path in (APPLY, CONTROL):
        executed = model_engine.execute(plan.build_request(json.loads(path.read_text()), data_root, model_engine))
        assert executed.result("iia").shape == (2,), path.name


def test_pca_harvest_then_control_is_a_round_trip_too(tmp_path, data_root, model_engine):
    harvested = model_engine.execute(plan.build_request(json.loads(HARVEST.read_text()), data_root, model_engine))
    harvested.write(tmp_path)
    basis = harvested.result("basis")
    assert tuple(basis.shape) == (16, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-5), "orthonormal by construction"

    raw = json.loads(CONTROL.read_text())
    raw["featurizers"]["rot"]["file_path"] = str(tmp_path / "pca.safetensors")
    controlled = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    (spec,) = controlled.step("featurizers", plan.Featurizers).specs
    assert spec.kind == "pca" and spec.trained is False
    assert controlled.result("iia").shape == (2,)


def test_pca_is_the_right_singular_vectors_of_the_centered_rows():
    rows = torch.randn(50, 6, generator=torch.Generator().manual_seed(0))
    basis = featurizer.pca(rows, 2)
    centered = rows - rows.mean(0, keepdim=True)
    _, s, vt = torch.linalg.svd(centered, full_matrices=False)
    assert torch.allclose(basis.abs(), vt[:2].T.abs(), atol=1e-5)  # up to sign
    assert isinstance(ops.FEATURIZERS["identity"], ops.Featurizer)
    assert isinstance(featurizer.Basis(basis), ops.Featurizer)


# --------------------------------------------------------------------- #
# what a load refuses
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "edit, key",
    [
        (lambda raw: raw["sites"]["target"].update(layers=[1]), "layer"),
        (lambda raw: raw["model"].update(key="hf-internal-testing/tiny-random-gpt2"), "model_key"),
        (lambda raw: raw["model"].update(dtype="bf16"), "model_dtype"),
    ],
    ids=["another layer", "another model", "another dtype"],
)
def test_a_bundle_from_another_experiment_is_refused_naming_the_key(
    tmp_path, data_root, model_engine, edit, key
):
    _, bundle = _fit_into(tmp_path, data_root, model_engine)
    raw = _apply_from(bundle)
    edit(raw)
    with pytest.raises(plan.PlanError, match=f"is not this featurizer — .*{key}: bundle says"):
        plan.build_request(raw, data_root, model_engine)


def test_a_bundle_of_the_wrong_width_is_refused(tmp_path, data_root, model_engine):
    _, bundle = _fit_into(tmp_path, data_root, model_engine)
    raw = _apply_from(bundle)
    raw["featurizers"]["rot"]["k"] = 4
    with pytest.raises(plan.PlanError, match="k: bundle says '8', document says '4'"):
        plan.build_request(raw, data_root, model_engine)


def test_a_bundle_that_is_not_a_featurizer_is_refused(tmp_path, data_root, model_engine):
    path = tmp_path / "junk.safetensors"
    save_file({"weight": torch.zeros(16, 8), "extra": torch.zeros(1)}, str(path), metadata={})
    raw = _apply_from(path)
    with pytest.raises(plan.PlanError, match="not one `weight`"):
        plan.build_request(raw, data_root, model_engine)
    raw = _apply_from(tmp_path / "missing.safetensors")
    with pytest.raises(plan.PlanError, match="no bundle at"):
        plan.build_request(raw, data_root, model_engine)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda f: f.update(kind="pca", file_path=None), "loaded from a file"),
        (lambda f: f.update(seed=3), "a loaded featurizer has no seed"),
    ],
    ids=["pca without a file", "loaded with a seed"],
)
def test_what_a_loaded_featurizer_may_not_say(edit, message):
    raw = json.loads(APPLY.read_text())
    edit(raw["featurizers"]["rot"])
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(raw)


def test_a_pca_basis_cannot_be_trained_or_written():
    raw = json.loads(DAS.read_text())
    raw["featurizers"]["rot"] = {"kind": "pca", "k": 8, "file_path": "x.safetensors"}
    with pytest.raises(ValidationError, match="a pca basis, which is fixed by definition"):
        Spec.model_validate(raw)
    harvest = json.loads(HARVEST.read_text())
    harvest["interventions"]["ablate"] = {
        "reads": {"logits": {"site": "target", "pos": -1, "model": "m", "input": "base"}},
        "writes": {"w": {"site": "target", "pos": -1, "mechanism": "swap", "operand": {"ref": "basis"}}},
        "models": {"m": {"input": "base", "writes": ["w"]}},
    }
    harvest["steps"]["harvest"]["intervention"] = "harvest"  # two interventions now: each step says which
    harvest["steps"]["ablate"] = {"kind": "observe", "intervention": "ablate", "rows": {"base": "weekdays/train"}}
    with pytest.raises(plan.PlanError, match="a basis is loaded as a featurizer, not written"):
        plan.build_request(harvest, REPO / "documents" / "data", __import__("causalab_mini.engine", fromlist=["NNterpEngine"]).NNterpEngine.load(Spec.model_validate(json.loads(DAS.read_text())).model, dispatch=False))


def test_a_pca_of_too_few_vectors_is_refused_before_any_forward(data_root, model_engine):
    """Four rows at one position each are four vectors; centered, they span
    three directions. The compiler knows the count from the positions."""
    raw = json.loads(HARVEST.read_text())
    raw["interventions"]["harvest"]["reads"]["acts"]["pos"] = -1
    with pytest.raises(plan.PlanError, match="4 principal directions of 4 vector"):
        plan.build_request(raw, data_root, model_engine)
