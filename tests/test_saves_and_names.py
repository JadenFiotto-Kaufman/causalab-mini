"""What a step saves, and what things may be called.

Found by exploratory testing (TESTING.md, batch 1–2): a save used to be
resolved against *every* intervention, a tensor written to a `.pt` path
wrote `[]`, generated ids collided between two forwards of one model, and an
unstamped bundle loaded anywhere because "check the keys the stamp has"
checks nothing on an empty stamp.
"""

import copy
import json
import pathlib

import pytest
import torch
from pydantic import ValidationError
from safetensors import safe_open
from safetensors.torch import save_file

from causalab_mini import plan
from causalab_mini.plan import sweep
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2 = REPO / "documents" / "v2"


def _doc(name):
    return sweep.points(json.loads((V2 / name).read_text()))[0][1]


# --------------------------------------------------------------------- #
# a save is its own step's
# --------------------------------------------------------------------- #


def test_a_save_uses_its_own_steps_metric_not_a_namesake(data_root, model_engine, tmp_path):
    """Two interventions both call a metric `score`. The table written for
    the step running B must carry B's unit and B's eligibility — it used to
    carry A's, shifting every value after an excluded row."""
    import shutil

    root = tmp_path / "data"
    shutil.copytree(data_root, root)
    path = root / "weekdays" / "train.json"
    table = json.loads(path.read_text())
    table[1]["cf_answer"] = None  # a hole in A's column, none in B's
    path.write_text(json.dumps(table))

    raw = _doc("patching.json")
    common = raw["interventions"].pop("patching")
    common["metrics"] = {}
    logits = {"kind": "token_prob", "of": "logits", "token_form": "space_prefixed"}
    raw["interventions"] = {
        "A": {**copy.deepcopy(common), "metrics": {"score": {**logits, "token": "cf_answer"}}},
        "B": {**copy.deepcopy(common), "metrics": {"score": {**logits, "kind": "token_logit", "token": "base_answer"}}},
    }
    raw["steps"] = {"sB": {"kind": "observe", "intervention": "B", "rows": raw["steps"]["score"]["rows"],
                           "saves": [{"value": "score", "file_path": "score.json"}]}}
    executed = model_engine.execute(plan.build_request(raw, root, model_engine))
    (save,) = executed.step("sB", plan.Observe).saves
    assert (save.unit, save.estimand_version, save.eligible) == ("logit", "token_logit/v1", ())

    executed.write(tmp_path / "out")
    rows = json.loads((tmp_path / "out" / "score.json").read_text())
    assert [row["value"] for row in rows] == pytest.approx(executed.result("score").tolist())
    assert all(row["produced_by"] for row in rows), "a table says which document produced it"


@pytest.mark.parametrize("path", ["rot.pt", "rot.json", "rot"])
def test_a_tensor_is_saved_as_safetensors_or_not_at_all(path, data_root, model_engine):
    """`rot.pt` used to write `[]` and exit 0: the fit's rotation, gone."""
    raw = _doc("das.json")
    raw["steps"]["weights"]["saves"][0]["file_path"] = path
    with pytest.raises(plan.PlanError, match="give it a .safetensors path"):
        plan.build_request(raw, data_root, model_engine)


def test_a_table_is_saved_as_json_or_not_at_all(data_root, model_engine):
    raw = _doc("patching.json")
    raw["steps"]["score"]["saves"][0]["file_path"] = "iia.safetensors"
    with pytest.raises(plan.PlanError, match="give it a .json path"):
        plan.build_request(raw, data_root, model_engine)


def test_a_fit_can_save_its_own_record(data_root, model_engine, tmp_path):
    """The validator has always listed `train/loss` and `train/eval` as what
    a fit produces; the compiler used to answer with a bare StopIteration."""
    raw = _doc("das.json")
    raw["steps"]["fit"]["saves"] = [{"value": "train/loss", "file_path": "loss.safetensors"}]
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    executed.write(tmp_path)
    with safe_open(str(tmp_path / "loss.safetensors"), "pt") as bundle:
        assert bundle.get_tensor("weight").ndim == 1


# --------------------------------------------------------------------- #
# generated ids
# --------------------------------------------------------------------- #


@pytest.fixture
def two_forwards():
    """`original` runs on the base and on the counterfactual."""
    raw = _doc("generate_probe.json")
    one = next(iter(raw["interventions"].values()))
    template = copy.deepcopy(next(iter(one["reads"].values())))
    one["reads"] = {
        "on_base": {**template, "model": "original", "input": "base", "pos": {"step": 0}},
        "on_cf": {**template, "model": "original", "input": "counterfactual", "pos": {"step": 0}},
    }
    one["writes"], one["models"], one["metrics"] = {}, {}, {}
    step = next(iter(raw["steps"].values()))
    step["saves"] = []
    step.pop("outputs", None)
    return raw


def test_two_forwards_of_one_model_publish_two_generations(two_forwards, data_root, model_engine):
    """Keyed by the model alone they collided, and whichever input sorted
    last won: saving `original.generated` returned the counterfactual's."""
    results = next(iter(model_engine.execute(plan.build_request(two_forwards, data_root, model_engine)).steps.values())).results
    assert {"original.base.generated", "original.counterfactual.generated"} <= set(results)
    assert "original.generated" not in results
    # (the tiny random model greedily emits the same tokens for every prompt,
    # so the two cannot be told apart by value here — only by existing)


def test_a_generation_that_no_forward_makes_cannot_be_saved(data_root, model_engine):
    """It used to validate, explain, run every forward, and then KeyError
    at write time."""
    raw = _doc("generate_probe.json")
    one = next(iter(raw["interventions"].values()))
    intervened = next(iter(one["models"]))
    for read in one["reads"].values():
        read["model"] = intervened  # nothing reads `original` any more
    one["metrics"] = {k: v for k, v in one["metrics"].items() if one["reads"].get(v["of"])}
    step = next(iter(raw["steps"].values()))
    step["saves"] = [{"value": "original.generated", "file_path": "ids.safetensors"}]
    with pytest.raises(ValidationError, match="which it does not produce"):
        Spec.model_validate(raw)


@pytest.mark.parametrize("name", ["patched.generated", "gate.mask", "train/loss"])
def test_names_the_run_produces_are_not_a_documents_to_take(name):
    """A metric called `patched.generated` was overwritten by the token ids."""
    raw = _doc("patching.json")
    metrics = raw["interventions"]["patching"]["metrics"]
    metrics[name] = copy.deepcopy(next(iter(metrics.values())))
    with pytest.raises(ValidationError, match="results the run produces itself"):
        Spec.model_validate(raw)


def test_an_intervened_model_may_not_be_called_original():
    raw = _doc("patching.json")
    one = raw["interventions"]["patching"]
    one["models"]["original"] = one["models"].pop("patched")
    with pytest.raises(ValidationError, match="'original' is the un-intervened model"):
        Spec.model_validate(raw)


def test_two_absolute_writes_at_one_place_are_refused():
    """The protocol format always refused this; here the second silently won."""
    raw = _doc("patching.json")
    one = raw["interventions"]["patching"]
    one["writes"]["again"] = {**one["writes"]["patch"], "operand": 0.0}
    one["models"]["patched"]["writes"] = ["patch", "again"]
    with pytest.raises(ValidationError, match="silently discard the first"):
        Spec.model_validate(raw)
    one["writes"]["again"].update(mechanism="add_scaled")  # an additive one composes
    Spec.model_validate(raw)


# --------------------------------------------------------------------- #
# the stamp
# --------------------------------------------------------------------- #


def test_a_harvested_basis_is_stamped_and_refused_at_another_layer(data_root, model_engine, tmp_path):
    """The documented harvest → control pair was the one path that wrote no
    header, so its load-time check was a no-op."""
    model_engine.execute(plan.build_request(_doc("pca_harvest.json"), data_root, model_engine)).write(tmp_path)
    (bundle,) = tmp_path.glob("*.safetensors")
    with safe_open(str(bundle), "pt") as f:
        assert f.metadata()["kind"] == "pca" and f.metadata()["layer"] == "0"

    control = _doc("pca_control.json")
    name = next(iter(control["featurizers"]))
    control["featurizers"][name]["file_path"] = str(bundle)
    plan.build_request(control, data_root, model_engine)  # the right layer loads
    site = next(s for s in control["sites"].values() if s.get("layers"))
    site["layers"] = [1]
    with pytest.raises(plan.PlanError, match="layer: bundle says '0', document says '1'"):
        plan.build_request(control, data_root, model_engine)


def test_an_unstamped_bundle_is_not_silently_trusted(data_root, model_engine, tmp_path):
    save_file({"weight": torch.eye(16)[:, :4].contiguous()}, str(tmp_path / "bare.safetensors"))
    control = _doc("pca_control.json")
    spec = control["featurizers"][next(iter(control["featurizers"]))]
    spec["file_path"] = str(tmp_path / "bare.safetensors")
    with pytest.raises(plan.PlanError, match="carries no identity stamp"):
        plan.build_request(control, data_root, model_engine)
    spec["trust_unstamped"] = True
    plan.build_request(control, data_root, model_engine)


def test_a_rotation_fitted_in_one_head_is_not_a_rotation_of_another(data_root, model_engine, tmp_path):
    raw = _doc("das.json")
    raw["sites"]["target"] = {"component": "attention_premix", "layers": [0], "heads": [1]}
    raw["featurizers"]["rot"]["k"] = 2
    raw["steps"]["fit"]["epochs"] = 1
    model_engine.execute(plan.build_request(raw, data_root, model_engine)).write(tmp_path)

    raw["featurizers"]["rot"]["file_path"] = str(tmp_path / "rot.safetensors")
    raw["steps"] = {"score": raw["steps"]["score"]}
    plan.build_request(raw, data_root, model_engine)
    raw["sites"]["target"]["heads"] = [2]
    with pytest.raises(plan.PlanError, match=r"heads: bundle says '\[1\]', document says '\[2\]'"):
        plan.build_request(raw, data_root, model_engine)


def test_features_are_bounded_by_the_sites_own_width(data_root, model_engine):
    """The bound was the whole component's width, so feature 7 of a 4-wide
    head compiled and died inside the trace with an IndexError."""
    raw = _doc("patching.json")
    raw["sites"]["target"] = {"component": "attention_premix", "layers": [0], "heads": [1]}
    raw["interventions"]["patching"]["writes"]["patch"]["features"] = [7]
    with pytest.raises(plan.PlanError, match="4-dimensional feature space"):
        plan.build_request(raw, data_root, model_engine)
