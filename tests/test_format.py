"""The steps-first document format.

A document's `steps` are what runs, in order; a step's name is how
everything after it reaches what it produced (`patched.logits`, `iia`,
`fit.rot`); `saves` names the references that go to disk. These tests hold
the format to what it says: that every reference resolves where it is
written or is refused with the fix, that each step compiles to one step of
the plan, that what reaches the plan is what the document wrote — and that
the protocol's format and this one, sharing every helper below their front
ends, give the same numbers.
"""

import copy
import json
import pathlib
import re

import pytest
import torch
from conftest import same_numbers
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2 = REPO / "documents" / "v2"


def _load(name):
    return json.loads((V2 / name).read_text())


@pytest.fixture
def patching():
    return _load("patching.json")


@pytest.fixture
def das():
    return _load("das.json")


@pytest.fixture
def mean():
    return _load("mean_ablation.json")


def _refused(raw, match):
    with pytest.raises(ValidationError, match=match):
        Spec.model_validate(raw)


def _compile(raw, data_root, engine):
    return plan.build_request(raw, data_root, engine)


# --------------------------------------------------------------------- #
# the shape
# --------------------------------------------------------------------- #


def test_a_documents_steps_are_the_plans_steps(patching, data_root, model_engine):
    """Two forwards and the two metrics of the second: four steps, named as
    the document names them, and every op by its reference."""
    built = _compile(patching, data_root, model_engine)
    assert list(built.steps) == ["counterfactual", "patched", "iia", "logit_diff"]
    (tap,) = [tap for tap in built.step("patched", plan.Forward).taps if tap.writes]
    assert tap.writes[0].name == "patched.patch" and tap.writes[0].operand == "counterfactual.v_cf"
    assert built.step("iia", plan.Metric).of == "patched.logits"


def test_a_fits_plan_is_its_steps_with_the_parameters_it_trained_after_it(das, data_root, model_engine):
    """What you read in the file is what runs, in that order. The compiler
    adds `featurizers`, because declaring a parameter set is what builds it,
    and `fit.weights`, because a save names `fit.rot`. Each update of the fit
    is its body's steps, named as the body names them."""
    built = _compile(das, data_root, model_engine)
    assert [name for name, _ in Spec.model_validate(das).steps.items()] == ["fit", "counterfactual", "patched", "iia", "ce"]
    assert list(built.steps) == ["featurizers", "fit", "fit.weights", "counterfactual", "patched", "iia", "ce"]
    fit = built.step("fit", plan.Fit)
    assert fit.objective == ((1.0, "ce"),) and fit.early_stop == "iia" and fit.params == ("rot",)
    update = fit.epochs[0][0]
    assert list(update.steps) == ["counterfactual", "patched", "iia", "ce"]
    (write,) = [op for tap in update.step("patched", plan.Forward).taps for op in tap.writes]
    assert (write.operand, write.featurizer) == ("counterfactual.v_cf", "rot")


def test_a_save_lands_on_the_step_that_produces_its_value(das, data_root, model_engine):
    """No prefixes to invent and no search: `iia` is the scoring's and
    `fit.iia` the held-out run's, and each file sits on its own step."""
    built = _compile(das, data_root, model_engine)
    assert [(one.value, one.file_path) for name in ("iia", "ce") for one in built.steps[name].saves] == [
        ("iia", "iia.json"), ("ce", "ce.json")
    ]
    evaluation = built.step("fit", plan.Fit).evaluation
    assert [(one.value, one.file_path) for one in evaluation.steps["iia"].saves] == [("iia", "held_out_iia.json")]
    assert [(one.value, one.file_path) for one in built.step("fit.weights", plan.Weights).saves] == [("rot", "rot.safetensors")]


def test_the_held_out_score_and_the_training_record_reach_disk(das, data_root, model_engine, tmp_path):
    """The held-out score is a value of the fit's body on its held-out pass,
    `fit.iia`; the fit itself is its training record, `fit`, one bundle."""
    das["steps"]["saves"]["fit"] = "fit.safetensors"
    executed = model_engine.execute(_compile(das, data_root, model_engine))
    written = {path.name for path in executed.write(tmp_path)}
    assert written == {
        "iia.json", "ce.json", "held_out_iia.json", "rot.safetensors", "fit.safetensors",
        "document.json", "run.json",
    }
    held_out = json.loads((tmp_path / "held_out_iia.json").read_text())
    assert [row["value"] for row in held_out] == pytest.approx(
        executed.step("fit", plan.Fit).evaluation.result("iia").tolist()
    )
    assert {row["metric"] for row in held_out} == {"iia"}
    scored = [row["value"] for row in json.loads((tmp_path / "iia.json").read_text())]
    assert scored != [row["value"] for row in held_out], "two passes, other rows, other numbers"

    from safetensors import safe_open

    with safe_open(str(tmp_path / "fit.safetensors"), "pt") as record:
        assert set(record.keys()) == {"loss", "eval"}


def test_a_mean_is_a_step_and_its_value_an_operand(mean, data_root, model_engine, tmp_path):
    """`mean` reduces a read of an earlier step, and a later step swaps it in
    by its name. It crosses between them through the walk's state — never a
    file — and reaches disk only because a save names it."""
    built = _compile(mean, data_root, model_engine)
    assert list(built.steps) == ["harvest", "mean", "clean", "clean_ld", "ablated", "ablated_ld"]
    reduced = built.step("mean", plan.Reduce)
    assert (reduced.of, reduced.reduce) == ("harvest.acts", "mean")

    executed = model_engine.execute(built)
    assert tuple(executed.result("mean").shape) == (1, 16)  # rows averaged away, the window kept
    assert not torch.equal(executed.result("clean_ld"), executed.result("ablated_ld")), "the swap landed"
    written = {path.name for path in executed.write(tmp_path)}
    assert written == {"mean.safetensors", "clean.json", "ablated.json", "document.json", "run.json"}
    assert {row["metric"] for row in json.loads((tmp_path / "ablated.json").read_text())} == {"ablated_ld"}


def test_the_two_engines_agree_on_mean_ablation(mean, data_root, model_engine):
    """A value an earlier step produced is handed to the call that takes it,
    on both engines, by the shared walk — so an engine cannot tell a mean
    from any other operand."""
    from causalab_mini.engine.engines.hooks import HooksEngine

    hooks = HooksEngine.load(Spec.model_validate(mean).model, device_map="cpu")
    traced = model_engine.execute(_compile(mean, data_root, model_engine))
    hooked = hooks.execute(_compile(mean, data_root, hooks))
    assert same_numbers(traced.result("mean"), hooked.result("mean"))
    assert same_numbers(traced.result("ablated_ld"), hooked.result("ablated_ld"))


def test_a_read_comes_home_when_a_save_names_it(mean, data_root, model_engine):
    """Every read is there for the steps after it; only one a save names is
    brought home, because on a real vocabulary one logits read over a
    thousand rows is half a gigabyte."""
    assert _compile(mean, data_root, model_engine).step("clean", plan.Forward).keep == ()
    mean["steps"]["saves"]["clean.logits"] = "clean_logits.safetensors"
    built = _compile(mean, data_root, model_engine)
    assert built.step("clean", plan.Forward).keep == ("clean.logits",)
    assert model_engine.execute(built).result("clean.logits").shape == (4, 1, 32000)


def test_an_unreduced_read_is_swapped_over_as_many_rows(mean, data_root, model_engine):
    """`(rows, width)` swapped over a different number of rows is a shape
    error the block would find; the compiler knows both counts and says so
    first."""
    mean["data"]["held_out"] = {"path": "weekdays/data#test"}  # 2 rows, not 4
    mean["steps"]["ablated"]["data"] = "held_out"
    mean["steps"]["ablated"]["interventions"]["writes"]["ablate"]["operand"] = "harvest.acts"
    mean["steps"]["ablated_ld"].update(a="held_out.cf_answer", b="held_out.base_answer")
    with pytest.raises(plan.PlanError, match="has 4 rows, over 2 rows"):
        _compile(mean, data_root, model_engine)


def test_a_site_declared_or_written_in_place_is_the_same_place(patching, data_root, model_engine):
    """A site written in place is named by its own spelling, and resolves to
    the same address a declared one does."""
    declared = copy.deepcopy(patching)
    declared["sites"]["head"] = {"component": "lm_head"}
    declared["steps"]["patched"]["reads"]["logits"]["site"] = "head"
    one, other = (model_engine.execute(_compile(raw, data_root, model_engine)).result("iia") for raw in (patching, declared))
    assert torch.equal(one, other)


# --------------------------------------------------------------------- #
# the two formats meet
# --------------------------------------------------------------------- #


def test_both_formats_compile_to_the_same_run(das, das_raw, data_root, model_engine):
    """`documents/v2/das.json` and `documents/das_cpu_reduction.json` are the
    same experiment written two ways. They share every helper below the front
    end, so they cannot drift into different numbers — and this is what says
    so."""
    from_protocol = model_engine.execute(_compile(das_raw, data_root, model_engine))
    from_spec = model_engine.execute(_compile(das, data_root, model_engine))
    assert torch.equal(from_protocol.result("iia"), from_spec.result("iia"))
    assert torch.equal(from_protocol.result("rot"), from_spec.result("rot"))


def test_the_patching_document_matches_its_protocol_twin(patching, minimal_raw, data_root, model_engine):
    from_protocol = model_engine.execute(_compile(minimal_raw, data_root, model_engine))
    from_spec = model_engine.execute(_compile(patching, data_root, model_engine))
    for name in ("iia", "logit_diff"):
        assert torch.equal(from_protocol.result(name), from_spec.result(name)), name


# --------------------------------------------------------------------- #
# what pydantic buys
# --------------------------------------------------------------------- #


def test_an_unknown_key_anywhere_is_refused_with_its_path(patching):
    """`extra="forbid"` is the catch-all `document.py` needed four silent
    bugs to learn it wanted, and the error says where."""
    patching["steps"]["counterfactual"]["reads"]["v_cf"]["shuffle"] = {"seed": 1}
    with pytest.raises(ValidationError) as refusal:
        Spec.model_validate(patching)
    assert "steps.counterfactual.forward.reads.v_cf.shuffle" in str(refusal.value)
    assert "Extra inputs are not permitted" in str(refusal.value)


def test_the_format_has_a_machine_readable_schema():
    """Which the protocol does not — 372 KB of authoritative prose and a
    Python implementation (NOTES §1). Here the schema is the model, and a
    step is any of its five kinds."""
    schema = Spec.model_json_schema()
    assert schema["required"] == ["model", "steps"]
    steps = schema["$defs"]["Steps"]
    assert set(steps["additionalProperties"]["discriminator"]["mapping"]) == {"forward", "generate", "metric", "reduce", "fit"}
    # and it takes a position the way a document writes it, `-1` included
    assert {"type": "integer"} in schema["$defs"]["Read"]["properties"]["pos"]["anyOf"]


def test_a_write_is_two_fields_a_schema_can_enumerate():
    """`{"swap": "v_cf"}` used the mechanism as a key, which JSON Schema
    cannot enumerate. `mechanism` is a literal, and the schema says so."""
    from causalab_mini.ops import intervene

    write = Spec.model_json_schema()["$defs"]["Write"]
    # `operand` is optional: `gaussian` draws its own and takes none
    assert write["required"] == ["site", "pos", "mechanism"]
    # the schema's list and the table of functions are the same list
    assert write["properties"]["mechanism"]["enum"] == list(intervene.MECHANISMS)


@pytest.mark.parametrize(
    "component, layers, message",
    [
        ("lm_head", [0], "lm_head takes no layers"),
        ("block_output", None, "block_output is addressed at one layer"),
    ],
    ids=["a whole-model site with a layer", "a per-layer site without one"],
)
def test_a_site_is_checked_against_its_component(patching, component, layers, message):
    """This format asks the same question the protocol's does, of the same
    table — whether a component is one place or one per layer is `address`'s
    to answer, and neither format keeps a list of its own."""
    site = {"component": component}
    if layers is not None:
        site["layers"] = layers
    patching["sites"]["probe"] = site
    _refused(patching, message)


def test_a_wrapper_left_in_a_document_is_refused_with_the_fix(das):
    das["featurizers"]["rot"]["seed"] = {"sweep": [0, 1]}
    _refused(das, "lowered before a document is validated")


def test_a_name_is_one_step_and_cannot_hold_a_dot(patching):
    patching["steps"]["a.b"] = patching["steps"].pop("iia")
    _refused(patching, "a name is a letter")
    patching["steps"].pop("a.b")
    patching["steps"]["featurizers"] = patching["steps"].pop("logit_diff")
    _refused(patching, "the name is reserved")


def test_undeclared_sites_datasets_and_interventions_are_refused(patching):
    one = copy.deepcopy(patching)
    one["steps"]["patched"]["reads"]["logits"]["site"] = "nowhere"
    _refused(one, "undeclared site 'nowhere'")
    two = copy.deepcopy(patching)
    two["steps"]["patched"]["data"] = "nothing"
    _refused(two, "undeclared dataset 'nothing'")
    patching["steps"]["patched"]["interventions"] = "nope"
    _refused(patching, "undeclared intervention 'nope'")


def test_a_write_at_a_read_only_place_is_refused(patching):
    patching["steps"]["patched"]["interventions"]["writes"]["patch"]["site"] = {"component": "input_ids"}
    _refused(patching, "is read-only")


# --------------------------------------------------------------------- #
# interventions: a list is composition
# --------------------------------------------------------------------- #


def test_a_list_of_interventions_applies_every_write_in_order(data_root, model_engine):
    """`["steer", "restore"]` is one forward with both writes in force, in
    the order the list gives — which is what makes a renormalize the last
    word at its site."""
    raw = _load("steer_renormalize.json")
    built = _compile(raw, data_root, model_engine)
    forward = built.step("steered_renormed", plan.Forward)
    assert [op.name for tap in forward.taps for op in tap.writes] == ["steered_renormed.steer", "steered_renormed.restore"]

    raw["steps"]["steered_renormed"]["interventions"] = ["restore", "steer"]
    _refused(raw, "renormalize 'restore' restores the norm")


def test_an_intervention_written_in_place_is_the_declared_one(patching, data_root, model_engine):
    """Declared under a name or written on the step, one intervention
    compiles to one forward."""
    declared = copy.deepcopy(patching)
    declared["interventions"] = {"patching": declared["steps"]["patched"]["interventions"]}
    declared["steps"]["patched"]["interventions"] = "patching"
    one, other = (_compile(raw, data_root, model_engine) for raw in (patching, declared))
    assert repr(one.step("patched", plan.Forward)) == repr(other.step("patched", plan.Forward))


def test_a_name_two_interventions_share_is_refused_naming_both(patching):
    patching["interventions"] = {"one": {"reads": {"logits": {"site": "target", "pos": -1}}}}
    patching["steps"]["patched"]["interventions"] = ["one", patching["steps"]["patched"]["interventions"]]
    _refused(patching, re.escape("read 'logits' is named by both the step and intervention 'one'"))


def test_a_baseline_is_a_forward_with_no_writes(data_root, model_engine):
    """No intervention is not a null: a forward with reads and nothing
    written, and a metric of it."""
    built = _compile(_load("two_observes.json"), data_root, model_engine)
    assert all(not tap.writes for tap in built.step("clean", plan.Forward).taps)
    assert model_engine.execute(built).result("clean_ld").shape == (4,)


def test_two_experiments_at_one_site_write_where_their_saves_say(data_root, model_engine, tmp_path):
    """Patching and mean ablation over the same rows, and the baseline they
    share: three logit differences, three names, and the save paths are the
    directories."""
    executed = model_engine.execute(_compile(_load("two_observes.json"), data_root, model_engine))
    clean, patched, ablated = (executed.result(name) for name in ("clean_ld", "patching_ld", "ablation_ld"))
    assert not torch.equal(clean, patched) and not torch.equal(clean, ablated) and not torch.equal(patched, ablated)
    written = {str(one.relative_to(tmp_path)) for one in executed.write(tmp_path)}
    assert {"clean.json", "compare/patching/iia.json", "compare/patching/logit_diff.json",
            "compare/ablation/logit_diff.json"} <= written


# --------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------- #


def test_an_operand_names_an_earlier_step(patching):
    steps = patching["steps"]
    patching["steps"] = {"patched": steps["patched"], "counterfactual": steps["counterfactual"], "iia": steps["iia"], "saves": {}}
    _refused(patching, "operand 'counterfactual.v_cf' is nothing before this step")


def test_an_operand_is_not_a_read_of_its_own_step(patching):
    patching["steps"]["patched"]["interventions"]["writes"]["patch"]["operand"] = "patched.logits"
    _refused(patching, "read by this same step")


def test_an_operand_is_a_read_or_a_mean(mean):
    one = copy.deepcopy(mean)
    one["steps"]["ablated"]["interventions"]["writes"]["ablate"]["operand"] = "clean_ld"
    _refused(one, "operand 'clean_ld' is a metric")
    mean["steps"]["mean"].update(reduce="pca", k=2)
    _refused(mean, "a pca basis is loaded as a featurizer")


def test_a_logits_view_is_not_an_operand(patching):
    patching["steps"]["counterfactual"]["reads"]["v_cf"]["view"] = "logits"
    _refused(patching, "is a logits view")


def test_an_operand_covers_the_window_the_write_does(patching):
    patching["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = {"span": [-3, -1]}
    _refused(patching, re.escape("covers 1 position(s) but its operand"))


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda steps: steps.update({"harvest": steps.pop("harvest"), "mean": steps.pop("mean")}),
         "operand 'mean' is nothing before this step"),
        (lambda steps: steps["ablated"]["interventions"]["writes"]["ablate"].update(operand="nope"),
         "operand 'nope' is nothing before this step"),
        (lambda steps: steps["mean"].update(of="harvest.nope"),
         "`of` is 'harvest.nope', which is not a read of a step before it"),
        (lambda steps: steps["saves"].update(mean="mean.json"),
         re.escape("'mean' is a tensor (mean): saved to a .safetensors file")),
    ],
    ids=["a mean taken after it is used", "an operand of nothing", "a mean of a read that is not there",
         "a tensor saved as a table"],
)
def test_references_to_reductions_are_checked(mean, edit, message):
    edit(mean["steps"])
    _refused(mean, message)


def test_a_metric_scores_one_position_of_a_read_before_it(patching):
    patching["steps"]["iia"]["of"] = "patched.nothing"
    _refused(patching, "not a read of a step before it")
    patching["steps"]["counterfactual"]["reads"]["wide"] = {"site": {"component": "lm_head"}, "pos": {"span": [-3, -1]}}
    patching["steps"]["iia"]["of"] = "counterfactual.wide"
    _refused(patching, "a window of 2 positions")


def test_a_metrics_columns_are_one_declared_datasets(patching):
    patching["steps"]["iia"]["expected"] = "cf_answer"
    _refused(patching, "is not `<dataset>.<column>`")
    patching["data"]["other"] = {"path": "weekdays/train"}
    patching["steps"]["logit_diff"]["b"] = "other.base_answer"
    patching["steps"]["iia"]["expected"] = "pairs.cf_answer"
    _refused(patching, re.escape("its columns come from ['other', 'pairs']"))


def test_a_dataset_written_in_place_has_no_columns_to_name(patching, data_root, model_engine):
    """A step may say `{"path": …}` for its rows; a metric then has no name
    to reach that dataset's columns by, and takes them from a declared one."""
    for name in ("counterfactual", "patched"):
        patching["steps"][name]["data"] = {"path": "weekdays/train"}
    assert model_engine.execute(_compile(patching, data_root, model_engine)).result("iia").shape == (4,)


def test_a_metric_pairs_row_i_with_row_i(patching, data_root, model_engine):
    patching["data"]["pairs"] = {"path": "weekdays/data#train"}
    patching["data"]["short"] = {"path": "weekdays/train"}
    patching["steps"]["iia"]["expected"] = "short.cf_answer"
    with pytest.raises(plan.PlanError, match="its columns are 4 rows of 'short'"):
        _compile(patching, data_root, model_engine)


# --------------------------------------------------------------------- #
# saves
# --------------------------------------------------------------------- #


def test_a_save_names_something_the_document_produces(patching):
    patching["steps"]["saves"]["nothing"] = "x.json"
    _refused(patching, "'nothing' is nothing this document produces")


def test_a_metric_is_a_table_and_everything_else_a_tensor(patching):
    one = copy.deepcopy(patching)
    one["steps"]["saves"]["iia"] = "iia.safetensors"
    _refused(one, "a metric, one row per example")
    patching["steps"]["saves"]["patched.logits"] = "logits.json"
    _refused(patching, re.escape("a tensor (read)"))


def test_two_saves_are_two_files_inside_the_output(patching):
    one = copy.deepcopy(patching)
    one["steps"]["saves"]["logit_diff"] = "iia.json"
    _refused(one, "are both saved to 'iia.json'")
    patching["steps"]["saves"]["iia"] = "../iia.json"
    _refused(patching, "a path inside the output directory")


# --------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------- #


def test_a_trained_featurizer_has_one_name(das):
    """Its bare name is the declaration; every use is `<fit>.<name>`, and a
    use before the fit has run would score the untrained parameter."""
    one = copy.deepcopy(das)
    one["interventions"]["das"]["writes"]["patch"]["featurizer"] = "rot"
    _refused(one, "featurizer 'rot' is trained by step 'fit'; name it fit.rot")
    two = copy.deepcopy(das)
    two["steps"] = {k: two["steps"][k] for k in ("counterfactual", "patched", "iia", "ce", "fit", "saves")}
    _refused(two, "featurizer 'fit.rot' is used before step 'fit' trains it")


def _untrained(das):
    """das.json with no fit: the rotation is its seeded random basis, named
    by its declared name."""
    raw = json.loads(json.dumps(das).replace('"fit.rot"', '"rot"'))
    del raw["steps"]["fit"]
    raw["steps"]["saves"] = {"iia": "iia.json"}
    return raw


def test_an_untrained_rotation_is_its_declared_name_in_any_order(das):
    """The rule is about training: with no fit, the rotation is the same
    parameter wherever it is used. (Scoring an untrained *and* a trained
    rotation in one document needs two featurizers — which is why the
    refusal above points at a second one.)"""
    Spec.model_validate(_untrained(das))


def test_a_declared_featurizer_nothing_uses_is_refused(das):
    das["featurizers"]["spare"] = {"kind": "subspace", "k": 4}
    _refused(das, re.escape("featurizer 'spare' is used at 0 site(s)"))


def test_a_swept_document_is_one_plan_per_point(das, data_root, model_engine, tmp_path):
    """A sweep is lowered on the raw JSON, so a swept featurizer seed is the
    random-subspace control in this format too."""
    raw = _untrained(das)
    raw["featurizers"]["rot"]["seed"] = {"sweep": [0, 1, 2]}
    root = _compile(raw, data_root, model_engine)

    assert list(root.steps) == ["seed=0", "seed=1", "seed=2"]
    for label in root.steps:
        (spec,) = root.step(label, plan.Plan).step("featurizers", plan.Featurizers).specs
        assert spec.seed == int(label.removeprefix("seed=")) and spec.trained is False

    executed = model_engine.execute(root)
    scored = [point.result("iia") for point in executed.steps.values()]
    assert not torch.equal(scored[0], scored[1])
    written = {str(p.relative_to(tmp_path)) for p in executed.write(tmp_path)}
    assert "seed=1/iia.json" in written and "seed=1/document.json" in written


def test_a_fits_terms_are_its_bodys(das):
    one = copy.deepcopy(das)
    one["steps"]["fit"]["objective"] = [[1.0, "nothing"]]
    _refused(one, "objective names 'nothing'")
    two = copy.deepcopy(das)
    two["steps"]["fit"]["early_stop"]["metric"] = "ce_typo"
    _refused(two, "early_stop watches 'ce_typo'")
    three = copy.deepcopy(das)
    three["steps"]["fit"]["eval"]["data"] = {"elsewhere": "test"}
    _refused(three, "eval replaces 'elsewhere'")


def test_a_fits_body_saves_nothing(das):
    das["steps"]["fit"]["steps"]["saves"] = {"iia": "x.json"}
    _refused(das, "a fit's body saves nothing")


def test_a_fits_body_may_train_through_a_mean_of_its_own_reads(das, data_root, model_engine):
    """A body's steps keep their graph until the update, so a mean taken in
    the body — here of the rotated counterfactuals of the minibatch — is
    something the fit differentiates through, like any other value."""
    body = das["steps"]["fit"]["steps"]
    das["steps"]["fit"]["steps"] = {
        "counterfactual": body["counterfactual"],
        "centre": {"kind": "reduce", "reduce": "mean", "of": "counterfactual.v_cf"},
        "patched": {**body["patched"], "interventions": {
            "writes": {"patch": {"site": "target", "pos": -1, "featurizer": "fit.rot", "mechanism": "swap", "operand": "centre"}},
            "reads": {"logits": {"site": "head", "pos": -1}}}},
        "iia": body["iia"],
        "ce": body["ce"],
    }
    losses = model_engine.execute(_compile(das, data_root, model_engine)).result("train/loss")
    assert losses[-1] < losses[0]


def test_a_read_from_outside_a_fits_body_is_not_its_operand(das):
    """The fit shuffles its rows, so an unreduced read from outside would
    meet the wrong row."""
    outside = {"kind": "forward", "data": "train", "field": "input", "reads": {"v": {"site": "target", "pos": -1}}}
    das["steps"] = {"outside": outside, **das["steps"]}
    das["interventions"]["das"]["writes"]["patch"].update(operand="outside.v", featurizer="identity")
    _refused(das, "is a read from outside the fit's body")


def test_the_held_out_pass_is_the_body_on_other_rows(das, data_root, model_engine):
    """`eval.data` puts one declared dataset in place of another, and the two
    must not share a row — unless they are the same dataset, which is the
    train-equals-test ablation and says so."""
    built = _compile(das, data_root, model_engine)
    assert len(built.step("fit", plan.Fit).evaluation.step("counterfactual", plan.Forward).input_ids) == 2
    das["data"]["overlap"] = {"path": "weekdays/data"}
    das["steps"]["fit"]["eval"]["data"] = {"train": "overlap"}
    with pytest.raises(plan.PlanError, match="share"):
        _compile(das, data_root, model_engine)
    das["steps"]["fit"]["eval"]["data"] = {"train": "train"}
    _compile(das, data_root, model_engine)


# --------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------- #


def _generating(raw, **arguments):
    raw["steps"]["patched"].update(kind="generate", max_new_tokens=3, **arguments)
    return raw


def test_a_generate_steps_arguments_reach_the_call(patching, data_root, model_engine):
    """Every key beyond the step's own is a `model.generate` argument,
    passed as written; the step's own name is the ids it generated."""
    raw = _generating(patching, min_new_tokens=3, do_sample=False)
    raw["steps"]["saves"]["patched"] = "generated.safetensors"
    built = _compile(raw, data_root, model_engine)
    forward = built.step("patched", plan.Generate)
    assert (forward.max_new_tokens, forward.generation) == (3, {"min_new_tokens": 3, "do_sample": False})
    assert model_engine.execute(built).result("patched").shape == (4, 3)


def test_generate_arguments_are_transformers_own(patching):
    _refused(_generating(copy.deepcopy(patching), min_new_tokens=3, tempreature=0.5), "`tempreature` is neither")
    _refused(_generating(copy.deepcopy(patching), min_new_tokens=3, max_length=9), "the bound is `max_new_tokens`")
    _refused(_generating(copy.deepcopy(patching), min_new_tokens=3, num_beams=2), "several rows of one")


def test_a_tapped_decode_runs_to_its_bound(patching):
    """Until the engine walks a decode that stops early, a generate with
    taps decodes exactly `max_new_tokens`, and says so."""
    _refused(_generating(patching), "say `min_new_tokens: 3`")


def test_a_generate_with_no_taps_may_stop_where_the_model_does(patching, data_root, model_engine):
    raw = {
        "model": patching["model"],
        "data": patching["data"],
        "steps": {
            "said": {"kind": "generate", "data": "pairs", "field": "input", "max_new_tokens": 4, "do_sample": False},
            "saves": {"said": "said.safetensors"},
        },
    }
    said = model_engine.execute(_compile(raw, data_root, model_engine)).result("said")
    assert said.shape[0] == 4 and said.shape[1] <= 4


def test_a_continuation_position_needs_a_generate_step(patching):
    patching["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 0}
    _refused(copy.deepcopy(patching), "needs a generate step")
    patching["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 5}
    _refused(_generating(patching, min_new_tokens=3), "step 5 of a 3-token decode")
