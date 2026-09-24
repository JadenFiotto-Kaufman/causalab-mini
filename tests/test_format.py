"""The steps-first document format.

A document's `steps` are what runs, in order; a step's name is how
everything after it reaches what it produced (`patched.logits`, `iia`,
`fit.rot`); `saves` names the references that go to disk. These tests hold
the format to what it says: that every reference resolves where it is
written or is refused with the fix, that each step compiles to one step of
the plan, and that what reaches the plan is what the document wrote.
"""

import copy
import json
import re

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.plan.spec import Spec

MODEL = {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "fp32",
}

#: The smallest real interchange, which every test below edits.
PATCHING = {
    "model": MODEL,
    "data": {"pairs": {"path": "weekdays/train"}},
    "sites": {"target": {"component": "block_output", "layers": [0]}, "head": {"component": "lm_head"}},
    "steps": {
        "counterfactual": {
            "kind": "forward", "data": "pairs", "field": "counterfactual_inputs[0]",
            "reads": {"v_cf": {"site": "target", "pos": -1}},
        },
        "patched": {
            "kind": "forward", "data": "pairs", "field": "input",
            "interventions": {"writes": {"patch": {"site": "target", "pos": -1, "mechanism": "swap", "operand": "counterfactual.v_cf"}}},
            "reads": {"logits": {"site": "head", "pos": -1}},
        },
        "iia": {"kind": "metric", "metric": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
        "logit_diff": {"kind": "metric", "metric": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
        "saves": {"iia": "iia.json", "logit_diff": "logit_diff.json"},
    },
}

#: DAS: a fit whose body is the experiment, and the same two declared
#: interventions scored after it.
DAS = {
    "model": MODEL,
    "data": {"train": {"path": "weekdays/data#train"}, "test": {"path": "weekdays/data#test"}},
    "sites": {"target": {"component": "block_output", "layers": [0]}, "head": {"component": "lm_head"}},
    "featurizers": {"rot": {"kind": "subspace", "k": 8}},
    "interventions": {
        "cf_read": {"reads": {"v_cf": {"site": "target", "pos": -1, "featurizer": "fit.rot"}}},
        "das": {
            "writes": {"patch": {"site": "target", "pos": -1, "featurizer": "fit.rot", "mechanism": "swap", "operand": "counterfactual.v_cf"}},
            "reads": {"logits": {"site": "head", "pos": -1}},
        },
    },
    "steps": {
        "fit": {
            "kind": "fit", "train": ["rot"], "objective": [[1.0, "ce"]],
            "epochs": 2, "batch_size": 16, "seed": 0,
            "optimizer": {"lr": 0.001}, "early_stop": {"metric": "iia", "patience": 3},
            "eval": {"data": {"train": "test"}},
            "steps": {
                "counterfactual": {"kind": "forward", "data": "train", "field": "counterfactual_inputs[0]", "interventions": "cf_read"},
                "patched": {"kind": "forward", "data": "train", "field": "input", "interventions": "das"},
                "iia": {"kind": "metric", "metric": "logit_diff", "of": "patched.logits", "a": "train.cf_answer", "b": "train.base_answer"},
                "ce": {"kind": "metric", "metric": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"},
            },
        },
        "counterfactual": {"kind": "forward", "data": "train", "field": "counterfactual_inputs[0]", "interventions": "cf_read"},
        "patched": {"kind": "forward", "data": "train", "field": "input", "interventions": "das"},
        "iia": {"kind": "metric", "metric": "logit_diff", "of": "patched.logits", "a": "train.cf_answer", "b": "train.base_answer"},
        "saves": {"fit.iia": "held_out_iia.json", "iia": "iia.json", "fit.rot": "rot.safetensors", "fit": "fit.safetensors"},
    },
}

#: A mean harvested by one step and swapped in by a later one.
MEAN = {
    "model": MODEL,
    "data": {"train": {"path": "weekdays/train"}},
    "sites": {"target": {"component": "block_output", "layers": [0]}, "head": {"component": "lm_head"}},
    "steps": {
        "harvest": {"kind": "forward", "data": "train", "field": "input", "reads": {"acts": {"site": "target", "pos": -1}}},
        "mean": {"kind": "reduce", "reduce": "mean", "of": "harvest.acts"},
        "clean": {"kind": "forward", "data": "train", "field": "input", "reads": {"logits": {"site": "head", "pos": -1}}},
        "clean_ld": {"kind": "metric", "metric": "logit_diff", "of": "clean.logits", "a": "train.cf_answer", "b": "train.base_answer"},
        "ablated": {
            "kind": "forward", "data": "train", "field": "input",
            "interventions": {"writes": {"ablate": {"site": "target", "pos": -1, "mechanism": "swap", "operand": "mean"}}},
            "reads": {"logits": {"site": "head", "pos": -1}},
        },
        "ablated_ld": {"kind": "metric", "metric": "logit_diff", "of": "ablated.logits", "a": "train.cf_answer", "b": "train.base_answer"},
        "saves": {"mean": "mean.safetensors", "clean_ld": "clean.json", "ablated_ld": "ablated.json"},
    },
}


@pytest.fixture
def patching():
    return copy.deepcopy(PATCHING)


@pytest.fixture
def das():
    return copy.deepcopy(DAS)


@pytest.fixture
def mean():
    return copy.deepcopy(MEAN)


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
    the document names them, and every op by its reference from the root."""
    built = _compile(patching, data_root, model_engine)
    assert list(built.steps) == ["counterfactual", "patched", "iia", "logit_diff"]
    patched = built.step("patched", plan.Forward)
    (tap,) = [tap for tap in patched.taps if tap.writes]
    assert tap.writes[0].name == "patched.patch" and tap.writes[0].operand == "counterfactual.v_cf"
    assert built.step("iia", plan.Metric).of == "patched.logits"


def test_a_mean_is_a_step_and_its_value_an_operand(mean, data_root, model_engine):
    """`mean` reduces a read of an earlier step, and a later step swaps it
    in by its name — the same as any other value."""
    built = _compile(mean, data_root, model_engine)
    assert list(built.steps) == ["harvest", "mean", "clean", "clean_ld", "ablated", "ablated_ld"]
    reduced = built.step("mean", plan.Reduce)
    assert (reduced.of, reduced.reduce) == ("harvest.acts", "mean")
    (write,) = [op for tap in built.step("ablated", plan.Forward).taps for op in tap.writes]
    assert write.operand == "mean"


def test_the_same_place_declared_or_written_in_place_is_one_site(patching, data_root, model_engine):
    """A site written in place is named by its own spelling, and resolves
    to the same address a declared one does."""
    inline = copy.deepcopy(patching)
    inline["steps"]["patched"]["reads"]["logits"]["site"] = {"component": "lm_head"}
    one, other = (
        model_engine.execute(_compile(raw, data_root, model_engine)).result("iia") for raw in (patching, inline)
    )
    assert torch.equal(one, other)


def test_a_read_comes_home_when_a_save_names_it(data_root, model_engine):
    """Every read is there for the steps after it; only one a save names is
    brought home, because on a real vocabulary one logits read over a
    thousand rows is half a gigabyte."""
    raw = copy.deepcopy(MEAN)
    assert _compile(raw, data_root, model_engine).step("clean", plan.Forward).keep == ()
    raw["steps"]["saves"]["clean.logits"] = "clean_logits.safetensors"
    built = _compile(raw, data_root, model_engine)
    assert built.step("clean", plan.Forward).keep == ("clean.logits",)
    assert model_engine.execute(built).result("clean.logits").shape == (4, 1, 32000)


def test_what_a_document_saves_is_what_reaches_disk(mean, data_root, model_engine, tmp_path):
    written = model_engine.execute(_compile(mean, data_root, model_engine)).write(tmp_path)
    assert {path.name for path in written} == {"mean.safetensors", "clean.json", "ablated.json", "document.json", "run.json"}
    rows = json.loads((tmp_path / "ablated.json").read_text())
    assert {row["metric"] for row in rows} == {"ablated_ld"}


# --------------------------------------------------------------------- #
# interventions: a list is composition
# --------------------------------------------------------------------- #


def test_a_list_of_interventions_applies_every_write_in_order(data_root, model_engine):
    """`["steer", "restore"]` is one forward with both writes in force, the
    order the list gives — which is what makes a renormalize the last word
    at its site."""
    raw = copy.deepcopy(MEAN)
    raw["interventions"] = {
        "steer": {"writes": {"steer": {"site": "target", "pos": -1, "mechanism": "add_scaled", "operand": "mean", "params": {"scale": 4.0}}}},
        "restore": {"writes": {"restore": {"site": "target", "pos": -1, "mechanism": "renormalize"}}},
    }
    raw["steps"]["ablated"]["interventions"] = ["steer", "restore"]
    built = _compile(raw, data_root, model_engine)
    forward = built.step("ablated", plan.Forward)
    assert [op.name for tap in forward.taps for op in tap.writes] == ["ablated.steer", "ablated.restore"]

    raw["steps"]["ablated"]["interventions"] = ["restore", "steer"]
    _refused(raw, "renormalize 'restore' restores the norm")


def test_a_name_two_interventions_share_is_refused_naming_both(patching):
    patching["interventions"] = {"one": {"reads": {"logits": {"site": "head", "pos": -1}}}}
    patching["steps"]["patched"]["interventions"] = ["one", patching["steps"]["patched"]["interventions"]]
    _refused(patching, re.escape("read 'logits' is named by both the step and intervention 'one'"))


def test_an_undeclared_intervention_is_refused(patching):
    patching["steps"]["patched"]["interventions"] = "nope"
    _refused(patching, "undeclared intervention 'nope'")


# --------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------- #


def test_an_operand_names_an_earlier_step(patching):
    steps = patching["steps"]
    patching["steps"] = {"patched": steps["patched"], "counterfactual": steps["counterfactual"],
                         "iia": steps["iia"], "saves": {}}
    _refused(patching, "operand 'counterfactual.v_cf' is nothing before this step")


def test_an_operand_is_not_a_read_of_its_own_step(patching):
    patching["steps"]["patched"]["interventions"]["writes"]["patch"]["operand"] = "patched.logits"
    _refused(patching, "read by this same step")


def test_an_operand_is_a_read_or_a_mean(patching, mean):
    patching["steps"]["patched"]["interventions"]["writes"]["patch"]["operand"] = "iia"
    _refused(patching, "operand 'iia' is nothing before this step")
    mean["steps"]["mean"]["reduce"] = "pca"
    mean["steps"]["mean"]["k"] = 2
    _refused(mean, "a pca basis is loaded as a featurizer")


def test_a_logits_view_is_not_an_operand(patching):
    patching["steps"]["counterfactual"]["reads"]["v_cf"]["view"] = "logits"
    _refused(patching, "is a logits view")


def test_an_operand_covers_the_window_the_write_does(patching):
    patching["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = {"span": [-3, -1]}
    _refused(patching, "covers 1 position\\(s\\) but its operand")


def test_a_metric_scores_one_position_of_a_read_before_it(patching):
    patching["steps"]["iia"]["of"] = "patched.nothing"
    _refused(patching, "not a read of a step before it")
    patching["steps"]["counterfactual"]["reads"]["wide"] = {"site": "head", "pos": {"span": [-3, -1]}}
    patching["steps"]["iia"]["of"] = "counterfactual.wide"
    _refused(patching, "a window of 2 positions")


def test_a_metrics_columns_are_one_declared_datasets(patching):
    patching["steps"]["iia"]["expected"] = "cf_answer"
    _refused(patching, "is not `<dataset>.<column>`")
    patching["data"]["other"] = {"path": "weekdays/train"}
    patching["steps"]["logit_diff"]["b"] = "other.base_answer"
    patching["steps"]["iia"]["expected"] = "pairs.cf_answer"
    _refused(patching, "its columns come from \\['other', 'pairs'\\]")


def test_a_dataset_written_in_place_has_no_columns_to_name(patching, data_root, model_engine):
    """A step may say `{"path": …}` for its rows; a metric then has no name
    to reach that dataset's columns by, and takes them from a declared one."""
    for name in ("counterfactual", "patched"):
        patching["steps"][name]["data"] = {"path": "weekdays/train"}
    built = model_engine.execute(_compile(patching, data_root, model_engine))
    assert built.result("iia").shape == (4,)


def test_a_metric_pairs_row_i_with_row_i(patching, data_root, model_engine):
    patching["data"]["pairs"] = {"path": "weekdays/data#train"}
    patching["data"]["short"] = {"path": "weekdays/train"}
    patching["steps"]["iia"]["expected"] = "short.cf_answer"
    with pytest.raises(plan.PlanError, match="its columns are 4 rows of 'short'"):
        _compile(patching, data_root, model_engine)


def test_a_name_is_one_step_and_cannot_hold_a_dot(patching):
    patching["steps"]["a.b"] = patching["steps"].pop("iia")
    _refused(patching, "a name is a letter")
    patching["steps"].pop("a.b")
    patching["steps"]["featurizers"] = patching["steps"].pop("logit_diff")
    _refused(patching, "the name is reserved")


def test_undeclared_sites_and_datasets_are_refused(patching):
    one = copy.deepcopy(patching)
    one["steps"]["patched"]["reads"]["logits"]["site"] = "nowhere"
    _refused(one, "undeclared site 'nowhere'")
    patching["steps"]["patched"]["data"] = "nothing"
    _refused(patching, "undeclared dataset 'nothing'")


def test_an_unknown_key_anywhere_is_refused_with_its_path(patching):
    patching["steps"]["patched"]["interventions"]["writes"]["patch"]["mechanisn"] = "swap"
    _refused(patching, "mechanisn")


def test_a_write_at_a_read_only_place_is_refused(patching):
    patching["sites"]["target"] = {"component": "embeddings"}
    patching["steps"]["patched"]["interventions"]["writes"]["patch"]["site"] = {"component": "input_ids"}
    with pytest.raises(ValidationError):
        Spec.model_validate(patching)


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
    _refused(patching, "a tensor \\(read\\)")


def test_two_saves_are_two_files_inside_the_output(patching):
    one = copy.deepcopy(patching)
    one["steps"]["saves"]["logit_diff"] = "iia.json"
    _refused(one, "are both saved to 'iia.json'")
    patching["steps"]["saves"]["iia"] = "../iia.json"
    _refused(patching, "a path inside the output directory")


# --------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------- #


def test_a_fit_trains_the_rotation_its_score_names(das, data_root, model_engine, tmp_path):
    """One declared intervention, named by the body and by the score after
    it: `fit.rot` is the rotation being trained inside, and the trained one
    after. The plan is the featurizers, the fit — each update its body's
    steps — the weights, and the score."""
    built = _compile(das, data_root, model_engine)
    assert list(built.steps) == ["featurizers", "fit", "fit.weights", "counterfactual", "patched", "iia"]
    fit = built.step("fit", plan.Fit)
    assert fit.objective == ((1.0, "ce"),) and fit.early_stop == "iia" and fit.params == ("rot",)
    update = fit.epochs[0][0]
    assert list(update.steps) == ["counterfactual", "patched", "iia", "ce"]
    (write,) = [op for tap in update.step("patched", plan.Forward).taps for op in tap.writes]
    assert (write.operand, write.featurizer) == ("counterfactual.v_cf", "rot")

    executed = model_engine.execute(built)
    written = {path.name for path in executed.write(tmp_path)}
    assert written == {"held_out_iia.json", "iia.json", "rot.safetensors", "fit.safetensors", "document.json", "run.json"}
    from safetensors import safe_open

    with safe_open(str(tmp_path / "fit.safetensors"), "pt") as record:
        assert set(record.keys()) == {"loss", "eval"}
    held_out = json.loads((tmp_path / "held_out_iia.json").read_text())
    assert {row["metric"] for row in held_out} == {"iia"}


def test_a_trained_featurizer_has_one_name(das):
    """Its bare name is the declaration; every use is `<fit>.<name>`."""
    one = copy.deepcopy(das)
    one["interventions"]["das"]["writes"]["patch"]["featurizer"] = "rot"
    _refused(one, "featurizer 'rot' is trained by step 'fit'; name it fit.rot")
    two = copy.deepcopy(das)
    two["steps"] = {"early": copy.deepcopy(das["steps"]["patched"]), **das["steps"]}
    two["steps"]["early"]["interventions"] = {"reads": {"x": {"site": "target", "pos": -1, "featurizer": "fit.rot"}}}
    _refused(two, "is used before step 'fit' trains it")


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


def test_a_fits_body_saves_nothing_and_holds_no_fit(das):
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
    executed = model_engine.execute(_compile(das, data_root, model_engine))
    losses = executed.result("train/loss")
    assert losses[-1] < losses[0]


def test_a_read_from_outside_a_fits_body_is_not_its_operand(das):
    """The fit shuffles its rows, so an unreduced read from outside would
    meet the wrong row."""
    das["steps"] = {"outside": {"kind": "forward", "data": "train", "field": "input", "reads": {"v": {"site": "target", "pos": -1}}}, **das["steps"]}
    das["interventions"]["das"]["writes"]["patch"]["operand"] = "outside.v"
    das["interventions"]["das"]["writes"]["patch"]["featurizer"] = "identity"
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
    step = raw["steps"]["patched"]
    step.update(kind="generate", max_new_tokens=3, **arguments)
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


def test_a_generate_with_no_taps_may_stop_where_the_model_does(data_root, model_engine):
    raw = {
        "model": MODEL,
        "data": {"pairs": {"path": "weekdays/train"}},
        "steps": {
            "said": {"kind": "generate", "data": "pairs", "field": "input", "max_new_tokens": 4, "do_sample": False},
            "saves": {"said": "said.safetensors"},
        },
    }
    executed = model_engine.execute(_compile(raw, data_root, model_engine))
    assert executed.result("said").shape[0] == 4 and executed.result("said").shape[1] <= 4


def test_a_continuation_position_needs_a_generate_step(patching):
    patching["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 0}
    _refused(patching, "needs a generate step")
    patching["steps"]["patched"]["reads"]["logits"]["pos"] = {"frame": "generated", "index": 5}
    _refused(_generating(patching, min_new_tokens=3), "step 5 of a 3-token decode")
