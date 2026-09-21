"""An excluded measurement is not a zero.

A row whose answer column is null cannot be scored by a metric that names
that column. It stays a row of the table — `eligible: false`, no value — and
it never enters a mean, a loss or an early-stop score. Which rows those are
is a question about the data, so the client answers it: the plan carries a
row list, and the run indexes. Nothing at run time masks, and a NaN is still
a bug rather than a convention.
"""

import json
import pathlib
import shutil

import pytest
import torch

from causalab_mini import plan
from causalab_mini.data import rows as rows_module
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan import explain
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCHING = REPO / "documents" / "v2" / "patching.json"
DAS = REPO / "documents" / "v2" / "das.json"
EXCLUDED = 1  # the row whose counterfactual answer is unknown


@pytest.fixture
def raw():
    """patching.json plus a metric that does not name the null column."""
    one = json.loads(PATCHING.read_text())
    one["interventions"]["patching"]["metrics"]["base_logit"] = {
        "kind": "token_logit", "of": "logits", "token": "base_answer", "token_form": "space_prefixed",
    }
    one["steps"]["score"]["saves"].append({"value": "base_logit", "file_path": "base_logit.json"})
    return one


@pytest.fixture
def holed_root(data_root, tmp_path):
    """The shipped data, with one row's `cf_answer` nulled."""
    root = tmp_path / "data"
    shutil.copytree(data_root, root)
    for file in ("train.json", "data.json"):
        path = root / "weekdays" / file
        table = json.loads(path.read_text())
        table[EXCLUDED]["cf_answer"] = None
        path.write_text(json.dumps(table))
    return root


def test_a_null_column_takes_the_row_out_of_that_metric_only(raw, holed_root, model_engine):
    built = plan.build_request(raw, holed_root, model_engine)
    by_name = {metric.name: metric for metric in built.step("score", plan.Observe).metrics}

    assert by_name["logit_diff"].rows == (0, 2, 3)
    assert [len(ids) for ids in by_name["logit_diff"].ids] == [3, 3]
    assert by_name["base_logit"].rows is None, "it does not name the null column"
    assert "logit_diff/logit_diff rows=[0, 2, 3]" in explain.explain(built)


def test_the_other_rows_score_exactly_what_they_did(raw, data_root, holed_root, model_engine):
    whole = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    holed = model_engine.execute(plan.build_request(raw, holed_root, model_engine))

    assert holed.result("logit_diff").shape == (3,)
    assert torch.equal(holed.result("logit_diff"), whole.result("logit_diff")[[0, 2, 3]])
    assert torch.equal(holed.result("base_logit"), whole.result("base_logit"))


def test_the_table_keeps_the_row_and_says_it_was_not_measured(raw, holed_root, model_engine, tmp_path):
    model_engine.execute(plan.build_request(raw, holed_root, model_engine)).write(tmp_path)

    table = json.loads((tmp_path / "logit_diff.json").read_text())
    assert [row["example_id"] for row in table] == ["0", "1", "2", "3"]
    assert [row["eligible"] for row in table] == [True, False, True, True]
    assert table[EXCLUDED]["value"] is None
    assert all(row["value"] is not None for row in table if row["eligible"])

    untouched = json.loads((tmp_path / "base_logit.json").read_text())
    assert all(row["eligible"] and row["value"] is not None for row in untouched)


def test_the_hooks_engine_excludes_the_same_row(raw, holed_root, model_engine):
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(raw, holed_root, model_engine))
    hooked = hooks.execute(plan.build_request(raw, holed_root, hooks))
    assert torch.equal(traced.result("logit_diff"), hooked.result("logit_diff"))


def test_a_metric_of_nothing_is_refused(raw, holed_root, model_engine):
    path = holed_root / "weekdays" / "train.json"
    table = json.loads(path.read_text())
    for row in table:
        row["cf_answer"] = ""
    path.write_text(json.dumps(table))
    with pytest.raises(plan.PlanError, match="a metric of nothing has no mean"):
        plan.build_request(raw, holed_root, model_engine)


def test_a_column_no_row_has_is_a_misspelling_not_an_exclusion(raw, data_root, model_engine):
    raw["interventions"]["patching"]["metrics"]["logit_diff"]["a"] = "cf_anwser"
    with pytest.raises(rows_module.DataError, match="no row has a column 'cf_anwser'"):
        plan.build_request(raw, data_root, model_engine)


# --------------------------------------------------------------------- #
# under a fit
# --------------------------------------------------------------------- #


def test_a_fit_trains_and_scores_over_the_eligible_rows(holed_root, model_engine, tmp_path):
    """The loss and the held-out score are means over measured rows, so the
    hole changes the numbers and breaks nothing."""
    raw = json.loads(DAS.read_text())
    executed = model_engine.execute(plan.build_request(raw, holed_root, model_engine))
    fit = executed.step("fit", plan.Fit)
    assert torch.isfinite(fit.results["train/loss"]).all()
    assert torch.isfinite(fit.results["train/eval"]).all()

    executed.write(tmp_path)
    table = json.loads((tmp_path / "iia.json").read_text())
    assert [row["eligible"] for row in table].count(False) >= 1


def test_a_minibatch_with_nothing_to_score_is_refused_before_anything_runs(holed_root, model_engine):
    """One pair per update makes the excluded row a whole update, whose loss
    would be the mean of nothing."""
    raw = json.loads(DAS.read_text())
    raw["steps"]["fit"]["pairs"] = 1
    with pytest.raises(plan.PlanError, match="a metric of nothing"):
        plan.build_request(raw, holed_root, model_engine)
