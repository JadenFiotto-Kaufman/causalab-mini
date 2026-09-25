"""kl, js and top_k: a divergence of one read from another, and a readout.

The arithmetic is checked against torch's own, the refusals where the
document is wrong, and the two corpus documents end to end. What each
writes is pinned in `golden/outputs.json`.
"""

import copy
import json
import math
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.data import tokens
from causalab_mini.ops import metrics
from causalab_mini.plan import PlanError
from causalab_mini.plan.spec import Spec

V2 = pathlib.Path(__file__).resolve().parents[1] / "documents" / "v2"


@pytest.fixture
def divergence():
    return json.loads((V2 / "divergence.json").read_text())


@pytest.fixture
def top_k():
    return json.loads((V2 / "top_k.json").read_text())


@pytest.fixture
def das():
    return json.loads((V2 / "das.json").read_text())


def _refused(raw, match):
    with pytest.raises(ValidationError, match=match):
        Spec.model_validate(raw)


def test_kl_and_js_are_the_divergences_in_nats():
    torch.manual_seed(0)
    p, q = torch.randn(3, 11), torch.randn(3, 11)
    expected = torch.nn.functional.kl_div(q.log_softmax(-1), p.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    assert torch.allclose(metrics.kl(p, q), expected)
    assert torch.allclose(metrics.kl(p, p), torch.zeros(3), atol=1e-6)
    js = metrics.js(p, q)
    assert torch.allclose(js, metrics.js(q, p)) and bool((js > 0).all()) and bool((js <= math.log(2)).all())


def test_a_token_with_no_probability_contributes_nothing():
    """−inf logits on both sides, or on the first side only, are a term of
    0 — in the value and in its gradient."""
    p = torch.tensor([[0.0, 1.0, -math.inf]], requires_grad=True)
    q = torch.tensor([[1.0, 0.0, -math.inf]])
    for kind in ("kl", "js"):
        value = metrics.compute(kind, p, (q,))
        (grad,) = torch.autograd.grad(value.sum(), p)
        assert torch.isfinite(value).all() and torch.isfinite(grad).all()
        assert torch.allclose(value, metrics.compute(kind, p[:, :2], (q[:, :2],)))


def test_top_k_is_the_k_most_likely_tokens():
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
    probs, ids = metrics.top_k(logits, 2)
    assert ids.tolist() == [[1, 3]]
    assert torch.allclose(probs, logits.softmax(-1)[:, [1, 3]])


def test_a_read_against_itself_diverges_by_nothing(divergence, data_root, model_engine):
    divergence["steps"]["kl"]["against"] = "patched.logits"
    divergence["steps"]["js"]["against"] = "patched.logits"
    executed = model_engine.execute(plan.build_request(divergence, data_root, model_engine))
    for name in ("kl", "js"):
        assert torch.allclose(executed.result(name), torch.zeros(4), atol=1e-6)


def test_a_divergence_needs_an_earlier_logits_read_against_it(divergence):
    for against, why in [
        (None, "against\n  Field required"),
        ("nothing.logits", "not a logits read of a step before it"),
        ("counterfactual.v_cf", "not a logits read of a step before it"),
    ]:
        raw = copy.deepcopy(divergence)
        if against is None:
            del raw["steps"]["kl"]["against"]
        else:
            raw["steps"]["kl"]["against"] = against
        _refused(raw, why)
    later = copy.deepcopy(divergence)
    later["steps"] = {name: later["steps"][name] for name in ("counterfactual", "patched", "kl", "clean", "saves")}
    _refused(later, "not a logits read of a step before it")


def test_a_divergence_over_rows_that_differ_is_refused(divergence, data_root, model_engine):
    """Another row count is refused, and so is a window, or a read only some
    rows may have: `against` is one position on every row."""
    six = copy.deepcopy(divergence)
    six["data"]["six"] = {"path": "weekdays_even/train"}
    six["steps"]["clean"]["data"] = "six"
    with pytest.raises(PlanError, match="read over 4 rows and 'clean.logits' over 6"):
        plan.build_request(six, data_root, model_engine)
    window = copy.deepcopy(divergence)
    window["steps"]["clean"]["reads"]["logits"]["pos"] = {"last": 2}
    _refused(window, "a window of 2 positions")
    divergence["steps"]["clean"]["reads"]["logits"]["pos"] = {"index": -1, "scope": {"variable": "entity"}}
    with pytest.raises(PlanError, match="'clean.logits' is read flat"):
        plan.build_request(divergence, data_root, model_engine)


def test_top_k_is_written_as_token_probability_pairs(top_k, data_root, model_engine, tmp_path):
    executed = model_engine.execute(plan.build_request(top_k, data_root, model_engine))
    executed.write(tmp_path)
    rows = json.loads((tmp_path / "top.json").read_text())
    assert [row["layer"] for row in rows] == [0] * 4 + [1] * 4
    for row in rows:
        assert len(row["value"]) == 3 and row["unit"] == "probability"
        assert all(isinstance(token, str) for token, _ in row["value"])
        assert [p for _, p in row["value"]] == sorted((p for _, p in row["value"]), reverse=True)


def test_top_k_must_fit_the_vocabulary(top_k, data_root, model_engine):
    top_k["steps"]["top"]["k"] = 0
    _refused(copy.deepcopy(top_k), "greater than 0")
    top_k["steps"]["top"]["k"] = 32001
    with pytest.raises(PlanError, match="top 32001 of a vocabulary of 32000"):
        model_engine.execute(plan.build_request(top_k, data_root, model_engine))


def test_a_top_k_is_neither_minimized_nor_watched(das):
    das["steps"]["fit"]["steps"]["top"] = {"kind": "metric", "metric": "top_k", "of": "patched.logits"}
    minimized = copy.deepcopy(das)
    minimized["steps"]["fit"]["objective"] = [[1.0, "top"]]
    _refused(minimized, "'top' is a top_k, a list of tokens")
    das["steps"]["fit"]["early_stop"]["metric"] = "top"
    _refused(das, "'top' is a top_k, a list of tokens")


def test_an_id_is_the_vocabulary_id_the_column_holds(model):
    assert tokens.token_id(model.tokenizer, 17, "id") == 17
    for value in ("17", True, -1, len(model.tokenizer)):
        with pytest.raises(tokens.TokenError, match="is not a vocabulary id"):
            tokens.token_id(model.tokenizer, value, "id")


def test_a_column_of_ids_scores_what_its_strings_do(data_root, model_engine, model, tmp_path):
    """The same answers, as the strings the corpus holds and as the ids they
    are, score the same; an id under a string form is refused."""
    rows = json.loads((data_root / "weekdays" / "train.json").read_text())
    for row in rows:
        row["cf_id"] = tokens.token_id(model.tokenizer, row["cf_answer"], "space_prefixed")
    (tmp_path / "ids.json").write_text(json.dumps(rows))
    raw = json.loads((V2 / "patching.json").read_text())
    raw["data"]["pairs"]["path"] = "ids"
    raw["steps"]["by_id"] = {"kind": "metric", "metric": "match", "of": "patched.logits", "expected": "pairs.cf_id", "token_form": "id"}
    executed = model_engine.execute(plan.build_request(raw, tmp_path, model_engine))
    assert torch.equal(executed.result("by_id"), executed.result("iia"))
    raw["steps"]["by_id"]["token_form"] = "bare"
    with pytest.raises(tokens.TokenError, match="is not a string"):
        plan.build_request(raw, tmp_path, model_engine)
