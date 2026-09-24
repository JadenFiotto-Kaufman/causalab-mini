"""Metrics of a distribution against a distribution, and of the mass on one token.

`kl` and `js` are the first kinds that take a second read, `against`; what
they take is their row in `ops.metrics.SIGNATURES`, which the document's
resolver, the compiler and the run all read — so these tests are also the
test that a two-read metric needed no plumbing of its own.
"""

import copy
import json
import math
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.plan.spec import Spec

FAITHFULNESS = pathlib.Path(__file__).resolve().parents[1] / "documents" / "v2" / "faithfulness.json"


@pytest.fixture
def faith():
    return json.loads(FAITHFULNESS.read_text())


def _run(raw, data_root, engine):
    return engine.execute(plan.build_request(raw, data_root, engine))


def test_a_distribution_against_itself_is_zero_to_the_bit(faith, data_root, model_engine):
    faith["steps"]["kl"]["of"] = faith["steps"]["js"]["of"] = "clean.logits"
    executed = _run(faith, data_root, model_engine)
    for name in ("kl", "js"):
        assert torch.equal(executed.result(name), torch.zeros(4)), name


def test_kl_is_directed_and_js_is_symmetric(faith, data_root, model_engine):
    turned = copy.deepcopy(faith)
    for name in ("kl", "js"):
        turned["steps"][name].update({"of": "clean.logits", "against": "patched.logits"})
    one, other = _run(faith, data_root, model_engine), _run(turned, data_root, model_engine)
    assert torch.equal(one.result("js"), other.result("js"))
    assert not torch.equal(one.result("kl"), other.result("kl"))
    assert (one.result("js") >= 0).all() and (one.result("js") <= math.log(2)).all()
    assert (one.result("kl") > 0).all()


def test_the_divergences_are_the_formulas(faith, data_root, model_engine):
    faith["steps"]["saves"] = ["kl", "js", "patched.logits", "clean.logits"]
    executed = _run(faith, data_root, model_engine)
    p = executed.result("patched.logits")[:, 0].log_softmax(-1)
    q = executed.result("clean.logits")[:, 0].log_softmax(-1)
    m = ((p.exp() + q.exp()) / 2).log()
    assert torch.allclose(executed.result("kl"), (p.exp() * (p - q)).sum(-1), atol=1e-6)
    js = 0.5 * ((p.exp() * (p - m)).sum(-1) + (q.exp() * (q - m)).sum(-1))
    assert torch.allclose(executed.result("js"), js, atol=1e-6)


def test_soft_accuracy_is_the_probability_of_the_expected_token(faith, data_root, model_engine):
    faith["steps"]["p"] = {"kind": "metric", "metric": "token_prob", "of": "patched.logits", "token": "pairs.cf_answer"}
    faith["steps"]["saves"] = ["soft_accuracy", "p"]
    executed = _run(faith, data_root, model_engine)
    assert torch.equal(executed.result("soft_accuracy"), executed.result("p"))


def test_the_two_reads_of_a_divergence_are_over_the_same_rows(faith, data_root, model_engine):
    faith["data"]["other"] = {"path": "weekdays/data#train"}
    faith["steps"]["clean"]["data"] = "other"
    with pytest.raises(plan.PlanError, match=r"'patched.logits' was read over 4 rows and 'clean.logits' over 2"):
        plan.build_request(faith, data_root, model_engine)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["steps"]["kl"].update(against="counterfactual.v_cf"), "`against` is 'counterfactual.v_cf', which is not logits"),
        (lambda raw: raw["steps"]["kl"].update(against="later.logits"), "`against` is 'later.logits', which is not a read of a step before it"),
        (lambda raw: raw["steps"]["kl"].pop("against"), "against"),
    ],
    ids=["not logits", "not a read", "missing"],
)
def test_what_a_second_read_may_not_be(faith, edit, message):
    edit(faith)
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(faith)
