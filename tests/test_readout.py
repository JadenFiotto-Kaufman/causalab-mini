"""Readouts, and how a column's value is spelled as a token.

`top_k` is the first metric whose value per row is not one number: a list
of the k most likely tokens, decoded, with their probabilities. It goes
through the same table writer as every other metric — the list is the row's
`value` — so a later kind whose value is a list per row needs nothing more.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.data import tokens
from causalab_mini.plan.spec import Spec

READOUT = pathlib.Path(__file__).resolve().parents[1] / "documents" / "v2" / "readout.json"


@pytest.fixture
def readout():
    return json.loads(READOUT.read_text())


def _run(raw, data_root, engine):
    return engine.execute(plan.build_request(raw, data_root, engine))


def test_top_k_is_k_decoded_tokens_a_row_with_their_probabilities(readout, data_root, model_engine, tmp_path):
    executed = _run(readout, data_root, model_engine)
    rows = executed.result("top3")
    assert len(rows) == 4 and all(len(row) == 3 for row in rows)
    for row in rows:
        probabilities = [p for _, p in row]
        assert probabilities == sorted(probabilities, reverse=True)
        assert 0 < sum(probabilities) <= 1
        assert all(isinstance(token, str) for token, _ in row)
    # the most likely token is the argmax, decoded
    readout["steps"]["saves"] = ["clean.logits"]
    logits = _run(readout, data_root, model_engine).result("clean.logits")[:, 0]
    top = logits.float().softmax(-1).max(-1)
    assert [row[0][1] for row in rows] == top.values.tolist()
    # spelled as provenance spells the tokens a run addressed
    assert [row[0][0] for row in rows] == [repr(model_engine.tokenizer.decode([one])) for one in top.indices.tolist()]

    executed.write(tmp_path)
    table = json.loads((tmp_path / "top3.json").read_text())
    assert [one["value"] for one in table] == rows
    assert {one["unit"] for one in table} == {"[token, probability] list"}


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda fit: fit.update(objective=[[1.0, "top"]]), "objective names 'top', a metric whose value is a list per row; a list cannot be minimized"),
        (lambda fit: fit["early_stop"].update(metric="top"), "early_stop watches 'top', a metric whose value is a list per row; a list cannot be watched"),
    ],
    ids=["objective", "early stop"],
)
def test_a_list_metric_is_not_something_to_minimize_or_watch(edit, message):
    das = json.loads((READOUT.parent / "das.json").read_text())
    das["steps"]["fit"]["steps"]["top"] = {"kind": "metric", "metric": "top_k", "of": "patched.logits"}
    edit(das["steps"]["fit"])
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(das)


@pytest.mark.parametrize("k", [True, "3", 0])
def test_k_is_a_strict_positive_integer(readout, k):
    readout["steps"]["top3"]["k"] = k
    with pytest.raises(ValidationError, match="k"):
        Spec.model_validate(readout)


def test_k_is_no_more_tokens_than_the_vocabulary_has(readout, data_root, model_engine):
    readout["steps"]["top3"]["k"] = 10**6
    with pytest.raises(plan.PlanError, match=r"metric 'top3': k=1000000 is more tokens than this 32000-token vocabulary"):
        plan.build_request(readout, data_root, model_engine)


def test_explain_prints_every_read_and_number_a_metric_takes(readout, data_root, model_engine):
    from causalab_mini.plan.explain import explain

    readout["steps"]["kl"] = {"kind": "metric", "metric": "kl", "of": "clean.logits", "against": "clean.logits"}
    text = explain(plan.build_request(readout, data_root, model_engine))
    assert "metric top_k(clean.logits, k=3)" in text
    assert "metric kl(clean.logits, clean.logits)" in text


def test_an_id_is_scored_as_the_token_it_is(readout, data_root, model_engine, tmp_path):
    """The same answer, spelled after a space and given as its id, is the
    same probability to the bit."""
    root = tmp_path / "data"
    rows = json.loads((data_root / "weekdays" / "train.json").read_text())
    ids = [tokens.token_id(model_engine.tokenizer, row["base_answer"], "space_prefixed") for row in rows]
    (root / "weekdays").mkdir(parents=True)
    (root / "weekdays" / "train.json").write_text(json.dumps([dict(row, answer_id=one) for row, one in zip(rows, ids)]))
    readout["steps"]["p_answer"].update(token_form="space_prefixed")
    readout["steps"]["by_id"] = {"kind": "metric", "metric": "token_prob", "of": "clean.logits",
                                 "token": "prompts.answer_id", "token_form": "id"}
    readout["steps"]["saves"] = ["p_answer", "by_id"]
    executed = _run(readout, root, model_engine)
    assert torch.equal(executed.result("p_answer"), executed.result("by_id"))


def test_bare_is_the_unprefixed_spelling(gpt2_engine):
    """Where a tokenizer spells a word differently at the start of a text —
    GPT-2's does — `bare` is that spelling and `space_prefixed` the other."""
    tokenizer = gpt2_engine.tokenizer
    assert tokens.token_id(tokenizer, " 2", "bare") == tokenizer.encode("2")[0]
    assert tokens.token_id(tokenizer, "2", "space_prefixed") == tokenizer.encode(" 2")[0]
    assert tokenizer.encode("2") != tokenizer.encode(" 2")


def test_an_id_outside_the_vocabulary_is_refused(model_engine):
    with pytest.raises(tokens.TokenError, match="not a token id"):
        tokens.token_id(model_engine.tokenizer, 10**9, "id")


@pytest.fixture(scope="module")
def gpt2_engine():
    from conftest import model_block

    from causalab_mini.engine import NNterpEngine

    return NNterpEngine.load(model_block(READOUT.parent / "gpt2_reach.json"), device_map="cpu", dispatch=False)
