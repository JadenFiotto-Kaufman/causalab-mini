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
    assert [row[0][0] for row in rows] == [model_engine.tokenizer.decode([one]) for one in top.indices.tolist()]

    executed.write(tmp_path)
    table = json.loads((tmp_path / "top3.json").read_text())
    assert [one["value"] for one in table] == rows
    assert {one["unit"] for one in table} == {"[token, probability] list"}


def test_a_list_metric_is_not_something_to_minimize(readout):
    das = json.loads((READOUT.parent / "das.json").read_text())
    das["steps"]["fit"]["steps"]["top"] = {"kind": "metric", "metric": "top_k", "of": "patched.logits"}
    das["steps"]["fit"]["objective"] = [[1.0, "top"]]
    with pytest.raises(ValidationError, match="objective names 'top'"):
        Spec.model_validate(das)


def test_an_id_is_scored_as_the_token_it_is(readout, data_root, model_engine, tmp_path):
    """The same answer, spelled after a space and given as its id, is the
    same probability to the bit."""
    root = tmp_path / "data"
    rows = json.loads((data_root / "weekdays" / "train.json").read_text())
    ids = [tokens.token_id(model_engine.tokenizer, row["base_answer"], "space_prefixed") for row in rows]
    (root / "weekdays").mkdir(parents=True)
    (root / "weekdays" / "train.json").write_text(json.dumps([dict(row, answer_id=one) for row, one in zip(rows, ids)]))
    readout["steps"]["p_answer"].update(token_form="space_prefixed")
    readout["steps"]["by_id"] = {"kind": "metric", "metric": "soft_accuracy", "of": "clean.logits",
                                 "expected": "prompts.answer_id", "token_form": "id"}
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
