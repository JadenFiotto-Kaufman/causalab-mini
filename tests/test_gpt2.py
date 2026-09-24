"""GPT-2 as a reach-only probe: does the *engine* generalize to a second family?

Not the metrics, and certainly not the results — the weights are random. The
question is whether `address.py` needed a family axis to read and write at
`block_output`, `lm_head` and the `attention_query` interior on a model whose
real module tree shares no path segment with the Llama's.
"""

import copy
import json
import pathlib

import pytest
import torch
from conftest import of_kind

from causalab_mini import ops, plan
from causalab_mini.data import tokens

from causalab_mini.address import Address
from causalab_mini.engine import steps
from causalab_mini.plan import document
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.nnterp import engine as nnterp

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "documents" / "gpt2_cpu.json"


@pytest.fixture(scope="session")
def gpt2_engine():
    """An nnterp engine holding the tiny random GPT-2 the document pins."""
    return NNterpEngine.load(document.Document.load(DOCUMENT).model, device_map="cpu")


@pytest.fixture(scope="session")
def gpt2(gpt2_engine):
    return gpt2_engine.model


@pytest.fixture
def gpt2_raw():
    return json.loads(DOCUMENT.read_text())


def _build(raw, data_root, engine):
    return plan.build_request(raw, data_root, engine)


def _at(raw, component, layer):
    """The same document with its written site moved to another component."""
    raw = copy.deepcopy(raw)
    site = {"component": component}
    if layer is not None:
        site["layers"] = [layer]
    raw["method"]["sites"]["target"] = site
    return raw


def _no_write(raw):
    raw = copy.deepcopy(raw)
    del raw["method"]["reads"]["v_cf"], raw["method"]["writes"], raw["method"]["intervened_models"]
    raw["method"]["reads"]["logits"]["model"] = "original"
    for entry in raw["method"]["save"]:
        entry["model"] = "original"
    return raw


def _identity_write(raw):
    raw = copy.deepcopy(raw)
    raw["method"]["reads"]["v_cf"]["input"] = "base"
    return raw


# --------------------------------------------------------------------- #
# the headline: one address table, two families
# --------------------------------------------------------------------- #


def test_one_address_serves_both_families(model_engine, gpt2_engine, model, gpt2):
    """The same `Address` — the same component, layer and resolved operation —
    reaches both models, although the modules it lands on share no path."""
    for component, layer in (("block_output", 0), ("lm_head", None), ("attention_query", 0)):
        assert model_engine.locate(component, layer).where == gpt2_engine.locate(component, layer).where

    # What nnterp is absorbing on our behalf, spelled out: these are the real
    # paths, and nothing in the project mentions either of them.
    assert model.attentions[0].path == "model.model.layers.0.self_attn"
    assert gpt2.attentions[0].path == "model.transformer.h.0.attn"
    assert model.layers[0].path == "model.model.layers.0"
    assert gpt2.layers[0].path == "model.transformer.h.0"


def test_the_interior_operation_is_the_same_call_on_both_families(model_engine, gpt2_engine, model, gpt2):
    """It is the same name for the same reason — transformers 5.17 spells both
    forwards' dispatch identically — not by luck of an occurrence count: on
    GPT-2 the call sits in an `else` branch, under two more assignments and a
    second candidate implementation, and the suffix still lands on 1 because a
    call-op suffix counts calls of one symbol."""
    assert gpt2_engine.locate("attention_query", 0).op == "attention_interface_1"
    assert model_engine.locate("attention_query", 0).op == "attention_interface_1"
    gpt2_ops = set(gpt2.attentions[0].source.names)
    assert "self__upcast_and_reordered_attn_0" in gpt2_ops  # the branch Llama has not
    assert "self__upcast_and_reordered_attn_0" not in set(model.attentions[0].source.names)


# --------------------------------------------------------------------- #
# reads and writes land, at every component
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "component, layer",
    [("block_output", 2), ("lm_head", None), ("attention_query", 2)],
)
def test_a_write_lands_and_an_identity_write_does_not_move_anything(gpt2_engine, 
    gpt2_raw, data_root, gpt2, component, layer
):
    raw = _at(gpt2_raw, component, layer)
    clean = gpt2_engine.execute(_build(_no_write(raw), data_root, gpt2_engine))
    identity = gpt2_engine.execute(_build(_identity_write(raw), data_root, gpt2_engine))
    swapped = gpt2_engine.execute(_build(raw, data_root, gpt2_engine))

    assert torch.equal(identity.result("logit_diff"), clean.result("logit_diff"))
    assert not torch.equal(swapped.result("logit_diff"), clean.result("logit_diff"))


def test_a_swap_at_the_head_makes_the_patched_run_score_the_counterfactual(gpt2_engine, 
    gpt2_raw, data_root, gpt2
):
    """The write at `lm_head` has a value we can name: the logits read in the
    patched model are, bit for bit, the counterfactual forward's."""
    swapped = gpt2_engine.execute(_build(_at(gpt2_raw, "lm_head", None), data_root, gpt2_engine))

    # The same run with no write, over the counterfactual prompts as its base.
    counterfactual = _no_write(gpt2_raw)
    counterfactual["data"]["base"]["field"] = counterfactual["data"]["counterfactual"]["field"]
    reference = gpt2_engine.execute(_build(counterfactual, data_root, gpt2_engine))

    assert torch.equal(swapped.result("logit_diff"), reference.result("logit_diff"))


def test_the_engines_head_read_is_the_models_own_logits_here_too(gpt2, gpt2_raw, data_root, gpt2_engine):
    """FINDINGS §1.2: reading the head *module's* output and reading the model's
    returned logits are two taps, equal only where nothing sits between them.
    They are equal on this family as well."""
    built = _build(gpt2_raw, data_root, gpt2_engine)
    results = gpt2_engine.execute(built)

    source, patched = of_kind(built, plan.Forward)
    with gpt2.trace(nnterp.batch(source)):
        v_cf = gpt2.layers_output[2][:, -1, :].clone().save()
    with gpt2.trace(nnterp.batch(patched)):
        gpt2.layers_output[2][:, -1, :] = v_cf
        logits = gpt2.logits[:, -1, :].clone().save()

    rows = torch.arange(4)
    a, b = of_kind(built, plan.Metric)[0].ids
    assert torch.equal(
        results.result("logit_diff"), logits[rows, torch.tensor(a)] - logits[rows, torch.tensor(b)]
    )


def test_the_document_runs_end_to_end(tmp_path, gpt2_raw, data_root, gpt2_engine):
    built = _build(gpt2_raw, data_root, gpt2_engine)
    written = gpt2_engine.execute(built).write(tmp_path)
    assert sorted(path.name for path in written) == ["document.json", "logit_diff.json", "run.json"]
    assert len(json.loads((tmp_path / "logit_diff.json").read_text())) == 4


# --------------------------------------------------------------------- #
# why this family needed its own document
# --------------------------------------------------------------------- #


def test_the_shipped_weekdays_answers_are_not_single_tokens_here(gpt2_engine, gpt2, minimal_raw, data_root):
    """The whole reason for documents/data/counting: the engine reaches
    everything on this model, and the *metric* still cannot run on the shipped
    rows."""
    assert gpt2.tokenizer.encode(" Friday", add_special_tokens=False) == [304, 82, 271, 288]
    with pytest.raises(tokens.TokenError, match="is 4 tokens"):
        plan.build(document.Document.from_json(minimal_raw), data_root, gpt2_engine)


def test_token_form_is_load_bearing_on_this_tokenizer_and_inert_on_the_llamas(gpt2, model):
    """FINDINGS §1.8 measured that `" Friday"` and `"Friday"` are one id on the
    tiny Llama, so nothing there can tell a `space_prefixed` from a `bare`. Here
    they differ, and the document's `token_form` decides which token a metric
    scores."""
    bare = gpt2.tokenizer.encode("two", add_special_tokens=False)
    assert tokens.token_id(gpt2.tokenizer, " two", "space_prefixed") not in bare
    assert tokens.token_id(model.tokenizer, " Friday", "space_prefixed") == model.tokenizer.encode(
        "Friday", add_special_tokens=False
    )[0]


def test_the_query_at_layer_0_carries_nothing_a_prompt_pair_differs_in(gpt2_engine, 
    gpt2_raw, data_root, gpt2
):
    """Not a family fact — a fact about what the address means, and a trap. The
    layer-0 query is a function of the last token and its position alone, so on
    four prompts of equal length ending in the same token it is the same tensor
    on every row, and an interchange there is a no-op that looks like a broken
    write. It is the reason the parametrized case above patches layer 2."""
    raw = _at(gpt2_raw, "attention_query", 0)
    clean = gpt2_engine.execute(_build(_no_write(raw), data_root, gpt2_engine))
    swapped = gpt2_engine.execute(_build(raw, data_root, gpt2_engine))
    assert torch.equal(swapped.result("logit_diff"), clean.result("logit_diff"))

    built = _build(raw, data_root, gpt2_engine)
    source, _ = steps.located(gpt2_engine, of_kind(built, plan.Forward)[0])
    tap = source.taps[0]
    with gpt2.trace(nnterp.batch(source)):
        query = ops.gather(
            nnterp.read(gpt2, tap.address), tap.reads[0].at.positions, tap.address.seq_axis
        ).clone().save()
    assert {sum(row) for row in source.attention_mask} == {12}
    assert all(torch.equal(query[0], query[row]) for row in range(4))
