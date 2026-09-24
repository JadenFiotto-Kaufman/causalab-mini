"""The plan: pure data, in forward order, with positions already resolved."""

import dataclasses
import pickle

import pytest
from conftest import of_kind

from causalab_mini import plan
from causalab_mini.data import tokens


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model_engine):
    return plan.build_request(minimal_raw, data_root, model_engine)


@pytest.fixture
def das_plan(das_raw, data_root, model_engine):
    return plan.build_request(das_raw, data_root, model_engine)


@pytest.fixture(params=["minimal_plan", "das_plan"])
def any_plan(request):
    """Both documents. A fit adds a featurizer table, a train block and the rows
    of every update it will make — and none of that may make a plan less pure."""
    return request.getfixturevalue(request.param)


# --------------------------------------------------------------------- #
# a plan is pure data
# --------------------------------------------------------------------- #

# bytes: a loaded featurizer's bundle, verbatim — data, and it pickles
ALLOWED = (str, int, float, bool, bytes, type(None), tuple, dict, list)
# `shapes.Selection` is where an op is — positions and feature groups, integers
# all the way down; the walk goes into it like any other node.
PLAN_TYPES = {"causalab_mini.plan.plan", "causalab_mini.address", "causalab_mini.shapes"}


def _walk(value, where="plan"):
    """Every value reachable from a plan, with the path that reached it."""
    yield where, value
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name), f"{where}.{field.name}")
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk(item, f"{where}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{where}[{key!r}]")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{where}[{index}]")


def test_a_fresh_plan_holds_strings_and_integers_and_nothing_else(any_plan):
    """A plan that has *run* holds its results, which are tensors. Before it
    runs, every `results` dict in the tree is empty and nothing in it is a
    tensor, an envoy or a handle — which is what makes it shippable."""
    for where, value in _walk(any_plan):
        if dataclasses.is_dataclass(value):
            # plan.py's own ops, plus Address — which is (component, layer),
            # pure data, and the reason a tap can be sorted and printed.
            assert type(value).__module__ in PLAN_TYPES, where
            continue
        assert isinstance(value, ALLOWED), f"{where} is a {type(value).__name__}"
        # No tensor, no envoy, no model, no tokenizer — named so a failure says
        # which one got in.
        assert not hasattr(value, "shape"), f"{where} looks like a tensor"
        assert not hasattr(value, "_module"), f"{where} looks like a model handle"
        assert not hasattr(value, "requires_grad"), f"{where} carries a graph"


def test_a_plan_pickles_with_plain_pickle(any_plan):
    assert pickle.loads(pickle.dumps(any_plan)) == any_plan


# --------------------------------------------------------------------- #
# the schedule
# --------------------------------------------------------------------- #


def test_the_schedule_is_two_forwards_counterfactual_then_base(minimal_plan):
    """Each step of the document is a step of the plan, under its own name,
    in the order the document wrote them."""
    assert list(minimal_plan.steps) == ["counterfactual", "patched", "iia", "logit_diff"]
    assert [f.input for f in of_kind(minimal_plan, plan.Forward)] == ["pairs", "pairs"]

    original, patched = of_kind(minimal_plan, plan.Forward)
    assert [(tap.address.path, tap.address.io) for tap in original.taps] == [
        ("layers.0", "output")
    ]
    assert [read.name for read in original.taps[0].reads] == ["counterfactual.v_cf"]
    assert original.taps[0].writes == ()

    # forward order: the write at layer 0 goes above the read at the head.
    assert [(tap.address.path, tap.address.io) for tap in patched.taps] == [
        ("layers.0", "output"),
        ("lm_head", "output"),
    ]
    assert [write.name for write in patched.taps[0].writes] == ["patched.patch"]
    assert patched.taps[0].writes[0].operand == "counterfactual.v_cf"
    assert [read.name for read in patched.taps[1].reads] == ["patched.logits"]


# --------------------------------------------------------------------- #
# what the plan resolved: positions and token ids
# --------------------------------------------------------------------- #


def test_the_batch_a_plan_carries_is_padded_to_one_width(model):
    # Two prompts of different length: the second is two tokens shorter, and
    # this tokenizer pads on the left. The plan carries the ids; where along
    # them a read acts is resolved where the model is (tests/test_locate.py).
    ids, mask, sample = tokens.encode(
        model.tokenizer,
        ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"],
    )
    assert [len(row) for row in ids] == [11, 11]
    assert [sum(row) for row in mask] == [11, 9]
    # and what row 0's ids say here, for the run to check its own reading against
    assert sample.endswith("If today is Thursday, tomorrow is")


def test_the_metric_columns_resolved_to_the_token_ids_notes_measured(minimal_plan, model):
    # NOTES.md §9.1: on this sentencepiece tokenizer " Friday" and "Friday" are
    # the same id, so the space-prefixed form is the bare one.
    assert model.tokenizer.encode(" Friday", add_special_tokens=False) == [28728]
    iia, logit_diff = of_kind(minimal_plan, plan.Metric)
    assert (iia.kind, iia.of) == ("match", "patched.logits")
    # row 0's cf_answer is " Sunday", row 2's is " Friday" (documents/data/weekdays/train.json)
    assert iia.ids == ((16340, 27822, 28728, 24211),)
    assert logit_diff.ids[0] == iia.ids[0]  # `a` is cf_answer too
    assert logit_diff.ids[1] == (28728, 24211, 16340, 27822)  # base_answer


def test_a_multi_token_answer_is_refused(model):
    with pytest.raises(tokens.TokenError, match="exactly one token"):
        tokens.token_id(model.tokenizer, "Thursday afternoon", "space_prefixed")


def test_a_layer_the_model_does_not_have_is_a_load_error(minimal_raw, data_root, model_engine):
    minimal_raw["sites"]["target"]["layers"] = 17
    with pytest.raises(plan.PlanError, match="outside the model's 2 layers"):
        plan.build_request(minimal_raw, data_root, model_engine)
