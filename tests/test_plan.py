"""The plan: pure data, in forward order, with positions already resolved."""

import dataclasses
import pickle

import pytest

from causalab_mini import document, encoding, plan


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model):
    return plan.build(document.Document.from_json(minimal_raw), data_root, model)


@pytest.fixture
def das_plan(das_raw, data_root, model):
    return plan.build(document.Document.from_json(das_raw), data_root, model)


@pytest.fixture(params=["minimal_plan", "das_plan"])
def any_plan(request):
    """Both documents. A fit adds a featurizer table, a train block and the rows
    of every update it will make — and none of that may make a plan less pure."""
    return request.getfixturevalue(request.param)


# --------------------------------------------------------------------- #
# a plan is pure data
# --------------------------------------------------------------------- #

ALLOWED = (str, int, float, bool, type(None), tuple)
PLAN_TYPES = {"causalab_mini.plan", "causalab_mini.address"}


def _walk(value, where="plan"):
    """Every value reachable from a plan, with the path that reached it."""
    yield where, value
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _walk(getattr(value, field.name), f"{where}.{field.name}")
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk(item, f"{where}[{index}]")


def test_a_plan_holds_strings_and_integers_and_nothing_else(any_plan):
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
    assert [(f.name, f.input) for f in minimal_plan.forwards] == [
        ("original", "counterfactual"),
        ("patched", "base"),
    ]

    original, patched = minimal_plan.forwards
    assert [(tap.address.path, tap.address.side) for tap in original.taps] == [
        ("layers.0", "output")
    ]
    assert [read.name for read in original.taps[0].reads] == ["v_cf"]
    assert original.taps[0].writes == ()

    # forward order: the write at layer 0 goes above the read at the head.
    assert [(tap.address.path, tap.address.side) for tap in patched.taps] == [
        ("layers.0", "output"),
        ("lm_head", "output"),
    ]
    assert [write.name for write in patched.taps[0].writes] == ["patch"]
    assert patched.taps[0].writes[0].operand == "v_cf"
    assert [read.name for read in patched.taps[1].reads] == ["logits"]


def test_a_write_whose_operand_is_read_in_its_own_model_is_a_cycle(minimal_raw, data_root, model):
    minimal_raw["method"]["reads"]["v_cf"].update(model="patched", input="base")
    with pytest.raises(plan.PlanError, match="cycle"):
        plan.build(document.Document.from_json(minimal_raw), data_root, model)


# --------------------------------------------------------------------- #
# what the plan resolved: positions and token ids
# --------------------------------------------------------------------- #


def test_positions_resolve_to_the_last_real_token_of_each_row(model):
    # Two prompts of different length: the second is two tokens shorter, and
    # this tokenizer pads on the left.
    batch = encoding.encode(
        model.tokenizer,
        ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"],
    )
    assert [len(row) for row in batch.input_ids] == [11, 11]
    assert batch.starts == (0, 2) and batch.ends == (11, 11)

    assert encoding.positions(batch, -1) == (10, 10)  # the last real token
    assert encoding.positions(batch, 0) == (0, 2)  # the first real token of each row
    with pytest.raises(encoding.EncodingError):
        encoding.positions(batch, -12)


def test_the_metric_columns_resolved_to_the_token_ids_notes_measured(minimal_plan, model):
    # NOTES.md §9.1: on this sentencepiece tokenizer " Friday" and "Friday" are
    # the same id, so the space-prefixed form is the bare one.
    assert model.tokenizer.encode(" Friday", add_special_tokens=False) == [28728]
    iia, logit_diff = minimal_plan.metrics
    assert (iia.name, iia.kind, iia.of) == ("iia", "match", "logits")
    # row 0's cf_answer is " Sunday", row 2's is " Friday" (documents/data/weekdays/train.json)
    assert iia.ids == ((16340, 27822, 28728, 24211),)
    assert logit_diff.ids[0] == iia.ids[0]  # `a` is cf_answer too
    assert logit_diff.ids[1] == (28728, 24211, 16340, 27822)  # base_answer


def test_a_multi_token_answer_is_refused(model):
    with pytest.raises(encoding.EncodingError, match="exactly one token"):
        encoding.token_id(model.tokenizer, "Thursday afternoon", "space_prefixed")


def test_a_layer_the_model_does_not_have_is_a_load_error(minimal_raw, data_root, model):
    minimal_raw["method"]["sites"]["target"]["layers"] = [17]
    with pytest.raises(plan.PlanError, match="outside the model's 2 layers"):
        plan.build(document.Document.from_json(minimal_raw), data_root, model)
