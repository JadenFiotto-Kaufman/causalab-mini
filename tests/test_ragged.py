"""Ragged positions: windows whose width varies by row, and rows with none.

`{"all": true}` is every content token; `{"column": "c"}` is the tokens of a
row's own text, located in its prompt — one token for ` Friday`, three for
` Thursday` on this tokenizer. A ragged read comes back flat, `(total,
width)`, with the run's own positions saying where each row is. A row whose
text is not in its prompt has an empty window: an excluded measurement,
still a row. A write may not have one, and the ragged write policy is
`refuse` — at the write, where the positions are.

What the forms resolve to is `tests/test_locate.py`; what a ragged form
does to a tensor, to a document and to a refusal is here.
"""

import json
import pathlib
import shutil

import pytest
import torch
from conftest import of_kind, provenance

from causalab_mini import ops, plan
from causalab_mini.data import rows as rows_module
from causalab_mini.engine import steps
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.explain import explain
from causalab_mini.plan.spec_v2 import Spec
from causalab_mini.shapes import Anchor, Where

REPO = pathlib.Path(__file__).resolve().parents[1]
ENTITY = REPO / "documents" / "v2" / "entity_mean_ablation.json"


@pytest.fixture
def entity_raw():
    return json.loads(ENTITY.read_text())


# --------------------------------------------------------------------- #
# resolving
# --------------------------------------------------------------------- #


def test_a_column_is_a_text_anchor_and_lands_token_by_token(entity_raw, data_root, model_engine):
    """` Thursday` is three tokens on this tokenizer and ` Friday` one, so
    the same column gives windows of different widths on different rows —
    which is what ragged means. The plan carries the *text* per row, and the
    run turns each one into indices."""
    built = plan.build_request(entity_raw, data_root, model_engine)
    forward = of_kind(built, plan.Forward)[0]
    read = forward.taps[0].reads[0]
    assert read.at.anchors == ("Thursday", "Friday", "Saturday", "Sunday")
    assert read.at.where == Where(all=True, scope=Anchor(variable="entity"))
    assert read.at.flat

    _, positions = steps.located(model_engine, forward)
    assert [len(one) for one in positions["harvest.acts"]["rows"]] == [3, 1, 1, 1]
    assert ops.is_ragged(positions["harvest.acts"]["rows"])


def test_a_dotted_field_reaches_into_a_dict_inside_a_list():
    row = {"counterfactual_inputs_variables": [{"entity": "Saturday"}]}
    assert rows_module.field_text(row, "counterfactual_inputs_variables[0].entity") == "Saturday"


# --------------------------------------------------------------------- #
# the tensors
# --------------------------------------------------------------------- #


def test_a_ragged_gather_is_flat_and_skips_empty_rows():
    tensor = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)
    windows = ((1, 2, 3), (), (4,))
    flat = ops.gather(tensor, windows)
    assert flat.shape == (4, 2)  # 3 + 0 + 1 positions, in row order
    assert torch.equal(flat[:3], tensor[0, 1:4]) and torch.equal(flat[3], tensor[2, 4])

    out = ops.scatter(tensor, windows, torch.zeros(4, 2))
    assert torch.equal(ops.gather(out, windows), torch.zeros(4, 2))
    assert torch.equal(out[1], tensor[1]), "the excluded row is untouched"


def test_a_ragged_mean_is_one_vector_that_lands_in_any_window():
    tensor = torch.ones(2, 6, 3)
    mean = torch.full((3,), 5.0)
    out = ops.apply_write(tensor, ((5,), (5,)), mean)
    assert torch.equal(ops.gather(out, ((5,), (5,))), torch.full((2, 1, 3), 5.0))
    out = ops.apply_write(tensor, ((1, 2, 3), (2,)), mean)  # a ragged target, too
    assert torch.equal(ops.gather(out, ((1, 2, 3), (2,))), torch.full((4, 3), 5.0))


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


def test_entity_mean_ablation_runs_and_both_engines_agree(entity_raw, data_root, model_engine):
    """The mean over every entity token in the corpus — a ragged read — is
    one vector, swapped in at the answer position."""
    built = plan.build_request(entity_raw, data_root, model_engine)
    read = of_kind(built, plan.Forward)[0].taps[0].reads[0]
    assert read.at.flat, "a text anchor gathers flat whatever the rows turn out to be"
    assert "pos={all scope:{variable:entity}}" in explain(built)

    traced = model_engine.execute(built)
    assert tuple(traced.result("mean").shape) == (16,)
    clean = traced.result("clean.logit_diff")
    ablated = traced.result("ablated.logit_diff")
    assert not torch.equal(clean, ablated)

    hooks = HooksEngine.load(Spec.model_validate(entity_raw).model, device_map="cpu")
    hooked = hooks.execute(plan.build_request(entity_raw, data_root, hooks))
    assert torch.equal(traced.result("mean"), hooked.result("mean")), "the ragged harvest is exact"
    # The ablated logits differ by 7.45e-09 — one write, one position, one
    # vector the two engines agree on to the bit. So the cross-engine ulp
    # (FINDINGS §8) is not about how many positions are replaced, as that
    # section first inferred; it depends on the values written. Zero and the
    # unit-window mean were exact. The tolerance is the measurement.
    other = hooked.result("ablated.logit_diff")
    assert torch.allclose(ablated, other, rtol=0, atol=1e-7)
    assert (ablated - other).abs().max() < 2e-8


# --------------------------------------------------------------------- #
# what is refused, and where
# --------------------------------------------------------------------- #


def test_a_metric_may_not_read_a_ragged_window(entity_raw):
    entity_raw["interventions"]["clean"]["reads"]["logits"]["pos"] = {"all": True}
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="a window of varying positions"):
        Spec.model_validate(entity_raw)


def test_a_write_with_nothing_to_write_on_a_row_is_refused_at_the_write(
    entity_raw, data_root, model_engine
):
    """A read may skip a row; a write may not — writing nothing somewhere is
    not an intervention, and the row would score as if it were.

    The refusal is at the write now, not at compile time: which rows a text
    anchor is in is a question about the model's own tokenization, so the
    compiler cannot answer it and no longer pretends to."""
    entity_raw["interventions"]["ablated"]["writes"]["ablate"]["pos"] = {
        "all": True, "scope": {"variable": "label"}
    }
    # `label` is " Sunday" etc. — an answer, never in the prompt
    built = plan.build_request(entity_raw, data_root, model_engine)
    with pytest.raises(plan.PlanError, match="has nothing to write on row"):
        model_engine.execute(built)


def test_an_entity_patch_between_rows_of_different_widths_is_refused_by_row(data_root, model_engine):
    """The realistic case: swap the counterfactual's entity span into the
    base's. ` Thursday` is three tokens and ` Saturday` one, so on those rows
    the windows differ and no landing policy exists here. Refused, naming
    the rows — the protocol's `exact_length_buckets` and `padded_masked` are
    what would make it land."""
    raw = json.loads((REPO / "documents" / "v2" / "patching.json").read_text())
    reads = raw["interventions"]["patching"]["reads"]
    entity = {"all": True, "scope": {"variable": "entity"}}
    reads["v_cf"]["pos"] = entity
    raw["interventions"]["patching"]["writes"]["patch"]["pos"] = entity
    built = plan.build_request(raw, data_root, model_engine)
    with pytest.raises(plan.PlanError, match="rows \\[.*\\] differ.*exact_length_buckets"):
        model_engine.execute(built)


def test_an_unreduced_ragged_output_cannot_be_an_operand(entity_raw, data_root, model_engine):
    entity_raw["steps"]["harvest"]["outputs"]["mean"] = {"read": "acts"}  # not reduced
    entity_raw["steps"]["harvest"]["saves"] = []
    with pytest.raises(plan.PlanError, match="unreduced; the windows must match, or reduce the output"):
        plan.build_request(entity_raw, data_root, model_engine)


# --------------------------------------------------------------------- #
# and what a scoped anchor makes runnable instead
# --------------------------------------------------------------------- #


@pytest.fixture
def patch_raw():
    return json.loads((REPO / "documents" / "v2" / "entity_patch.json").read_text())


def _holed(data_root, tmp_path, entity="Neptune", drop=None):
    """weekdays/train with row 1's entity replaced by a word that is not in
    its prompt — or with that row taken out altogether."""
    root = tmp_path / "data"
    shutil.copytree(data_root, root)
    path = root / "weekdays" / "train.json"
    table = json.loads(path.read_text())
    if drop is None:
        table[1]["entity"] = entity
    else:
        table.pop(drop)
    path.write_text(json.dumps(table))
    return root


def test_an_entity_patch_at_the_last_token_of_the_entity_lands_on_every_row(
    patch_raw, data_root, model_engine
):
    """The reversal this whole vocabulary is for.

    The same interchange refused above — each row's entity swapped in from
    its counterfactual — runs, because `{"index": -1}` *inside* the entity is
    one token on every row however many tokens the entity is. One spec, two
    roles, and each role resolves its own row's text: the base patches the
    piece `day` of ` Thursday` where the counterfactual read ` Saturday`
    whole.
    """
    executed = model_engine.execute(plan.build_request(patch_raw, data_root, model_engine))
    where = {
        **executed.step("original", plan.Forward).results["positions"],
        **executed.step("patched", plan.Forward).results["positions"],
    }

    assert where["patch"]["tokens"] == ("'day'", "' Friday'", "' Saturday'", "' Sunday'")
    assert where["v_cf"]["tokens"] == ("' Saturday'", "' Sunday'", "'day'", "' Friday'")
    # One token per row on both sides — which is why the swap lands at all.
    assert [len(one) for one in where["patch"]["rows"]] == [1] * 4
    assert set(where["patch"]["reason"]) == {""}
    # These prompts differ only in the entity and are padded on the left, so
    # the *index* coincides while the token addressed does not. Where the
    # tail varies, so does the index — tests/test_locate.py.
    assert executed.result("logit_diff").shape == (4,)


def test_the_two_engines_patch_the_same_entity_to_the_bit(patch_raw, data_root, model_engine):
    """The resolver is engine-agnostic: it is the model's tokenizer that
    answers, and both engines hold the same one."""
    hooks = HooksEngine.load(Spec.model_validate(patch_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(patch_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(patch_raw, data_root, hooks))

    assert provenance(traced) == provenance(hooked)
    assert torch.equal(traced.result("iia"), hooked.result("iia"))
    assert torch.allclose(traced.result("logit_diff"), hooked.result("logit_diff"), rtol=0, atol=1e-7)


def test_a_row_whose_entity_is_not_in_its_prompt_refuses_the_write_by_name(
    patch_raw, data_root, tmp_path, model_engine
):
    """And taking that row out is all it takes: the others score what they
    scored, because a row is a row of a batch and nothing about the swap
    depended on it."""
    holed = _holed(data_root, tmp_path)
    with pytest.raises(plan.PlanError, match=r"has nothing to write on row\(s\) .1: 'alignment_missing'."):
        model_engine.execute(plan.build_request(patch_raw, holed, model_engine))

    whole = model_engine.execute(plan.build_request(patch_raw, data_root, model_engine))
    without = model_engine.execute(
        plan.build_request(patch_raw, _holed(data_root, tmp_path / "b", drop=1), model_engine)
    )
    assert torch.equal(without.result("logit_diff"), whole.result("logit_diff")[[0, 2, 3]])


def test_a_text_anchor_resolves_the_same_way_through_the_serialized_path(
    patch_raw, data_root, model_engine
):
    """The claim the whole design rests on: the block reaches the *model's*
    tokenizer, and a remote run has no second code path.

    `remote="local"` serializes the session, hides this project's modules and
    deserializes against the persistent objects a server would supply — which
    is how `model.tokenizer` resolves to the served checkpoint's own. If that
    did not hold, an anchored document would come back with different
    positions, or with none. It comes back with the same ones, and the same
    numbers. (A run against a real NDIF deployment is still outstanding.)
    """
    here = model_engine.execute(plan.build_request(patch_raw, data_root, model_engine))
    shipped = model_engine.execute(
        plan.build_request(patch_raw, data_root, model_engine), remote="local"
    )
    assert provenance(here) == provenance(shipped)
    assert shipped.step("patched", plan.Forward).results["positions"]["patch"]["tokens"] == (
        "'day'", "' Friday'", "' Saturday'", "' Sunday'"
    )
    assert torch.equal(here.result("logit_diff"), shipped.result("logit_diff"))


def test_a_write_refusal_says_which_reason_each_row_had(patch_raw, data_root, tmp_path, model_engine):
    """"Nothing to write" has three causes and they want three different
    fixes. A value that is in the prompt twice is not a value that is
    missing, and the fix for it — scope the anchor — is the one the message
    used to hide."""
    root = tmp_path / "data"
    shutil.copytree(data_root, root)
    path = root / "weekdays" / "train.json"
    table = json.loads(path.read_text())
    table[1]["entity"] = "day"  # in "today" and in the weekday name: twice over
    path.write_text(json.dumps(table))

    with pytest.raises(plan.PlanError, match=r"row\(s\) .1: 'alignment_ambiguous'."):
        model_engine.execute(plan.build_request(patch_raw, root, model_engine))
