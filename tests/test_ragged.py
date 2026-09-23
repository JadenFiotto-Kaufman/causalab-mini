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

import pytest
import torch

from causalab_mini import ops, plan
from causalab_mini.data import rows as rows_module
from causalab_mini.engine import steps
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.explain import explain
from causalab_mini.plan.spec import Spec
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
    forward = built.step("harvest", plan.Observe).forwards[0]
    read = forward.taps[0].reads[0]
    assert read.at.anchors == ("Thursday", "Friday", "Saturday", "Sunday")
    assert read.at.where == Where(all=True, scope=Anchor(variable="entity"))
    assert read.at.flat

    _, positions = steps.located(model_engine, forward)
    assert [len(one) for one in positions["acts"]] == [3, 1, 1, 1]
    assert ops.is_ragged(positions["acts"])


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
    read = built.step("harvest", plan.Observe).forwards[0].taps[0].reads[0]
    assert read.at.flat, "a text anchor gathers flat whatever the rows turn out to be"
    assert "pos={all scope:{variable:entity}}" in explain(built)

    traced = model_engine.execute(built)
    assert tuple(traced.result("mean").shape) == (16,)
    clean = traced.step("clean", plan.Observe).results["logit_diff"]
    ablated = traced.step("ablated", plan.Observe).results["logit_diff"]
    assert not torch.equal(clean, ablated)

    hooks = HooksEngine.load(Spec.model_validate(entity_raw).model, device_map="cpu")
    hooked = hooks.execute(plan.build_request(entity_raw, data_root, hooks))
    assert torch.equal(traced.result("mean"), hooked.result("mean")), "the ragged harvest is exact"
    # The ablated logits differ by 7.45e-09 — one write, one position, one
    # vector the two engines agree on to the bit. So the cross-engine ulp
    # (FINDINGS §8) is not about how many positions are replaced, as that
    # section first inferred; it depends on the values written. Zero and the
    # unit-window mean were exact. The tolerance is the measurement.
    other = hooked.step("ablated", plan.Observe).results["logit_diff"]
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
    entity_raw["interventions"]["ablated"]["writes"]["ablate"]["pos"] = {"column": "label"}
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
    reads["v_cf"]["pos"] = {"column": "counterfactual_inputs_variables[0].entity"}
    raw["interventions"]["patching"]["writes"]["patch"]["pos"] = {"column": "entity"}
    built = plan.build_request(raw, data_root, model_engine)
    with pytest.raises(plan.PlanError, match="rows \\[.*\\] differ.*exact_length_buckets"):
        model_engine.execute(built)


def test_an_unreduced_ragged_output_cannot_be_an_operand(entity_raw, data_root, model_engine):
    entity_raw["steps"]["harvest"]["outputs"]["mean"] = {"read": "acts"}  # not reduced
    entity_raw["steps"]["harvest"]["saves"] = []
    with pytest.raises(plan.PlanError, match="unreduced; the windows must match, or reduce the output"):
        plan.build_request(entity_raw, data_root, model_engine)
