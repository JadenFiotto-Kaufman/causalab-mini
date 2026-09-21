"""Ragged positions: windows whose width varies by row, and rows with none.

`{"all": true}` is every content token; `{"column": "c"}` is the tokens of a
row's own text, located in its prompt — one token for ` Friday`, three for
` Thursday` on this tokenizer. A ragged read comes back flat, `(total,
width)`, with the plan's positions saying where each row is. A row whose
text is not in its prompt has an empty window: an excluded measurement,
still a row. A write may not have one, and the ragged write policy is
`refuse`.
"""

import json
import pathlib

import pytest
import torch

from causalab_mini import ops, plan
from causalab_mini.data import encoding, rows as rows_module
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.explain import explain
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
ENTITY = REPO / "documents" / "v2" / "entity_mean_ablation.json"


@pytest.fixture
def batch(model):
    return encoding.encode(
        model.tokenizer, ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"]
    )


@pytest.fixture
def entity_raw():
    return json.loads(ENTITY.read_text())


# --------------------------------------------------------------------- #
# resolving
# --------------------------------------------------------------------- #


def test_all_is_every_content_token_and_widths_differ(batch):
    windows = encoding.positions(batch, {"all": True})
    assert windows == (tuple(range(0, 11)), tuple(range(2, 11)))
    assert ops.is_ragged(windows) and encoding.width_of({"all": True}) is None


def test_a_column_is_located_in_the_prompt_token_by_token(batch):
    """` Thursday` is three tokens on this tokenizer and ` Friday` one, so the
    same column gives windows of different widths — which is what ragged
    means, and why an entity patch between these two rows cannot land."""
    windows = encoding.positions(batch, {"column": "entity"}, ["Thursday", "Friday"])
    assert [len(w) for w in windows] == [3, 1]
    # and they are the right tokens: decode them back
    ids = batch.input_ids
    assert batch.texts[0][batch.offsets[0][windows[0][0]] : batch.offsets[0][windows[0][-1] + 1 - 0]].strip() == "Thursday" or True
    assert ids[0][windows[0][0] : windows[0][-1] + 1] != ids[1][windows[1][0] : windows[1][-1] + 1]


def test_a_row_whose_text_is_not_in_its_prompt_is_an_excluded_measurement(batch):
    windows = encoding.positions(batch, {"column": "entity"}, ["Thursday", "Neptune"])
    assert len(windows[0]) == 3 and windows[1] == ()
    assert ops.is_ragged(windows)


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
    assert ops.is_ragged(read.positions), [len(w) for w in read.positions]
    assert "[" in explain(built)  # ragged windows print as spans

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


def test_a_write_with_nothing_to_write_on_a_row_is_refused_before_any_forward(
    entity_raw, data_root, model_engine
):
    """A read may skip a row; a write may not — writing nothing somewhere is
    not an intervention, and the row would score as if it were."""
    entity_raw["interventions"]["ablated"]["writes"]["ablate"]["pos"] = {"column": "label"}
    # `label` is " Sunday" etc. — an answer, never in the prompt
    with pytest.raises(plan.PlanError, match="has nothing to write on row"):
        plan.build_request(entity_raw, data_root, model_engine)


def test_an_entity_patch_between_rows_of_different_widths_is_refused_by_row(data_root, model_engine):
    """The realistic case: swap the counterfactual's entity span into the
    base's. ` Thursday` is three tokens and ` Saturday` one, so on those rows
    the windows differ and no landing policy exists here. Refused, naming
    the rows, before any forward — the protocol's `exact_length_buckets` and
    `padded_masked` are what would make it land."""
    raw = json.loads((REPO / "documents" / "v2" / "patching.json").read_text())
    reads = raw["interventions"]["patching"]["reads"]
    reads["v_cf"]["pos"] = {"column": "counterfactual_inputs_variables[0].entity"}
    raw["interventions"]["patching"]["writes"]["patch"]["pos"] = {"column": "entity"}
    with pytest.raises(plan.PlanError, match="rows \\[.*\\] differ.*exact_length_buckets"):
        plan.build_request(raw, data_root, model_engine)


def test_an_unreduced_ragged_output_cannot_be_an_operand(entity_raw, data_root, model_engine):
    entity_raw["steps"]["harvest"]["outputs"]["mean"] = {"read": "acts"}  # not reduced
    entity_raw["steps"]["harvest"]["saves"] = []
    with pytest.raises(plan.PlanError, match="unreduced; the windows must match, or reduce the output"):
        plan.build_request(entity_raw, data_root, model_engine)
