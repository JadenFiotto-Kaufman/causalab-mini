"""Positions: one index, or a window of the same width on every row.

A resolved position is a window per row, `((10,), (10,))` for the unit case,
so a read is always `(rows, w, width)` — a rectangle. Every form here keeps
it one, which is the cheap half of the position story; `{"all": true}` and
per-row variables would not, and are refused until reads can be ragged.

The claim that decides whether a window is the right abstraction: patching
the last three tokens in ONE write must equal patching them in three, to the
bit. `documents/v2/window_patch.json` and
`documents/multi_position_patch_cpu.json` are that pair.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import ops, plan
from causalab_mini.data import encoding
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
WINDOW = REPO / "documents" / "v2" / "window_patch.json"
THREE = REPO / "documents" / "multi_position_patch_cpu.json"


@pytest.fixture
def batch(model):
    # left-padded to width 11: row 0 has 11 content tokens, row 1 has 9
    return encoding.encode(
        model.tokenizer, ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"]
    )


# --------------------------------------------------------------------- #
# resolving
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pos, expected",
    [
        (-1, ((10,), (10,))),
        ({"index": -1}, ((10,), (10,))),
        (0, ((0,), (2,))),
        ({"last": 3}, ((8, 9, 10), (8, 9, 10))),
        ({"span": [0, 2]}, ((0, 1), (2, 3))),
        ({"span": [-3, -1]}, ((8, 9), (8, 9))),
    ],
    ids=["-1", "index", "0", "last 3", "span from the start", "negative span"],
)
def test_every_form_is_content_relative_and_uniform(batch, pos, expected):
    """Row 1 starts at index 2 because of its padding, and every form lands
    on its content the same way — which is why `0` is `(0, 2)`."""
    assert encoding.positions(batch, pos) == expected
    assert encoding.width_of(pos) == len(expected[0])


@pytest.mark.parametrize(
    "pos, message",
    [
        ({"last": 12}, "outside the row's content"),
        ({"span": [-1, -3]}, "outside the row's content"),
        ({"span": [0, 0]}, "outside the row's content"),
        ({"all": True}, "ragged reads are not implemented"),
        ({"variable": "entity"}, "not a form this slice runs"),
        ("last", "not a form this slice runs"),
    ],
    ids=["too wide", "backwards", "empty", "all", "variable", "not a form"],
)
def test_what_a_window_may_not_be(batch, pos, message):
    with pytest.raises(encoding.EncodingError, match=message):
        encoding.positions(batch, pos)


# --------------------------------------------------------------------- #
# the tensors
# --------------------------------------------------------------------- #


def test_gather_and_scatter_carry_the_window_axis():
    tensor = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    window = ((1, 2, 3), (2, 3, 4))
    gathered = ops.gather(tensor, window)
    assert gathered.shape == (2, 3, 3)
    assert torch.equal(gathered[0], tensor[0, 1:4]) and torch.equal(gathered[1], tensor[1, 2:5])

    out = ops.scatter(tensor, window, torch.zeros(2, 3, 3))
    assert torch.equal(ops.gather(out, window), torch.zeros(2, 3, 3))
    assert torch.equal(out[0, 0], tensor[0, 0]) and torch.equal(out[0, 4], tensor[0, 4])


def test_a_published_mean_broadcasts_into_a_window():
    """A `(w, width)` mean — rows averaged away — swaps into `(rows, w,
    width)` without anyone reshaping it. The unit case is `(1, width)`."""
    tensor = torch.zeros(4, 6, 3)
    out = ops.apply_write(tensor, ((5,), (5,), (5,), (5,)), torch.ones(1, 3))
    assert torch.equal(ops.gather(out, ((5,), (5,), (5,), (5,))), torch.ones(4, 1, 3))


# --------------------------------------------------------------------- #
# the document, and the claim
# --------------------------------------------------------------------- #


def test_one_window_write_equals_three_single_writes_to_the_bit(data_root, model_engine):
    """The test of what a window is. `window_patch.json` swaps -4, -3, -2 in
    one write; `multi_position_patch_cpu.json` swaps them in three. Same rows,
    same site, same operand tokens — same logits.

    (The first draft of the document said `{"last": 3}`, which is -3, -2, -1,
    and this test caught it at 0.0036 — which is what it is for.)"""
    one = model_engine.execute(
        plan.build_request(json.loads(WINDOW.read_text()), data_root, model_engine)
    )
    three = model_engine.execute(
        plan.build_request(json.loads(THREE.read_text()), data_root, model_engine)
    )
    assert torch.equal(one.result("logit_diff"), three.result("logit_diff"))


def test_the_window_document_agrees_across_engines_to_the_ulp(data_root, model_engine):
    """FINDINGS §8, sharpened. The three-write document diverges across
    engines by 1.49e-08 on one row; this one-write document diverges by
    exactly the same amount on the same row, and the single-position
    document is exact. So it is about how many positions are replaced at an
    address, not how many writes do it — and it is nnsight's, downstream of
    the replacement. The tolerance is the measurement."""
    raw = json.loads(WINDOW.read_text())
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(raw, data_root, hooks))
    a, b = traced.result("logit_diff"), hooked.result("logit_diff")
    assert torch.allclose(a, b, rtol=0, atol=1e-7)
    assert (a - b).abs().max() < 2e-8


def test_the_window_shows_in_the_plan(data_root, model_engine):
    from causalab_mini.plan.explain import explain

    built = plan.build_request(json.loads(WINDOW.read_text()), data_root, model_engine)
    write = built.step("score", plan.Observe).forwards[1].taps[0].writes[0]
    assert write.positions == ((7, 8, 9),) * 4  # -4, -3, -2 of an 11-token row
    assert "pos=[7:10], [7:10], [7:10], [7:10]" in explain(built)


# --------------------------------------------------------------------- #
# what the compiler refuses
# --------------------------------------------------------------------- #


def test_a_metric_reads_one_position():
    raw = json.loads(WINDOW.read_text())
    raw["interventions"]["window"]["reads"]["logits"]["pos"] = {"last": 2}
    with pytest.raises(ValidationError, match="a window of 2 positions; a metric scores one position"):
        Spec.model_validate(raw)


def test_a_write_and_its_operand_cover_the_same_window():
    raw = json.loads(WINDOW.read_text())
    raw["interventions"]["window"]["writes"]["patch"]["pos"] = {"last": 2}
    with pytest.raises(ValidationError, match="covers 2 position\\(s\\) but its operand 'v_cf' was read over 3"):
        Spec.model_validate(raw)


def test_a_referenced_output_must_match_the_window_too(data_root, model_engine):
    raw = json.loads((REPO / "documents" / "v2" / "mean_ablation.json").read_text())
    raw["interventions"]["ablated"]["writes"]["ablate"]["pos"] = {"last": 2}
    with pytest.raises(plan.PlanError, match="covers 2 position\\(s\\) but 'mean' was read over 1"):
        plan.build_request(raw, data_root, model_engine)
