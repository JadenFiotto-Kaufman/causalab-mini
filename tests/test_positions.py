"""Positions: a spec the document writes, and the width it promises.

A document says *where* — `-1`, `{"last": 3}`, `{"span": [a, b)}` — and what
a run resolves that to is a window per row, `((10,), (10,))` for the unit
case, so a read of a form with a fixed width is `(rows, w, width)`: a
rectangle. What the forms mean, row by row, is `tests/test_locate.py`; what
they promise a compiler, and what a document may not write, is here.

The claim that decides whether a window is the right abstraction: patching
the last three tokens in ONE write must equal patching them in three, to the
bit. `documents/v2/window_patch.json` and
`documents/multi_position_patch_cpu.json` are that pair.
"""

import json
import pathlib

import pytest
import torch
from pydantic import TypeAdapter, ValidationError
from conftest import of_kind

from causalab_mini import ops, plan
from causalab_mini.engine import steps
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.spec import Position, Spec
from causalab_mini.shapes import Where

REPO = pathlib.Path(__file__).resolve().parents[1]
WINDOW = REPO / "documents" / "v2" / "window_patch.json"
THREE = REPO / "documents" / "multi_position_patch_cpu.json"


# --------------------------------------------------------------------- #
# what a form promises
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pos, width",
    [
        (-1, 1),
        ({"index": -1}, 1),
        (0, 1),
        ({"last": 3}, 3),
        ({"span": [0, 2]}, 2),
        ({"span": [-3, -1]}, 2),
        ({"all": True}, None),
    ],
    ids=["-1", "index", "0", "last 3", "span from the start", "negative span", "all"],
)
def test_a_forms_width_is_known_without_a_row(pos, width):
    """`width` is what lets the compiler check a write against its operand
    before either has been resolved. `None` is the ragged case, and it is
    decided by the form rather than by the data."""
    where = TypeAdapter(Position).validate_python(pos)
    assert where.width == width
    assert where.ragged == (width is None)


@pytest.mark.parametrize(
    "pos, message",
    [
        ({"span": [-1, -3]}, "is not a forward window"),
        ({"span": [0, 0]}, "is not a forward window"),
        ({"last": 0}, "not a positive number of tokens"),
        ({"index": -1, "last": 2}, "exactly one of index/span/last/all"),
        ({}, "exactly one of index/span/last/all"),
        ({"variable": "entity"}, "Unexpected keyword argument"),
        ("last", "should be a dictionary or an instance of Where"),
        ({"scope": {}, "index": -1}, "names a variable, a segment, or both"),
        ({"index": -1, "scope": {"segment": "eos"}}, "'eos' is a run of the generated frame"),
    ],
    ids=["backwards", "empty", "no tokens", "two cuts", "no cut", "a bare variable",
         "not a form", "an empty anchor", "eos in the prompt"],
)
def test_what_a_position_may_not_be(pos, message):
    """Every one of these is refused by pydantic with the path to the key,
    before a model is loaded — a position is a spec and a spec is checkable."""
    with pytest.raises(ValidationError, match=message):
        TypeAdapter(Position).validate_python(pos)


def test_a_fixed_width_cut_the_row_cannot_fit_is_refused_at_compile_time(
    data_root, model_engine
):
    """`{"index": 40}` names one token on every row, so a row with eleven of
    them is a document that is wrong about its own prompts. The client
    tokenized the rows, so the compiler refuses it — naming the op, the rows
    and the reason — before any model runs."""
    raw = json.loads(WINDOW.read_text())
    raw["steps"]["patched"]["reads"]["logits"]["pos"] = {"index": 40}
    with pytest.raises(plan.PlanError, match=r"read 'patched.logits' at .index:40. has no position"):
        plan.build_request(raw, data_root, model_engine)


def test_a_fixed_width_write_the_row_cannot_fit_is_refused_too(data_root, model_engine):
    """Same rule at a write, refused as a document rather than as a shape
    error inside the seam."""
    raw = json.loads(WINDOW.read_text())
    raw["steps"]["patched"]["interventions"]["writes"]["patch"]["pos"] = {"last": 12}
    raw["steps"]["counterfactual"]["reads"]["v_cf"]["pos"] = {"last": 12}
    with pytest.raises(plan.PlanError, match=r"'counterfactual.v_cf' at .last:12. has no position on row"):
        plan.build_request(raw, data_root, model_engine)


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


def test_the_window_is_a_spec_in_the_plan_and_integers_in_the_run(data_root, model_engine):
    """A fresh plan carries the spec and no integers; the run resolves it
    against its own tokenizer, and gets -4, -3, -2 of an 11-token row."""
    from causalab_mini.plan.explain import explain

    built = plan.build_request(json.loads(WINDOW.read_text()), data_root, model_engine)
    forward = built.step("patched", plan.Forward)
    write = forward.taps[0].writes[0]
    assert write.at.positions == () and write.at.where == Where(span=(-4, -1))
    assert "pos={span:[-4, -1]}" in explain(built)

    ready, positions = steps.located(model_engine, forward)
    assert positions["patched.patch"]["rows"] == ((7, 8, 9),) * 4
    assert ready.taps[0].writes[0].at.positions == positions["patched.patch"]["rows"]


# --------------------------------------------------------------------- #
# what the compiler refuses
# --------------------------------------------------------------------- #


def test_a_metric_reads_one_position():
    raw = json.loads(WINDOW.read_text())
    raw["steps"]["patched"]["reads"]["logits"]["pos"] = {"last": 2}
    with pytest.raises(ValidationError, match="a window of 2 positions; a metric scores one position"):
        Spec.model_validate(raw)


def test_a_write_and_its_operand_cover_the_same_window():
    raw = json.loads(WINDOW.read_text())
    raw["steps"]["patched"]["interventions"]["writes"]["patch"]["pos"] = {"last": 2}
    with pytest.raises(ValidationError, match=r"covers 2 position\(s\) but its operand 'counterfactual.v_cf' was read over 3"):
        Spec.model_validate(raw)


def test_a_referenced_mean_must_match_the_window_too():
    raw = json.loads((REPO / "documents" / "v2" / "mean_ablation.json").read_text())
    raw["steps"]["ablated"]["interventions"]["writes"]["ablate"]["pos"] = {"last": 2}
    with pytest.raises(ValidationError, match="covers 2 position\\(s\\) but the mean 'mean' keeps a window of 1"):
        Spec.model_validate(raw)
