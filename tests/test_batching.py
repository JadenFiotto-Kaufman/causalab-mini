"""`batch_size`: how many rows one model call may hold.

It is a property of the run, not of the experiment, so it arrives with
`execute` and is absent from the plan. The shared walk does the windowing —
a window of rows is the same model call over fewer rows, handed to an
unchanged `engine.forward` — so every engine gets it, and every step after
the call (metrics, reduces, saves, a fit's loss) sees all its rows.
"""

import json
import pathlib

import pytest
import torch
from conftest import of_kind

from causalab_mini import plan
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import intervene
from causalab_mini.plan import sweep
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2 = REPO / "documents" / "v2"


def _point(name):
    return sweep.points(json.loads((V2 / name).read_text()))[0][1]


def _same_everywhere(whole, windowed):
    """Step by step: two steps may both produce `logit_diff`, and scope is
    what tells them apart."""
    def steps_of(node, path=""):
        for name, child in node.steps.items():
            if isinstance(child, plan.Plan):
                yield from steps_of(child, f"{path}{name}/")
            else:
                yield f"{path}{name}", child
    theirs = dict(steps_of(windowed))
    for where, step in steps_of(whole):
        assert set(step.results) == set(theirs[where].results), where
        for key, value in step.results.items():
            assert _agree(value, theirs[where].results[key]), f"{where}.{key}"


def _agree(a, b):
    """A smaller batch is a different GEMM, so the last bit may move. A
    result that is not a tensor — where the run read, which rows it could
    score — is plain data and must be equal, window for window: that is the
    claim that a window of rows resolves its own positions and concatenates
    back in row order."""
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return a == b
    return a.shape == b.shape and torch.allclose(a.float(), b.float(), rtol=0, atol=1e-6)


# --------------------------------------------------------------------- #
# the two slices
# --------------------------------------------------------------------- #


def test_a_window_of_a_forward_is_the_same_taps_over_fewer_rows(data_root, model_engine):
    built = plan.build_request(_point("window_patch.json"), data_root, model_engine)
    forward = of_kind(built, plan.Forward)[-1]
    small = plan.window(forward, 1, 3)

    assert small.input_ids == forward.input_ids[1:3]
    assert [tap.address for tap in small.taps] == [tap.address for tap in forward.taps]
    for mine, theirs in zip(small.taps, forward.taps):
        for a, b in zip(mine.writes + mine.reads, theirs.writes + theirs.reads):
            assert a.at.positions == b.at.positions[1:3]
            assert (a.name, a.at.groups, a.at.take) == (b.name, b.at.groups, b.at.take)
    assert plan.window(forward, 0, len(forward.input_ids)) == forward


def test_rows_of_a_value_follow_its_layout():
    rectangle = torch.arange(24.0).reshape(4, 2, 3)
    assert torch.equal(intervene.rows(rectangle, ((0, 1),) * 4, 1, 3), rectangle[1:3])

    ragged = ((0, 1), (), (2,), (0, 1, 2))  # 2 + 0 + 1 + 3 positions, flat
    flat = torch.arange(18.0).reshape(6, 3)
    assert torch.equal(intervene.rows(flat, ragged, 1, 3), flat[2:3])
    assert torch.equal(intervene.rows(flat, ragged, 3, 4), flat[3:6])

    mean = torch.ones(3)
    assert intervene.rows(mean, None, 1, 3) is mean, "a value with no rows is shared whole"


# --------------------------------------------------------------------- #
# every kind of step, windowed
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    ["patching.json", "window_patch.json", "zero_ablation.json", "logit_lens.json",
     "generate_probe.json", "head_patching.json", "neuron_patching.json"],
)
@pytest.mark.parametrize("batch_size", [1, 3])
def test_a_windowed_run_is_the_whole_run(name, batch_size, data_root, model_engine):
    """3 does not divide the 4 rows, so the last window is short."""
    raw = _point(name)
    whole = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    windowed = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=batch_size)

    _same_everywhere(whole, windowed)


def test_no_batch_size_is_one_window_and_changes_nothing(data_root, model_engine):
    raw = _point("patching.json")
    a = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    b = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=64)
    for key, value in a.all_results().items():
        assert torch.equal(value, b.result(key)), key


def test_the_hooks_engine_is_windowed_by_the_same_walk(data_root):
    raw = _point("patching.json")
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    whole = hooks.execute(plan.build_request(raw, data_root, hooks))
    windowed = hooks.execute(plan.build_request(raw, data_root, hooks), batch_size=2)
    assert _agree(whole.result("logit_diff"), windowed.result("logit_diff"))


# --------------------------------------------------------------------- #
# values that cross steps
# --------------------------------------------------------------------- #


def test_a_published_mean_is_shared_by_every_window(data_root, model_engine):
    raw = _point("mean_ablation.json")
    whole = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    windowed = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=1)
    _same_everywhere(whole, windowed)


def test_a_ragged_harvest_is_concatenated_in_row_order(data_root, model_engine):
    raw = _point("entity_mean_ablation.json")
    whole = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    windowed = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=3)
    _same_everywhere(whole, windowed)


def test_a_per_row_value_reaches_the_window_of_its_own_rows(data_root, model_engine):
    """The case a careless windowing gets wrong: one step reads a value that
    still has its rows, a later step swaps it in, and each step is windowed
    on its own. Window k of the later step must be handed rows k of it —
    were it handed the whole thing, or window 0's, the patch would land on
    the wrong examples and still have the right shape. In `patching.json`
    that value is `counterfactual.v_cf`, taken by `patched`, and one row at
    a time is the whole run's numbers."""
    raw = _point("patching.json")
    whole = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    one_by_one = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=1)
    assert _agree(whole.result("logit_diff"), one_by_one.result("logit_diff"))


# --------------------------------------------------------------------- #
# under a fit, and in the record
# --------------------------------------------------------------------- #


def test_a_fit_still_trains_when_its_passes_are_windowed(data_root, model_engine):
    """The loss is the mean over the concatenated rows, so the gradient is
    the whole minibatch's. (That also means windowing frees no memory inside
    an update: a fit's memory knob is its own `batch_size`. It does bound the
    eval pass.)"""
    raw = _point("das.json")
    whole = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    windowed = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=1)
    assert torch.allclose(whole.result("rot"), windowed.result("rot"), atol=1e-5)


def test_the_run_record_says_how_it_was_batched(data_root, model_engine, tmp_path):
    executed = model_engine.execute(
        plan.build_request(_point("patching.json"), data_root, model_engine), batch_size=2
    )
    executed.write(tmp_path)
    assert json.loads((tmp_path / "run.json").read_text())["batch_size"] == 2
    # and the plan itself does not: the same plan, however it is run
    assert "batch_size" not in json.dumps(executed.source)


def test_windowing_ships_with_the_walk(data_root, model_engine):
    """It lives in the shared walk, so it runs wherever the walk runs:
    `remote="local"` serializes the session and windows it identically."""
    raw = _point("patching.json")
    here = model_engine.execute(plan.build_request(raw, data_root, model_engine), batch_size=2)
    shipped = model_engine.execute(
        plan.build_request(raw, data_root, model_engine), remote="local", batch_size=2
    )
    assert torch.equal(here.result("logit_diff"), shipped.result("logit_diff"))
