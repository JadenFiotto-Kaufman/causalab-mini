"""The engine seam: what a plan needs from a runtime, and how little that is.

A plan says what to do; an engine says how to reach tensors. The test of that
claim is an engine that is not nnsight at all — no session, no envoys, no
model — running the same plan the real one runs.
"""

from typing import Any

import pytest
import torch

from causalab_mini.engine import Engine, NNterpEngine, steps
from causalab_mini.plan import Forward, Plan, build, document


class FakeEngine(Engine):
    """An engine for a runtime that does not exist: it opens nothing and its
    forwards are made up. It implements the whole contract, which is the
    point — two methods, and one of them is three lines."""

    VOCAB = 32000  # the metrics index by token id, so the width has to be real
    calls: list[str] = []

    @classmethod
    def execute(cls, model: Any, plan: Plan, remote: bool | str = False) -> Plan:
        steps.run(cls, model, plan)
        return plan

    @classmethod
    def forward(
        cls,
        model: Any,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        cls.calls.append(forward.name)
        rows = len(forward.input_ids)
        for tap in forward.taps:
            for read in tap.reads:
                # One row per input row, wide enough to be logits. A real
                # engine would take this off a model; nothing downstream can
                # tell the difference.
                values[read.name] = torch.arange(rows * cls.VOCAB, dtype=torch.float32).reshape(
                    rows, cls.VOCAB
                )


def test_the_base_engine_has_no_implementation():
    """`Engine` is the contract. An engine that opens nothing still has to say
    what running a request means for it, rather than inheriting a session it
    does not want."""
    with pytest.raises(NotImplementedError):
        Engine.execute(None, Plan())
    with pytest.raises(NotImplementedError):
        Engine.forward(None, None, {}, {})  # type: ignore[arg-type]


def test_the_engine_specific_surface_is_exactly_two_methods():
    """The finding this project exists to produce: everything else — the walk
    over steps, the fit loop, the metrics, the write algebra — is shared."""
    overridden = {name for name in vars(NNterpEngine) if not name.startswith("__")}
    assert overridden == {"execute", "forward"}


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model):
    return build(document.Document.from_json(minimal_raw), data_root, model)


def test_an_engine_with_no_model_and_no_session_runs_the_same_plan(minimal_plan, tmp_path):
    """The plan does not know which engine is running it, and a plan compiled
    for the nnterp engine runs unchanged on one that has never heard of
    nnsight."""
    FakeEngine.calls = []
    executed = FakeEngine.execute(None, minimal_plan)

    assert FakeEngine.calls == ["original", "patched"]
    assert sorted(executed.all_results()) == ["iia", "logit_diff"]
    assert executed.result("iia").shape == (4,)

    written = executed.write(tmp_path)
    assert sorted(path.name for path in written) == ["iia.json", "logit_diff.json"]
