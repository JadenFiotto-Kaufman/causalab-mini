"""The engine seam: what a plan needs from a runtime, and how little that is.

A plan says what to do; an engine says how to reach tensors. The test of that
claim is an engine that is not nnsight at all — no session, no envoys, no
model — running the same plan the real one runs.
"""

from typing import Any

import pytest
import torch

from causalab_mini.address import Address
from causalab_mini.engine import Engine, NNterpEngine, steps
from causalab_mini.plan import Forward, Plan, build, document
from causalab_mini.plan.document import ModelSpec


class FakeEngine(Engine):
    """An engine for a runtime that does not exist: it opens nothing and its
    forwards are made up. It implements the whole run half of the contract,
    which is the point — two methods, and one of them is two lines."""

    VOCAB = 32000  # the metrics index by token id, so the width has to be real

    def __init__(self) -> None:
        self.model = None
        self.calls: list[str] = []

    def execute(self, plan: Plan, remote: bool | str = False) -> Plan:
        steps.run(self, plan)
        return plan

    def forward(
        self,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        self.calls.append(forward.name)
        rows = len(forward.input_ids)
        for tap in forward.taps:
            for read in tap.reads:
                # One row per input row, wide enough to be logits. A real
                # engine would take this off a model; nothing downstream can
                # tell the difference.
                values[read.name] = torch.arange(rows * self.VOCAB, dtype=torch.float32).reshape(
                    rows, self.VOCAB
                )


def test_the_base_engine_has_no_implementation():
    """`Engine` is the contract and nothing else. An engine that opens nothing
    still has to say what running a request means for it, rather than
    inheriting a session it does not want."""
    bare = Engine()
    with pytest.raises(NotImplementedError):
        Engine.load(ModelSpec("k", "r", "fp32"))
    with pytest.raises(NotImplementedError):
        bare.execute(Plan())
    with pytest.raises(NotImplementedError):
        bare.forward(None, {}, {})  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        bare.locate("block_output", 0)
    with pytest.raises(NotImplementedError):
        bare.width(Address("block_output", 0))
    with pytest.raises(NotImplementedError):
        bare.tokenizer
    with pytest.raises(NotImplementedError):
        bare.num_layers


def test_the_engine_specific_surface_is_exactly_the_contract():
    """The finding this project exists to produce: an engine is how you load a
    model, how you address it and how you run one forward — six members. The
    walk over steps, the fit loop, the metrics and the write algebra are
    shared, and an engine adds nothing of its own to them."""
    overridden = {name for name in vars(NNterpEngine) if not name.startswith("_")}
    assert overridden == {"load", "tokenizer", "num_layers", "locate", "width",
                          "execute", "forward"}


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model_engine):
    return build(document.Document.from_json(minimal_raw), data_root, model_engine)


def test_an_engine_that_ships_holds_nothing_but_its_model(model_engine):
    """A traced block ships every name it loads, and the engine is one of
    those names — so whatever the engine holds rides along with it.

    Measured on the `remote="local"` path with the tiny Llama: the whole
    session payload is 115,746 bytes, the engine class alone is 77,736 (our
    package ships by value once registered), the instance is 86,227, and the
    model it holds is 7,079 of that. So the object costs about 1.4 KB over
    passing the class and the model separately — which is why `execute` may
    pass `self` rather than `type(self)`.

    That stays true only while the engine holds the model and nothing else.
    An engine that also held a tokenizer, a dataset or a document would ship
    it. (The hooks engine does hold a tokenizer, and refuses `remote`
    outright, which is the other way to be safe.)
    """
    assert set(vars(model_engine)) == {"model"}


def test_an_engine_with_no_model_and_no_session_runs_the_same_plan(minimal_plan, tmp_path):
    """The plan does not know which engine is running it, and a plan compiled
    for the nnterp engine runs unchanged on one that has never heard of
    nnsight."""
    engine = FakeEngine()
    executed = engine.execute(minimal_plan)

    assert engine.calls == ["original", "patched"]
    assert sorted(executed.all_results()) == ["iia", "logit_diff"]
    assert executed.result("iia").shape == (4,)

    written = executed.write(tmp_path)
    assert sorted(path.name for path in written) == ["iia.json", "logit_diff.json"]
