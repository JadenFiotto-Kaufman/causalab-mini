"""The engine seam: what a plan needs from a runtime, and how little that is.

A plan says what to do; an engine says how to reach tensors. The test of that
claim is an engine that is not nnsight at all — no session, no envoys, no
model — running the same plan the real one runs.
"""

from typing import Any

import pytest
import torch

from causalab_mini.address import Address
from causalab_mini.engine import Engine, EngineError, NNterpEngine, steps
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan import Forward, Plan, build, document
from causalab_mini.plan.document import ModelSpec


class FakeEngine(Engine):
    """An engine for a runtime that does not exist: it opens nothing and its
    forwards are made up. It implements what a plan that does not decode
    asks of the run, which is the point — two methods and a tokenizer,
    because resolving a position against the text the model will see is
    something the *run* does."""

    VOCAB = 32000  # the metrics index by token id, so the width has to be real

    def __init__(self, tokenizer: Any) -> None:
        self.model = None
        self.calls: list[str] = []
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def execute(self, plan: Plan, remote: bool | str = False, batch_size: int | None = None) -> Plan:
        steps.run(self, plan, steps.start(plan, batch_size))
        return plan

    def forward(
        self,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        self.calls.append(forward.input)
        rows = len(forward.input_ids)
        for tap in forward.taps:
            for read in tap.reads:
                # One row per input row, wide enough to be logits. A real
                # engine would take this off a model; nothing downstream can
                # tell the difference.
                values[read.name] = torch.arange(rows * self.VOCAB, dtype=torch.float32).reshape(
                    rows, 1, self.VOCAB  # a unit window per row, as a real read is
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
        bare.generate(None, {}, {})  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        bare.locate("block_output", 0)
    with pytest.raises(NotImplementedError):
        bare.width(Address("block_output", 0))
    with pytest.raises(NotImplementedError):
        bare.heads(Address("attention_z", 0))
    with pytest.raises(NotImplementedError):
        bare.tokenizer
    with pytest.raises(NotImplementedError):
        bare.num_layers


def test_the_engine_specific_surface_is_exactly_the_contract():
    """The finding this project exists to produce: an engine is how you load a
    model, how you address it and how you call it — nine members (`heads`
    joined when a site could name them: like `width`, it is a question about
    the checkpoint that only its holder can answer; `generate` beside
    `forward`, because a decode is a different call with a different
    result). The walk over steps, the fit loop, the metrics and the write
    algebra are shared, and an engine adds nothing of its own to them."""
    overridden = {name for name in vars(NNterpEngine) if not name.startswith("_")}
    assert overridden == {"load", "tokenizer", "num_layers", "locate", "width", "heads",
                          "execute", "forward", "generate"}


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model_engine):
    return build(document.Document.from_json(minimal_raw), data_root, model_engine)


def test_an_engine_that_ships_holds_nothing_but_its_model(model_engine):
    """A traced block ships every name it loads, and the engine is one of
    those names — so whatever the engine holds rides along with it.

    Its class ships as a reference: nothing of this package travels by
    value, and the server imports its own copy (FINDINGS §24). What the
    instance adds is what it holds, and the model it holds is itself a
    reference to the module the server has loaded — which is why `execute`
    may pass `self` rather than `type(self)`.

    That stays true only while the engine holds the model and nothing else.
    An engine that also held a tokenizer, a dataset or a document would ship
    it. (The hooks engine does hold a tokenizer, and refuses `remote`
    outright, which is the other way to be safe.)
    """
    assert set(vars(model_engine)) == {"model"}


def test_an_engine_with_no_model_and_no_session_runs_the_same_plan(minimal_plan, tmp_path, model_engine):
    """The plan does not know which engine is running it, and a plan compiled
    for the nnterp engine runs unchanged on one that has never heard of
    nnsight."""
    engine = FakeEngine(model_engine.tokenizer)
    executed = engine.execute(minimal_plan)

    assert engine.calls == ["counterfactual", "base"]
    assert sorted(executed.all_results()) == ["iia", "logit_diff"]
    assert executed.result("iia").shape == (4,)

    written = executed.write(tmp_path)
    assert sorted(path.name for path in written) == ["iia.json", "logit_diff.json"]


# --------------------------------------------------------------------- #
# compiling without weights
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("engine_class", [NNterpEngine, HooksEngine])
def test_a_meta_shell_compiles_the_same_plan(engine_class, minimal_raw, data_root):
    """The compiler asks for the tokenizer, the layer count, the widths and an
    interior's operation — none of which is a weight. `dispatch=False`, in
    nnsight's own spelling, answers all four in well under a second, so a
    document can be validated and explained on a machine that will never run
    it — and, for nnterp, the same shell is what runs on NDIF."""
    spec = document.Document.from_json(minimal_raw).model
    shell = engine_class.load(spec, dispatch=False)
    full = engine_class.load(spec, device_map="cpu")

    assert shell.num_layers == full.num_layers
    assert shell.width(shell.locate("block_output", 0)) == full.width(full.locate("block_output", 0))
    assert build(document.Document.from_json(minimal_raw), data_root, shell) == build(
        document.Document.from_json(minimal_raw), data_root, full
    )


def test_a_meta_shell_still_locates_an_interior(minimal_raw):
    """Where the query is, is nnterp's row — no weight is needed to say so."""
    shell = NNterpEngine.load(document.Document.from_json(minimal_raw).model, dispatch=False)
    located = shell.locate("attention_query", 0)
    assert (located.accessor, located.inside, located.rank) == ("attention_queries", True, (0, 13))


def test_a_hooks_shell_refuses_to_run_because_it_has_nowhere_to(minimal_raw, data_root):
    """nnsight's shell can run remotely; a hooks shell has no server, so the
    engine — not the base contract — is the one that refuses."""
    shell = HooksEngine.load(document.Document.from_json(minimal_raw).model, dispatch=False)
    with pytest.raises(EngineError, match="dispatch=False"):
        shell.execute(build(document.Document.from_json(minimal_raw), data_root, shell))


def test_a_nested_plan_gets_its_own_values_but_the_same_featurizers():
    """The state is scoped to a `steps` list. A sweep point, or a fit's
    update, is a nested plan: it shares the live parameter sets, reads what
    was produced before it, and what it produces stays its own — so no
    sibling sees another's values."""
    parent = steps.State(featurizers={"rot": object()}, values={"mean": object()})
    child = parent.child()
    assert child.featurizers is parent.featurizers
    assert child.values == parent.values and child.values is not parent.values
    child.values["delta"] = object()
    assert "delta" not in parent.values and "delta" not in parent.child().values
