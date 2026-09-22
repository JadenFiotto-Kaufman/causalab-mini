"""The second engine: torch forward hooks, and whether it agrees with nnsight.

The test that matters is parity. Two runtimes that share no execution code —
one recompiles the forward and traces it, the other registers a callback on a
`nn.Module` — compile the same document and must produce the same numbers. If
they do, what a plan means is genuinely engine-independent; if they do not,
one of them is encoding something the document did not say.

The rest is what the hooks engine refuses, and what it refuses is as much the
point: the interior and remote are not approximated here, they are named.
"""

import json
import pathlib
from dataclasses import replace

import pytest
import torch
from conftest import same_numbers

from causalab_mini import plan
from causalab_mini.address import AddressError
from causalab_mini.engine import HooksEngine, NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngineError
from causalab_mini.engine.engines.hooks import engine as hooks
from causalab_mini.engine.engines.hooks.loading import standardized
from causalab_mini.ops import intervene
from causalab_mini.plan import Observe, ReadOp, document

REPO = pathlib.Path(__file__).resolve().parents[1]
GPT2_DOCUMENT = REPO / "documents" / "gpt2_cpu.json"


@pytest.fixture(scope="session")
def hooks_engine():
    """A hooks engine holding the same tiny Llama `model_engine` holds, loaded
    the way this engine loads: `AutoModelForCausalLM` and `AutoTokenizer`."""
    return HooksEngine.load(
        document.Document.load(REPO / "documents" / "minimal_cpu.json").model,
        device_map="cpu",
    )


def _build(raw, data_root, engine):
    return plan.build_request(raw, data_root, engine)


# --------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------- #


def test_the_two_engines_answer_the_compiler_identically(model_engine, hooks_engine):
    """Before the numbers: the four things a plan is compiled against. A
    difference here would make the parity below a comparison of two different
    experiments."""
    assert hooks_engine.num_layers == model_engine.num_layers
    assert hooks_engine.tokenizer.padding_side == model_engine.tokenizer.padding_side
    for component, layer in (("block_output", 0), ("lm_head", None)):
        address = hooks_engine.locate(component, layer)
        assert address == model_engine.locate(component, layer)
        assert hooks_engine.width(address) == model_engine.width(address)


def test_the_two_engines_produce_the_same_numbers(
    minimal_raw, data_root, model_engine, hooks_engine
):
    """The cross-engine golden: same document, two runtimes, bit-identical.

    Each engine compiles its own plan, because compiling is an engine
    question — the tokenizer and the widths come from it. That the two plans
    then produce equal tensors is the claim.
    """
    traced = model_engine.execute(_build(minimal_raw, data_root, model_engine))
    hooked = hooks_engine.execute(_build(minimal_raw, data_root, hooks_engine))

    assert set(traced.all_results()) == set(hooked.all_results())
    for name, values in traced.all_results().items():
        assert same_numbers(values, hooked.result(name)), name


def test_the_two_engines_fit_the_same_rotation(das_raw, data_root):
    """Parity through a training loop, which is the stronger claim: ten
    AdamW updates, a backward through each engine's own intervention path,
    early stopping — and the fitted `(16, 8)` rotation comes out equal to the
    bit, not just the metrics over it."""
    spec = document.Document.from_json(das_raw).model
    traced = NNterpEngine.load(spec, device_map="cpu")
    hooked = HooksEngine.load(spec, device_map="cpu")

    fitted = traced.execute(_build(das_raw, data_root, traced))
    hooks = hooked.execute(_build(das_raw, data_root, hooked))

    assert set(fitted.all_results()) == set(hooks.all_results())
    for name, values in fitted.all_results().items():
        assert same_numbers(values, hooks.result(name), atol=1e-4), name  # through ten AdamW updates


# --------------------------------------------------------------------- #
# the write lands
# --------------------------------------------------------------------- #


def test_a_swap_lands_the_source_read_bit_for_bit(hooks_engine, minimal_raw, data_root):
    """The hook's whole job in one assertion: after the write, the activation
    at the address *is* the tensor the other forward read.

    Reading it back is a `ReadOp` added to the tap that already carries the
    write, which also pins the ordering rule — at one address the writes run
    before the reads, so `landed` sees the patched value and not the clean one.
    """
    compiled = _build(minimal_raw, data_root, hooks_engine)
    source, patched = compiled.step("observe", Observe).forwards
    tap = patched.taps[0]
    watched = replace(
        patched,
        taps=(
            replace(tap, reads=(ReadOp(name="landed", at=tap.writes[0].at),)),
            *patched.taps[1:],
        ),
    )

    values: dict[str, torch.Tensor] = {}
    featurizers = dict(intervene.FEATURIZERS)
    hooks_engine.forward(source, values, featurizers)
    hooks_engine.forward(watched, values, featurizers)

    assert values["landed"].shape == (4, 1, hooks_engine.model.config.hidden_size)
    assert torch.equal(values["landed"], values["v_cf"])


def test_a_forward_leaves_no_hook_behind(hooks_engine, minimal_raw, data_root):
    """A leaked handle would intervene on the next forward — a wrong number,
    not an error — so the count is asserted rather than trusted."""
    compiled = _build(minimal_raw, data_root, hooks_engine)
    source, _ = compiled.step("observe", Observe).forwards
    layer = hooks.resolve(hooks_engine.locate("block_output", 0), standardized(hooks_engine.model))

    before = len(layer._forward_hooks)
    hooks_engine.forward(source, {}, dict(intervene.FEATURIZERS))
    assert len(layer._forward_hooks) == before


# --------------------------------------------------------------------- #
# what it refuses
# --------------------------------------------------------------------- #


def test_the_interior_is_refused_by_name(hooks_engine):
    """`attention_query` is one argument of one call inside a forward. A hook
    fires at the boundary, so this engine says so instead of reaching for
    something nearby."""
    with pytest.raises(AddressError, match="interior"):
        hooks_engine.locate("attention_query", 0)


def test_a_document_that_names_the_interior_fails_at_compile_time(data_root, hooks_engine):
    """And it fails where every other unsupported document fails: on the
    client, while the plan is being built."""
    raw = json.loads((REPO / "documents" / "attention_query_cpu.json").read_text())
    with pytest.raises(AddressError, match="attention_query"):
        _build(raw, data_root, hooks_engine)


@pytest.mark.parametrize("remote", [True, "local", "true"])
def test_remote_is_refused(hooks_engine, remote):
    """There is no remote for hooks, and silently running locally would be the
    worst outcome: the same call, the same numbers, and a claim about where
    they came from that is false."""
    with pytest.raises(HooksEngineError, match="no remote for hooks"):
        hooks_engine.execute(plan.Plan(), remote=remote)


# --------------------------------------------------------------------- #
# the contract, and the translation
# --------------------------------------------------------------------- #


def test_the_engine_specific_surface_is_exactly_the_contract():
    """The same assertion `test_engine.py` makes about the nnterp engine, and
    it is worth more here: a second runtime, sharing nothing with the first,
    still added no member of its own to the contract."""
    overridden = {name for name in vars(HooksEngine) if not name.startswith("_")}
    assert overridden == {"load", "tokenizer", "num_layers", "locate", "width", "heads",
                          "execute", "forward"}


def test_the_standardized_names_reach_a_second_family(data_root):
    """The translation is what this engine exists to measure, so it is checked
    on a tree that shares no path segment with the Llama's: GPT-2's stack is
    `transformer.h`, not `model.layers`, and the same two addresses resolve to
    the right modules on both without a family table."""
    engine = HooksEngine.load(document.Document.load(GPT2_DOCUMENT).model, device_map="cpu")
    model = engine.model

    assert engine.num_layers == len(model.transformer.h)
    assert hooks.resolve(engine.locate("block_output", 3), standardized(model)) is model.transformer.h[3]
    assert hooks.resolve(engine.locate("lm_head"), standardized(model)) is model.lm_head
    assert engine.width(engine.locate("block_output", 0)) == model.config.n_embd
