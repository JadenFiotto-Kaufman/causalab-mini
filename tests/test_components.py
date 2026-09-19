"""The component vocabulary: eleven names, and what each one is.

`address.py` is the only file that knows where a tensor lives, so this is
where that knowledge is checked. Three things are asked of every component:
that both families and both engines resolve it to the same place, that the
order the addresses sort into is the order the forward actually runs them in,
and that a write at it lands bit for bit.

The invariants at the end are the ones that would catch an address that
resolves, reads a real tensor, and reads the wrong one — `embeddings` and
`block_input` at layer 0 are the same tensor, and so are `block_output` at a
layer and `block_input` at the next.
"""

import copy
import json
import pathlib

import nnsight
import pytest
import torch

from causalab_mini import ops, plan
from causalab_mini.address import Address, _COMPONENTS
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.engine.engines.nnterp import engine as nnterp
from causalab_mini.plan import document

REPO = pathlib.Path(__file__).resolve().parents[1]
GPT2 = REPO / "documents" / "gpt2_cpu.json"

#: Every module boundary. The interiors are covered by `test_interior.py`.
BOUNDARIES = [name for name, entry in _COMPONENTS.items() if entry.op is None]
LAYERED = [name for name in BOUNDARIES if "{layer}" in _COMPONENTS[name].path]


@pytest.fixture(scope="session")
def gpt2_engine():
    return NNterpEngine.load(document.Document.load(GPT2).model, device_map="cpu")


@pytest.fixture(scope="session")
def hooks_engine():
    return HooksEngine.load(
        document.Document.load(REPO / "documents" / "minimal_cpu.json").model,
        device_map="cpu",
    )


@pytest.fixture(scope="session")
def hooks_engines(hooks_engine):
    """The same engine over both families, for the translation table."""
    return {
        "llama": hooks_engine,
        "gpt2": HooksEngine.load(document.Document.load(GPT2).model, device_map="cpu"),
    }


def _at(raw, component, layer):
    """The minimal document with its written site moved to another component."""
    raw = copy.deepcopy(raw)
    raw["method"]["sites"]["target"] = {"component": component}
    if layer is not None:
        raw["method"]["sites"]["target"]["layers"] = [layer]
    return raw


# --------------------------------------------------------------------- #
# where a component is
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("component", list(_COMPONENTS))
def test_one_address_serves_both_families(component, model_engine, gpt2_engine):
    """FINDINGS §1.11, extended from three components to eleven: the address
    is *equal* on tiny Llama and tiny GPT-2, whose module trees share no path,
    because nnterp absorbs the family axis and the interior's operation is
    resolved per checkpoint rather than tabulated."""
    layer = 0 if _COMPONENTS[component].band == 1 else None
    assert model_engine.locate(component, layer) == gpt2_engine.locate(component, layer)


#: What each standardized name resolves to in a raw HuggingFace tree. This is
#: the whole of what nnterp's standardization is worth for these two
#: families, written down — see FINDINGS §6.
RAW_PATHS = {
    "llama": {
        "embeddings": "model.embed_tokens",
        "block_input": "model.layers.0",
        "attention_output": "model.layers.0.self_attn",
        "mlp_input": "model.layers.0.mlp",
        "mlp_output": "model.layers.0.mlp",
        "block_output": "model.layers.0",
        "ln_final": "model.norm",
        "lm_head": "lm_head",
    },
    "gpt2": {
        "embeddings": "transformer.wte",
        "block_input": "transformer.h.0",
        "attention_output": "transformer.h.0.attn",
        "mlp_input": "transformer.h.0.mlp",
        "mlp_output": "transformer.h.0.mlp",
        "block_output": "transformer.h.0",
        "ln_final": "transformer.ln_f",
        "lm_head": "lm_head",
    },
}


def _qualified(root, module):
    for name, child in root.named_modules():
        if child is module:
            return name
    raise AssertionError(f"{module} is not in this tree")


@pytest.mark.parametrize("family", list(RAW_PATHS))
def test_the_hooks_engine_translates_every_name_to_the_right_raw_module(family, hooks_engines):
    """The nnterp engine resolves a standardized path against a handle that
    already has those names; the hooks engine resolves the same path against
    a raw tree that does not. Two families, no shared spelling, one address
    table — and this is the translation, stated rather than assumed."""
    engine = hooks_engines[family]
    for component, expected in RAW_PATHS[family].items():
        layer = 0 if _COMPONENTS[component].band == 1 else None
        module = engine.locate(component, layer).resolve(engine._names)
        assert _qualified(engine.model, module) == expected, component


def test_the_addresses_sort_into_forward_order():
    """The sort key is what keeps a read at the head from being issued before
    a read at layer 0 — nnsight refuses that outright. Bands first: the
    embeddings are the one tap upstream of the whole stack."""
    layered = [Address(name, 0) for name in LAYERED]
    order = [one.component for one in sorted(layered, key=lambda one: one.key)]
    assert order == ["block_input", "attention_output", "mlp_input", "mlp_output", "block_output"]

    everything = [Address(name, 0 if _COMPONENTS[name].band == 1 else None) for name in BOUNDARIES]
    ordered = [one.component for one in sorted(everything, key=lambda one: one.key)]
    assert ordered[0] == "embeddings" and ordered[-2:] == ["ln_final", "lm_head"]


def test_every_component_can_be_read_in_one_forward(model_engine):
    """The ordering claim, against the library rather than against itself:
    nnsight raises `OutOfOrderError` if an address is reached after the model
    has run past it, so reading all eleven in sorted order in one trace is
    what proves the key."""
    model = model_engine.model
    addresses = sorted(
        (
            model_engine.locate(name, 0 if entry.band == 1 else None)
            for name, entry in _COMPONENTS.items()
        ),
        key=lambda one: one.key,
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    with model.session():
        seen = nnsight.save({})
        with model.trace(batch):
            for address in addresses:
                seen[address.component] = nnterp.read(model, address).clone()

    assert set(seen) == set(_COMPONENTS)
    assert seen["lm_head"].shape[-1] == model_engine.width(Address("lm_head"))


# --------------------------------------------------------------------- #
# what a component is
# --------------------------------------------------------------------- #


def test_the_residual_stream_taps_are_the_tensors_they_claim(model_engine):
    """Two identities that hold by construction, and would not hold if an
    address were off by one module: the embeddings are what enters layer 0,
    and a block's output is what enters the next block."""
    model = model_engine.model
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    with model.session():
        seen = nnsight.save({})
        with model.trace(batch):
            for name, layer in (
                ("embeddings", None),
                ("block_input", 0),
                ("block_output", 0),
                ("block_input", 1),
            ):
                address = model_engine.locate(name, layer)
                seen[f"{name}{layer}"] = nnterp.read(model, address).clone()

    assert torch.equal(seen["embeddingsNone"], seen["block_input0"])
    assert torch.equal(seen["block_output0"], seen["block_input1"])


def test_the_attention_interior_taps_are_head_shaped(model_engine):
    """`attention_key` is the second argument of the same call the query is
    the first of, and `attention_z` is that call's return — the one tap whose
    handle is an output rather than an argument."""
    model = model_engine.model
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    with model.session():
        seen = nnsight.save({})
        with model.trace(batch):
            for name in ("attention_query", "attention_key", "attention_z"):
                seen[name] = nnterp.read(model, model_engine.locate(name, 0)).clone()

    # (batch, head, seq, head_dim) for the two arguments, sequence on axis 2 —
    # and the keys are in key-head space, which is narrower under GQA.
    assert seen["attention_query"].shape[0] == 1 and seen["attention_query"].shape[2] == 4
    assert seen["attention_key"].shape[2] == 4
    assert seen["attention_key"].shape[1] <= seen["attention_query"].shape[1]
    # the return is (batch, seq, head, head_dim): sequence on axis 1
    assert seen["attention_z"].shape[1] == 4


# --------------------------------------------------------------------- #
# writing at a component
# --------------------------------------------------------------------- #


#: At layer 0 these two taps are the token embeddings, and the weekdays pairs
#: share their last token — so an interchange at the declared position swaps a
#: tensor for itself. FINDINGS §1.14 records the same thing for a layer-0
#: query. Tested below as a no-op rather than skipped, because it looks
#: exactly like a broken write.
SAME_AT_LAYER_0 = ("embeddings", "block_input")

#: Which layer each component is tested at. `block_input` moves at layer 1 for
#: the reason above.
LAYER_UNDER_TEST = {"block_input": 1}


@pytest.mark.parametrize(
    "component",
    [name for name in BOUNDARIES if name not in ("lm_head", "embeddings")],
)
def test_a_swap_at_every_component_lands_and_moves_the_logits(
    component, minimal_raw, data_root, model_engine
):
    """One document per component, each the minimal interchange with its site
    moved. A write that landed nowhere would leave the logits equal to the
    un-intervened run's, and a write at the wrong tensor would still move
    them — so the test is both: it moves, and an identity write does not."""
    layer = LAYER_UNDER_TEST.get(component, 0) if _COMPONENTS[component].band == 1 else None
    raw = _at(minimal_raw, component, layer)

    swapped = model_engine.execute(plan.build_request(raw, data_root, model_engine))

    clean = copy.deepcopy(raw)
    del clean["method"]["reads"]["v_cf"], clean["method"]["writes"]
    del clean["method"]["intervened_models"]
    clean["method"]["reads"]["logits"]["model"] = "original"
    for entry in clean["method"]["save"]:
        entry["model"] = "original"
    plain = model_engine.execute(plan.build_request(clean, data_root, model_engine))

    identity = copy.deepcopy(raw)
    identity["method"]["reads"]["v_cf"]["input"] = "base"
    same = model_engine.execute(plan.build_request(identity, data_root, model_engine))

    assert torch.equal(same.result("logit_diff"), plain.result("logit_diff")), component
    assert not torch.equal(swapped.result("logit_diff"), plain.result("logit_diff")), component


@pytest.mark.parametrize("component", SAME_AT_LAYER_0)
def test_an_interchange_at_the_embeddings_of_a_shared_last_token_is_a_no_op(
    component, minimal_raw, data_root, model_engine
):
    """It looks exactly like a write that landed nowhere, and it is not: the
    two prompts of a weekdays pair differ in the day, not in the last token,
    and at layer 0 the declared position has attended to nothing yet. The
    same swap moves the logits one layer up (see the test above, which takes
    `block_input` at layer 1)."""
    raw = _at(minimal_raw, component, 0 if _COMPONENTS[component].band == 1 else None)
    swapped = model_engine.execute(plan.build_request(raw, data_root, model_engine))

    clean = copy.deepcopy(raw)
    del clean["method"]["reads"]["v_cf"], clean["method"]["writes"]
    del clean["method"]["intervened_models"]
    clean["method"]["reads"]["logits"]["model"] = "original"
    for entry in clean["method"]["save"]:
        entry["model"] = "original"
    plain = model_engine.execute(plan.build_request(clean, data_root, model_engine))

    assert torch.equal(swapped.result("logit_diff"), plain.result("logit_diff"))


@pytest.mark.parametrize("component", ["block_input", "attention_output", "ln_final"])
def test_the_two_engines_agree_at_the_new_components(
    component, minimal_raw, data_root, model_engine, hooks_engine
):
    """Bit-identical across a traced engine and a hooked one, including at an
    input-side boundary, which the hooks engine reaches with a pre-hook."""
    layer = LAYER_UNDER_TEST.get(component, 0) if _COMPONENTS[component].band == 1 else None
    raw = _at(minimal_raw, component, layer)

    traced = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    hooked = hooks_engine.execute(plan.build_request(raw, data_root, hooks_engine))

    for name in ("iia", "logit_diff"):
        assert torch.equal(traced.result(name), hooked.result(name)), (component, name)
