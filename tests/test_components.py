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
from conftest import same_numbers

from causalab_mini import ops, plan
from causalab_mini.address import Address, _COMPONENTS
from causalab_mini.engine.engines.hooks import engine as hooks
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.engine.engines.nnterp import engine as nnterp
from causalab_mini.plan import document

REPO = pathlib.Path(__file__).resolve().parents[1]
GPT2 = REPO / "documents" / "gpt2_cpu.json"

#: Every module boundary. The interiors are covered by `test_interior.py`.
BOUNDARIES = [name for name, entry in _COMPONENTS.items() if entry.op is None]
LAYERED = [name for name in BOUNDARIES if _COMPONENTS[name].per_layer]


@pytest.fixture(scope="session")
def gpt2_engine():
    return NNterpEngine.load(document.Document.load(GPT2).model, device_map="cpu")


@pytest.fixture(scope="session")
def eager_engine():
    """The same tiny Llama running eager attention — the only implementation
    under which the attention pattern is a tensor at all."""
    return NNterpEngine.load(
        document.Document.load(REPO / "documents" / "minimal_cpu.json").model,
        device_map="cpu", attn_implementation="eager",
    )


@pytest.fixture(scope="session")
def eager_gpt2_engine():
    return NNterpEngine.load(document.Document.load(GPT2).model, device_map="cpu", attn_implementation="eager")


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
def test_one_address_serves_both_families(component, eager_engine, eager_gpt2_engine):
    """FINDINGS §1.11, extended from three components to eleven: the address
    is *equal* on tiny Llama and tiny GPT-2, whose module trees share no path,
    because nnterp absorbs the family axis and the interior's operation is
    resolved per checkpoint rather than tabulated."""
    layer = 0 if _COMPONENTS[component].per_layer else None
    assert eager_engine.locate(component, layer).where == eager_gpt2_engine.locate(component, layer).where


#: What each standardized name resolves to in a raw HuggingFace tree. This is
#: the whole of what nnterp's standardization is worth for these two
#: families, written down — see FINDINGS §6.
RAW_PATHS = {
    "llama": {
        "input_ids": "model.embed_tokens",
        "embeddings": "model.embed_tokens",
        "block_input": "model.layers.0",
        "attention_input_norm": "model.layers.0.input_layernorm",
        "attention_premix": "model.layers.0.self_attn.o_proj",
        "attention_output": "model.layers.0.self_attn",
        "block_mid": "model.layers.0.post_attention_layernorm",
        "mlp_input_norm": "model.layers.0.post_attention_layernorm",
        "mlp_activation": "model.layers.0.mlp.act_fn",
        "mlp_neuron_output": "model.layers.0.mlp.down_proj",
        "mlp_input": "model.layers.0.mlp",
        "mlp_output": "model.layers.0.mlp",
        "block_output": "model.layers.0",
        "ln_final": "model.norm",
        "lm_head": "lm_head",
    },
    "gpt2": {
        "input_ids": "transformer.wte",
        "embeddings": "transformer.wte",
        "block_input": "transformer.h.0",
        "attention_input_norm": "transformer.h.0.ln_1",
        "attention_premix": "transformer.h.0.attn.c_proj",
        "attention_output": "transformer.h.0.attn",
        "block_mid": "transformer.h.0.ln_2",
        "mlp_input_norm": "transformer.h.0.ln_2",
        "mlp_activation": "transformer.h.0.mlp.act",
        "mlp_neuron_output": "transformer.h.0.mlp.c_proj",
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
        layer = 0 if _COMPONENTS[component].per_layer else None
        module = hooks.resolve(engine.locate(component, layer), engine._names)
        assert _qualified(engine.model, module) == expected, component


def test_the_addresses_sort_into_forward_order(eager_engine):
    """The sort key is what keeps a read at the head from being issued before
    a read at layer 0 — nnsight refuses that outright. It is nnterp's
    `internals.rank`, stamped by `locate`, with mini's own numbering only for
    the interiors it addresses and nnterp does not."""
    model_engine = eager_engine  # eager, so the pattern is among them
    layered = [model_engine.locate(name, 0) for name in LAYERED]
    order = [one.component for one in sorted(layered, key=lambda one: one.key)]
    assert order == [
        "block_input", "attention_input_norm", "attention_probs", "attention_premix",
        "attention_output", "block_mid", "mlp_input_norm", "mlp_input", "mlp_activation",
        "mlp_neuron_output", "mlp_output", "block_output",
    ]

    everything = [
        model_engine.locate(name, 0 if _COMPONENTS[name].per_layer else None)
        for name in BOUNDARIES
    ]
    ordered = [one.component for one in sorted(everything, key=lambda one: one.key)]
    assert ordered[:2] == ["input_ids", "embeddings"]
    # the head's output, then the model's own — which is not the same tensor
    # on a family that caps its logits
    assert ordered[-3:] == ["ln_final", "lm_head", "logits"]


def test_every_component_can_be_read_in_one_forward(eager_engine):
    model_engine = eager_engine  # eager, so the pattern is among them
    """The ordering claim, against the library rather than against itself:
    nnsight raises `OutOfOrderError` if an address is reached after the model
    has run past it, so reading all eleven in sorted order in one trace is
    what proves the key."""
    model = model_engine.model
    addresses = sorted(
        (
            model_engine.locate(name, 0 if entry.per_layer else None)
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
SAME_AT_LAYER_0 = ("embeddings", "block_input", "attention_input_norm")

#: Which layer each component is tested at. `block_input` moves at layer 1 for
#: the reason above.
LAYER_UNDER_TEST = {"block_input": 1, "attention_input_norm": 1}


@pytest.mark.parametrize(
    "component",
    # `lm_head` and `logits` are where the metric reads, so a write there is
    # the read; `embeddings`/`input_ids` are upstream of everything and are
    # covered by the no-op test below; a place that needs eager attention is
    # not in this sdpa document and has its own tests above
    [
        name
        for name in BOUNDARIES
        if name not in ("lm_head", "logits", "embeddings", "input_ids")
        and _COMPONENTS[name].needs is None
    ],
)
def test_a_swap_at_every_component_lands_and_moves_the_logits(
    component, minimal_raw, data_root, model_engine
):
    """One document per component, each the minimal interchange with its site
    moved. A write that landed nowhere would leave the logits equal to the
    un-intervened run's, and a write at the wrong tensor would still move
    them — so the test is both: it moves, and an identity write does not."""
    layer = LAYER_UNDER_TEST.get(component, 0) if _COMPONENTS[component].per_layer else None
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
    raw = _at(minimal_raw, component, 0 if _COMPONENTS[component].per_layer else None)
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
    layer = LAYER_UNDER_TEST.get(component, 0) if _COMPONENTS[component].per_layer else None
    raw = _at(minimal_raw, component, layer)

    traced = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    hooked = hooks_engine.execute(plan.build_request(raw, data_root, hooks_engine))

    for name in ("iia", "logit_diff"):
        assert same_numbers(traced.result(name), hooked.result(name)), (component, name)


def test_the_token_ids_can_be_read_and_never_written(minimal_raw, data_root, model_engine):
    """`input_ids` is the model's input: integers, `(batch, seq)`, no width.
    A read gathers it like anything else; a write is refused where the
    document is read, because a float activation swapped into token ids
    means nothing."""
    from pydantic import ValidationError

    from causalab_mini.plan.spec import Spec

    raw = __import__("json").loads((REPO / "documents" / "v2" / "patching.json").read_text())
    raw["sites"]["ids"] = {"component": "input_ids"}
    raw["interventions"]["patching"]["reads"]["tokens"] = {"site": "ids", "pos": {"last": 2}, "input": "base"}
    raw["steps"]["score"]["outputs"] = {"last_two": "tokens"}
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    tokens = executed.step("score", plan.Observe).results["last_two"]
    assert tokens.shape == (4, 2) and not tokens.is_floating_point()

    raw["interventions"]["patching"]["writes"]["patch"]["site"] = "ids"
    with pytest.raises(ValidationError, match="'input_ids' is read-only"):
        Spec.model_validate(raw)


def test_a_childs_spelling_is_nnterps_to_know(model_engine, gpt2_engine):
    """`input_layernorm` on Llama, `ln_1` on GPT-2: which a checkpoint has is a
    fact nnterp reads off its module tree, and `locate` writes into the
    address so the plan says where."""
    assert model_engine.locate("attention_input_norm", 0).module == "input_layernorm"
    assert gpt2_engine.locate("attention_input_norm", 0).module == "ln_1"
    assert model_engine.locate("attention_premix", 0) == Address(
        "attention_premix", 0, module="self_attn.o_proj", io="input", rank=(0, 25)
    )


# --------------------------------------------------------------------- #
# the head's output and the model's own
# --------------------------------------------------------------------- #


from causalab_mini.engine import NNterpEngine  # noqa: E402
from causalab_mini.plan.spec import Model  # noqa: E402

GEMMA2 = {"key": "trl-internal-testing/tiny-Gemma2ForCausalLM", "revision": "main", "dtype": "fp32"}
TINY_LLAMA = {
    "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
    "dtype": "fp32",
}


def _both_heads(model: dict) -> dict:
    """A document that swaps a big constant into the head's output and scores
    it twice: once where the head put it, once where the model reads it."""
    return {
        "model": model,
        "roles": {"base": {"field": "input"}},
        "sites": {"head": {"component": "lm_head"}, "out": {"component": "logits"}},
        "interventions": {
            "one": {
                "reads": {
                    "raw": {"site": "head", "pos": -1, "model": "loud", "input": "base"},
                    "capped": {"site": "out", "pos": -1, "model": "loud", "input": "base"},
                },
                "writes": {"shout": {"site": "head", "pos": -1, "mechanism": "swap", "operand": 100.0}},
                "models": {"loud": {"input": "base", "writes": ["shout"]}},
                "metrics": {
                    "at_head": {"kind": "token_logit", "of": "raw", "token": "base_answer"},
                    "at_logits": {"kind": "token_logit", "of": "capped", "token": "base_answer"},
                },
            }
        },
        "steps": {"score": {"kind": "observe", "rows": {"base": "weekdays/train"}}},
    }


def test_a_metric_at_the_head_and_at_the_logits_differ_where_the_family_caps(data_root):
    """Gemma-2 bounds its logits with `final_logit_softcapping`, so the head's
    output is not what the model predicts from. They are two places and two
    components, and a metric scored on the wrong one is scored on numbers the
    model never used."""
    raw = _both_heads(GEMMA2)
    engine = NNterpEngine.load(Model.model_validate(GEMMA2), device_map="cpu")
    cap = engine.model.config.final_logit_softcapping
    assert cap == 30.0

    scored = engine.execute(plan.build_request(raw, data_root, engine))
    at_head = scored.result("at_head")
    at_logits = scored.result("at_logits")

    assert torch.allclose(at_head, torch.full_like(at_head, 100.0))
    assert torch.allclose(
        at_logits, torch.full_like(at_logits, float(torch.tanh(torch.tensor(100.0 / cap)) * cap)),
        atol=1e-4,
    )
    assert not torch.allclose(at_head, at_logits)


def test_the_two_are_the_same_tensor_where_it_does_not(data_root, model_engine):
    """And on a family with no cap they are one number twice, which is why
    every shipped document can go on naming either."""
    scored = model_engine.execute(
        plan.build_request(_both_heads(TINY_LLAMA), data_root, model_engine)
    )
    assert getattr(model_engine.model.config, "final_logit_softcapping", None) is None
    assert torch.equal(scored.result("at_head"), scored.result("at_logits"))


def test_the_hooks_engine_reaches_the_logits_too(data_root, model_engine):
    """`logits` is the model's own output with the tensor one field inside
    it, and a forward hook sees exactly that value — so the row is reachable
    by asking nnterp where the tensor is, which is what unwrapping a tuple
    was already doing. Both engines, one number."""
    raw = _both_heads(TINY_LLAMA)
    hooks = HooksEngine.load(Model.model_validate(TINY_LLAMA), device_map="cpu")
    traced = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(raw, data_root, hooks))
    assert torch.equal(traced.result("at_logits"), hooked.result("at_logits"))
    assert torch.equal(traced.result("at_head"), hooked.result("at_head"))


def test_the_hooks_engine_sees_the_cap_as_well(data_root):
    """And it is the model's own output, so it carries the model's cap: the
    same two numbers on tiny Gemma-2 that the traced engine reports."""
    raw = _both_heads(GEMMA2)
    hooks = HooksEngine.load(Model.model_validate(GEMMA2), device_map="cpu")
    scored = hooks.execute(plan.build_request(raw, data_root, hooks))
    at_head, at_logits = scored.result("at_head"), scored.result("at_logits")
    assert torch.allclose(at_head, torch.full_like(at_head, 100.0))
    assert torch.allclose(at_logits, torch.full_like(at_logits, 29.9237), atol=1e-3)
