"""A Gated DeltaNet layer's recurrent state, as a place with a position.

The state is the one thing in a hybrid (Qwen3-Next, Qwen3.5) that is not a
stream: a (key, value) matrix per head with no sequence axis, one per forward.
It is still the state *after a token* — the forward's last — so it is addressed
with the positions every other place has: `{"index": -1}` is the state after the
prompt, and the continuation frame's step k the state after the token that
step processed. Nothing in the executor knows it is a state; nnterp's row says
it has no sequence axis, and `ops` treats such a tensor as one position.

On `yujiepan/qwen3.5-tiny-random`: dense, layers 0-2 linear and 3 softmax.
"""

import copy
import json
import pathlib

import nnsight
import pytest
import torch
from conftest import model_block

from causalab_mini import plan
from causalab_mini.address import AddressError
from causalab_mini.engine import NNterpEngine
from causalab_mini.plan.plan import PlanError

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCH = REPO / "documents" / "v2" / "deltanet_state_patch.json"
DECODE = REPO / "documents" / "v2" / "deltanet_state_decode.json"


@pytest.fixture(scope="session")
def hybrid_engine():
    return NNterpEngine.load(model_block(PATCH), device_map="cpu")


@pytest.fixture
def patch_raw():
    return copy.deepcopy(json.loads(PATCH.read_text()))


def test_the_state_is_a_row_with_no_sequence_axis(hybrid_engine):
    """Its layout is nnterp's row: no sequence axis, per value head, and a
    (key, value) matrix wide per head — a product of three config numbers."""
    config = hybrid_engine.model._module.config.get_text_config()
    located = hybrid_engine.locate("linear_attention_state", 0)
    assert located.accessor == "linear_attentions_state_output"
    assert located.seq_axis is None and located.heads == "linear_num_value_heads" and located.inside
    assert hybrid_engine.heads(located) == config.linear_num_value_heads
    assert hybrid_engine.width(located) == (
        config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim
    )
    query = hybrid_engine.locate("linear_attention_query", 0)
    assert query.seq_axis == 1
    assert hybrid_engine.width(query) == config.linear_num_value_heads * config.linear_key_head_dim


def test_a_softmax_layer_and_a_model_without_one_say_why(hybrid_engine, model_engine):
    with pytest.raises(AddressError, match="has no 'linear_attn'"):
        hybrid_engine.locate("linear_attention_state", 3)
    with pytest.raises(AddressError, match="has no 'linear_attn'"):
        model_engine.locate("linear_attention_state", 0)


@pytest.mark.parametrize(
    "pos", [-2, {"index": 0}, {"span": [2, 3]}, {"index": -1, "scope": {"variable": "entity"}}]
)
def test_a_state_inside_the_prompt_is_refused_where_the_document_is_compiled(
    pos, patch_raw, data_root, hybrid_engine
):
    """The forward holds the state after its last token and no other: a
    position inside the prompt — or one only the row's text can place — names
    a state it never holds, and the compiler says so before anything runs."""
    patch_raw["steps"]["counterfactual"]["reads"]["s_cf"]["pos"] = pos
    with pytest.raises(PlanError, match="has no sequence axis"):
        plan.build_request(patch_raw, data_root, hybrid_engine)


def test_a_state_swapped_in_after_the_prompt_moves_what_follows_and_nothing_before(
    patch_raw, data_root, hybrid_engine
):
    """The swap lands after the prompt's last token: the prediction the prompt's
    own forward makes does not move, to the bit, and the one after the first
    decode step does. Swapped with itself, nothing moves at all."""
    executed = hybrid_engine.execute(plan.build_request(patch_raw, data_root, hybrid_engine))
    assert torch.equal(executed.result("unmoved"), torch.zeros(4))
    assert (executed.result("moved") > 0).all()

    patch_raw["steps"]["counterfactual"]["field"] = "input"
    same = hybrid_engine.execute(plan.build_request(patch_raw, data_root, hybrid_engine))
    assert torch.equal(same.result("moved"), torch.zeros(4))


def test_the_state_at_a_position_is_the_one_the_forward_hands_on(data_root, hybrid_engine):
    """Against nnterp's row read by hand in one generate over the same batch:
    `{"index": -1}` is what the prompt's forward hands on, and the continuation
    frame's step k what step k's forward hands on — bit for bit. Step 0 is the
    prompt's forward, so it is the state after the prompt again."""
    raw = json.loads(DECODE.read_text())
    executed = hybrid_engine.execute(plan.build_request(raw, data_root, hybrid_engine))
    after_prompt = executed.result("decode.after_prompt")
    per_step = executed.result("decode.per_step")
    step = executed.step("decode", plan.Generate)
    model = hybrid_engine.model
    batch = {"input_ids": torch.tensor(step.input_ids), "attention_mask": torch.tensor(step.attention_mask)}
    with model.generate(batch, max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
        by_hand = nnsight.save([])
        for _ in tracer.iter[:3]:
            by_hand.append(model.linear_attentions_state_output[1].clone())
    rows = len(step.input_ids)
    assert after_prompt.shape == (rows, 1, 8 * 32 * 32)
    assert torch.equal(after_prompt[:, 0], by_hand[0].flatten(1))
    # the continuation is gathered flat, a row's steps together, since rows may stop apart
    assert torch.equal(per_step.view(rows, 3, -1), torch.stack([one.flatten(1) for one in by_hand], 1))


def test_a_featurizer_on_the_state_names_heads(patch_raw, data_root, hybrid_engine):
    """The state is a (key, value) matrix per head with no feature axis of its
    own, so a featurizer there acts on the heads the site names, and one over
    the whole state is refused where the document is compiled."""
    patch_raw["featurizers"] = {"rot": {"kind": "subspace", "k": 4}}
    patch_raw["steps"]["patched"]["interventions"]["writes"]["swap"]["featurizer"] = "rot"
    with pytest.raises(PlanError, match="give it `heads`"):
        plan.build_request(patch_raw, data_root, hybrid_engine)
    patch_raw["sites"]["state"]["heads"] = [0]
    built = plan.build_request(patch_raw, data_root, hybrid_engine)
    featurizers = [one for step in built.steps.values() if isinstance(step, plan.Featurizers) for one in step.specs]
    assert [one.d for one in featurizers if one.name == "rot"] == [32 * 32]


def test_a_prompt_tap_and_a_step_tap_in_one_forward_run_in_rank_order(data_root, hybrid_engine):
    """The prompt frame and the continuation's first step (and `"all"`) act in
    the same forward, the prefill: the state after the prompt, read at its rank
    beside a per-head output read at every step, which comes before it."""
    raw = json.loads(DECODE.read_text())
    raw["sites"]["z"] = {"component": "linear_attention_z", "layers": 1}
    raw["steps"]["decode"]["reads"]["z"] = {"site": "z", "pos": {"frame": "generated", "all": True}}
    raw["steps"]["saves"]["decode.z"] = "z.safetensors"
    executed = hybrid_engine.execute(plan.build_request(raw, data_root, hybrid_engine))
    assert executed.result("decode.z").shape[-1] == 8 * 32
    assert executed.result("decode.after_prompt").shape[-1] == 8 * 32 * 32
