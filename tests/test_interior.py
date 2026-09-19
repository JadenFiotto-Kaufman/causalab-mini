"""`attention_query`: the one address that is not a module boundary.

The query tensor as the attention implementation receives it — after the
projection, after the head reshape, after RoPE — reached inside the attention
module's forward through nnsight's `.source`.
"""

import copy
import json
import pathlib

import pytest
import torch

from causalab_mini import ops, plan
from causalab_mini.model.address import Address, AddressError, find_op
from causalab_mini.plan import document
from causalab_mini.session import observe, run

DOCUMENT = pathlib.Path(__file__).resolve().parents[1] / "documents" / "attention_query_cpu.json"


@pytest.fixture
def interior_raw():
    return json.loads(DOCUMENT.read_text())


def _build(raw, data_root, model):
    return plan.build(document.Document.from_json(raw), data_root, model)


def _no_write(raw):
    raw = copy.deepcopy(raw)
    del raw["method"]["reads"]["v_cf"], raw["method"]["writes"], raw["method"]["intervened_models"]
    raw["method"]["reads"]["logits"]["model"] = "original"
    for entry in raw["method"]["save"]:
        entry["model"] = "original"
    return raw


def _identity_write(raw):
    """The operand read off the base input at the same address, so the write
    puts back exactly what was already there."""
    raw = copy.deepcopy(raw)
    raw["method"]["reads"]["v_cf"]["input"] = "base"
    return raw


# --------------------------------------------------------------------- #
# the address
# --------------------------------------------------------------------- #


def test_the_interior_is_addressed_by_the_interface_call(model):
    located = Address.locate(model, "attention_query", 0)

    assert located.path == "attentions.0"
    assert located.op == "attention_interface_1"
    assert located.seq_axis == 2
    # It is still pure data, and it is still what the document said.
    assert located == Address("attention_query", 0, "attention_interface_1")


def test_a_binding_and_a_call_share_one_namespace_so_the_name_alone_is_ambiguous(model):
    """The reason the needle is call-shaped. nnsight's occurrence suffix counts
    assignments and calls together, so this forward has both
    `attention_interface_0` (the assignment that looks the implementation up)
    and `attention_interface_1` (the call that runs it)."""
    source = model.attentions[0].source
    assert {"attention_interface_0", "attention_interface_1"} <= set(source.names)

    with pytest.raises(AddressError, match="serves exactly one") as refusal:
        find_op(source, "attention_interface")
    # Three, in fact: the lookup call, the name it binds, and the call itself.
    assert "matches 3 operations" in str(refusal.value)


def test_a_needle_that_matches_nothing_prints_the_inventory_it_saw(model):
    with pytest.raises(AddressError, match="attention_probs") as refusal:
        find_op(model.attentions[0].source, "attention_probs(")
    assert "matches 0 operations" in str(refusal.value)
    assert "self_q_proj_0" in str(refusal.value)  # the inventory is printed


def test_an_interior_address_built_without_a_model_refuses_rather_than_guesses(model):
    with pytest.raises(AddressError, match="Address.locate"):
        with model.trace(
            {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        ):
            Address("attention_query", 0).read(model)


# --------------------------------------------------------------------- #
# what is there
# --------------------------------------------------------------------- #


def test_the_query_is_head_shaped_and_already_rotated(model, data_root, interior_raw):
    """Two claims about the tensor: the sequence is axis 2 because the heads are
    axis 1, and it is past RoPE — so it is not the projection's output."""
    built = _build(interior_raw, data_root, model)
    source_forward = built.forwards[0]
    tap = source_forward.taps[0]

    with model.trace(observe.batch(source_forward)):
        # q_proj first: nnsight enforces forward order inside a module's forward
        # exactly as it does between modules, and the projection runs before the
        # call that consumes it.
        projected = model.attentions[0].q_proj.output.clone().save()
        whole = tap.address.read(model).clone().save()
        at_position = ops.gather(
            tap.address.read(model), tap.reads[0].positions, tap.address.seq_axis
        ).clone().save()

    rows, heads, seq, head_dim = whole.shape
    assert (rows, seq) == (4, len(source_forward.input_ids[0]))
    assert heads * head_dim == model.hidden_size
    assert at_position.shape == (4, heads, head_dim)

    # The same numbers, reshaped the way the forward reshapes them, before RoPE.
    before = projected.view(rows, seq, heads, head_dim).transpose(1, 2)
    assert before.shape == whole.shape
    assert not torch.equal(before, whole)


def test_the_interior_is_reached_under_the_checkpoints_own_attention(model):
    """Reading and writing q needs no eager attention — only scores and
    probabilities do, and those are one level deeper, inside the implementation
    this call dispatches to."""
    assert model.config._attn_implementation == "sdpa"


# --------------------------------------------------------------------- #
# read and write
# --------------------------------------------------------------------- #


def test_a_swap_at_the_interior_lands_bit_for_bit(model, data_root, interior_raw):
    built = _build(interior_raw, data_root, model)
    source_forward, patched_forward = built.forwards
    read = source_forward.taps[0].reads[0]
    tap = patched_forward.taps[0]
    write = tap.writes[0]

    with model.session():
        landed = run.nnsight.save({})
        with model.trace(observe.batch(source_forward)):
            v_cf = ops.gather(
                tap.address.read(model), read.positions, tap.address.seq_axis
            ).clone()
        with model.trace(observe.batch(patched_forward)):
            tap.address.write(
                model,
                ops.apply_write(
                    tap.address.read(model),
                    write.positions,
                    v_cf,
                    write.mechanism,
                    write.featurizer,
                    tap.address.seq_axis,
                ),
            )
            landed["after"] = ops.gather(
                tap.address.read(model), write.positions, tap.address.seq_axis
            ).clone()
            landed["source"] = v_cf

    assert torch.equal(landed["after"], landed["source"])


def test_a_write_at_the_interior_moves_the_logits(model, data_root, interior_raw):
    """The one that proves the write is not decoration: the same document with
    the operand taken from the counterfactual rows scores differently from the
    same document with nothing written, and identically when what is written is
    what was already there."""
    clean = run.execute(model, _build(_no_write(interior_raw), data_root, model))
    identity = run.execute(model, _build(_identity_write(interior_raw), data_root, model))
    swapped = run.execute(model, _build(interior_raw, data_root, model))

    assert torch.equal(identity["logit_diff"], clean["logit_diff"])
    assert not torch.equal(swapped["logit_diff"], clean["logit_diff"])


def test_only_the_declared_position_of_the_query_changes(model, data_root, interior_raw):
    built = _build(interior_raw, data_root, model)
    source_forward, patched_forward = built.forwards
    tap = patched_forward.taps[0]
    write = tap.writes[0]
    first_token = (0, 2, 0, 2)  # the content start of each row, left-padded

    with model.session():
        seen = run.nnsight.save({})
        with model.trace(observe.batch(source_forward)):
            v_cf = ops.gather(
                tap.address.read(model), write.positions, tap.address.seq_axis
            ).clone()
        with model.trace(observe.batch(patched_forward)):
            seen["clean_first"] = ops.gather(
                tap.address.read(model), first_token, tap.address.seq_axis
            ).clone()
        with model.trace(observe.batch(patched_forward)):
            tap.address.write(
                model,
                ops.apply_write(
                    tap.address.read(model),
                    write.positions,
                    v_cf,
                    write.mechanism,
                    write.featurizer,
                    tap.address.seq_axis,
                ),
            )
            seen["patched_first"] = ops.gather(
                tap.address.read(model), first_token, tap.address.seq_axis
            ).clone()

    assert torch.equal(seen["clean_first"], seen["patched_first"])


def test_the_interior_document_runs_end_to_end(tmp_path, data_root, model, interior_raw):
    from causalab_mini import output

    built = _build(interior_raw, data_root, model)
    results = run.execute(model, built)
    written = output.write_results(tmp_path, built, results)

    assert [path.name for path in written] == ["logit_diff.json"]
    assert len(json.loads((tmp_path / "logit_diff.json").read_text())) == 4


def test_the_interior_is_ordered_before_its_own_blocks_output(interior_raw, data_root, model):
    """The forward order now has a rank *inside* a block: layer 0's query is
    read before layer 0's output. nnsight enforces it — get this wrong and the
    run raises rather than returning a wrong number."""
    raw = copy.deepcopy(interior_raw)
    raw["method"]["sites"]["block"] = {"component": "block_output", "layers": [0]}
    raw["method"]["reads"]["after"] = {
        "site": "block", "pos": -1, "model": "patched", "input": "base",
    }
    built = _build(raw, data_root, model)

    assert [tap.address.component for tap in built.forwards[1].taps] == [
        "attention_query",
        "block_output",
        "lm_head",
    ]
    run.execute(model, built)


def test_the_interior_plan_survives_the_remote_path(model, data_root, interior_raw):
    """`remote="local"` serializes the session exactly as a remote run would.
    An interior address is a component, a layer and an operation name, so there
    is nothing in it that cannot make the trip."""
    built = _build(interior_raw, data_root, model)
    here = run.execute(model, built)
    shipped = run.execute(model, built, remote="local")
    assert torch.equal(here["logit_diff"], shipped["logit_diff"])
