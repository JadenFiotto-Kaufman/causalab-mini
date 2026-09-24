"""`attention_query`: an address that is not a module boundary.

The query tensor as the attention implementation receives it — after the
projection, after the head reshape, after RoPE — reached inside the attention
module's forward through nnsight's `.source`, by nnterp's `attention_queries`.
"""

import copy
import json
import pathlib

import nnsight
import pytest
import torch
from conftest import of_kind

from causalab_mini import ops, plan
from causalab_mini.address import Address
from causalab_mini.plan import document
from causalab_mini.engine import NNterpEngine, steps
from causalab_mini.engine.engines.nnterp import engine as nnterp

DOCUMENT = pathlib.Path(__file__).resolve().parents[1] / "documents" / "attention_query_cpu.json"


@pytest.fixture
def interior_raw():
    return json.loads(DOCUMENT.read_text())


def _build(raw, data_root, engine):
    return plan.build_request(raw, data_root, engine)


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


def test_the_interior_is_nnterps_row_for_the_interface_call(model_engine, model):
    """The address is the accessor and the layer; where that is — the
    attention's call into its implementation, positional argument 1 of the
    `(args, kwargs)` it is called with — is nnterp's row, one of those its
    source-ops suite checks on every family."""
    located = model_engine.locate("attention_query", 0)

    assert located.accessor == "attention_queries"
    assert (located.path, located.io, located.inside) == ("layers.0.self_attn", "inputs", True)
    assert located.seq_axis == 2
    # It is still pure data, and it is still what the document said.
    assert located == Address("attention_query", 0, module="self_attn", io="inputs", rank=(0, 13), inside=True)
    row = model.internals["attention_queries"].address
    assert row.op == ("attention_interface_1",) and row.select.steps == (0, 1)


def test_an_address_is_only_the_accessor_and_the_layer(model):
    """Reading needs nothing `locate` resolved: the accessor knows the rest,
    so a hand-built address reads the tensor the accessor does."""
    tokens = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    with torch.no_grad(), model.trace(tokens):
        got = nnsight.save({})
        got["address"] = nnterp.read(model, Address("attention_query", 0)).clone()
    with torch.no_grad(), model.trace(tokens):
        got["accessor"] = model.attention_queries[0].clone()
    assert torch.equal(got["address"], got["accessor"])


# --------------------------------------------------------------------- #
# what is there
# --------------------------------------------------------------------- #


def test_the_query_is_head_shaped_and_already_rotated(model_engine, model, data_root, interior_raw):
    """Two claims about the tensor: the sequence is axis 2 because the heads are
    axis 1, and it is past RoPE — so it is not the projection's output."""
    built = _build(interior_raw, data_root, model_engine)
    source_forward, _ = steps.located(model_engine, of_kind(built, plan.Forward)[0])
    tap = source_forward.taps[0]

    with model.trace(nnterp.batch(source_forward)):
        # q_proj first: nnsight enforces forward order inside a module's forward
        # exactly as it does between modules, and the projection runs before the
        # call that consumes it.
        projected = model.attentions[0].q_proj.output.clone().save()
        whole = nnterp.read(model, tap.address).clone().save()
        at_position = ops.gather(
            nnterp.read(model, tap.address), tap.reads[0].at.positions, tap.address.seq_axis
        ).clone().save()

    rows, heads, seq, head_dim = whole.shape
    assert (rows, seq) == (4, len(source_forward.input_ids[0]))
    assert heads * head_dim == model.hidden_size
    assert at_position.shape == (4, 1, heads, head_dim)  # the unit window, kept

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


def test_a_swap_at_the_interior_lands_bit_for_bit(model_engine, model, data_root, interior_raw):
    built = _build(interior_raw, data_root, model_engine)
    source_forward, patched_forward = (
        steps.located(model_engine, one)[0]
        for one in of_kind(built, plan.Forward)
    )
    read = source_forward.taps[0].reads[0]
    tap = patched_forward.taps[0]
    write = tap.writes[0]

    with model.session():
        landed = nnsight.save({})
        with model.trace(nnterp.batch(source_forward)):
            v_cf = ops.gather(
                nnterp.read(model, tap.address), read.at.positions, tap.address.seq_axis
            ).clone()
        with model.trace(nnterp.batch(patched_forward)):
            nnterp.write(
                model,
                tap.address,
                ops.apply_write(
                    nnterp.read(model, tap.address),
                    write.at.positions,
                    v_cf,
                    write.mechanism,
                    None,  # a plain patch: no featurizer
                    tap.address.seq_axis,
                ),
            )
            landed["after"] = ops.gather(
                nnterp.read(model, tap.address), write.at.positions, tap.address.seq_axis
            ).clone()
            landed["source"] = v_cf

    assert torch.equal(landed["after"], landed["source"])


def test_a_write_at_the_interior_moves_the_logits(model_engine, model, data_root, interior_raw):
    """The one that proves the write is not decoration: the same document with
    the operand taken from the counterfactual rows scores differently from the
    same document with nothing written, and identically when what is written is
    what was already there."""
    clean = model_engine.execute(_build(_no_write(interior_raw), data_root, model_engine))
    identity = model_engine.execute(_build(_identity_write(interior_raw), data_root, model_engine))
    swapped = model_engine.execute(_build(interior_raw, data_root, model_engine))

    assert torch.equal(identity.result("logit_diff"), clean.result("logit_diff"))
    assert not torch.equal(swapped.result("logit_diff"), clean.result("logit_diff"))


def test_only_the_declared_position_of_the_query_changes(model_engine, model, data_root, interior_raw):
    built = _build(interior_raw, data_root, model_engine)
    source_forward, patched_forward = (
        steps.located(model_engine, one)[0]
        for one in of_kind(built, plan.Forward)
    )
    tap = patched_forward.taps[0]
    write = tap.writes[0]
    first_token = ((0,), (2,), (0,), (2,))  # the content start of each row, left-padded

    with model.session():
        seen = nnsight.save({})
        with model.trace(nnterp.batch(source_forward)):
            v_cf = ops.gather(
                nnterp.read(model, tap.address), write.at.positions, tap.address.seq_axis
            ).clone()
        with model.trace(nnterp.batch(patched_forward)):
            seen["clean_first"] = ops.gather(
                nnterp.read(model, tap.address), first_token, tap.address.seq_axis
            ).clone()
        with model.trace(nnterp.batch(patched_forward)):
            nnterp.write(
                model,
                tap.address,
                ops.apply_write(
                    nnterp.read(model, tap.address),
                    write.at.positions,
                    v_cf,
                    write.mechanism,
                    None,  # a plain patch: no featurizer
                    tap.address.seq_axis,
                ),
            )
            seen["patched_first"] = ops.gather(
                nnterp.read(model, tap.address), first_token, tap.address.seq_axis
            ).clone()

    assert torch.equal(seen["clean_first"], seen["patched_first"])


def test_the_interior_document_runs_end_to_end(model_engine, tmp_path, data_root, model, interior_raw):
    built = _build(interior_raw, data_root, model_engine)
    results = model_engine.execute(built)
    written = results.write(tmp_path)

    assert sorted(path.name for path in written) == ["document.json", "logit_diff.json", "run.json"]
    assert len(json.loads((tmp_path / "logit_diff.json").read_text())) == 4


def test_the_interior_is_ordered_before_its_own_blocks_output(interior_raw, data_root, model_engine):
    """The forward order now has a rank *inside* a block: layer 0's query is
    read before layer 0's output. nnsight enforces it — get this wrong and the
    run raises rather than returning a wrong number."""
    raw = copy.deepcopy(interior_raw)
    raw["method"]["sites"]["block"] = {"component": "block_output", "layers": [0]}
    raw["method"]["reads"]["after"] = {
        "site": "block", "pos": -1, "model": "patched", "input": "base",
    }
    built = _build(raw, data_root, model_engine)

    assert [tap.address.component for tap in of_kind(built, plan.Forward)[1].taps] == [
        "attention_query",
        "block_output",
        "lm_head",
    ]
    model_engine.execute(built)


def test_the_interior_plan_survives_the_remote_path(model_engine, model, data_root, interior_raw):
    """`remote="local"` serializes the session exactly as a remote run would.
    An interior address is a component, a layer and an operation name, so there
    is nothing in it that cannot make the trip."""
    built = _build(interior_raw, data_root, model_engine)
    here = model_engine.execute(built)
    shipped = model_engine.execute(built, remote="local")
    assert torch.equal(here.result("logit_diff"), shipped.result("logit_diff"))
