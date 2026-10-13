"""`attention_query`: an address that is not a module boundary.

The query tensor as the attention implementation receives it — after the
projection, after the head reshape, after RoPE — reached inside the attention
module's forward through nnsight's `.source`, by nnterp's `attention_queries`,
which nnterp serves under eager attention.
"""

import copy
import json
import pathlib

import nnsight
import pytest
import torch
from conftest import of_kind

from causalab_mini import ops, plan
from causalab_mini.address import Address, AddressError
from causalab_mini.engine import NNterpEngine, steps
from causalab_mini.engine.engines.nnterp import engine as nnterp

DOCUMENT = pathlib.Path(__file__).resolve().parents[1] / "documents" / "v2" / "attention_query.json"


@pytest.fixture(scope="module")
def model(eager_engine):
    """This file's model: the document's, which runs eager attention."""
    return eager_engine.model


@pytest.fixture
def interior_raw():
    return json.loads(DOCUMENT.read_text())


def _build(raw, data_root, engine):
    return plan.build_request(raw, data_root, engine)


def _no_write(raw):
    raw = copy.deepcopy(raw)
    del raw["steps"]["patched"]["interventions"]
    return raw


def _identity_write(raw):
    """The operand read off the base input at the same address, so the write
    puts back exactly what was already there."""
    raw = copy.deepcopy(raw)
    raw["steps"]["counterfactual"]["field"] = "input"
    return raw


# --------------------------------------------------------------------- #
# the address
# --------------------------------------------------------------------- #


def test_the_interior_is_nnterps_value_on_the_attention(eager_engine):
    """Where the query is — the attention's call into its implementation, the
    query argument of the `(args, kwargs)` it is called with — is nnterp's,
    and so are its axes: heads before tokens, so the sequence is axis 2."""
    located = eager_engine.locate("attention_query", 0)

    assert located.seq_axis == 2 and located.heads_kind == "q"
    # It is still pure data, and it is still what the document said.
    assert located == Address("attention_query", 0, ("batch", "num_heads", "seq", "qk_head_dim"))


def test_an_address_reads_what_nnterps_value_reads(eager_engine, model):
    """A read at an address is the value off the envoy, nothing of mini's in
    between: the same tensor `model.layers[0].self_attn.attention_queries`
    is."""
    located = eager_engine.locate("attention_query", 0)
    tokens = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    with torch.no_grad(), model.trace(tokens):
        got = nnsight.save({})
        got["address"] = nnterp.read(model, located).clone()
    with torch.no_grad(), model.trace(tokens):
        got["value"] = model.layers[0].self_attn.attention_queries.clone()
    assert torch.equal(got["address"], got["value"])


# --------------------------------------------------------------------- #
# what is there
# --------------------------------------------------------------------- #


def test_the_query_is_head_shaped_and_already_rotated(eager_engine, model, data_root, interior_raw):
    """Two claims about the tensor: the sequence is axis 2 because the heads are
    axis 1, and it is past RoPE — so it is not the projection's output."""
    built = _build(interior_raw, data_root, eager_engine)
    source_forward, _ = steps.located(eager_engine, of_kind(built, plan.Forward)[0])
    tap = source_forward.taps[0]

    with model.trace(nnterp.batch(source_forward)):
        # q_proj first: nnsight enforces forward order inside a module's forward
        # exactly as it does between modules, and the projection runs before the
        # call that consumes it.
        projected = model.layers[0].self_attn.q_proj.output.clone().save()
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


def test_the_interior_is_refused_under_sdpa_and_says_how_to_get_it(model_engine):
    """nnterp serves the attention's interior under eager attention only — the
    query as well as the scores and the pattern, though the call whose
    argument the query is runs under sdpa too — so the document says eager,
    and a model running sdpa refuses the address with the fix."""
    assert model_engine.model.config._attn_implementation == "sdpa"
    with pytest.raises(AddressError, match="load with attn_implementation='eager'"):
        model_engine.locate("attention_query", 0)


# --------------------------------------------------------------------- #
# read and write
# --------------------------------------------------------------------- #


def test_a_swap_at_the_interior_lands_bit_for_bit(eager_engine, model, data_root, interior_raw):
    built = _build(interior_raw, data_root, eager_engine)
    source_forward, patched_forward = (
        steps.located(eager_engine, one)[0]
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


def test_a_write_at_the_interior_moves_the_logits(eager_engine, model, data_root, interior_raw):
    """The one that proves the write is not decoration: the same document with
    the operand taken from the counterfactual rows scores differently from the
    same document with nothing written, and identically when what is written is
    what was already there."""
    clean = eager_engine.execute(_build(_no_write(interior_raw), data_root, eager_engine))
    identity = eager_engine.execute(_build(_identity_write(interior_raw), data_root, eager_engine))
    swapped = eager_engine.execute(_build(interior_raw, data_root, eager_engine))

    assert torch.equal(identity.result("logit_diff"), clean.result("logit_diff"))
    assert not torch.equal(swapped.result("logit_diff"), clean.result("logit_diff"))


def test_only_the_declared_position_of_the_query_changes(eager_engine, model, data_root, interior_raw):
    built = _build(interior_raw, data_root, eager_engine)
    source_forward, patched_forward = (
        steps.located(eager_engine, one)[0]
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


def test_the_interior_document_runs_end_to_end(eager_engine, tmp_path, data_root, model, interior_raw):
    built = _build(interior_raw, data_root, eager_engine)
    results = eager_engine.execute(built)
    written = results.write(tmp_path)

    assert sorted(path.name for path in written) == ["document.json", "logit_diff.json", "run.json"]
    assert len(json.loads((tmp_path / "logit_diff.json").read_text())) == 4


def test_the_interior_is_ordered_before_its_own_blocks_output(interior_raw, data_root, eager_engine):
    """The forward order now has a rank *inside* a block: layer 0's query is
    read before layer 0's output. nnsight enforces it — get this wrong and the
    run raises rather than returning a wrong number.

    The plan keeps the document's order — the write, then the reads as
    written — and the engine puts them in the model's at execute."""
    raw = copy.deepcopy(interior_raw)
    raw["sites"]["block"] = {"component": "block_output", "layers": 0}
    raw["steps"]["patched"]["reads"]["after"] = {"site": "block", "pos": -1}
    built = _build(raw, data_root, eager_engine)

    taps = of_kind(built, plan.Forward)[1].taps
    assert [tap.address.component for tap in taps] == ["attention_query", "lm_head", "block_output"]
    assert [tap.address.component for tap in nnterp.in_forward_order(eager_engine.model, taps)] == [
        "attention_query",
        "block_output",
        "lm_head",
    ]
    eager_engine.execute(built)


def test_taps_the_model_reaches_together_keep_the_documents_order(eager_engine):
    """The query and the key are served at one call, so neither is before
    the other; the sort is stable, and the document's order stands."""
    query, key = (plan.Tap(eager_engine.locate(name, 0), (), ()) for name in ("attention_query", "attention_key"))
    model = eager_engine.model
    assert nnterp.in_forward_order(model, (query, key)) == (query, key)
    assert nnterp.in_forward_order(model, (key, query)) == (key, query)


def test_an_engine_that_has_not_measured_its_order_measures_it_in_the_session(
    eager_engine, data_root, interior_raw
):
    """A server's model has not been asked for its order when a plan
    arrives; the sort asks inside the session, before each trace, and nnterp
    measures it there."""
    built = _build(interior_raw, data_root, eager_engine)
    here = eager_engine.execute(built)
    eager_engine.model._order = None
    shipped = eager_engine.execute(built, remote="local")
    assert torch.equal(here.result("logit_diff"), shipped.result("logit_diff"))


def test_the_interior_plan_survives_the_remote_path(eager_engine, model, data_root, interior_raw):
    """`remote="local"` serializes the session exactly as a remote run would.
    An interior address is a component, a layer and an operation name, so there
    is nothing in it that cannot make the trip."""
    built = _build(interior_raw, data_root, eager_engine)
    here = eager_engine.execute(built)
    shipped = eager_engine.execute(built, remote="local")
    assert torch.equal(here.result("logit_diff"), shipped.result("logit_diff"))
