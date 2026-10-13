"""`Address`: the one type that knows about models, and holds none of one."""

import pickle

import pytest

from causalab_mini import address
from causalab_mini.address import AddressError


def test_an_address_is_stamped_whole_and_pickles_as_such(model_engine, monkeypatch):
    """Everything an address says is written into it by `locate`, so a plan
    unpickled where another table is installed — or none — means what it
    meant where it was compiled."""
    one = model_engine.locate("block_mid", 1)
    shipped = pickle.dumps(one)
    monkeypatch.setattr(address, "COMPONENTS", {})
    back = pickle.loads(shipped)
    assert back == one
    assert (back.path, back.value, back.seq_axis) == ("layers.1.post_attention_layernorm", "input", 1)


def test_addresses_sort_into_forward_order(eager_engine):
    """Depth first, then position inside the block, then everything after the
    stack. The query is inside the attention, so it comes before the block's
    output at its layer.

    The rank is nnterp's `rank`, stamped by `locate`, because a family may
    build its block differently and mini keeping a second numbering beside
    nnterp's is how the two drift."""
    stack = [
        eager_engine.locate("lm_head"),
        eager_engine.locate("block_output", 1),
        eager_engine.locate("block_output", 0),
        eager_engine.locate("attention_query", 1),
        eager_engine.locate("attention_query", 0),
    ]
    assert [(one.component, one.layer) for one in sorted(stack, key=lambda one: one.rank)] == [
        ("attention_query", 0),
        ("block_output", 0),
        ("attention_query", 1),
        ("block_output", 1),
        ("lm_head", None),
    ]


def test_a_component_neither_table_has_is_refused_by_name(model_engine):
    with pytest.raises(AddressError, match="no address here"):
        model_engine.locate("attention_pattern", 0)


def test_a_component_only_nnterp_knows_is_addressable(eager_engine):
    """nnterp's values are an extension point — `envoys=` installs one on a
    family's envoy — and a document should be able to name what nnterp
    serves by its dotted `support()` name. Its axes are its value's own
    layout, so it is sized like any row here: the values entering the
    attention are the key heads', `head_dim` deep."""
    located = eager_engine.locate("self_attn.attention_values", 0)
    model = eager_engine.model
    assert (located.path, located.value, located.inside) == ("layers.0.self_attn", "attention_values", True)
    assert located.rank == eager_engine.locate("attention_key", 0).rank, "served at the same call"
    assert located.heads_kind == "kv" and located.seq_axis == 2
    attention = model.layers[0].self_attn
    assert eager_engine.width(located) == attention.num_kv_heads * attention.head_dim



def test_the_tables_axis_names_are_nnterps():
    # The table spells axis names as literals so a plan loads without nnterp;
    # this holds the literals to nnterp's own constants, where an axis that is
    # a model fact is named as the size attribute that resolves it.
    from nnterp import Mlp
    from nnterp.components import BATCH, HEAD_DIM, HEADS, HIDDEN, KV_HEADS, QK_HEAD_DIM, SEQ, VOCAB

    assert address._STREAM == (BATCH, SEQ, HIDDEN)
    assert hasattr(Mlp, "intermediate_size")
    known = {BATCH, SEQ, HIDDEN, VOCAB, HEADS, KV_HEADS, HEAD_DIM, QK_HEAD_DIM, "intermediate_size"}
    used = {axis for entry in address.COMPONENTS.values() if entry.boundary for axis in entry.boundary[0]}
    assert used <= known
