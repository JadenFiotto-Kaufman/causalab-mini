"""`Address`: the one type that knows about models, and holds none of one."""

import pickle

import pytest

from causalab_mini.address import Address, AddressError


def test_an_address_is_the_documents_words_and_pickles_as_such():
    one = Address("block_output", 0)
    assert pickle.loads(pickle.dumps(one)) == one
    # every place is served by an envoy which one this checkpoint has decides
    # — the document's words alone cannot say — and has axes its value's
    # layout on the checkpoint says
    for bare in (one, Address("lm_head"), Address("attention_query", 0)):
        with pytest.raises(AddressError, match="engine.locate"):
            bare.path
        with pytest.raises(AddressError, match="engine.locate"):
            bare.seq_axis
    assert Address("lm_head", module="lm_head").path == "lm_head"
    assert Address("block_output", 0, module="").path == "layers.0"
    assert Address("block_mid", 1, module="post_attention_layernorm").path == "layers.1.post_attention_layernorm"


def test_addresses_sort_into_forward_order(eager_engine):
    """Depth first, then position inside the block, then everything after the
    stack. The query is inside the attention, so it comes before the block's
    output at its layer.

    The rank is nnterp's — `rank`, stamped by `locate` — because a family may
    build its block differently and mini keeping a second numbering beside
    nnterp's is how the two drift."""
    stack = [
        eager_engine.locate("lm_head"),
        eager_engine.locate("block_output", 1),
        eager_engine.locate("block_output", 0),
        eager_engine.locate("attention_query", 1),
        eager_engine.locate("attention_query", 0),
    ]
    assert [(one.component, one.layer) for one in sorted(stack, key=lambda one: one.key)] == [
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
    assert (located.module, located.value, located.inside) == ("self_attn", "attention_values", True)
    assert located.rank == eager_engine.locate("attention_key", 0).rank, "served at the same call"
    assert located.heads_kind == "kv" and located.seq_axis == 2
    attention = model.layers[0].self_attn
    assert eager_engine.width(located) == attention.num_kv_heads * attention.head_dim

