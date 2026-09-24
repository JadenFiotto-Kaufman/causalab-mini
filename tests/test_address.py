"""`Address`: the one type that knows about models, and holds none of one."""

import pickle

import pytest

from causalab_mini.address import Address, AddressError


def test_an_address_is_the_documents_words_and_pickles_as_such():
    one = Address("block_output", 0)
    assert pickle.loads(pickle.dumps(one)) == one
    # every place is a nnterp accessor, whose module the checkpoint spells —
    # which the document's words alone cannot say
    for bare in (one, Address("lm_head"), Address("attention_query", 0)):
        with pytest.raises(AddressError, match="engine.locate"):
            bare.path
    assert Address("lm_head", module="lm_head").path == "lm_head"
    assert Address("block_output", 0, module="").path == "layers.0"
    assert Address("block_mid", 1, module="post_attention_layernorm").path == "layers.1.post_attention_layernorm"


def test_addresses_sort_into_forward_order(model_engine):
    """Depth first, then position inside the block, then everything after the
    stack. The query is inside the attention, so it comes before the block's
    output at its layer.

    The rank is nnterp's — `internals.rank`, stamped by `locate` — because a
    family may build its block differently and mini keeping a second
    numbering beside nnterp's is how the two drift."""
    stack = [
        model_engine.locate("lm_head"),
        model_engine.locate("block_output", 1),
        model_engine.locate("block_output", 0),
        model_engine.locate("attention_query", 1),
        model_engine.locate("attention_query", 0),
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


def test_a_component_only_nnterp_knows_is_addressable(model_engine):
    """nnterp's accessors are an extension point — `RenameConfig(addresses=
    {...})` — and a document should be able to name what a user added there.
    Mini claims nothing else about such a place: no width, so a featurizer
    there is refused rather than sized wrongly."""
    located = model_engine.locate("attentions_input", 0)
    assert located.accessor == "attentions_input" and located.rank is not None
    assert located.width_attribute is None
    with pytest.raises(AddressError, match="not a fact about the model"):
        model_engine.width(located)
