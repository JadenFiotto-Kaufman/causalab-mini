"""`Address`: the one type that knows about models, and holds none of one."""

import pickle

import pytest

from causalab_mini.address import Address, AddressError


def test_an_address_is_the_documents_words_and_pickles_as_such():
    one = Address("block_output", 0)
    assert (one.path, one.side) == ("layers.0", "output")
    assert pickle.loads(pickle.dumps(one)) == one
    assert Address("lm_head").path == "lm_head"


def test_addresses_sort_into_forward_order():
    stack = [Address("lm_head"), Address("block_output", 1), Address("block_output", 0)]
    assert sorted(stack, key=lambda one: one.key) == [
        Address("block_output", 0),
        Address("block_output", 1),
        Address("lm_head"),
    ]


def test_a_component_with_no_address_is_refused_here():
    with pytest.raises(AddressError, match="no address here"):
        Address("attention_probs", 0)
