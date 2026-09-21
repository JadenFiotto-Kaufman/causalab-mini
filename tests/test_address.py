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
    """Depth first, then position inside the block, then everything after the
    stack. An interior is not a module boundary, so it needs the middle rank."""
    stack = [
        Address("lm_head"),
        Address("block_output", 1),
        Address("block_output", 0),
        Address("attention_query", 1, "attention_interface_1"),
        Address("attention_query", 0, "attention_interface_1"),
    ]
    assert [(one.component, one.layer) for one in sorted(stack, key=lambda one: one.key)] == [
        ("attention_query", 0),
        ("block_output", 0),
        ("attention_query", 1),
        ("block_output", 1),
        ("lm_head", None),
    ]


def test_a_module_boundary_cannot_carry_an_operation():
    with pytest.raises(AddressError, match="module boundary"):
        Address("block_output", 0, "attention_interface_1")


def test_a_component_with_no_address_is_refused_here():
    with pytest.raises(AddressError, match="no address here"):
        Address("attention_pattern", 0)
