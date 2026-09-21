"""Where the activation sits inside what a module boundary hands over.

By default that is decided from the value — a tuple's first element, or the
tensor itself — because which of the two a block returns is a property of
the transformers version. When a family puts it somewhere else, the table
says so: a `select` path, or a `Lens` of two functions, in an override keyed
by `config.model_type`. A read needs the way in and a write needs the way
back, so both forms are a pair.
"""

import json
import pathlib
from collections import OrderedDict, namedtuple

import pytest
import torch

from causalab_mini import address as address_module
from causalab_mini import plan
from causalab_mini.address import Address, AddressError, Lens
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCHING = REPO / "documents" / "v2" / "patching.json"

HIDDEN, OTHER, NEW = torch.ones(2, 3), torch.zeros(5), torch.full((2, 3), 7.0)


@pytest.fixture
def override(monkeypatch):
    """Install one exception for the length of a test."""

    def install(family, component, **differs):
        monkeypatch.setitem(address_module._OVERRIDES, (family, component), differs)

    return install


# --------------------------------------------------------------------- #
# the pair
# --------------------------------------------------------------------- #


def test_with_no_select_the_value_decides():
    at = Address("block_output", 0)
    assert at.get(HIDDEN) is HIDDEN and at.put(HIDDEN, NEW) is NEW
    assert at.get((HIDDEN, OTHER)) is HIDDEN
    rebuilt = at.put((HIDDEN, OTHER), NEW)
    assert rebuilt[0] is NEW and rebuilt[1] is OTHER


def test_a_path_is_walked_in_and_rebuilt_on_the_way_out(override):
    override("some_moe", "block_output", select=(1,))
    at = Address("block_output", 0, family="some_moe")
    value = (OTHER, HIDDEN, "cache")

    assert at.get(value) is HIDDEN
    assert at.put(value, NEW) == (OTHER, NEW, "cache") and value[1] is HIDDEN, "the caller's tuple is not edited"
    # every other family still gets the default
    assert Address("block_output", 0, family="llama").get(value) is OTHER
    assert Address("block_output", 0).get(value) is OTHER


def test_a_path_goes_through_keys_named_tuples_and_nesting(override):
    Output = namedtuple("Output", ["router", "states"])
    override("nested", "block_output", select=("states", 0))
    at = Address("block_output", 0, family="nested")
    value: OrderedDict = OrderedDict(router=OTHER, states=Output(HIDDEN, OTHER))

    assert at.get(value) is HIDDEN
    rebuilt = at.put(value, NEW)
    assert isinstance(rebuilt["states"], Output) and rebuilt["states"].router is NEW
    assert value["states"].router is HIDDEN


def test_a_lens_is_two_functions(override):
    """Anything a path cannot say — here an activation handed over packed as
    `(tokens, d)`, which the rest of the library wants as `(batch, seq, d)`."""
    lens = Lens(get=lambda v: v.reshape(2, -1, 3), put=lambda v, t: t.reshape(-1, 3))
    override("packed", "mlp_output", select=lens)
    at = Address("mlp_output", 0, family="packed")
    packed = torch.arange(12.0).reshape(4, 3)

    assert at.get(packed).shape == (2, 2, 3)
    assert at.put(packed, at.get(packed) * 2).shape == (4, 3)


def test_a_select_that_does_not_reach_a_tensor_is_refused(override):
    override("wrong", "block_output", select=(1,))
    at = Address("block_output", 0, family="wrong")
    with pytest.raises(AddressError, match="reaches a NoneType, not a tensor"):
        at.get((HIDDEN, None))
    with pytest.raises(AddressError, match=r"no 1 in a tuple"):
        at.get((HIDDEN,))


def test_an_address_is_still_data(override):
    """The family is a string and the functions stay in the table, so a plan
    that holds this address pickles and compares as before."""
    import pickle

    override("packed", "mlp_output", select=Lens(get=lambda v: v, put=lambda v, t: t))
    at = Address("mlp_output", 0, family="packed")
    assert pickle.loads(pickle.dumps(at)) == at


# --------------------------------------------------------------------- #
# through both engines
# --------------------------------------------------------------------- #


def _flip_get(value):
    value = value[0] if isinstance(value, tuple) else value
    CALLS.append("get")
    return value.flip(-1)


def _flip_put(value, tensor):
    CALLS.append("put")
    tensor = tensor.flip(-1)
    return (tensor, *value[1:]) if isinstance(value, tuple) else tensor


CALLS: list[str] = []


@pytest.fixture
def raw():
    one = json.loads(PATCHING.read_text())
    one["sites"]["target"]["component"] = "attention_output"
    return one


@pytest.mark.parametrize("engine_name", ["nnterp", "hooks"])
def test_both_engines_read_and_write_through_the_select(engine_name, raw, data_root, model_engine, override):
    """A lens that reverses the width axis on the way in and again on the way
    out is invisible to a full-width swap — so the run must be bit-equal to
    the run without it, and the lens must have been used for the read and
    for the write. That is the wiring, end to end, on a real model."""
    engine = model_engine if engine_name == "nnterp" else HooksEngine.load(
        Spec.model_validate(raw).model, device_map="cpu"
    )
    plain = engine.execute(plan.build_request(raw, data_root, engine)).result("logit_diff")

    override("llama", "attention_output", select=Lens(get=_flip_get, put=_flip_put))
    CALLS.clear()
    through = engine.execute(plan.build_request(raw, data_root, engine)).result("logit_diff")

    assert torch.equal(through, plain)
    assert "get" in CALLS and "put" in CALLS


def test_an_explicit_path_on_a_real_tuple(raw, data_root, override):
    """A raw HuggingFace attention module returns `(output, weights)`. Saying
    `(0,)` out loud is the default, bit for bit; saying `(1,)` reaches the
    weights, which sdpa does not return, and is refused by name rather than
    swapped in."""
    engine = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    plain = engine.execute(plan.build_request(raw, data_root, engine)).result("logit_diff")

    override("llama", "attention_output", select=(0,))
    assert torch.equal(engine.execute(plan.build_request(raw, data_root, engine)).result("logit_diff"), plain)

    override("llama", "attention_output", select=(1,))
    with pytest.raises(AddressError, match="reaches a NoneType, not a tensor"):
        engine.execute(plan.build_request(raw, data_root, engine))
