"""The resolver, against the client-side one it replaces.

`ops/locate.py` answers "which indices does this position name, on this row"
where the model is. Everything it has to agree with is here: for every
position form and every batch the suite tokenizes, the windows it returns are
the ones `data/encoding.positions` returned on the client, index for index.

The expected windows are written out as literals as well as compared, so this
file keeps saying what the forms mean after the client-side resolver is gone.
"""

import pytest

from causalab_mini.data import encoding
from causalab_mini.ops import locate
from causalab_mini.shapes import Anchor, Where

# The prompts the suite tokenizes, and what each one is for.
PAIR = ["If today is Thursday, tomorrow is", "If today is Friday, tomorrow is"]
FOUR = [
    "If today is Thursday, tomorrow is",
    "If today is Friday, tomorrow is",
    "If today is Saturday, tomorrow is",
    "If today is Sunday, tomorrow is",
]
EVEN = ["If today is Friday, tomorrow is", "If today is Sunday, tomorrow is"]


@pytest.fixture
def frames(model):
    """Each batch as both resolvers see it: the client's `Batch`, and the
    `Frame` the block builds from the same ids."""
    made = {}
    for name, texts in (("pair", PAIR), ("four", FOUR), ("even", EVEN)):
        batch = encoding.encode(model.tokenizer, texts)
        frame = locate.frame_of(model.tokenizer, batch.input_ids, batch.attention_mask)
        made[name] = (batch, frame)
    return made


# --------------------------------------------------------------------- #
# every form, on every batch
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pos, where, expected",
    [
        (-1, Where(index=-1), {"pair": ((10,), (10,)), "even": ((8,), (8,))}),
        ({"index": 0}, Where(index=0), {"pair": ((0,), (2,)), "even": ((0,), (0,))}),
        ({"last": 3}, Where(last=3), {"pair": ((8, 9, 10),) * 2, "even": ((6, 7, 8),) * 2}),
        ({"span": [0, 2]}, Where(span=(0, 2)), {"pair": ((0, 1), (2, 3)), "even": ((0, 1),) * 2}),
        ({"span": [-3, -1]}, Where(span=(-3, -1)), {"pair": ((8, 9),) * 2, "even": ((6, 7),) * 2}),
        (
            {"all": True},
            Where(all=True),
            {"pair": (tuple(range(11)), tuple(range(2, 11))), "even": (tuple(range(9)),) * 2},
        ),
    ],
    ids=["-1", "index 0", "last 3", "span from the start", "negative span", "all"],
)
def test_locate_resolves_what_the_client_resolved(frames, pos, where, expected):
    for name, (batch, frame) in frames.items():
        windows, reasons = locate.locate(frame, where)
        assert windows == encoding.positions(batch, pos), name
        assert set(reasons) == {""}, name
        if name in expected:
            assert windows == expected[name], name


def test_a_variable_anchor_resolves_what_a_column_resolved(frames):
    """`{"column": c}` was the only text form the client ran, and
    `{"all": true, "scope": {"variable": c}}` is what it becomes: the tokens
    of this row's own text, located in its prompt. ` Thursday` is three
    tokens on this tokenizer and ` Friday` one."""
    batch, frame = frames["pair"]
    anchors = ("Thursday", "Friday")
    windows, reasons = locate.locate(frame, Where(all=True, scope=Anchor(variable="entity")), anchors)

    assert windows == encoding.positions(batch, {"column": "entity"}, list(anchors))
    assert [len(one) for one in windows] == [3, 1]
    assert set(reasons) == {""}


def test_a_row_whose_text_is_not_in_its_prompt_says_why(frames):
    batch, frame = frames["pair"]
    where = Where(all=True, scope=Anchor(variable="entity"))
    windows, reasons = locate.locate(frame, where, ("Thursday", "Neptune"))

    assert windows == encoding.positions(batch, {"column": "entity"}, ["Thursday", "Neptune"])
    assert windows[1] == () and reasons == ("", "alignment_missing")


def test_a_value_that_occurs_twice_is_ambiguous_rather_than_the_first_one(model):
    """No occurrence index is minted: the fix an author wants is to scope the
    anchor, and a silently-first window is the wrong number."""
    _, _, frame = locate.frame_of_texts(model.tokenizer, ["Friday, then Friday again"])
    windows, reasons = locate.locate(frame, Where(all=True, scope=Anchor(variable="x")), ("Friday",))
    assert windows == ((),) and reasons == ("alignment_ambiguous",)


@pytest.mark.parametrize(
    "pos, where",
    [
        ({"last": 12}, Where(last=12)),
        ({"span": [0, 40]}, Where(span=(0, 40))),
        ({"index": 40}, Where(index=40)),
        ({"index": -40}, Where(index=-40)),
    ],
    ids=["too wide", "past the end", "index past the end", "index before the start"],
)
def test_a_cut_outside_the_run_is_a_reason_where_it_was_a_refusal(frames, pos, where):
    """The one behaviour that changes shape: the client raised before the run,
    and the block reports per row, because on a text-anchored run only the row
    knows how long it is."""
    batch, frame = frames["pair"]
    with pytest.raises(encoding.EncodingError, match="outside the row's content"):
        encoding.positions(batch, pos)
    windows, reasons = locate.locate(frame, where)
    assert windows == ((), ()) and set(reasons) == {"out_of_range"}


# --------------------------------------------------------------------- #
# the frame itself
# --------------------------------------------------------------------- #


def test_the_frame_is_the_batch_the_client_encoded(frames, model):
    for name, (batch, frame) in frames.items():
        assert (frame.starts, frame.ends) == (batch.starts, batch.ends), name
        assert (frame.texts, frame.offsets) == (batch.texts, batch.offsets), name
    ids, mask, frame = locate.frame_of_texts(model.tokenizer, PAIR)
    assert (ids, mask) == (frames["pair"][0].input_ids, frames["pair"][0].attention_mask)


def test_the_continuation_stops_at_the_first_eos_and_names_it(model):
    """The decode runs to the bound; the frame does not. A row that stopped
    has an `eos` segment and one that did not has none, which is how "did the
    model stop?" becomes a reported reason."""
    eos = int(model.tokenizer.eos_token_id)
    friday = model.tokenizer.encode(" Friday", add_special_tokens=False)[0]
    generated = ((friday, eos, friday), (friday, friday, friday))
    frame = locate.continuation(model.tokenizer, generated, (eos,))

    assert frame.starts == (0, 0) and frame.ends == (2, 3), "row 0 is cut after its stop token"
    assert "eos" in frame.segments[0] and frame.segments[1] == {}
    assert locate.locate(frame, Where(index=-1))[0] == ((1,), (2,))
    windows, reasons = locate.locate(frame, Where(all=True, scope=Anchor(segment="eos"), frame="generated"))
    assert windows == ((1,), ()) and reasons == ("", "alignment_missing")


def test_a_window_decodes_back_to_what_it_addressed(frames):
    """The provenance claim: the last token of ` Thursday` is the piece
    `day`, and a run can say so because the block has the tokenizer."""
    _, frame = frames["pair"]
    windows, _ = locate.locate(frame, Where(index=-1, scope=Anchor(variable="e")), ("Thursday", "Friday"))
    assert [locate.tokens_of(frame, one, row) for row, one in enumerate(windows)] == ["'day'", "' Friday'"]
