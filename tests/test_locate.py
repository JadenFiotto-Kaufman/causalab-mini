"""The resolver: which indices a position names, on this row.

`ops/locate.py` answers that where the model is, against the model's own
tokenizer. Every form is here, over the batches the suite tokenizes, with
the windows written out — these literals were checked against the
client-side resolver this replaced, form for form and index for index,
before it was deleted.
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


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    """A second, unrelated tokenizer, for the one thing that needs two."""
    from causalab_mini.engine import NNterpEngine
    from causalab_mini.plan.spec import Model

    return NNterpEngine.load(
        Model(key="hf-internal-testing/tiny-random-gpt2", revision="main", dtype="fp32"),
        dispatch=False,
    ).tokenizer


@pytest.fixture
def frames(model):
    """Each batch as the resolver sees it: the ids a plan would carry, and
    the frame built from those same ids."""
    return {
        name: locate.frame_of_texts(model.tokenizer, texts)
        for name, texts in (("pair", PAIR), ("four", FOUR), ("even", EVEN))
    }


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
def test_every_form_is_content_relative_and_uniform(frames, pos, where, expected):
    """Row 1 of `pair` starts at index 2 because of its padding, and every
    form lands on its content the same way — which is why `0` is `(0, 2)`."""
    for name, (_ids, _mask, frame) in frames.items():
        windows, reasons = locate.locate(frame, where)
        assert set(reasons) == {""}, name
        if name in expected:
            assert windows == expected[name], name
        for row, window in enumerate(windows):
            assert set(window) <= set(range(frame.starts[row], frame.ends[row])), name
            assert where.width is None or len(window) == where.width, name


def test_a_variable_anchor_resolves_what_a_column_resolved(frames):
    """`{"column": c}` was the only text form the client ran, and
    `{"all": true, "scope": {"variable": c}}` is what it becomes: the tokens
    of this row's own text, located in its prompt. ` Thursday` is three
    tokens on this tokenizer and ` Friday` one."""
    _ids, _mask, frame = frames["pair"]
    anchors = ("Thursday", "Friday")
    windows, reasons = locate.locate(frame, Where(all=True, scope=Anchor(variable="entity")), anchors)

    assert windows == ((4, 5, 6), (6,))
    assert [len(one) for one in windows] == [3, 1]
    assert set(reasons) == {""}


def test_a_row_whose_text_is_not_in_its_prompt_says_why(frames):
    _ids, _mask, frame = frames["pair"]
    where = Where(all=True, scope=Anchor(variable="entity"))
    windows, reasons = locate.locate(frame, where, ("Thursday", "Neptune"))

    assert windows[0] == (4, 5, 6)
    assert windows[1] == () and reasons == ("", "alignment_missing")


def test_a_value_that_occurs_twice_is_ambiguous_rather_than_the_first_one(model):
    """No occurrence index is minted: the fix an author wants is to scope the
    anchor, and a silently-first window is the wrong number."""
    _, _, frame = locate.frame_of_texts(model.tokenizer, ["Friday, then Friday again"])
    windows, reasons = locate.locate(frame, Where(all=True, scope=Anchor(variable="x")), ("Friday",))
    assert windows == ((),) and reasons == ("alignment_ambiguous",)


@pytest.mark.parametrize(
    "where",
    [Where(last=12), Where(span=(0, 40)), Where(index=40), Where(index=-40)],
    ids=["too wide", "past the end", "index past the end", "index before the start"],
)
def test_a_cut_outside_the_run_is_a_reason_rather_than_a_refusal(frames, where):
    """A cut that does not fit is reported per row, not raised: on a
    text-anchored run only the row knows how long the run is, and the
    scope-free case is the same three steps."""
    _ids, _mask, frame = frames["pair"]
    windows, reasons = locate.locate(frame, where)
    assert windows == ((), ()) and set(reasons) == {"out_of_range"}


# --------------------------------------------------------------------- #
# the frame itself
# --------------------------------------------------------------------- #


def test_the_frame_is_the_batch_the_plan_carries(frames, model):
    """The client tokenizes and the plan carries the ids; the frame is built
    from those same ids, where the run is. The two constructors are one
    function, which is what makes that true."""
    ids, mask, frame = frames["pair"]
    assert encoding.encode(model.tokenizer, PAIR) == (ids, mask)
    assert locate.frame_of(model.tokenizer, ids, mask) == frame
    assert (frame.starts, frame.ends) == ((0, 2), (11, 11))
    assert frame.texts[0].endswith("Thursday, tomorrow is")


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
    _ids, _mask, frame = frames["pair"]
    windows, _ = locate.locate(frame, Where(index=-1, scope=Anchor(variable="e")), ("Thursday", "Friday"))
    assert [locate.tokens_of(frame, one, row) for row, one in enumerate(windows)] == ["'day'", "' Friday'"]


def test_an_anchor_resolves_to_a_different_index_on_different_rows(model):
    """The whole point of a spec: one form, and the integer is not the same
    on every row.

    These prompts are left-padded, so a row's text is pushed rightwards by
    however many tokens the rows before the anchor take — ` Tuesday` is
    three and ` Monday` is one. The number is therefore the row's, and a
    document that wrote `-4` would be addressing a different word in each.
    """
    prompts = [
        "Q: What day is one days after Monday?\nA:",
        "Q: What day is three days after Monday?\nA:",
        "Q: What day is two days after Tuesday?\nA:",
        "Q: What day is five days after Tuesday?\nA:",
    ]
    _ids, _mask, frame = locate.frame_of_texts(model.tokenizer, prompts)
    where = Where(index=-1, scope=Anchor(variable="number"))
    windows, reasons = locate.locate(frame, where, ("one", "three", "two", "five"))

    assert windows == ((8,), (8,), (6,), (6,))
    assert set(reasons) == {""}
    assert [locate.tokens_of(frame, one, row) for row, one in enumerate(windows)] == [
        "' one'", "' three'", "' two'", "' five'"
    ]


def test_a_tokenizer_that_disagrees_with_the_plan_is_refused_by_name(model, gpt2_tokenizer):
    """The failure this design could have had. The client encodes and the
    *run* resolves, so the two sides must be the same tokenizer; when they
    are not, every position is in range and in the wrong place. One row is
    re-encoded before anything is placed, which turns that into a refusal."""
    ids, mask, _frame = locate.frame_of_texts(model.tokenizer, PAIR)
    with pytest.raises(locate.LocateError, match="disagrees with the one that encoded this plan"):
        locate.frame_of(gpt2_tokenizer, ids, mask)
