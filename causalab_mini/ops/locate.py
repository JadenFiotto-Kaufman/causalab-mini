"""AGNOSTIC: a position spec, a padded batch, and the indices it names.

Nothing here knows what a layer is, what a document said or where the rows
came from. It takes a tokenizer and integers and gives back integers, which is
what lets it run **where the model is** — including inside someone else's
process, where the tokenizer is the served checkpoint's own. For that to hold
this file may import nothing but the standard library, `torch` and
`shapes.py`: a stock NDIF server has our package by value and nothing else.

A position is three independent questions and one function answers them:

    1. the run     the row's content, or an anchor's characters mapped to the
                   tokens that overlap them
    2. the cut     index / span / last / all, *inside that run*
    3. the check   every index inside the row's content, or a reason

An integer position is the scope-free case of those same three steps. That is
what makes this one resolver and not two, and it is why `{"index": -1}` and
`{"index": -1, "scope": {"variable": "entity"}}` cost the same code.

The character map comes from the **ids**, by decoding growing prefixes:
`offsets[k]` is the character at which token `k` starts. It needs no fast
tokenizer and no `offset_mapping`, and — the point — it describes the ids the
run will actually feed the model rather than a re-encoding that may not
reproduce them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..shapes import Indices, Positions, TokenRows, Where

#: Why a row has no window. A closed set: the first two are the protocol's
#: own names for an alignment that did not resolve, and the third is the case
#: it refuses rather than reports — a cut that falls outside the run it cuts.
#: An empty string is a row that resolved.
REASONS = ("alignment_missing", "alignment_ambiguous", "out_of_range")


class LocateError(ValueError):
    pass


@dataclass(frozen=True)
class Frame:
    """One padded batch as the resolver sees it.

    `starts`/`ends` are each row's half-open content span inside the padded
    sequence, which is what makes resolution independent of the padding side.
    `texts[row]` is what those content tokens decode to and `offsets[row][k]`
    is the character at which token `k` of the content starts, with a final
    entry at the end, so a character span maps to a token window.

    One type for the prompt and for the continuation, so there is one frame
    and not a client one, a server one and a generated one.
    """

    starts: Indices
    ends: Indices
    texts: tuple[str, ...]
    offsets: tuple[tuple[int, ...], ...]  # len == the row's token count + 1
    #: Per row, the character span of each run the *frame* located: `eos` in
    #: the continuation. Empty for a plain prompt, which is every row of
    #: every document today.
    segments: tuple[dict[str, tuple[int, int]], ...] = ()


def frame_of(tokenizer: Any, ids: TokenRows, mask: TokenRows, text: bool = True) -> Frame:
    """The frame of a padded batch the plan already carries.

    `text=False` builds the content spans and nothing else. They come from
    the mask, and a position that names no anchor is index arithmetic over
    them — while the character map is O(L) decode calls of O(L) work per
    row, which is 840 ms for 64 rows of 260 tokens and is built once per
    forward per pass. A fit whose every position is a bare `-1` would spend
    all of that on a map nothing reads.
    """
    starts, ends, texts, offsets = [], [], [], []
    for row, flags in zip(ids, mask):
        real = [index for index, flag in enumerate(flags) if flag]
        if real != list(range(real[0], real[-1] + 1)):
            raise LocateError("padding is not contiguous; cannot place positions")
        starts.append(real[0])
        ends.append(real[-1] + 1)
        if text:
            content, marks = _chars(tokenizer, row[real[0] : real[-1] + 1])
            texts.append(content)
            offsets.append(marks)
    return Frame(tuple(starts), tuple(ends), tuple(texts), tuple(offsets))


def frame_of_texts(
    tokenizer: Any, texts: list[str], text: bool = True
) -> tuple[TokenRows, TokenRows, Frame]:
    """Prompts in; the padded batch and its frame out. The client's `encode`
    is this with `text=False`, because it needs the ids and not the map."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(list(texts), padding=True)
    ids = tuple(tuple(int(one) for one in row) for row in encoded["input_ids"])
    mask = tuple(tuple(int(one) for one in row) for row in encoded["attention_mask"])
    return ids, mask, frame_of(tokenizer, ids, mask, text=text)


def continuation(tokenizer: Any, generated: TokenRows) -> Frame:
    """The decode's own frame, each row cut at its first stop token.

    The decode ran to the bound — mini holds EOS off so the loop is a bound
    and the batch stays rectangular — but how far a row *meant* to generate
    is a result, so the frame stops there and the stop token is the last one
    in it. A row that never stopped has the whole bound and no `eos`
    segment, which is how "did the model stop?" comes home as a reported
    reason rather than an exception.

    The character map is built from the ids the decode produced, never from
    re-encoding the finished text: a tokenizer is free to merge across a
    boundary the decode never saw. Which ids stop a row is the tokenizer's
    to say, and it may name one or several.
    """
    stop_ids = getattr(tokenizer, "eos_token_id", None)
    eos_ids = (
        ()
        if stop_ids is None
        else tuple(int(one) for one in stop_ids)
        if isinstance(stop_ids, (list, tuple))
        else (int(stop_ids),)
    )
    starts, ends, texts, offsets, segments = [], [], [], [], []
    for row in generated:
        stop = next((k for k, one in enumerate(row) if one in eos_ids), None)
        end = len(row) if stop is None else stop + 1
        text, marks = _chars(tokenizer, row[:end])
        starts.append(0)
        ends.append(end)
        texts.append(text)
        offsets.append(marks)
        segments.append({} if stop is None else {"eos": (marks[stop], marks[stop + 1])})
    return Frame(tuple(starts), tuple(ends), tuple(texts), tuple(offsets), tuple(segments))


def _chars(tokenizer: Any, content: Any) -> tuple[str, tuple[int, ...]]:
    """One row's text and the character each of its tokens starts at.

    `len(decode(ids[:k]))` is a character offset only while the decode of a
    prefix really *is* a prefix of the whole, and for a byte-fallback token
    it is not: each incomplete byte of one character decodes to its own
    U+FFFD, so three bytes of an emoji contribute three characters to the
    prefixes and one to the whole, and the offsets run backwards. Every
    window located after such a character is then wrong, contiguity and all,
    with no reason reported — so a prefix that is not one is parked where
    the character it belongs to starts. The incomplete bytes get an empty
    span, the byte that completes the character gets the whole of it, and
    the offsets are non-decreasing whatever the tokenizer does.
    """
    tokens = list(content)
    whole = tokenizer.decode(tokens)
    marks = [0]
    for k in range(1, len(tokens) + 1):
        prefix = tokenizer.decode(tokens[:k])
        marks.append(max(marks[-1], len(prefix)) if whole.startswith(prefix) else marks[-1])
    return whole, tuple(marks)


def locate(frame: Frame, where: Where, anchors: tuple[str, ...] = ()) -> tuple[Positions, tuple[str, ...]]:
    """Per row: the window of absolute indices into the padded sequence that
    `where` names, and why it is empty when it is — `""` when it is not.

    `anchors[row]` is the text this row's `scope.variable` binds to — the one
    thing this file cannot get for itself, because it has no dataset.
    """
    windows: list[tuple[int, ...]] = []
    reasons: list[str] = []
    for row in range(len(frame.starts)):
        run, reason = _run(frame, where, row, anchors[row] if anchors else None)
        window, reason = ((), reason) if reason else _cut(run, where)
        windows.append(window)
        reasons.append(reason)
    return tuple(windows), tuple(reasons)


def _run(frame: Frame, where: Where, row: int, anchor: str | None) -> tuple[list[int], str]:
    """Step one: which tokens of this row the spec is *about*."""
    start, end = frame.starts[row], frame.ends[row]
    if where.scope is None:
        return list(range(start, end)), ""
    if not frame.texts:
        raise LocateError(
            f"position scope {where.scope} needs the row's text, and this frame was built "
            "without one; build it with text=True"
        )
    text = frame.texts[row]
    lo, hi = 0, len(text)
    if where.scope.segment is not None:
        located = (frame.segments[row] if frame.segments else {}).get(where.scope.segment)
        if located is None:
            return [], "alignment_missing"
        lo, hi = located
    if where.scope.variable is not None:
        if anchor is None:
            raise LocateError(
                f"position scope {where.scope.variable!r}: this row carries no text to "
                "look for; a variable anchor travels with the plan, per row"
            )
        hits = _hits(text, lo, hi, anchor)
        if not hits:
            return [], "alignment_missing"
        if len(hits) > 1:
            return [], "alignment_ambiguous"
        lo, hi = hits[0]
    offsets = frame.offsets[row]
    # the tokens that OVERLAP the character span: a value starting mid-piece
    # brings its whole piece, because the addressed thing is a token
    return [
        start + k
        for k in range(len(offsets) - 1)
        if offsets[k] < hi and offsets[k + 1] > lo
    ], ""


def _hits(text: str, lo: int, hi: int, anchor: str) -> list[tuple[int, int]]:
    """Every occurrence of `anchor` in `text[lo:hi]`, as character spans.

    The search is in the **decoded** text, because decoding is what
    normalizes a sentencepiece prefix space: `value`, then `" " + value`,
    then `value.strip()`, and the first spelling that hits at all decides.
    Occurrences are counted for that spelling only, so " Thursday" appearing
    once is one hit even though "Thursday" also appears inside it.
    """
    for spelling in (anchor, " " + anchor, anchor.strip()):
        if not spelling:
            continue
        found, at = [], text.find(spelling, lo, hi)
        while at != -1:
            found.append((at, at + len(spelling)))
            at = text.find(spelling, at + 1, hi)
        if found:
            return found
    return []


def _cut(run: list[int], where: Where) -> tuple[tuple[int, ...], str]:
    """Step two and three: how much of the run, and whether it is in it."""
    if where.index is not None:
        if not -len(run) <= where.index < len(run):
            return (), "out_of_range"
        return (run[where.index],), ""
    if where.last is not None:
        if where.last > len(run):
            return (), "out_of_range"
        return tuple(run[-where.last :]), ""
    if where.span is not None:
        a, b = where.span
        lo = len(run) + a if a < 0 else a
        hi = len(run) + b if b < 0 else b
        if not 0 <= lo < hi <= len(run):
            return (), "out_of_range"
        return tuple(run[lo:hi]), ""
    return tuple(run), "" if run else "out_of_range"


def tokens_of(frame: Frame, window: tuple[int, ...], row: int) -> str:
    """What a resolved window actually addressed, decoded — the provenance a
    number needs when "the last token of ` Thursday`" turns out to be the
    piece `day`. One short string per row per op."""
    if not window or not frame.offsets:
        return ""
    offsets, start = frame.offsets[row], frame.starts[row]
    lo = offsets[window[0] - start]
    hi = offsets[window[-1] - start + 1]
    return repr(frame.texts[row][lo:hi])
