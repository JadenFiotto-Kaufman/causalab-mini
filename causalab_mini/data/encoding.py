"""Text -> tokens, and a position spec -> per-row indices.

Both jobs happen on the client, before anything runs, and both produce plain
integers. That is what lets the plan be pure data: the block never tokenizes and
never resolves a position.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Any

from ..shapes import Indices, Positions, TokenRows


class EncodingError(ValueError):
    pass


@dataclass(frozen=True)
class Batch:
    """One padded batch of prompts, as integers.

    `starts`/`ends` are the half-open span of each row's real content inside the
    padded sequence, which is what makes position resolution independent of the
    padding side.
    """

    input_ids: TokenRows
    attention_mask: TokenRows
    starts: Indices
    ends: Indices
    #: Per row: the decoded content text, and the character offset at which
    #: each content token starts (with a final entry at the end), so a
    #: substring of the text maps to a token window. Built by decoding
    #: growing prefixes, which works on any tokenizer and needs no
    #: offset_mapping.
    texts: tuple[str, ...] = ()
    offsets: tuple[tuple[int, ...], ...] = ()


def encode(tokenizer: Any, texts: list[str]) -> Batch:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(list(texts), padding=True)
    input_ids = tuple(tuple(int(i) for i in row) for row in encoded["input_ids"])
    mask = tuple(tuple(int(m) for m in row) for row in encoded["attention_mask"])
    starts, ends, decoded, offsets = [], [], [], []
    for ids, row in zip(input_ids, mask):
        real = [index for index, flag in enumerate(row) if flag]
        if real != list(range(real[0], real[-1] + 1)):
            raise EncodingError("padding is not contiguous; cannot place positions")
        starts.append(real[0])
        ends.append(real[-1] + 1)
        content = list(ids[real[0] : real[-1] + 1])
        prefixes = [tokenizer.decode(content[:k]) for k in range(len(content) + 1)]
        decoded.append(prefixes[-1])
        offsets.append(tuple(len(prefix) for prefix in prefixes))
    return Batch(input_ids, mask, tuple(starts), tuple(ends), tuple(decoded), tuple(offsets))


def positions(batch: Batch, pos: Any, column_texts: list[str] | None = None) -> Positions:
    """A window of absolute indices into the padded sequence, per row.

    Every form is relative to the row's *content*, so it means the same thing
    whichever side the padding is on. Negative counts from the end.

        -1 / 3               one position: the unit window
        {"index": i}         the same, spelled out
        {"last": n}          the last n content tokens
        {"span": [a, b]}     content-relative, half-open, negatives allowed
        {"step": k}          decode step k, at the one position it processes —
                             the continuation frame; k=0 is the prefill's last
                             prompt token. Needs a forward that decodes.
        {"all": true}        every content token — RAGGED: widths differ by row
        {"column": "c"}      the tokens of the row's column-`c` text, located
                             in the prompt — RAGGED, and a row whose text is
                             not there gets an EMPTY window: an excluded
                             measurement, still a row

    The first four give every row the same width, which keeps a read a
    rectangle. The last two do not, and a ragged read is flat rows with the
    plan's own positions saying where each row's begin and end.
    """
    resolved = []
    for index, (start, end) in enumerate(zip(batch.starts, batch.ends)):
        length = end - start
        if isinstance(pos, dict) and set(pos) == {"step"}:
            # the last prompt token: right for the prefill, and for a decode
            # step the engine takes the last position of whatever it sees
            resolved.append((end - 1,))
            continue
        if isinstance(pos, dict) and set(pos) == {"all"}:
            resolved.append(tuple(range(start, end)))
            continue
        if isinstance(pos, dict) and set(pos) == {"column"}:
            if column_texts is None:
                raise EncodingError("a {'column': …} position needs the rows it is resolved over")
            resolved.append(_locate(batch, index, column_texts[index]))
            continue
        lo, hi = _window(pos, length)
        if not 0 <= lo < hi <= length:
            raise EncodingError(
                f"position {pos!r} falls outside the row's content ({length} tokens)"
            )
        resolved.append(tuple(range(start + lo, start + hi)))
    return tuple(resolved)


def _locate(batch: Batch, row: int, text: str | None) -> tuple[int, ...]:
    """The token window covering `text` inside the row's decoded content, or
    `()` when the text is not there — or the row has none: the row is then
    an excluded measurement, still a row, contributing no positions.

    A position names one place. Text that occurs twice is refused rather
    than resolved to its first occurrence, and a match has to stand on its
    own — `day` is not found inside `Thursday` — because on a write either
    mistake patches a wrong token with every shape correct."""
    if text is None or not text.strip():
        return ()
    decoded, offsets = batch.texts[row], batch.offsets[row]
    wanted = text.strip()
    hits = [
        found.start()
        for found in re.finditer(re.escape(wanted), decoded)
        if not (found.start() > 0 and decoded[found.start() - 1].isalnum() and wanted[0].isalnum())
        and not (found.end() < len(decoded) and decoded[found.end()].isalnum() and wanted[-1].isalnum())
    ]
    if not hits:
        return ()
    if len(hits) > 1:
        raise EncodingError(
            f"a {{'column': …}} position: row {row}'s text {wanted!r} occurs {len(hits)} times in its "
            f"prompt {decoded!r}; a position names one place. Make the column's value unique in the prompt"
        )
    lo_char, hi_char = hits[0], hits[0] + len(wanted)
    # the first token that ends after the substring starts, up to the last
    # token that starts before it ends
    first = next(k for k in range(len(offsets) - 1) if offsets[k + 1] > lo_char)
    last = max(k for k in range(len(offsets) - 1) if offsets[k] < hi_char)
    return tuple(range(batch.starts[row] + first, batch.starts[row] + last + 1))


#: Every position form, in the words an error and `causalab-mini vocab` use.
FORMS = (
    "an integer (negative counts from the end of the row's content)",
    '{"index": i} — the same, spelled out',
    '{"last": n} — the last n content tokens, n >= 1',
    '{"span": [a, b]} — content-relative and half-open, negatives allowed',
    '{"all": true} — every content token; the width varies by row',
    '{"column": "c"} — the tokens of the row\'s column-c text inside its prompt; varies by row',
    '{"step": k} or {"step": "all"} — a decode step, which needs decode > 0',
)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def check_form(pos: Any) -> None:
    """Refuse a position no row could make sense of, before any row exists.

    A misspelt form, a span given as a string, `{"all": false}` (which used
    to mean all), a window that is empty whatever the row — these are
    mistakes in the document, and they used to surface as a TypeError from
    inside the resolver or as "falls outside the row's content"."""
    def refuse(why: str) -> None:
        raise EncodingError(f"position {pos!r}: {why}. The forms are: " + "; ".join(FORMS))

    if _integer(pos):
        return
    if not isinstance(pos, dict) or len(pos) != 1:
        refuse("not a position form")
    (form, value), = pos.items()
    if form == "index":
        if not (_integer(value)):
            refuse("an index is an integer")
    elif form == "last":
        if not ((_integer(value) and value >= 1)):
            refuse("`last` is how many tokens, at least 1")
    elif form == "span":
        ok = isinstance(value, list) and len(value) == 2 and all(_integer(one) for one in value)
        if not (ok):
            refuse("a span is [start, stop] in integers")
        a, b = value
        if (a < 0) == (b < 0) and a >= b:
            refuse("this span is empty on every row")
    elif form == "all":
        if not (value is True):
            refuse('it is spelled {"all": true}')
    elif form == "column":
        if not ((isinstance(value, str) and bool(value))):
            refuse("a column is named by a non-empty string")
    elif form == "step":
        if not ((value == "all" or (_integer(value) and value >= 0))):
            refuse("a step is a non-negative integer, or 'all'")
    elif form == "variable":
        refuse("{'variable': v} is not implemented; {'column': c} locates a column's text in the prompt")
    else:
        refuse("not a position form")


def is_ragged(pos: Any) -> bool:
    """Whether a form's width varies by row."""
    return isinstance(pos, dict) and (set(pos) == {"all"} or set(pos) == {"column"})


def step_of(pos: Any) -> int | str | None:
    """The decode step a form names, or None for the prompt frame."""
    return pos["step"] if isinstance(pos, dict) and set(pos) == {"step"} else None


def width_of(pos: Any) -> int | None:
    """How many positions a form names, knowable without a row — which is
    what lets the compiler check a write against its operand — or `None`
    for a ragged form, whose widths are only known once the rows are."""
    check_form(pos)
    if is_ragged(pos):
        return None
    if step_of(pos) is not None:
        return 1
    lo, hi = _window(pos, 1 << 30)
    return hi - lo


def _window(pos: Any, length: int) -> tuple[int, int]:
    """The half-open content-relative window a form names, on a row of
    `length` tokens."""
    if isinstance(pos, bool) or not isinstance(pos, (int, dict)):
        raise EncodingError(f"position {pos!r}: not a form this slice runs")
    if isinstance(pos, int):
        index = length + pos if pos < 0 else pos
        return index, index + 1
    if set(pos) == {"index"}:
        return _window(pos["index"], length)
    if set(pos) == {"last"}:
        return length - pos["last"], length
    if set(pos) == {"span"}:
        a, b = pos["span"]
        return (length + a if a < 0 else a), (length + b if b < 0 else b)
    if set(pos) == {"variable"}:
        raise EncodingError(
            "{'variable': v} is not implemented; {'column': c} locates a column's "
            "text in the prompt, which is what a prompt variable is here"
        )
    raise EncodingError(f"position {pos!r}: not a form this slice runs")


def token_id(tokenizer: Any, text: str, token_form: str) -> int:
    """One vocabulary id for an authored answer string.

    A leading space in the column value is normalized away first, so `" X"` and
    `"X"` name the same answer and `token_form` alone decides the surface form.
    A value that is not exactly one token is refused, never scored on its first
    piece.
    """
    if token_form != "space_prefixed":
        raise EncodingError(f"token_form {token_form!r} is not implemented")
    surface = " " + text.lstrip()
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(ids) != 1:
        raise EncodingError(
            f"answer {text!r} is {len(ids)} tokens as {surface!r}; a metric column "
            "must resolve to exactly one token"
        )
    return int(ids[0])
