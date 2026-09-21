"""Text -> tokens, and a position spec -> per-row indices.

Both jobs happen on the client, before anything runs, and both produce plain
integers. That is what lets the plan be pure data: the block never tokenizes and
never resolves a position.
"""

from __future__ import annotations

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


def _locate(batch: Batch, row: int, text: str) -> tuple[int, ...]:
    """The token window covering `text` inside the row's decoded content, or
    `()` when the text is not there: the row is then an excluded measurement
    — still a row, contributing no positions."""
    decoded, offsets = batch.texts[row], batch.offsets[row]
    for candidate in (text, " " + text, text.strip()):
        at = decoded.find(candidate) if candidate else -1
        if at != -1:
            break
    else:
        return ()
    lo_char, hi_char = at, at + len(candidate)
    # the first token that ends after the substring starts, up to the last
    # token that starts before it ends
    first = next(k for k in range(len(offsets) - 1) if offsets[k + 1] > lo_char)
    last = max(k for k in range(len(offsets) - 1) if offsets[k] < hi_char)
    return tuple(range(batch.starts[row] + first, batch.starts[row] + last + 1))


def is_ragged(pos: Any) -> bool:
    """Whether a form's width varies by row."""
    return isinstance(pos, dict) and (set(pos) == {"all"} or set(pos) == {"column"})


def width_of(pos: Any) -> int | None:
    """How many positions a form names, knowable without a row — which is
    what lets the compiler check a write against its operand — or `None`
    for a ragged form, whose widths are only known once the rows are."""
    if is_ragged(pos):
        return None
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
