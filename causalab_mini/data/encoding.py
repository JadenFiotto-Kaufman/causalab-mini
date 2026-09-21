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


def encode(tokenizer: Any, texts: list[str]) -> Batch:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(list(texts), padding=True)
    input_ids = tuple(tuple(int(i) for i in row) for row in encoded["input_ids"])
    mask = tuple(tuple(int(m) for m in row) for row in encoded["attention_mask"])
    starts, ends = [], []
    for row in mask:
        real = [index for index, flag in enumerate(row) if flag]
        if real != list(range(real[0], real[-1] + 1)):
            raise EncodingError("padding is not contiguous; cannot place positions")
        starts.append(real[0])
        ends.append(real[-1] + 1)
    return Batch(input_ids, mask, tuple(starts), tuple(ends))


def positions(batch: Batch, pos: Any) -> Positions:
    """A window of absolute indices into the padded sequence, per row.

    Every form is relative to the row's *content*, so it means the same thing
    whichever side the padding is on. Negative counts from the end.

        -1 / 3               one position: the unit window
        {"index": i}         the same, spelled out
        {"last": n}          the last n content tokens
        {"span": [a, b]}     content-relative, half-open, negatives allowed

    Every row gets a window of the same width — the forms above cannot make
    one otherwise — and that is what keeps a read a rectangle. `{"all":
    true}` would not, and is refused here by name until reads can be ragged.
    """
    resolved = []
    for start, end in zip(batch.starts, batch.ends):
        length = end - start
        lo, hi = _window(pos, length)
        if not 0 <= lo < hi <= length:
            raise EncodingError(
                f"position {pos!r} falls outside the row's content ({length} tokens)"
            )
        resolved.append(tuple(range(start + lo, start + hi)))
    return tuple(resolved)


def width_of(pos: Any) -> int:
    """How many positions a form names — knowable without a row, which is
    what lets the compiler check a write against its operand."""
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
    if set(pos) == {"all"}:
        raise EncodingError(
            "{'all': true} is a window whose width varies by row; ragged reads are "
            "not implemented — use {'span': [a, b]} or {'last': n}"
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
