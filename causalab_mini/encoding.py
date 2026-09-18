"""Text -> tokens, and a position spec -> per-row indices.

Both jobs happen on the client, before anything runs, and both produce plain
integers. That is what lets the plan be pure data: the block never tokenizes and
never resolves a position.
"""

from __future__ import annotations

from dataclasses import dataclass


class EncodingError(ValueError):
    pass


@dataclass(frozen=True)
class Batch:
    """One padded batch of prompts, as integers.

    `starts`/`ends` are the half-open span of each row's real content inside the
    padded sequence, which is what makes position resolution independent of the
    padding side.
    """

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    starts: tuple[int, ...]
    ends: tuple[int, ...]


def encode(tokenizer, texts: list[str]) -> Batch:
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


def positions(batch: Batch, pos: int) -> tuple[int, ...]:
    """One absolute index into the padded sequence, per row.

    Negative counts from the end of the row's content, non-negative from its
    start — so `-1` is the last real token whichever side the padding is on.
    """
    resolved = []
    for start, end in zip(batch.starts, batch.ends):
        index = end + pos if pos < 0 else start + pos
        if not start <= index < end:
            raise EncodingError(f"position {pos} falls outside the row's content")
        resolved.append(index)
    return tuple(resolved)


def token_id(tokenizer, text: str, token_form: str) -> int:
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
