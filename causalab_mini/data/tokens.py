"""Text -> tokens, and the padding arithmetic over them.

Tokenization stays here and only tokenization does. A plan carries the padded
ids because everything downstream of them is a client-side decision — the
padded width, a fit's minibatching, `plan.window`, the attention-pattern
layout check, a metric's token ids and cross-engine bit-identity all hang off
one batch. *Where along that batch* a read or a write acts does not: that is a
spec, and `ops/locate.py` resolves it where the model is.

Three jobs, all of them about a batch of ids and none about a model:
`encode` makes one, `token_id` says what a single answer string is in it, and
`same_layout` compares two of them. Who needs those answers, and what they
refuse when the answer is no, is their business.
"""

from __future__ import annotations

from typing import Any

from ..ops import locate
from ..shapes import TokenRows


class TokenError(ValueError):
    pass


def encode(tokenizer: Any, texts: list[str]) -> tuple[TokenRows, TokenRows, str]:
    """One padded batch of prompts: the ids, their mask, and what one row's
    ids say here.

    The frame the resolver works against is built from these same ids, where
    the run is, so the two can only disagree if the two tokenizers do. That
    third value is how the run finds out: it decodes the same row with its
    own tokenizer and compares the strings.
    """
    ids, mask, frame = locate.frame_of_texts(tokenizer, texts, text=False)
    if not ids:
        return ids, mask, ""
    return ids, mask, tokenizer.decode(ids[0][frame.starts[0] : frame.ends[0]])


def same_layout(one: TokenRows, other: TokenRows) -> list[int] | None:
    """Whether two padded batches are laid out the same way, and where they
    differ if they are not.

    `None` when they are identical. Otherwise the rows whose real-token
    counts differ — empty when the counts all agree and only the padded
    width does not. It is arithmetic over two attention masks, which is
    what this module is: who needs the answer, and what they refuse when it
    is no, is their business.
    """
    if one == other:
        return None
    ours = [sum(row) for row in one]
    theirs = [sum(row) for row in other]
    return [row for row, (a, b) in enumerate(zip(ours, theirs)) if a != b]


def token_id(tokenizer: Any, text: str, token_form: str) -> int:
    """One vocabulary id for an authored answer string.

    A leading space in the column value is normalized away first, so `" X"` and
    `"X"` name the same answer and `token_form` alone decides the surface form.
    A value that is not exactly one token is refused, never scored on its first
    piece.
    """
    if token_form != "space_prefixed":
        raise TokenError(f"token_form {token_form!r} is not implemented")
    surface = " " + text.lstrip()
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(ids) != 1:
        raise TokenError(
            f"answer {text!r} is {len(ids)} tokens as {surface!r}; a metric column "
            "must resolve to exactly one token"
        )
    return int(ids[0])
