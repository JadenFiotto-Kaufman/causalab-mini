"""Text -> tokens, on the client, before anything runs.

Tokenization stays here and only tokenization does. A plan carries the padded
ids because everything downstream of them is a client-side decision — the
padded width, a fit's minibatching, `plan.window`, the attention-pattern
layout check, a metric's token ids and cross-engine bit-identity all hang off
one batch. *Where along that batch* a read or a write acts does not: that is a
spec, and `ops/locate.py` resolves it where the model is.
"""

from __future__ import annotations

from typing import Any

from ..ops import locate
from ..shapes import TokenRows


class EncodingError(ValueError):
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
