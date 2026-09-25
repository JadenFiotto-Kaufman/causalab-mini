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

from collections import Counter
from typing import Any

from ..ops import locate
from ..shapes import TokenRows


class TokenError(ValueError):
    pass


def rendered(tokenizer: Any, value: Any, where: str) -> str:
    """One role's prompt text: what the row holds, or what the model's own
    chat template makes of it.

    A conversation is a list of `{"role", "content"}` messages, which is a
    thing a dataset column already can be — so a document says "this role is
    a chat" by *its data being one*, exactly as it says which text an anchor
    looks for by the row carrying it. There is no flag.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not all(
        isinstance(one, dict) and isinstance(one.get("role"), str) and isinstance(one.get("content"), str)
        for one in value
    ):
        raise TokenError(
            f"{where}: a role's text is a string, or a conversation — a list of "
            f"{{'role': …, 'content': …}} messages. This row has {type(value).__name__}"
        )
    if getattr(tokenizer, "chat_template", None) is None:
        raise TokenError(
            f"{where}: this row is a conversation and this checkpoint has no chat "
            "template, so there is no text to run. Give the role a string column"
        )
    return tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=True)


def turns(
    tokenizer: Any, ids: TokenRows, mask: TokenRows, conversations: list[Any]
) -> tuple[dict[str, tuple[int, int]], ...]:
    """Per row, the character span of each turn inside the row's own text.

    The coordinates are the frame's — `locate.frame_of`'s decoded text, the
    same function the run builds its frame with — so the spans the plan
    carries and the offsets the run resolves against cannot drift apart.
    The span is the message's *content*: the template's own control tokens
    are between the turns and in none of them, because each content is
    searched for as itself, forward from where the last one ended.

    A turn is named by its own role, plus `role[k]` for the k-th of that
    role. The bare name is recorded only when that role appears once; a
    reader asking for it when there are two gets `alignment_ambiguous`.
    """
    frame = locate.frame_of(tokenizer, ids, mask)
    found = []
    for row, messages in enumerate(conversations):
        spans: dict[str, tuple[int, int]] = {}
        if isinstance(messages, list):
            text, cursor, seen = frame.texts[row], 0, Counter()
            for one in messages:
                role, content = one["role"], one["content"]
                at = text.find(content, cursor)
                index = seen[role]
                seen[role] += 1
                if at == -1:  # the template did not keep it whole
                    continue
                spans[f"{role}[{index}]"] = (at, at + len(content))
                cursor = at + len(content)
            for role, count in seen.items():
                if count == 1 and f"{role}[0]" in spans:
                    spans[role] = spans[f"{role}[0]"]
        found.append(spans)
    return tuple(found)


def encode(
    tokenizer: Any, texts: list[str], add_special: bool = True
) -> tuple[TokenRows, TokenRows, str]:
    """One padded batch of prompts: the ids, their mask, and what one row's
    ids say here.

    The frame the resolver works against is built from these same ids, where
    the run is, so the two can only disagree if the two tokenizers do. That
    third value is how the run finds out: it decodes the same row with its
    own tokenizer and compares the strings.

    `add_special=False` for text a chat template rendered: the template has
    already put the family's opening token in, and asking for it again puts
    it in twice (measured on the tiny Llama: `[1, 1, …]`).
    """
    ids, mask, frame = locate.frame_of_texts(tokenizer, texts, text=False, add_special=add_special)
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


def token_id(tokenizer: Any, text: Any, token_form: str) -> int:
    """One vocabulary id for an authored answer.

    `space_prefixed` and `bare` spell a string: a leading space in the column
    value is normalized away first, so `" X"` and `"X"` name the same answer
    and `token_form` alone decides the surface form. A value that is not
    exactly one token is refused, never scored on its first piece. `id` takes
    the column value as the vocabulary id itself, which is the only spelling
    of a token no string round-trips to.
    """
    if token_form == "id":
        if not isinstance(text, int) or isinstance(text, bool) or not 0 <= text < len(tokenizer):
            raise TokenError(
                f"answer {text!r} is not a vocabulary id; under token_form 'id' a metric column "
                f"holds integers in [0, {len(tokenizer)})"
            )
        return text
    if token_form not in ("space_prefixed", "bare"):
        raise TokenError(f"token_form {token_form!r} is not implemented")
    surface = (" " if token_form == "space_prefixed" else "") + text.lstrip()
    ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(ids) != 1:
        raise TokenError(
            f"answer {text!r} is {len(ids)} tokens as {surface!r}; a metric column "
            "must resolve to exactly one token"
        )
    return int(ids[0])
