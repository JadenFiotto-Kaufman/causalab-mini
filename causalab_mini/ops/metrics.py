"""match, logit_diff, cross_entropy — pure tensor functions.

Each takes the read it binds to, shape (rows, vocab), plus one vocabulary id per
row per operand, and returns one value per row. The unit and estimand version
are derived from the kind and are never authored.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ..shapes import TokenIds


def _rows(logits: Any, ids: TokenIds) -> tuple[Any, Any]:
    return torch.arange(logits.shape[0], device=logits.device), torch.as_tensor(
        ids, device=logits.device
    )


def match(logits: Any, expected: TokenIds) -> Any:
    """1.0 when the argmax is the expected token, else 0.0."""
    rows, ids = _rows(logits, expected)
    return (logits.argmax(dim=-1) == ids).to(torch.float32)


def logit_diff(logits: Any, a: TokenIds, b: TokenIds) -> Any:
    rows, a_ids = _rows(logits, a)
    _, b_ids = _rows(logits, b)
    return logits[rows, a_ids] - logits[rows, b_ids]


def cross_entropy(logits: Any, target: TokenIds) -> Any:
    rows, ids = _rows(logits, target)
    return -logits.log_softmax(dim=-1)[rows, ids]


def token_prob(logits: Any, token: TokenIds) -> Any:
    """The probability the model puts on one token, per row — a logit lens
    reads this at every layer to see where an answer emerges."""
    rows, ids = _rows(logits, token)
    return logits.softmax(dim=-1)[rows, ids]


def token_logit(logits: Any, token: TokenIds) -> Any:
    """The raw logit of one token, per row. Not a difference and not a
    probability: the Hydra-effect experiments measure a *direct effect* as a
    logit moved by an intervention, and a difference would hide which side
    moved."""
    rows, ids = _rows(logits, token)
    return logits[rows, ids]


KINDS: dict[str, Callable[..., Any]] = {
    "match": match,
    "logit_diff": logit_diff,
    "cross_entropy": cross_entropy,
    "token_logit": token_logit,
    "token_prob": token_prob,
}

UNITS: dict[str, tuple[str, str]] = {
    "match": ("fraction", "match/v1"),
    "logit_diff": ("logit", "logit_diff/v1"),
    "cross_entropy": ("nat", "cross_entropy/v1"),
    "token_logit": ("logit", "token_logit/v1"),
    "token_prob": ("probability", "token_prob/v1"),
}


def compute(kind: str, logits: Any, ids: tuple[TokenIds, ...]) -> Any:
    """`ids` is the kind's operands in order: match takes (expected,),
    logit_diff takes (a, b), cross_entropy takes (target,), token_logit
    takes (token,)."""
    return KINDS[kind](logits, *ids)
