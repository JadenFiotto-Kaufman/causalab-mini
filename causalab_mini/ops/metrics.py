"""match, logit_diff, cross_entropy, kl, js, top_k — pure tensor functions.

Each takes the read it binds to, shape (rows, vocab), plus one vocabulary id per
row per operand, and returns one value per row. `kl` and `js` take a second
read of the same shape in place of the ids, and `top_k` takes its `k` and
returns the k probabilities and their ids. The unit and estimand version are
derived from the kind and are never authored.
"""

from __future__ import annotations

import math
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


def kl(logits: Any, against: Any) -> Any:
    """KL(of ‖ against) in nats, per row, in float32 from the log-softmax. A
    token the first puts no probability on contributes nothing, even where
    the second puts none either."""
    return _kl(logits.float().log_softmax(dim=-1), against.float().log_softmax(dim=-1))


def js(logits: Any, against: Any) -> Any:
    """The Jensen–Shannon divergence in nats, per row: each side's KL to their
    mixture, averaged. Symmetric, and at most ln 2."""
    p, q = logits.float().log_softmax(dim=-1), against.float().log_softmax(dim=-1)
    # clamped, so a token neither side can say has a finite mixture and a
    # gradient; the term it would carry is 0 either way
    low = torch.finfo(p.dtype).min
    mixture = torch.logaddexp(p.clamp(min=low), q.clamp(min=low)) - math.log(2)
    return (_kl(p, mixture) + _kl(q, mixture)) / 2


def _kl(p: Any, q: Any) -> Any:
    # masking the difference, not the product, keeps −inf − −inf out of the
    # gradient as well as the value
    probs = p.exp()
    return (probs * torch.where(probs > 0, p - q, 0.0)).sum(dim=-1)


def top_k(logits: Any, k: int) -> Any:
    """The k most likely tokens per row: their probabilities and their ids,
    most likely first."""
    return logits.float().softmax(dim=-1).topk(k, dim=-1)


KINDS: dict[str, Callable[..., Any]] = {
    "match": match,
    "logit_diff": logit_diff,
    "cross_entropy": cross_entropy,
    "token_logit": token_logit,
    "token_prob": token_prob,
    "kl": kl,
    "js": js,
    "top_k": top_k,
}

#: Which of a metric kind's own document fields name data columns, in the
#: order `compute` takes them. One table, because a front end that disagreed
#: with `compute` would score the wrong column.
COLUMNS: dict[str, tuple[str, ...]] = {
    "match": ("expected",),
    "logit_diff": ("a", "b"),
    "cross_entropy": ("target",),
    "token_logit": ("token",),
    "token_prob": ("token",),
    "kl": (),
    "js": (),
    "top_k": (),
}

UNITS: dict[str, tuple[str, str]] = {
    "match": ("fraction", "match/v1"),
    "logit_diff": ("logit", "logit_diff/v1"),
    "cross_entropy": ("nat", "cross_entropy/v1"),
    "token_logit": ("logit", "token_logit/v1"),
    "token_prob": ("probability", "token_prob/v1"),
    "kl": ("nat", "kl/v1"),
    "js": ("nat", "js/v1"),
    "top_k": ("probability", "top_k/v1"),
}


def compute(kind: str, logits: Any, ids: tuple[Any, ...]) -> Any:
    """`ids` is the kind's operands in order: match takes (expected,),
    logit_diff takes (a, b), cross_entropy takes (target,), token_logit
    takes (token,), kl and js take (against,), the second read, and top_k
    takes (k,)."""
    return KINDS[kind](logits, *ids)
