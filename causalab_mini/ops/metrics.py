"""The metric kinds — pure tensor functions — and what each takes.

A kind takes the read it scores, `of`, shape (rows, vocab); then any further
reads its signature names, each the same shape over the same rows; then one
vocabulary id per row per data column. It returns one value per row. The
unit and estimand version are derived from the kind and are never authored.

`SIGNATURES` is the extension point: a new kind is a function and a row
there — which of its document fields are reads, which are columns — and the
document's resolver, the compiler and the run all read the row, so a kind
that scores one read against another needs no new plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..shapes import TokenIds
from .intervene import exact


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


def soft_accuracy(logits: Any, expected: TokenIds) -> Any:
    """The probability mass on the expected token: `match` without the
    argmax, so a near miss counts for what it is."""
    return token_prob(logits, expected)


def kl(logits: Any, against: Any) -> Any:
    """KL(of ‖ against), per row, in nats: how far the distribution `of`
    predicts is from `against`'s, weighted by `of`'s — so a patched run
    scored against a clean one says what the patch cost where the patched
    model put its mass. In float32 from log-softmax, outside autocast."""
    with exact(logits):
        p, q = logits.float().log_softmax(dim=-1), against.float().log_softmax(dim=-1)
        return (p.exp() * (p - q)).sum(dim=-1)


def js(logits: Any, against: Any) -> Any:
    """The Jensen–Shannon divergence, per row, in nats: KL of each to their
    mixture, averaged — symmetric, and bounded by log 2."""
    with exact(logits):
        p, q = logits.float().log_softmax(dim=-1), against.float().log_softmax(dim=-1)
        # log((P + Q) / 2), written so it is P's own log to the bit where P is Q
        m = torch.maximum(p, q) + torch.log((1 + (-(p - q).abs()).exp()) / 2)
        return 0.5 * ((p.exp() * (p - m)).sum(dim=-1) + (q.exp() * (q - m)).sum(dim=-1))


@dataclass(frozen=True)
class Signature:
    """What a kind takes beside `of`, as the document fields that name it:
    further reads, each `<step>.<read>` of the same rows, then data columns,
    each `<dataset>.<column>` — in the order its function takes them."""

    reads: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()


SIGNATURES: dict[str, Signature] = {
    "match": Signature(columns=("expected",)),
    "logit_diff": Signature(columns=("a", "b")),
    "cross_entropy": Signature(columns=("target",)),
    "token_logit": Signature(columns=("token",)),
    "token_prob": Signature(columns=("token",)),
    "soft_accuracy": Signature(columns=("expected",)),
    "kl": Signature(reads=("against",)),
    "js": Signature(reads=("against",)),
}

KINDS: dict[str, Callable[..., Any]] = {
    "match": match,
    "logit_diff": logit_diff,
    "cross_entropy": cross_entropy,
    "token_logit": token_logit,
    "token_prob": token_prob,
    "soft_accuracy": soft_accuracy,
    "kl": kl,
    "js": js,
}

UNITS: dict[str, tuple[str, str]] = {
    "match": ("fraction", "match/v1"),
    "logit_diff": ("logit", "logit_diff/v1"),
    "cross_entropy": ("nat", "cross_entropy/v1"),
    "token_logit": ("logit", "token_logit/v1"),
    "token_prob": ("probability", "token_prob/v1"),
    "soft_accuracy": ("probability", "soft_accuracy/v1"),
    "kl": ("nat", "kl/v1"),
    "js": ("nat", "js/v1"),
}


def compute(kind: str, logits: Any, reads: tuple[Any, ...], ids: tuple[TokenIds, ...]) -> Any:
    """`reads` and `ids` are the kind's further reads and its columns, each in
    its signature's order: `kl` takes (against,) and no ids, `logit_diff` no
    reads and (a, b)."""
    return KINDS[kind](logits, *reads, *ids)
