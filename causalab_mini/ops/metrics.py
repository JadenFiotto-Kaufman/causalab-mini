"""The metric kinds — pure tensor functions — and what each takes.

A kind takes the read it scores, `of`, one position a row; then any further
reads its signature names, over the same rows; then one vocabulary id per
row per data column; then its numbers, by keyword. It returns one value per
row. The unit and estimand version are the kind's, never authored.

`SIGNATURES` is the extension point and the one table: a new kind is a
function and a row — what the function is, which of its document fields are
reads, columns and numbers, whether its reads must be logits, its unit — and
the document's model, its resolver, the compiler, the run, the table writer
and `vocab` all read the row. `cosine` is the proof: one function, one row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    """The probability the model puts on one token, per row. With the
    expected answer as the token it is the *soft accuracy* — `match` without
    the argmax, so a near miss counts for what it is — and a logit lens
    reads it at every layer to see where an answer emerges."""
    rows, ids = _rows(logits, token)
    return logits.softmax(dim=-1)[rows, ids]


def token_logit(logits: Any, token: TokenIds) -> Any:
    """The raw logit of one token, per row. Not a difference and not a
    probability: the Hydra-effect experiments measure a *direct effect* as a
    logit moved by an intervention, and a difference would hide which side
    moved."""
    rows, ids = _rows(logits, token)
    return logits[rows, ids]


def _weighted(p: Any, q: Any) -> Any:
    """Σ P · (log P − log Q) over the last axis, with a token `of` gives no
    mass contributing nothing — 0 · log 0 is 0 by the limit, where the
    arithmetic would make it NaN."""
    return torch.where(p == -math.inf, 0.0, p.exp() * (p - q)).sum(dim=-1)


def kl(logits: Any, against: Any) -> Any:
    """KL(of ‖ against), per row, in nats: how far the distribution `of`
    predicts is from `against`'s, weighted by `of`'s — so a patched run
    scored against a clean one says what the patch cost where the patched
    model put its mass. In float32 from log-softmax, outside autocast."""
    with exact(logits):
        p, q = logits.float().log_softmax(dim=-1), against.float().log_softmax(dim=-1)
        return _weighted(p, q)


def js(logits: Any, against: Any) -> Any:
    """The Jensen–Shannon divergence, per row, in nats: KL of each to their
    mixture, averaged — symmetric, and bounded by log 2."""
    with exact(logits):
        p, q = logits.float().log_softmax(dim=-1), against.float().log_softmax(dim=-1)
        # log((P + Q) / 2), written so it is P's own log to the bit where P is Q
        m = torch.maximum(p, q) + torch.log((1 + (-(p - q).abs()).exp()) / 2)
        return 0.5 * (_weighted(p, m) + _weighted(q, m))


def cosine(value: Any, against: Any) -> Any:
    """The cosine of the angle between two reads, per row: any two laid out
    alike — two layers' residual streams, a head's output before and after a
    patch. In float32, outside autocast."""
    with exact(value):
        return torch.nn.functional.cosine_similarity(value.float(), against.float(), dim=-1)


def top_k(logits: Any, k: int) -> tuple[Any, Any]:
    """The k most likely tokens per row, and the probability of each:
    `(ids, probabilities)`, each `(rows, k)`, most likely first. The run
    decodes the ids — it has the tokenizer — into each row's value, a list
    of `[token, probability]`."""
    with exact(logits):
        probabilities, ids = logits.float().softmax(dim=-1).topk(k, dim=-1)
    return ids, probabilities


@dataclass(frozen=True)
class Signature:
    """One kind: its function, what it takes beside `of` — further reads,
    each `<step>.<read>` of the same rows, then data columns, each
    `<dataset>.<column>`, in the order the function takes them, then its
    numbers, each a count of tokens with its default, a positive integer no
    larger than the vocabulary — and what it is.

    `logits`: its reads must be distributions over the vocabulary — a read
    of `lm_head` or `logits`, or one viewed as logits. Otherwise they need
    only be laid out alike. `tokens`: its value per row is a list, of
    `(token id, number)` pairs its function returns as two tensors, which
    the run decodes."""

    function: Callable[..., Any]
    unit: str
    version: str
    doc: str
    reads: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    params: dict[str, int] = field(default_factory=dict)
    logits: bool = True
    tokens: bool = False


SIGNATURES: dict[str, Signature] = {
    "match": Signature(match, "fraction", "match/v1", "1 when the argmax is the expected token", columns=("expected",)),
    "logit_diff": Signature(logit_diff, "logit", "logit_diff/v1", "the logit of a minus the logit of b", columns=("a", "b")),
    "cross_entropy": Signature(cross_entropy, "nat", "cross_entropy/v1", "−log P(target)", columns=("target",)),
    "token_logit": Signature(token_logit, "logit", "token_logit/v1", "the logit of one token", columns=("token",)),
    "token_prob": Signature(
        token_prob, "probability", "token_prob/v1",
        "P(token); with the expected answer as the token, the soft accuracy", columns=("token",),
    ),
    "kl": Signature(kl, "nat", "kl/v1", "KL(of ‖ against), weighted by of", reads=("against",)),
    "js": Signature(js, "nat", "js/v1", "the Jensen–Shannon divergence of of and against, symmetric", reads=("against",)),
    "cosine": Signature(
        cosine, "cosine", "cosine/v1", "the cosine between two reads laid out alike", reads=("against",), logits=False
    ),
    "top_k": Signature(
        top_k, "[token, probability] list", "top_k/v1", "the k most likely tokens, each with its probability",
        params={"k": 5}, tokens=True,
    ),
}


def compute(
    kind: str, logits: Any, reads: tuple[Any, ...], ids: tuple[TokenIds, ...], params: dict[str, Any]
) -> Any:
    """`reads` and `ids` are the kind's further reads and its columns, each in
    its signature's order, and `params` its numbers: `kl` takes (against,)
    and no ids, `logit_diff` no reads and (a, b), `top_k` its `k`."""
    return SIGNATURES[kind].function(logits, *reads, *ids, **params)
