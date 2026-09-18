"""match, logit_diff, cross_entropy — pure tensor functions.

Each takes the read it binds to, shape (rows, vocab), plus one vocabulary id per
row per operand, and returns one value per row. The unit and estimand version
are derived from the kind and are never authored.
"""

from __future__ import annotations

import torch


def _rows(logits, ids):
    return torch.arange(logits.shape[0], device=logits.device), torch.as_tensor(
        ids, device=logits.device
    )


def match(logits, expected):
    """1.0 when the argmax is the expected token, else 0.0."""
    rows, ids = _rows(logits, expected)
    return (logits.argmax(dim=-1) == ids).to(torch.float32)


def logit_diff(logits, a, b):
    rows, a_ids = _rows(logits, a)
    _, b_ids = _rows(logits, b)
    return logits[rows, a_ids] - logits[rows, b_ids]


def cross_entropy(logits, target):
    rows, ids = _rows(logits, target)
    return -logits.log_softmax(dim=-1)[rows, ids]


KINDS = {"match": match, "logit_diff": logit_diff, "cross_entropy": cross_entropy}

UNITS = {
    "match": ("fraction", "match/v1"),
    "logit_diff": ("logit", "logit_diff/v1"),
    "cross_entropy": ("nat", "cross_entropy/v1"),
}


def compute(kind, logits, ids):
    """`ids` is the kind's operands in order: match takes (expected,),
    logit_diff takes (a, b), cross_entropy takes (target,)."""
    return KINDS[kind](logits, *ids)
