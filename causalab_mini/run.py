"""Open one session, execute the plan.

The whole request is one `model.session(...)`: not a session per forward, not a
lazy per-read path. `remote` is passed to that one session and is the only
difference between running here and running on NDIF — there is no second code
path.

Everything inside the session block is a module-level function taking the plan
and tensors. No `self`, no document, no tokenizer, no executor: nnsight ships
every name a block loads as a whole pickled object, so a block that reads one
of those ships it.
"""

from __future__ import annotations

import nnsight
import torch

from . import metrics, ops


def execute(model, plan, remote: bool | str = False) -> dict:
    """Run `plan` against `model` and return {metric name: per-row tensor}."""
    if remote:
        # Our own package is not installed on an NDIF server, so the functions
        # the block calls have to ship by value.
        nnsight.register("causalab_mini")
    with model.session(remote=remote):
        results = nnsight.save({})
        values = {}
        for forward in plan.forwards:
            with model.trace(batch(forward)):
                apply_taps(model, forward, values)
        score(plan, values, results)
    return dict(results)


def batch(forward) -> dict:
    """The plan's integers, as the tensors a forward takes."""
    return {
        "input_ids": torch.tensor(forward.input_ids),
        "attention_mask": torch.tensor(forward.attention_mask),
    }


def apply_taps(model, forward, values) -> None:
    """One pass over the addresses of one forward, in forward order.

    `values` carries reads between forwards inside the session: an operand is a
    read name, and its tensor was produced by an earlier forward of this plan.
    """
    for tap in forward.taps:
        for write in tap.writes:
            patched = ops.apply_write(
                tap.address.read(model),
                write.positions,
                values[write.operand],
                write.mechanism,
                write.featurizer,
            )
            tap.address.write(model, patched)
        for read in tap.reads:
            values[read.name] = ops.gather(tap.address.read(model), read.positions).clone()


def score(plan, values, results) -> None:
    """Metrics run inside the session, so what leaves is one number per row."""
    for metric in plan.metrics:
        results[metric.name] = metrics.compute(
            metric.kind, values[metric.of], metric.ids
        ).detach().cpu()
