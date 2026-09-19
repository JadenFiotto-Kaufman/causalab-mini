"""One execution of a plan: its forwards, its taps, its metrics.

This is the primitive the rest of the session is built from. The scored run is
one call; a training update is one call over a minibatch's plan; an eval pass is
one call under `no_grad`. Nothing here knows whether it is being trained.

`values` is the only channel between forwards, and it never leaves: an operand
is the *name* of a read, and the tensor it names was produced by an earlier
forward of the same plan, in the same session, on the same machine.
"""

from __future__ import annotations

from typing import Any

import torch

from ..ops import intervene, metrics
from ..plan import Forward, Plan


def batch(forward: Forward) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes."""
    return {
        "input_ids": torch.tensor(forward.input_ids),
        "attention_mask": torch.tensor(forward.attention_mask),
    }



def observe(model: Any, plan: Plan, featurizers: dict[str, Any]) -> dict[str, Any]:
    """One execution of a plan's forwards: metrics, one number per row.

    Metrics are computed here, inside the session, so what could leave is
    already reduced — and, during a fit, so that the graph the loss needs is
    still the one the forward built.
    """
    values: dict[str, Any] = {}
    for forward in plan.forwards:
        with model.trace(batch(forward)):
            apply_taps(model, forward, values, featurizers)
    return {
        metric.name: metrics.compute(metric.kind, values[metric.of], metric.ids)
        for metric in plan.metrics
    }



def apply_taps(
    model: Any, forward: Forward, values: dict[str, Any], featurizers: dict[str, Any]
) -> None:
    """One pass over the addresses of one forward, in forward order.

    `values` carries reads between forwards inside the session: an operand is a
    read name, and its tensor was produced by an earlier forward of this plan.
    A read's featurizer is applied here too — `v_cf` is `Qᵀx`, not `x` — and it
    is the same object the write's `inverse` will use, which is what makes one
    featurizer name one parameter set.
    """
    for tap in forward.taps:
        for write in tap.writes:
            patched = intervene.apply_write(
                tap.address.read(model),
                write.positions,
                values[write.operand],
                write.mechanism,
                featurizers[write.featurizer],
                tap.address.seq_axis,
            )
            tap.address.write(model, patched)
        for read in tap.reads:
            gathered = intervene.gather(
                tap.address.read(model), read.positions, tap.address.seq_axis
            )
            values[read.name] = featurizers[read.featurizer].featurize(gathered)[0].clone()
