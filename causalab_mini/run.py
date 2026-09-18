"""Open one session, execute the plan — fit included.

The whole request is one `model.session(...)`: not a session per forward, not a
session per training step, not a loop outside that opens sessions. `remote` is
passed to that one session and is the only difference between running here and
running on NDIF — there is no second code path.

A fit is a request, so the fit is inside that session. The featurizers, the
optimizer and the parameter RNG are all built **in the block, from the plan**,
because a client-side optimizer would be stepping tensors the block only ever
saw copies of, and would train nothing. What comes home is the trained
parameters and the scores; the activations stay where they were computed, and so
does `backward()`, which is run where the graph is.

Everything inside the session block is a module-level function taking the plan
and tensors. No `self`, no document, no tokenizer, no executor: nnsight ships
every name a block loads as a whole pickled object, so a block that reads one
of those ships it.
"""

from __future__ import annotations

from typing import Any

import nnsight
import torch

from . import featurizer, metrics, ops
from .plan import Forward, Plan, TrainPlan


def execute(model: Any, plan: Plan, remote: bool | str = False) -> dict[str, Any]:
    """Run `plan` against `model`, fitting first if it declares a fit, and
    return {metric name: per-row tensor} plus {featurizer name: its weight}."""
    if remote:
        # Our own package is not installed on an NDIF server, so the functions
        # the block calls have to ship by value.
        nnsight.register("causalab_mini")
    with model.session(remote=remote):
        results = nnsight.save({})
        featurizers = build_featurizers(plan)
        if plan.train is not None:
            fit(model, plan.train, featurizers, results)
        for name, value in observe(model, plan, featurizers).items():
            results[name] = value.detach().cpu()
        for spec in plan.featurizers:
            results[spec.name] = featurizers[spec.name].weight.detach().cpu()
    return dict(results)


def build_featurizers(plan: Plan) -> dict[str, Any]:
    """The plan's featurizer declarations, as live objects.

    This runs inside the session, and that is the whole point: the parameter it
    draws is the tensor the optimizer will step and the write will use. Drawn on
    the client it would be a different tensor from the one the block rotates by.
    """
    built: dict[str, Any] = dict(ops.FEATURIZERS)
    for spec in plan.featurizers:
        weight = featurizer.start_weight(spec.d, spec.k, spec.seed)
        built[spec.name] = featurizer.KINDS[spec.kind](weight.requires_grad_(spec.trained))
    return built


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
            patched = ops.apply_write(
                tap.address.read(model),
                write.positions,
                values[write.operand],
                write.mechanism,
                featurizers[write.featurizer],
                tap.address.seq_axis,
            )
            tap.address.write(model, patched)
        for read in tap.reads:
            gathered = ops.gather(
                tap.address.read(model), read.positions, tap.address.seq_axis
            )
            values[read.name] = featurizers[read.featurizer].featurize(gathered)[0].clone()


def fit(
    model: Any, train: TrainPlan, featurizers: dict[str, Any], results: dict[str, Any]
) -> None:
    """The training loop, inside the session, over the same plan N times.

    Nothing about this loop is a second engine: an update is `observe` over one
    minibatch's plan, and the only things that happen between updates are an
    optimizer step and an eval pass. `train.params` is the protocol's only
    trainability declaration, so the optimizer's parameter list *is* it — the
    model is frozen and nothing else in the run carries a gradient.
    """
    optimizer = torch.optim.AdamW(
        [featurizers[name].weight for name in train.params],
        lr=train.lr,
        weight_decay=train.weight_decay,
    )
    losses, scores = [], []
    best, waited = None, 0
    for epoch in train.epochs:
        for update in epoch:
            loss = objective(train.objective, observe(model, update, featurizers))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        # The eval pass runs in eval mode: no gradients, and on rows the fit
        # never saw.
        with torch.no_grad():
            evaluated = observe(model, train.evaluation, featurizers)
        scores.append(torch.stack([evaluated[name].mean() for name in train.eval_metrics]))
        watched = float(evaluated[train.early_stop].mean())
        # `mode` is "max"; the document refuses the other one.
        if best is None or watched > best:
            best, waited = watched, 0
        else:
            waited += 1
            if waited >= train.patience:
                break
    results["train/loss"] = torch.stack(losses).cpu()
    results["train/eval"] = torch.stack(scores).cpu()


def objective(terms: tuple[tuple[float, str], ...], scored: dict[str, Any]) -> Any:
    """Σ wᵢ · termᵢ, minimized. The sign of the weight is the direction — a
    positive weight on a cross-entropy minimizes it, a −1 on a logit_diff
    maximizes the margin — and there is no `maximize` flag anywhere."""
    total = None
    for weight, name in terms:
        term = weight * scored[name].mean()
        total = term if total is None else total + term
    return total
