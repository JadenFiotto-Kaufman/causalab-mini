"""What each step means — the part no engine gets to have an opinion about.

Every function here takes the engine as its first argument and calls it for
exactly one thing: running a forward. The dispatch, the order, the optimizer,
the early stop, the metrics and the results are the same on every runtime, so
they are written once, here, as plain functions.

Two rules hold throughout:

* **a step writes its own results.** Every `Observe` records what it scored,
  including the ones inside a fit, so a run is navigable at any depth
  (`root.steps["fit"].epochs[0][1].results["iia"]`). On a long fit that is a
  real amount of small tensors coming home; it is worth it here and would be
  worth a `record` flag there.
* **nothing crosses steps except the featurizers.** `values` — the activations
  a write's operand refers to — are born and die inside one `Observe`, because
  a pass's forwards are compiled into one step. The featurizers are the only
  live state, and they are shared, so a fit trains the rotation a later step
  scores with.
"""

from __future__ import annotations

from typing import Any

import torch

from ..ops import featurizer as featurizer_module, intervene, metrics
from ..plan import Featurizers, Fit, Observe, Plan, Step, Weights


def run(engine: Any, model: Any, step: Step, featurizers: dict[str, Any] | None = None) -> None:
    """Execute one step. A `Plan` is a step, so this is the whole walk."""
    if featurizers is None:
        # The stateless featurizers exist before any document declares
        # anything: `identity` is what a read or a write with no `featurizer`
        # names, and it is never declared.
        featurizers = dict(intervene.FEATURIZERS)
    if isinstance(step, Plan):
        for child in step.steps.values():
            run(engine, model, child, featurizers)
    elif isinstance(step, Featurizers):
        build(step, featurizers)
    elif isinstance(step, Observe):
        observe(engine, model, step, featurizers)
    elif isinstance(step, Fit):
        fit(engine, model, step, featurizers)
    elif isinstance(step, Weights):
        weights(step, featurizers)
    else:
        raise TypeError(f"{type(step).__name__} is not a step this engine runs")


def build(step: Featurizers, featurizers: dict[str, Any]) -> None:
    """The plan's featurizer declarations, as live objects.

    This runs where the run runs, and that is the whole point: the parameter it
    draws is the tensor the optimizer will step and the write will use. Drawn
    on the client it would be a different tensor from the one the block
    rotates by.
    """
    for spec in step.specs:
        weight = featurizer_module.start_weight(spec.d, spec.k, spec.seed)
        featurizers[spec.name] = featurizer_module.KINDS[spec.kind](
            weight.requires_grad_(spec.trained)
        )


def observe(engine: Any, model: Any, step: Observe, featurizers: dict[str, Any]) -> dict[str, Any]:
    """One pass: the forwards in order, then the metrics over what they read.

    Returns the metrics live, because a fit differentiates them; records them
    detached, because what comes home should not carry a graph.
    """
    values: dict[str, Any] = {}
    for forward in step.forwards:
        engine.forward(model, forward, values, featurizers)
    scored = {
        metric.name: metrics.compute(metric.kind, values[metric.of], metric.ids)
        for metric in step.metrics
    }
    step.results.update({name: value.detach().cpu() for name, value in scored.items()})
    return scored


def fit(engine: Any, model: Any, step: Fit, featurizers: dict[str, Any]) -> None:
    """The same pass, N times, with an optimizer between.

    Nothing about this loop is a second engine: an update is `observe` over one
    minibatch's rows, and the only things that happen between updates are an
    optimizer step and an eval pass. `step.params` is the protocol's only
    trainability declaration, so the optimizer's parameter list *is* it — the
    model is frozen and nothing else in the run carries a gradient.
    """
    optimizer = torch.optim.AdamW(
        [featurizers[name].weight for name in step.params],
        lr=step.lr,
        weight_decay=step.weight_decay,
    )
    losses, scores = [], []
    best, waited = None, 0
    for epoch in step.epochs:
        for update in epoch:
            loss = objective(step.objective, observe(engine, model, update, featurizers))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        # The eval pass runs in eval mode: no gradients, and on rows the fit
        # never saw.
        with torch.no_grad():
            evaluated = observe(engine, model, step.evaluation, featurizers)
        scores.append(torch.stack([evaluated[name].mean() for name in step.eval_metrics]))
        watched = float(evaluated[step.early_stop].mean())
        # `mode` is "max"; the document refuses the other one.
        if best is None or watched > best:
            best, waited = watched, 0
        else:
            waited += 1
            if waited >= step.patience:
                break
    step.results["train/loss"] = torch.stack(losses).cpu()
    step.results["train/eval"] = torch.stack(scores).cpu()


def weights(step: Weights, featurizers: dict[str, Any]) -> None:
    """The fitted parameters, as results. This is the step that makes a
    rotation something a save entry can name."""
    for name in step.names:
        step.results[name] = featurizers[name].weight.detach().cpu()


def objective(terms: tuple[tuple[float, str], ...], scored: dict[str, Any]) -> Any:
    """Σ wᵢ · termᵢ, minimized. The sign of the weight is the direction — a
    positive weight on a cross-entropy minimizes it, a −1 on a logit_diff
    maximizes the margin — and there is no `maximize` flag anywhere."""
    total = None
    for weight, name in terms:
        term = weight * scored[name].mean()
        total = term if total is None else total + term
    return total
