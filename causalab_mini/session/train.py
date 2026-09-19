"""The fit: the same plan, N times, with an optimizer between.

Nothing here is a second engine. An update is `observe` over one minibatch's
plan; the only things that happen between updates are an optimizer step and an
eval pass. The loop runs *inside* the session, because the parameter it steps is
the tensor the write rotated by and the graph it differentiates is the one the
forward built.
"""

from __future__ import annotations

from typing import Any

import torch

from ..plan import TrainPlan
from .observe import observe


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
