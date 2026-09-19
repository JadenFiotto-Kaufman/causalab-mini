"""Open one session and execute the plan — fit included.

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

from ..ops import featurizer, intervene
from ..plan import Plan
from .observe import observe
from .train import fit


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
    built: dict[str, Any] = dict(intervene.FEATURIZERS)
    for spec in plan.featurizers:
        weight = featurizer.start_weight(spec.d, spec.k, spec.seed)
        built[spec.name] = featurizer.KINDS[spec.kind](weight.requires_grad_(spec.trained))
    return built
