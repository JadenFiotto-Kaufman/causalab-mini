"""What each step means — the part no engine gets to have an opinion about.

Every function here takes the engine as its first argument and calls it for
exactly one thing: running a forward. It never touches the model — the engine
holds that, and what kind of object it is, is the engine's business. The dispatch, the order, the optimizer,
the early stop, the metrics and the results are the same on every runtime, so
they are written once, here, as plain functions.

Two rules hold throughout:

* **a step writes its own results.** Every `Observe` records what it scored,
  including the ones inside a fit, so a run is navigable at any depth
  (`root.steps["fit"].epochs[0][1].results["iia"]`). On a long fit that is a
  real amount of small tensors coming home; it is worth it here and would be
  worth a `record` flag there.
* **what crosses steps is one `State`, scoped to a `steps` list.** Its
  `featurizers` are the live parameter sets, shared so a fit trains the
  rotation a later step scores with; its `outputs` are what earlier siblings
  published — a harvested activation, a mean — for a later step to reference.
  A nested plan (a sweep point) gets its own `outputs`, so points cannot see
  each other's. The state is a local of the walk: never saved, never shipped,
  gone with the plan. `values` — the activations a write's operand names —
  still live inside one `Observe`, seeded from `outputs` so an operand may be
  either a read of this pass or a published value from before it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import safetensors.torch
import torch

from ..ops import featurizer as featurizer_module, intervene, locate, metrics
from ..plan import Featurizers, Fit, Forward, Observe, Plan, PlanError, Step, Weights
from ..plan import plan as plan_module
from ..shapes import Positions


@dataclass
class State:
    """What one `steps` list shares, for as long as it runs."""

    featurizers: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    #: For an output that still has its rows: the positions it was read over,
    #: which say where each row is in it. Absent for one reduced over rows.
    layout: dict[str, Any] = field(default_factory=dict)
    #: How many rows one model call may hold. None: all of them. A property
    #: of the run and not of the experiment — it bounds memory and moves the
    #: last bit, nothing else — so it arrives with `execute`, not the plan.
    batch_size: int | None = None

    def child(self) -> "State":
        """A nested plan's state: the same live featurizers, its own outputs."""
        return State(featurizers=self.featurizers, batch_size=self.batch_size)


def run(engine: Any, step: Step, state: State | None = None, batch_size: int | None = None) -> None:
    """Execute one step. A `Plan` is a step, so this is the whole walk."""
    if state is None:
        # The stateless featurizers exist before any document declares
        # anything: `identity` is what a read or a write with no `featurizer`
        # names, and it is never declared.
        state = State(featurizers=dict(intervene.FEATURIZERS), batch_size=batch_size)
    if isinstance(step, Plan):
        for child in step.steps.values():
            run(engine, child, state.child() if isinstance(child, Plan) else state)
    elif isinstance(step, Featurizers):
        build(step, state)
    elif isinstance(step, Observe):
        observe(engine, step, state)
    elif isinstance(step, Fit):
        fit(engine, step, state)
    elif isinstance(step, Weights):
        weights(step, state)
    else:
        raise TypeError(f"{type(step).__name__} is not a step this engine runs")


def build(step: Featurizers, state: State) -> None:
    """The plan's featurizer declarations, as live objects.

    This runs where the run runs, and that is the whole point: the parameter it
    draws is the tensor the optimizer will step and the write will use. Drawn
    on the client it would be a different tensor from the one the block
    rotates by.
    """
    for spec in step.specs:
        if spec.weight is not None:
            # a bundle's own bytes, carried verbatim: compact, shippable, and
            # exactly what was checked on the client
            tensors = {name: one.to(torch.float32) for name, one in safetensors.torch.load(spec.weight).items()}
        else:
            tensors = {"weight": featurizer_module.start(spec.kind, spec.d, spec.k, spec.seed)}
        if spec.trained:
            tensors["weight"].requires_grad_(True)
        state.featurizers[spec.name] = featurizer_module.KINDS[spec.kind](**tensors)


def observe(engine: Any, step: Observe, state: State) -> dict[str, Any]:
    """One pass: the forwards in order, then the metrics over what they read.

    Returns the metrics live, because a fit differentiates them; records them
    detached, because what comes home should not carry a graph.

    `values` starts as a copy of the state's outputs, so a write's operand may
    name either a read of this pass or something an earlier step published —
    the engine cannot tell the difference and does not need to. What this
    pass declares as its own outputs is published at the end.
    """
    values, positions = passes(engine, step, state)
    # A metric reads one position per row — the compiler refused anything
    # else — so its (rows, 1, vocab) is (rows, vocab) with the unit window off.
    # A metric with excluded rows scores the others: the compiler said which,
    # so this indexes and the mean downstream needs no mask.
    scored = {
        metric.name: metrics.compute(
            metric.kind,
            values[metric.of][:, 0] if metric.rows is None else values[metric.of][list(metric.rows), 0],
            metric.ids,
        )
        for metric in step.metrics
    }
    step.results.update({name: value.detach().cpu() for name, value in scored.items()})
    # a decoding forward leaves its generated ids in `values`; they are a
    # result of the pass like a metric is
    step.results.update(
        {name: value.detach().cpu() for name, value in values.items() if name.endswith(".generated")}
    )
    for output in step.outputs:
        tensor = values[output.read]
        if output.reduce == "mean":
            # over rows for a rectangle, keeping the window; over every
            # position for a ragged read, which has no window axis to keep
            tensor = tensor.mean(dim=0)
        elif output.reduce == "pca":
            assert output.k is not None
            tensor = featurizer_module.pca(tensor, output.k)
        state.outputs[output.name] = tensor.detach()
        if output.reduce == "none":
            state.layout[output.name] = positions[output.read]
        step.results[output.name] = tensor.detach().cpu()
    return scored


def passes(engine: Any, step: Observe, state: State) -> tuple[dict[str, Any], dict[str, Positions]]:
    """Every forward of a pass, over every row, `batch_size` rows at a time.

    A window of rows is a whole small pass: the same forwards in the same
    order over a slice of the rows, with the published values it reads
    sliced the same way — so a write still meets the operand of *its* row.
    An engine is handed a forward that is merely shorter and cannot tell;
    what the windows read is concatenated back in row order, and everything
    after this function sees one pass. With no `batch_size` there is one
    window, which is the pass exactly as compiled.

    Rows were padded to one width on the client, so a window's positions are
    already right. What a smaller batch does change is the last bit: a GEMM
    over fewer rows rounds differently (FINDINGS §8).
    """
    count = len(step.forwards[0].input_ids) if step.forwards else 0
    size = state.batch_size or count or 1
    parts: list[dict[str, Any]] = []
    windows: list[dict[str, Positions]] = []
    for start in range(0, count, size):
        stop = min(start + size, count)
        values = {
            name: intervene.rows(tensor, state.layout.get(name), start, stop)
            for name, tensor in state.outputs.items()
        }
        published = set(values)
        found: dict[str, Positions] = {}
        for forward in step.forwards:
            ready, resolved = located(engine, plan_module.window(forward, start, stop))
            found.update(resolved)
            _writes_land(ready, found, start)
            engine.forward(ready, values, state.featurizers)
        parts.append({name: value for name, value in values.items() if name not in published})
        windows.append(found)
    merged = dict(state.outputs)
    for name in parts[0] if parts else ():
        merged[name] = parts[0][name] if len(parts) == 1 else torch.cat([part[name] for part in parts])
    # each window resolved its own rows; in row order they are the pass's
    return merged, {
        name: tuple(one for part in windows for one in part[name]) for name in (windows[0] if windows else {})
    }


def located(engine: Any, forward: Forward) -> tuple[Forward, dict[str, Positions]]:
    """`forward` with every op's positions resolved, and what each one got.

    This is where a spec becomes integers, and it happens here — in the
    walk, inside the session, with the *model's* tokenizer — rather than on
    the client, so that a text anchor is looked for in the text the model
    will actually see. One `Frame` is built per forward and every tap shares
    it; building one per tap is the one place this could be slow.

    A tap in the continuation frame is not resolved against the prompt at
    all: it acts at the one position its decode step processes, which is
    what `intervene.at_step` puts any non-empty window on.
    """
    frame = locate.frame_of(engine.tokenizer, forward.input_ids, forward.attention_mask)
    rows = len(forward.input_ids)
    found: dict[str, Positions] = {}

    def resolve(op: Any) -> Any:
        where = op.at.where
        if where is None or where.frame == "generated":
            windows: Positions = ((0,),) * rows
        else:
            windows, _ = locate.locate(frame, where, op.at.anchors)
        found[op.name] = windows
        return replace(op, at=replace(op.at, positions=windows))

    return (
        replace(
            forward,
            taps=tuple(
                replace(
                    tap,
                    writes=tuple(resolve(op) for op in tap.writes),
                    reads=tuple(resolve(op) for op in tap.reads),
                )
                for tap in forward.taps
            ),
        ),
        found,
    )


def _writes_land(forward: Forward, located: dict[str, Positions], start: int) -> None:
    """The ragged write policy, and it is `refuse`.

    A read may have an empty window on a row — the anchor's text was not in
    that prompt — and that row is simply an excluded measurement. A *write*
    may not: writing nothing somewhere is not an intervention, and the row
    would score as if it were. And a ragged write's operand must have, row
    by row, exactly the width the write covers; the protocol's other
    landing policies are not implemented.

    Both are checked here, at the write, because here is where the positions
    are. `start` puts the row numbers back in the pass's own terms, so a
    batched run names the row an author would count to.
    """
    for tap in forward.taps:
        for write in tap.writes:
            empty = [start + row for row, window in enumerate(write.at.positions) if not window]
            if empty:
                raise PlanError(
                    f"write {write.name!r} has nothing to write on row(s) {empty}: its "
                    "position's text is not in those prompts. A read may skip a row; a "
                    "write may not"
                )
            if isinstance(write.operand, str) and write.operand in located:
                have = [len(window) for window in located[write.operand]]
                want = [len(window) for window in write.at.positions]
                mismatched = [start + row for row, (a, b) in enumerate(zip(have, want)) if a != b]
                if mismatched:
                    raise PlanError(
                        f"write {write.name!r} covers {want} positions per row but its "
                        f"operand {write.operand!r} was read over {have}; rows "
                        f"{mismatched} differ. Landing a window of one width in another "
                        "is a policy this slice does not implement — the protocol's "
                        "`exact_length_buckets` and `padded_masked` — so it refuses"
                    )


def fit(engine: Any, step: Fit, state: State) -> None:
    """The same pass, N times, with an optimizer between.

    Nothing about this loop is a second engine: an update is `observe` over one
    minibatch's rows, and the only things that happen between updates are an
    optimizer step and an eval pass. `step.params` is the protocol's only
    trainability declaration, so the optimizer's parameter list *is* it — the
    model is frozen and nothing else in the run carries a gradient.
    """
    featurizers = state.featurizers
    optimizer = torch.optim.AdamW(
        [featurizers[name].weight for name in step.params],
        lr=step.lr,
        weight_decay=step.weight_decay,
    )
    # A gate's mask is a term the objective may name (`<name>.mask`, its mean
    # is the L1 penalty) and a column of the eval record (the fraction kept).
    gates = {name: featurizers[name] for name in step.params if hasattr(featurizers[name], "mask")}
    total, done = sum(len(epoch) for epoch in step.epochs), 0
    losses, scores = [], []
    best, waited = None, 0
    for epoch in step.epochs:
        _training(featurizers, step.params, True)
        for update in epoch:
            for name, first, last in step.anneal:
                # geometric, from `first` on the first update to `last` on the last
                featurizers[name].temperature = first * (last / first) ** (done / max(total - 1, 1))
            done += 1
            scored = dict(observe(engine, update, state))
            scored.update({f"{name}.mask": gate.mask for name, gate in gates.items()})
            loss = objective(step.objective, scored)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())
        # The eval pass runs in eval mode: no gradients, and on rows the fit
        # never saw.
        _training(featurizers, step.params, False)
        with torch.no_grad():
            evaluated = dict(observe(engine, step.evaluation, state))
            evaluated.update({f"{name}.mask": gate.mask for name, gate in gates.items()})
        # each on the CPU first: a metric is wherever the model is, a gate's mask
        # wherever its parameter is, and a record is neither's
        scores.append(torch.stack([evaluated[name].mean().cpu() for name in step.eval_metrics]))
        watched = float(evaluated[step.early_stop].mean())
        improved = best is None or (watched > best if step.mode == "max" else watched < best)
        if improved:
            best, waited = watched, 0
        else:
            waited += 1
            if waited >= step.patience:
                break
    step.results["train/loss"] = torch.stack(losses).cpu()
    step.results["train/eval"] = torch.stack(scores).cpu()


def _training(featurizers: dict[str, Any], names: tuple[str, ...], on: bool) -> None:
    """The one piece of mode: a gate is soft while it is being updated and
    hard whenever it is scored. A rotation has no use for the flag."""
    for name in names:
        featurizers[name].training = on


def weights(step: Weights, state: State) -> None:
    """The fitted parameters, as results. This is the step that makes a
    rotation something a save entry can name."""
    for name in step.names:
        step.results[name] = state.featurizers[name].weight.detach().cpu()


def objective(terms: tuple[tuple[float, str], ...], scored: dict[str, Any]) -> Any:
    """Σ wᵢ · termᵢ, minimized. The sign of the weight is the direction — a
    positive weight on a cross-entropy minimizes it, a −1 on a logit_diff
    maximizes the margin — and there is no `maximize` flag anywhere."""
    total = None
    for weight, name in terms:
        term = weight * scored[name].mean()
        total = term if total is None else total + term
    return total
