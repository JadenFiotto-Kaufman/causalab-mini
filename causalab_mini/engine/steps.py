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
from ..ops.locate import Frame
from ..plan import Featurizers, Fit, Forward, Observe, Plan, PlanError, Step, Weights
from ..plan import plan as plan_module
from ..shapes import Positions, Selection

#: What a run reports about where it read and wrote: per op, each row's
#: window, why it is empty when it is (`locate.REASONS`), and what the
#: window decoded back to. Plain tuples of integers and strings, so it comes
#: home in the plan's `results` like any other result does.
Record = dict[str, dict[str, Any]]


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
        """A nested plan's state: the same live featurizers, and the outputs
        published before it as a copy — it reads what an earlier step
        published, and what it publishes stays its own."""
        return State(
            featurizers=self.featurizers,
            outputs=dict(self.outputs),
            layout=dict(self.layout),
            batch_size=self.batch_size,
        )


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
    rows = len(step.forwards[0].input_ids) if step.forwards else 0
    # A metric's rows are the intersection of two halves: the column half,
    # which the client decided from the data, and the position half, which
    # only the run can know. Neither is authoritative alone.
    eligible = {
        metric.name: tuple(
            (metric.rows is None or row in metric.rows)
            and bool(positions[metric.of]["rows"][row])
            for row in range(rows)
        )
        for metric in step.metrics
    }
    scored = {
        metric.name: metrics.compute(
            metric.kind,
            *_measured(step, metric, values[metric.of], eligible[metric.name], positions[metric.of]["rows"], rows),
        )
        for metric in step.metrics
    }
    step.results.update({name: value.detach().cpu() for name, value in scored.items()})
    if positions and _dynamic(step):
        # Where this pass read and wrote, and which rows it could score.
        # Plain tuples of integers and strings, so they come home in the plan
        # like a metric does and a table can print them beside a number.
        step.results["positions"] = positions
        if eligible:
            step.results["eligible"] = eligible
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
            state.layout[output.name] = positions[output.read]["rows"]
        step.results[output.name] = tensor.detach().cpu()
    return scored


def _dynamic(step: Observe) -> bool:
    """Whether this pass has a position the document does not already fix.

    A text anchor is one — which rows carry a word is data — and so is any
    cut of the continuation, because the continuation is what the decode
    turned out to produce. A pass with neither resolves the same integers on
    every row and every run, and the document already says so: it needs no
    character map and reports nothing, so a plan compiled before any of this
    existed still writes the table it used to.
    """
    return any(
        op.at.where is not None
        and (op.at.where.scope is not None or op.at.where.frame == "generated")
        for forward in step.forwards
        for tap in forward.taps
        for op in (*tap.reads, *tap.writes)
    )


def _measured(
    step: Observe,
    metric: Any,
    value: Any,
    eligible: tuple[bool, ...],
    located: Positions,
    rows: int,
) -> tuple[Any, tuple[Any, ...]]:
    """What this metric scores: the read's eligible rows, and their token ids.

    A metric reads one position per row — the compiler refused anything else
    — so a rectangular read's `(rows, 1, vocab)` is `(rows, vocab)` with the
    unit window off. A text-anchored read gathered flat instead, one row per
    row it *found*, so the eligible rows are indexed by their place among
    those. The ids came compiled for the *column*-eligible rows, and the
    position half may drop more, so they are indexed the same way. Either
    side holds one entry per eligible row, in row order, and the mean
    downstream needs no mask.
    """
    keep = [row for row, one in enumerate(eligible) if one]
    if not keep:
        raise PlanError(
            f"metric {metric.name!r}: none of these {len(eligible)} row(s) is both in the "
            f"metric's columns and at a position the run could resolve; a metric of "
            "nothing has no mean"
        )
    scored = metric.rows if metric.rows is not None else range(rows)
    place = {row: index for index, row in enumerate(scored)}
    ids = tuple(tuple(one[place[row]] for row in keep) for one in metric.ids)
    if not _gathered_flat(step, metric.of):
        return value[keep, 0], ids
    found = {row: index for index, row in enumerate(row for row, one in enumerate(located) if one)}
    return value[[found[row] for row in keep]], ids


def _gathered_flat(step: Observe, name: str) -> bool:
    """Whether the value called `name` came back flat — one row per row that
    resolved — rather than as a rectangle.

    The op that produced it is the read of that name, or, for a read the run
    cut out of the continuation, any one of the per-step ops that made it.
    For a read of the prompt the selection answers; for one cut out of the
    continuation the spec does, because the cut happened over the steps and
    not at the tap.
    """
    op = next(
        one
        for forward in step.forwards
        for tap in forward.taps
        for one in tap.reads
        if name in (one.name, one.stack)
    )
    if not op.stack:
        return op.at.flat
    return op.at.where is not None and op.at.where.ragged


def passes(engine: Any, step: Observe, state: State) -> tuple[dict[str, Any], Record]:
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
    text = _dynamic(step)
    parts: list[dict[str, Any]] = []
    windows: list[Record] = []
    for start in range(0, count, size):
        stop = min(start + size, count)
        values = {
            name: intervene.rows(tensor, state.layout.get(name), start, stop)
            for name, tensor in state.outputs.items()
        }
        published = set(values)
        found: Record = {}
        for forward in step.forwards:
            ready, resolved = located(engine, plan_module.window(forward, start, stop), start, text)
            found.update(resolved)
            _writes_land(ready, found, start)
            engine.forward(ready, values, state.featurizers)
            found.update(_continuation(engine, ready, values))
        parts.append({name: value for name, value in values.items() if name not in published})
        windows.append(found)
    merged = dict(state.outputs)
    for name in parts[0] if parts else ():
        merged[name] = parts[0][name] if len(parts) == 1 else torch.cat([part[name] for part in parts])
    # each window resolved its own rows; in row order they are the pass's
    return merged, {
        name: {
            key: tuple(one for part in windows for one in part[name][key])
            for key in ("rows", "reason", "tokens")
        }
        for name in (windows[0] if windows else {})
    }


def located(engine: Any, forward: Forward, start: int = 0, text: bool = True) -> tuple[Forward, Record]:
    """`forward` with every op's positions resolved, and what each one got.

    This is where a spec becomes integers, and it happens here — in the
    walk, inside the session, with the *model's* tokenizer — rather than on
    the client, so that a text anchor is looked for in the text the model
    will actually see. One `Frame` is built per forward and every tap shares
    it; building one per tap is the one place this could be slow. `text` is
    the character map, which is the expensive half and which only a pass
    that reports where it acted has any use for — see `_dynamic`.

    What comes back beside the forward is what the run reports: per op, each
    row's `rows` window, the `reason` it is empty when it is, and the
    `tokens` it actually addressed. The last of those is the provenance a
    number needs, and it is free here because the tokenizer is here.

    A tap in the continuation frame is not resolved against the prompt at
    all: it acts at the one position its decode step processes, which is
    what `intervene.at_step` puts any non-empty window on, and *where* in
    the continuation that was is reported by `_continuation` — which is the
    only thing that has the right frame to say.
    """
    frame = locate.frame_of(
        engine.tokenizer, forward.input_ids, forward.attention_mask, text=text
    )
    # the runs the *client* located — a chat template's turns, which only it
    # saw — in the same coordinates this frame uses
    frame = replace(frame, segments=forward.segments)
    _same_text(forward, frame)
    rows = len(forward.input_ids)
    found: Record = {}

    def resolve(kind: str, op: Any) -> Any:
        where = op.at.where
        if where is None or where.frame == "generated":
            return replace(op, at=replace(op.at, positions=((0,),) * rows))
        windows, reasons = locate.locate(frame, where, op.at.anchors)
        if not where.ragged:
            _fits_every_row(kind, op, windows, reasons, start)
        found[op.name] = {
            "rows": windows,
            "reason": reasons,
            "tokens": tuple(locate.tokens_of(frame, one, row) for row, one in enumerate(windows)),
        }
        return replace(op, at=replace(op.at, positions=windows))

    return (
        replace(
            forward,
            taps=tuple(
                replace(
                    tap,
                    writes=tuple(resolve("write", op) for op in tap.writes),
                    reads=tuple(resolve("read", op) for op in tap.reads),
                )
                for tap in forward.taps
            ),
        ),
        found,
    )


def _continuation(engine: Any, forward: Forward, values: dict[str, Any]) -> Record:
    """What the continuation frame's taps did, once there is a continuation.

    Two jobs, and they need the same `Frame`: the ids the decode produced,
    cut at each row's first stop token.

    A read whose position only the finished text can settle fired at every
    decode step and left one value per step; those become one tensor and the
    spec is resolved against the continuation to cut it — so `{"index": -1}`
    is the row's own last generated token, `{"scope": {"segment": "eos"}}` is
    where it stopped, and a row that never stopped says so rather than ending
    the run.

    And *every* tap in that frame reports where it was, including the ones
    that needed no stack. A tap at a named step acts at whatever one position
    its step processes, which is a fact about the decode and not about the
    prompt — so the prompt frame has nothing true to say about it, and this
    is the only place that does.
    """
    generated = values.get(f"{forward.name}.generated")
    in_frame = [
        (tap.step, op)
        for tap in forward.taps
        for op in (*tap.reads, *tap.writes)
        if op.at.where is not None and op.at.where.frame == "generated"
    ]
    if not in_frame or generated is None:
        return {}
    frame = locate.continuation(
        engine.tokenizer, tuple(tuple(int(one) for one in row) for row in generated)
    )
    found: Record = {}
    for name in dict.fromkeys(getattr(op, "stack", "") or op.name for _, op in in_frame):
        parts = [op for _, op in sorted(in_frame, key=_by_step) if (getattr(op, "stack", "") or op.name) == name]
        where = parts[0].at.where
        assert where is not None
        windows, reasons = locate.locate(frame, where, parts[0].at.anchors)
        if getattr(parts[0], "stack", ""):
            steps = [values.pop(op.name) for op in parts]
            # (rows, steps, width): each step read the one position it processed
            whole = torch.stack(steps, dim=1).squeeze(2)
            values[name] = intervene.gather(
                whole, Selection(positions=windows, flat=where.ragged), seq_axis=1
            )
        found[name] = {
            "rows": windows,
            "reason": reasons,
            "tokens": tuple(locate.tokens_of(frame, one, row) for row, one in enumerate(windows)),
        }
    return found


def _by_step(one: tuple[Any, Any]) -> int:
    """A stack's parts in decode order. A tap at every step (`"all"`) is not
    one of them, and sorts first."""
    return one[0] if isinstance(one[0], int) else -1


def _fits_every_row(kind: str, op: Any, windows: Positions, reasons: tuple[str, ...], start: int) -> None:
    """A fixed-width cut that does not fit a row is an authoring error.

    `{"last": 12}` on a nine-token row, `{"index": 40}` on any of these —
    the form names the same number of tokens on every row, so a row it does
    not fit is a document that is wrong about its own prompts, not a row
    with nothing to say. An anchored cut is the other case and is reported
    per row instead: which rows carry a word is data.
    """
    missed = {start + row: reasons[row] for row, window in enumerate(windows) if not window}
    if missed:
        raise PlanError(
            f"{kind} {op.name!r} at {op.at.where.spelling()} has no position on row(s) "
            f"{missed}. A position of a fixed width names the same number of tokens on "
            "every row, so a row it does not fit is refused rather than skipped; a "
            "position anchored to the row's own text may skip a row, and says why"
        )


def _same_text(forward: Forward, frame: Frame) -> None:
    """One row of this batch, as the client read it and as the run reads it.

    The ids were encoded by the client's tokenizer, and every anchor is
    looked for in what *this* tokenizer says they say. Those are one object
    on a local run and two on a remote one, and a disagreement would put
    every position in range and in the wrong place — silently, because both
    sides produce integers. Decoding the same row on both sides and
    comparing the strings is a check the two sides can both make, which
    re-encoding is not: `encode(decode(ids)) == ids` is not a property
    tokenizers have, and an emoji is enough to break it.

    One row is a sample. A skew that first shows on row 7 passes.
    """
    if not forward.sample or not frame.texts:
        return
    if frame.texts[0] != forward.sample:
        raise PlanError(
            "the tokenizer here disagrees with the one that encoded this plan: row 0's ids "
            f"decode to {frame.texts[0]!r} here and to {forward.sample!r} where the plan was "
            "compiled. Positions are resolved against the model's own tokenizer, so a plan "
            "encoded by a different one would be addressed in the wrong place"
        )


def _writes_land(forward: Forward, record: Record, start: int) -> None:
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
            reasons = record.get(write.name, {}).get("reason", ())
            missed = {
                start + row: (reasons[row] if row < len(reasons) else "") or "out_of_range"
                for row, window in enumerate(write.at.positions)
                if not window
            }
            if missed:
                raise PlanError(
                    f"write {write.name!r} has nothing to write on row(s) {missed}. A read "
                    "may skip a row; a write may not — writing nothing somewhere is not an "
                    "intervention, and the row would score as if one had happened. "
                    "`alignment_ambiguous` is a value that is in the prompt more than once, "
                    "and scoping the anchor is what makes it one"
                )
            if isinstance(write.operand, str) and write.operand in record:
                have = [len(window) for window in record[write.operand]["rows"]]
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
