"""What each step means — the part no engine gets to have an opinion about.

Every function here takes the engine as its first argument and calls it for
exactly one thing: running a model. It never touches the model — the engine
holds that, and what kind of object it is, is the engine's business. The
dispatch, the order, the optimizer, the early stop, the metrics and the
results are the same on every runtime, so they are written once, here, as
plain functions.

The walk is: for each step, run it. Three rules hold throughout:

* **a step writes its own results.** Every step records what it produced,
  including the ones inside a fit, so a run is navigable at any depth
  (`root.steps["fit"].epochs[0][1].steps["iia"].results["iia"]`). On a long
  fit that is a real amount of small tensors coming home; it is worth it here
  and would be worth a `record` flag there.
* **what crosses steps is one `State`, scoped to a `steps` list.** Its
  `featurizers` are the live parameter sets, shared so a fit trains the
  rotation a later step scores with; its `values` are everything the steps
  before produced — every read, a generate's ids, a metric, a reduction — by
  name, so a write's operand is simply the name of an earlier value. A nested
  plan (a sweep point) gets a copy, so points cannot see each other's. The
  state is a local of the walk: never saved, never shipped, gone with the
  plan.
* **a value keeps its graph only inside a fit's update.** There the metrics
  are differentiated, so everything an update's steps publish stays attached
  until its optimizer step; everywhere else a value is published detached,
  and a value leaving a fit is always a result, which is detached too.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import safetensors.torch
import torch

from ..ops import featurizer as featurizer_module, intervene, locate, metrics
from ..ops.locate import Frame
from ..plan import Featurizers, Fit, Forward, Generate, Metric, Plan, PlanError, Reduce, Step
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

    #: The live parameter sets. The stateless ones exist before any document
    #: declares anything: `identity` is what a read or a write with no
    #: `featurizer` names, and it is never declared.
    featurizers: dict[str, Any] = field(default_factory=lambda: dict(intervene.FEATURIZERS))
    #: Every value the steps so far produced, by name.
    values: dict[str, Any] = field(default_factory=dict)
    #: For a value that still has its rows, whether it came back flat — one
    #: entry per position found, where its record's windows say — or as a
    #: rectangle, a row per row. Absent for a value with no row axis: a mean,
    #: a basis, a metric, a read stacked over every layer.
    flat: dict[str, bool] = field(default_factory=dict)
    #: Where each op acted, per row — the whole of a run's `Record`.
    records: Record = field(default_factory=dict)
    #: The ops whose record comes home: those of a step with a position the
    #: document does not already fix (`_dynamic`).
    reported: set[str] = field(default_factory=set)
    #: How many rows one model call may hold. None: all of them. A property
    #: of the run and not of the experiment — it bounds memory and moves the
    #: last bit, nothing else — so it arrives with `execute`, not the plan.
    batch_size: int | None = None
    #: Inside a fit's update: values keep their graph, for the backward.
    attached: bool = False

    def child(self, attached: bool = False) -> "State":
        """A nested scope: the same live featurizers, and what was produced
        before it as a copy — it reads what came earlier, and what it
        produces stays its own."""
        return State(
            featurizers=self.featurizers,
            values=dict(self.values),
            flat=dict(self.flat),
            records=dict(self.records),
            reported=set(self.reported),
            batch_size=self.batch_size,
            attached=attached,
        )

    def publish(self, name: str, value: Any, flat: bool | None = None) -> None:
        """Keep a value for the steps after this one; `flat` as the field
        says, None for a value with no rows."""
        self.values[name] = value if self.attached else value.detach()
        if flat is not None:
            self.flat[name] = flat

    def window(self, name: str, start: int, stop: int) -> Any:
        """Rows `start:stop` of a value, as a window of rows needs it: sliced
        by its own form, or whole if it has no rows to slice."""
        value = self.values[name]
        if name not in self.flat:
            return value
        return intervene.rows(value, self.records[name]["rows"] if self.flat[name] else None, start, stop)


def run(engine: Any, step: Step, state: State, name: str = "") -> None:
    """Execute one step. A `Plan` is a step, so this is the whole walk; `name`
    is the key a step has in its plan, which is what it publishes under. An
    engine starts it with a fresh `State` of its run's `batch_size`."""
    if isinstance(step, Plan):
        for key, child in step.steps.items():
            run(engine, child, state.child() if isinstance(child, Plan) else state, name=key)
    elif isinstance(step, Featurizers):
        build(step, state)
    elif isinstance(step, Forward):
        call(engine, name, step, state)
    elif isinstance(step, Metric):
        metric(name, step, state)
    elif isinstance(step, Reduce):
        reduce(name, step, state)
    elif isinstance(step, Fit):
        fit(engine, step, state)
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


def call(engine: Any, name: str, step: Forward, state: State) -> None:
    """One model call over every row, `batch_size` rows at a time.

    A window of rows is the same call over a slice of them, with each
    operand sliced the same way — a read that still has its rows is cut by
    its own layout, so a write in this step meets the operand of *its* row
    whichever earlier step read it, and a mean, which has no rows, is handed
    whole. An engine is handed a step that is merely shorter and cannot tell;
    what the windows read is concatenated back in row order, and every step
    after this one sees one call. With no `batch_size` there is one window,
    which is the step exactly as compiled.

    Rows were padded to one width on the client, so a window's positions are
    already right. What a smaller batch does change is the last bit: a GEMM
    over fewer rows rounds differently (FINDINGS §8).
    """
    rows = len(step.input_ids)
    size = state.batch_size or rows or 1
    operands = {op.operand for tap in step.taps for op in tap.writes if isinstance(op.operand, str)}
    dynamic = _dynamic(step)
    produced: list[dict[str, Any]] = []
    windows: list[Record] = []
    for start in range(0, rows, size):
        stop = min(start + size, rows)
        values = {one: state.window(one, start, stop) for one in operands}
        ready, found = located(engine, plan_module.window(step, start, stop), start, dynamic)
        # the operands' own windows over these rows, for the checks at the write
        taken = {one: {key: part[start:stop] for key, part in state.records[one].items()} for one in operands if one in state.records}
        _writes_land(ready, {**taken, **found}, start)
        if isinstance(step, Generate):
            values[name] = engine.generate(ready, values, state.featurizers)
            found.update(_continuation(engine, ready, values, values[name]))
        elif (logits := engine.forward(ready, values, state.featurizers)) is not None:
            values[name] = logits
        produced.append({one: value for one, value in values.items() if one not in operands})
        windows.append(found)
    record: Record = {
        op: {key: tuple(one for part in windows for one in part[op][key]) for key in ("rows", "reason", "tokens")}
        for op in (windows[0] if windows else {})
    }
    made = {
        one: produced[0][one] if len(produced) == 1 else torch.cat([part[one] for part in produced])
        for one in (produced[0] if produced else ())
    }
    _stack_layers(step, made, record)
    state.records.update(record)
    # a read has its own form; a call's own result is a rectangle; a read at
    # every layer has the layers first, and no rows a later window could take
    forms = {op.stack or op.name: op.flat for tap in step.taps for op in tap.reads if not op.layered}
    for one, whole in made.items():
        state.publish(one, whole, forms.get(one, False if one == name else None))
    if isinstance(step, Generate) or step.logits:
        step.results[name] = state.values[name].detach().cpu()
    step.results.update({one: state.values[one].detach().cpu() for one in step.keep})
    if dynamic:
        # Where this step read and wrote, and what it decoded to. Plain
        # tuples of integers and strings, so they come home in the plan like
        # a metric does and a table can print them beside a number.
        step.results["positions"] = record
        state.reported |= set(record)


def _stack_layers(step: Forward, made: dict[str, Any], record: Record) -> None:
    """A read at every layer, as the one value it is: its per-layer reads
    stacked in layer order, the layer axis first. Every layer read the same
    positions, so one record says where."""
    layers: dict[str, list[tuple[int, str]]] = {}
    for tap in step.taps:
        for op in tap.reads:
            if op.layered:
                layers.setdefault(op.layered, []).append((tap.address.layer or 0, op.name))
    for stacked, parts in layers.items():
        names = [one for _, one in sorted(parts)]
        made[stacked] = torch.stack([made.pop(one) for one in names])
        record[stacked] = record[names[0]]
        for one in names:
            del record[one]


def _dynamic(step: Forward) -> bool:
    """Whether this step has a position the document does not already fix.

    A text anchor is one — which rows carry a word is data — and so is any
    cut of the continuation, because the continuation is what the decode
    turned out to produce. A step with neither resolves the same integers on
    every row and every run, and the document already says so: it needs no
    character map and reports nothing, so a plan compiled before any of this
    existed still writes the table it used to.
    """
    return any(
        op.at.where is not None
        and (op.at.where.scope is not None or op.at.where.frame == "generated")
        for tap in step.taps
        for op in (*tap.reads, *tap.writes)
    )


def metric(name: str, step: Metric, state: State) -> None:
    """One score per scored row of one read.

    A metric's rows are the intersection of two halves: the column half,
    which the client decided from the data, and the position half, which
    only the run can know. Neither is authoritative alone. When the read's
    step reported where it acted, the metric carries the intersection and
    that record too, so the table it is saved to can say which token each
    number came from.
    """
    located_rows = state.records[step.of]["rows"]
    eligible = tuple(
        (step.rows is None or row in step.rows) and bool(one) for row, one in enumerate(located_rows)
    )
    value = state.values[step.of]
    # a read at every layer is scored layer by layer, a row of scores each
    scores = [
        metrics.compute(step.kind, *_measured(name, step, one, eligible, located_rows))
        for one in (value if step.layers else [value])
    ]
    state.publish(name, torch.stack(scores) if step.layers else scores[0])
    step.results[name] = state.values[name].detach().cpu()
    if step.of in state.reported:
        step.results["eligible"] = {name: eligible}
        step.results["positions"] = {step.of: state.records[step.of]}


def _measured(
    name: str,
    step: Metric,
    value: Any,
    eligible: tuple[bool, ...],
    located_rows: Positions,
) -> tuple[Any, tuple[Any, ...]]:
    """What this metric scores: the read's eligible rows, and their token ids.

    A metric reads one position per row — the compiler refused anything else
    — so a rectangular read's `(rows, 1, vocab)` is `(rows, vocab)` with the
    unit window off. A read gathered flat instead has one row per row it
    *found*, so the eligible rows are indexed by their place among those.
    The ids came compiled for the *column*-eligible rows, and the position
    half may drop more, so they are indexed the same way. Either side holds
    one entry per eligible row, in row order, and the mean downstream needs
    no mask.
    """
    keep = [row for row, one in enumerate(eligible) if one]
    if not keep:
        raise PlanError(
            f"metric {name!r}: none of these {len(eligible)} row(s) is both in the "
            f"metric's columns and at a position the run could resolve; a metric of "
            "nothing has no mean"
        )
    scored = step.rows if step.rows is not None else range(len(eligible))
    place = {row: index for index, row in enumerate(scored)}
    ids = tuple(tuple(one[place[row]] for row in keep) for one in step.ids)
    if not step.flat:
        return value[keep, 0], ids
    found = {row: index for index, row in enumerate(row for row, one in enumerate(located_rows) if one)}
    return value[[found[row] for row in keep]], ids


def reduce(name: str, step: Reduce, state: State) -> None:
    """One read, reduced over its rows: over rows for a rectangle, keeping
    the window; over every position for a ragged read, which has no window
    axis to keep. What is left has no rows, so every window shares it."""
    value = state.values[step.of]
    if step.reduce == "mean":
        value = value.mean(dim=0)
    else:
        assert step.k is not None
        value = featurizer_module.pca(value, step.k)
    state.publish(name, value)
    step.results[name] = state.values[name].detach().cpu()


def located(engine: Any, forward: Forward, start: int = 0, text: bool = True) -> tuple[Forward, Record]:
    """`forward` with every op's positions resolved, and what each one got.

    This is where a spec becomes integers, and it happens here — in the
    walk, inside the session, with the *model's* tokenizer — rather than on
    the client, so that a text anchor is looked for in the text the model
    will actually see. One `Frame` is built per forward and every tap shares
    it; building one per tap is the one place this could be slow. `text` is
    the character map, which is the expensive half and which only a step
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
        found[op.name] = _record(frame, windows, reasons)
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


def _continuation(engine: Any, forward: Forward, values: dict[str, Any], generated_ids: Any) -> Record:
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
    def generated(op: Any) -> bool:
        return op.at.where is not None and op.at.where.frame == "generated"

    # a read cut out of the continuation is its decode steps' reads, in step
    # order; every other op in the frame is one
    stacks: dict[str, list[Any]] = {}
    for _, op in sorted(((tap.step, op) for tap in forward.taps for op in tap.reads if generated(op)), key=_by_step):
        stacks.setdefault(op.stack or op.name, []).append(op)
    ops = {name: parts[0] for name, parts in stacks.items()}
    ops |= {op.name: op for tap in forward.taps for op in tap.writes if generated(op)}
    if not ops:
        return {}
    frame = locate.continuation(engine.tokenizer, tuple(tuple(int(one) for one in row) for row in generated_ids))
    found: Record = {}
    for name, op in ops.items():
        where = op.at.where
        windows, reasons = locate.locate(frame, where, op.at.anchors)
        if name in stacks and op.stack:
            # (rows, steps, width): each step read the one position it processed
            whole = torch.stack([values.pop(one.name) for one in stacks[name]], dim=1).squeeze(2)
            values[name] = intervene.gather(whole, Selection(positions=windows, flat=where.ragged), seq_axis=1)
        found[name] = _record(frame, windows, reasons)
    return found


def _record(frame: Frame, windows: Positions, reasons: tuple[str, ...]) -> dict[str, Any]:
    """What a run reports of one op: each row's window, why it is empty when
    it is, and what it decoded to in `frame`."""
    tokens = tuple(locate.tokens_of(frame, one, row) for row, one in enumerate(windows))
    return {"rows": windows, "reason": reasons, "tokens": tokens}


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
    are. `start` puts the row numbers back in the step's own terms, so a
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
    """The same steps, once per minibatch, with an optimizer between.

    Nothing about this loop is a second engine: an update is the walk over
    one minibatch's steps, in a scope whose values keep their graph, and the
    only things that happen between updates are an optimizer step and an
    evaluation over the held-out rows. `step.params` is the protocol's only
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
            scored = _scored(engine, update, state)
            scored.update({f"{name}.mask": gate.mask for name, gate in gates.items()})
            loss = objective(step.objective, scored)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())
        # The evaluation runs in eval mode: no gradients, and on rows the fit
        # never saw.
        _training(featurizers, step.params, False)
        with torch.no_grad():
            evaluated = _scored(engine, step.evaluation, state)
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
    # the fit's own results: its record, and each parameter as it trained it
    step.results["train"] = {"loss": torch.stack(losses).cpu(), "eval": torch.stack(scores).cpu()}
    step.results.update({name: featurizers[name].weight.detach().cpu() for name in step.params})


def _scored(engine: Any, steps: Plan, state: State) -> dict[str, Any]:
    """Every value a fit's subtree produced, live: the metrics its objective
    and its early stop name are among them, still attached to their graph."""
    inner = state.child(attached=True)
    run(engine, steps, inner)
    return inner.values


def _training(featurizers: dict[str, Any], names: tuple[str, ...], on: bool) -> None:
    """The one piece of mode: a gate is soft while it is being updated and
    hard whenever it is scored. A rotation has no use for the flag."""
    for name in names:
        featurizers[name].training = on


def objective(terms: tuple[tuple[float, str], ...], scored: dict[str, Any]) -> Any:
    """Σ wᵢ · termᵢ, minimized. The sign of the weight is the direction — a
    positive weight on a cross-entropy minimizes it, a −1 on a logit_diff
    maximizes the margin — and there is no `maximize` flag anywhere."""
    total = None
    for weight, name in terms:
        term = weight * scored[name].mean()
        total = term if total is None else total + term
    return total
