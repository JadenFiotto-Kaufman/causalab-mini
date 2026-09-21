"""The plan: a tree of steps, in the order they will run.

A plan is data. It holds strings, integers and other steps — no envoys, no
tokenizer, no document, and (until it has been run) no tensors. It does not
know how to execute itself: an `Engine` walks it. That separation is the whole
reason there can be more than one engine.

The tree, top down:

    Plan                        a list of steps, and the files they produce
      steps: {name: Step}       executed in order; a step may be a Plan
      saves: (SaveFile, …)      what leaves the run, written by `Plan.write`
      results: {name: tensor}   filled in as it runs

    Featurizers(specs)          construct the parameter sets
    Observe(forwards, metrics)  one pass: forwards in order, then the metrics
    Fit(epochs, evaluation, …)  that pass N times, with an optimizer between
    Weights(names)              the fitted parameters, as results

and, inside an `Observe`, the ops that make one pass:

    Forward(name, input, input_ids, attention_mask, taps)
      taps: one per address, in forward order
        Tap(address, writes, reads)     writes run before reads at the same
          WriteOp(name, positions, operand, mechanism, featurizer)   address
          ReadOp(name, positions, featurizer)

**`results` is the only mutable thing on a step, and `provenance` the only
other one on a plan.** Every other field is frozen: a plan cannot be edited,
only filled. Filling it is how results come home —
the engine saves the root plan at the top of its session, so what the run
produced is navigable exactly where it happened,
`root.steps["fit"].results["train/loss"]`.

A fit is a request, so a fit is part of one plan: `Fit` holds the rows of every
update it will make, already batched, already tokenized, already in the order
the seed puts them in — as `Observe` steps, because a training update is this
same pass over different rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from ..address import Address
from ..shapes import ExampleIds, Selection, TokenIds, TokenRows


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class ReadOp:
    name: str
    at: Selection  # where in the tensor: positions, and which features
    featurizer: str = "identity"
    #: "raw" is the tensor at the address. "logits" is that tensor pushed
    #: through the model's final norm and head — the logit lens: what the
    #: model would say if this layer were its last.
    view: str = "raw"


@dataclass(frozen=True)
class WriteOp:
    name: str
    at: Selection
    #: A name — a read of this pass, or an output published before it — a
    #: literal number (zero ablation is `0.0`), or nothing for a mechanism
    #: that takes none.
    operand: str | float | None
    mechanism: str
    featurizer: str
    #: The mechanism's numbers: a scale, a seed.
    params: dict[str, float] = field(default_factory=dict)
    #: Which coordinates of the featurizer's space the mechanism acts on.
    #: None: all of them.
    features: tuple[int, ...] | None = None


@dataclass(frozen=True)
class Tap:
    """One place in one forward: an address, and — when the forward decodes —
    which step. `None` is the prompt frame: the prefill, with positions
    resolved against the prompt. An integer is that decode step, at the one
    position it processes. `"all"` is every step, and only a write may say
    it: steering is a write at every step."""

    address: Address
    writes: tuple[WriteOp, ...]
    reads: tuple[ReadOp, ...]
    step: int | str | None = None


@dataclass(frozen=True)
class Forward:
    name: str  # "original" or an intervened model's name
    input: str  # the data role its rows come from
    input_ids: TokenRows
    attention_mask: TokenRows
    taps: tuple[Tap, ...]
    #: How many tokens to generate after the prompt. 0 is one forward pass;
    #: N is the prefill plus N decode steps, greedy, EOS held off so the
    #: bound holds, and the generated ids come back as a value named
    #: `<forward>.generated`.
    decode: int = 0


def window(forward: Forward, start: int, stop: int) -> Forward:
    """Rows `start:stop` of a forward: the same taps over fewer rows. Every
    per-row thing a forward holds is a tuple with one entry per row — its
    token ids, its mask, each op's positions — so a window is a slice of
    each, and an engine cannot tell it from a forward compiled that small.
    Positions are absolute indices into the batch's padded width, which the
    client fixed once for all rows, so they survive the slice unchanged."""
    return replace(
        forward,
        input_ids=forward.input_ids[start:stop],
        attention_mask=forward.attention_mask[start:stop],
        taps=tuple(
            replace(
                tap,
                writes=tuple(replace(op, at=replace(op.at, positions=op.at.positions[start:stop])) for op in tap.writes),
                reads=tuple(replace(op, at=replace(op.at, positions=op.at.positions[start:stop])) for op in tap.reads),
            )
            for tap in forward.taps
        ),
    )


@dataclass(frozen=True)
class OutputOp:
    """One value a pass makes available to the steps after it: a read, kept
    — optionally reduced over rows — under a name later steps reference."""

    name: str
    read: str
    #: "none" keeps the read; "mean" averages the rows away; "pca" reduces
    #: the rows to their top-k principal directions, a `(d, k)` basis.
    reduce: str = "none"
    k: int | None = None


@dataclass(frozen=True)
class MetricOp:
    name: str
    kind: str
    of: str  # the read it binds to
    ids: tuple[TokenIds, ...]  # one vocabulary id per scored row, per operand
    #: The rows this metric is computed for, when that is not all of them: a
    #: row whose answer column is null is an excluded measurement. Decided on
    #: the client, so the run indexes and never masks — a mean is a mean, and
    #: a NaN is still a bug rather than a convention.
    rows: tuple[int, ...] | None = None


@dataclass(frozen=True)
class FeaturizerOp:
    """One parameter set. `d` is derived from (model, site) here on the client,
    because the block may not decide anything from a tensor — including how wide
    the tensor it is about to rotate is."""

    name: str
    kind: str
    k: int
    d: int
    parametrization: str
    seed: int
    trained: bool
    #: Loaded tensors, as the bytes of a safetensors bundle — the one
    #: tensor-shaped thing a fresh plan carries, and it is an *input* of the
    #: experiment like the token ids are. Bytes because they are still plain
    #: data (they pickle, compare and ship), at four bytes a number: a real
    #: SAE is tens of millions of them, which a tuple of Python floats is not
    #: a format for. None: drawn from `seed` where the run runs.
    weight: bytes | None = None
    #: Where it was loaded from, for the record.
    source: str = ""


@dataclass(frozen=True)
class SaveFile:
    file_path: str
    value: str  # the result this file holds, by name, from this plan's subtree
    example_ids: ExampleIds = ()
    #: Per example id, whether the metric was computed for it. Empty: all.
    #: The result holds one value per *eligible* row, in order.
    eligible: tuple[bool, ...] = ()
    unit: str = ""
    estimand_version: str = ""
    produced_by: str = ""
    #: For a `.safetensors` bundle: the ArtifactIdentity stamped into its
    #: header. Empty for a metric table.
    identity: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Step:
    """One thing that happens, what it produced, and what of that is written.

    Every step is frozen except `results`, which the engine fills in as it
    runs. A step has no `execute`: what it means to run one is the engine's
    business, and a plan that knew would only work on one engine.

    **`saves` is on every step, not only on a plan**, and that is what makes
    a result addressable without a naming convention: a save names a result
    of *the step it sits on*. A fit's held-out score and the scored run's are
    both `iia`, and they never collide, because scope does the work that a
    prefix would otherwise have to do.
    """

    results: dict[str, Any] = field(default_factory=dict)
    saves: tuple["SaveFile", ...] = ()


@dataclass(frozen=True, kw_only=True)
class Featurizers(Step):
    """Construct the parameter sets this plan's writes and reads name."""

    specs: tuple[FeaturizerOp, ...]


@dataclass(frozen=True, kw_only=True)
class Observe(Step):
    """One execution of the forwards, and the metrics over what they read."""

    forwards: tuple[Forward, ...]
    metrics: tuple[MetricOp, ...]
    #: What this pass publishes to its later siblings. Ephemeral: it lives in
    #: the walk's state and dies with the plan, unless a save on this step
    #: names it too.
    outputs: tuple[OutputOp, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Fit(Step):
    """The same pass, N times, with an optimizer between.

    `epochs` is already shuffled: the seed covers data order, and data order
    decides which rows share a padded batch, which is a tokenizer question and
    therefore a client-side one. The *parameter* seed travels as data and is
    drawn where the parameter is built.
    """

    epochs: tuple[tuple[Observe, ...], ...]
    evaluation: Observe
    objective: tuple[tuple[float, str], ...]
    params: tuple[str, ...]
    lr: float
    weight_decay: float
    eval_metrics: tuple[str, ...]
    early_stop: str
    patience: int
    mode: str
    #: `(gate, first, last)`: the temperature of a gate's soft mask, annealed
    #: geometrically across the fit's updates.
    anneal: tuple[tuple[str, float, float], ...] = ()


@dataclass(frozen=True, kw_only=True)
class Weights(Step):
    """The fitted parameters, as results, so they can be saved."""

    names: tuple[str, ...]


S = TypeVar("S", bound=Step)


@dataclass(frozen=True, kw_only=True)
class Plan(Step):
    """A list of steps, executed in order — and itself a step, so plans nest.

    Three experiments in a row is a plan whose three steps are plans. Nothing
    else is needed for that, which is the test the shape has to pass.
    """

    steps: dict[str, Step] = field(default_factory=dict)
    #: The document this plan was compiled from, verbatim, so an output
    #: directory can carry the experiment that produced it. JSON, so pure
    #: data; set by the compiler on the root and on each sweep point.
    source: dict[str, Any] | None = None
    #: What ran: engine, remote, versions, code digest. The one thing besides
    #: `results` that is filled in rather than compiled — by the engine at the
    #: top of `execute`, before the session opens.
    provenance: dict[str, Any] = field(default_factory=dict)

    def step(self, name: str, kind: type[S] = Step) -> S:  # type: ignore[assignment]
        """The step called `name`, checked to be the kind you expected.

        `plan.steps["observe"]` is a `Step` as far as a type checker knows, so
        reading `.forwards` off it is unchecked. `plan.step("observe",
        Observe)` is the same lookup with the kind stated, which a checker can
        follow and a wrong document shape trips on immediately.
        """
        found = self.steps[name]
        if not isinstance(found, kind):
            raise PlanError(
                f"step {name!r} is a {type(found).__name__}, not a {kind.__name__}"
            )
        return found

    def result(self, name: str) -> Any:
        """The one result called `name` in this subtree.

        The search descends through **steps** and stops at them: a `Fit`'s own
        results are found, the hundreds of per-update passes inside it are not,
        or every fitted document would have an ambiguous `iia`. Those are still
        there to read, by attribute, where they happened —
        `root.steps["fit"].epochs[0][0].results["iia"]`.

        Result names come from the document and are unique within a plan; a
        sweep repeats them across its points, and naming the point
        (`root.steps["seed=0"].result("iia")`) is what disambiguates. An
        ambiguous lookup is an error rather than a first match, because the
        wrong number quietly is the worst outcome here.
        """
        found = _find(self, name)
        if len(found) != 1:
            raise PlanError(
                f"result {name!r}: {len(found)} in this plan. Results here: "
                f"{sorted(_names(self))}"
            )
        return found[0]

    def all_results(self) -> dict[str, Any]:
        """Every result in this subtree, flattened by name.

        A run with one point has unique names throughout, so this is the whole
        of what it produced. A sweep repeats them and this refuses; ask a point
        (`root.steps["seed=0"].all_results()`) instead.
        """
        found: dict[str, Any] = {}
        for name in _names(self):
            found[name] = self.result(name)
        return found

    def write(self, out_dir: str | Path) -> list[Path]:
        """Write this plan's save manifest, and its children's below it."""
        from . import write as write_module  # local: writing is not part of the shape

        return write_module.write(self, out_dir)


def steps_of(step: Step) -> tuple[Step, ...]:
    """The steps one step contains, for a *name* lookup: what a plan
    declares, and nothing else. A `Fit`'s passes are its own workings rather
    than steps of the plan, so a name search stops there — see
    `Plan.result`. `children` is the wider walk, for writing.
    """
    return tuple(step.steps.values()) if isinstance(step, Plan) else ()


def children(step: Step) -> tuple[tuple[str, Step], ...]:
    """Every step inside this one, named, for a walk that is about *places*
    rather than names: writing files, or reading what a run produced.

    Unlike `steps_of` this descends into a fit, because a fit's eval pass is
    a place a save may sit. It stops at the per-update passes, which are the
    fit's own workings and carry no saves.
    """
    if isinstance(step, Plan):
        return tuple(step.steps.items())
    if isinstance(step, Fit):
        return (("eval", step.evaluation),)
    return ()


def results_of(step: Step, path: str = "") -> dict[str, dict[str, Any]]:
    """Everything a run produced, as `{step path: {name: value}}` — plain
    strings and tensors, with none of this package's classes in it. That is
    what may cross back from a server: the plan went out by value, so the
    server can *use* its classes but cannot pickle an instance of one home.
    The client still has the plan; only what filled it in has to travel."""
    found = {path: dict(step.results)} if step.results else {}
    for name, child in children(step):
        found.update(results_of(child, f"{path}/{name}" if path else name))
    return found


def fill(step: Step, results: dict[str, dict[str, Any]], path: str = "") -> None:
    """The inverse of `results_of`: put a run's results into this plan."""
    step.results.update(results.get(path, {}))
    for name, child in children(step):
        fill(child, results, f"{path}/{name}" if path else name)


def _find(step: Step, name: str) -> list[Any]:
    found = [step.results[name]] if name in step.results else []
    for child in steps_of(step):
        found.extend(_find(child, name))
    return found


def _names(step: Step) -> set[str]:
    return set(step.results) | {n for child in steps_of(step) for n in _names(child)}
