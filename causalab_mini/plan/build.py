"""The compiler: a document becomes a plan.

Everything that has to be decided against something the block will not have —
the tokenizer, the rows on disk, the model's layer count and hidden width, the
shuffle a seed defines — is decided here, once, on the client. What comes out
the other side is `plan.Plan`: strings and integers.

The order of the work is the order of the file: resolve every site to an
`Address`, load every role's rows, derive each featurizer's width, compile one
`_pass` per set of rows (the scored run, and one per training update), and
finally the save manifest.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..address import Address
from ..data import encoding, rows as rows_module
from ..ops import metrics as metrics_module
from . import sweep
from .document import Document, SaveSpec
from .plan import (
    FeaturizerOp,
    Featurizers,
    Fit,
    Forward,
    MetricOp,
    Observe,
    OutputOp,
    Plan,
    PlanError,
    ReadOp,
    SaveFile,
    Step,
    Tap,
    Weights,
    WriteOp,
)


def build_spec(spec: Any, data_root: str | Path, engine: Any) -> Plan:
    """Compile a plan-shaped document (`spec.py`) into a plan.

    Shorter than `build` below, and not because it does less: the document
    already says what the steps are and where the saves go, so this resolves
    what needs a model — the addresses, the widths, the tokenizer, the rows —
    and copies the structure across. That difference *is* the argument for
    the format.
    """
    for name, site in spec.sites.items():
        if site.layers is not None and not 0 <= site.layers[0] < engine.num_layers:
            raise PlanError(
                f"site {name!r}: layer {site.layers[0]} is outside the model's "
                f"{engine.num_layers} layers"
            )
    addresses = {
        name: engine.locate(site.component, site.layers[0] if site.layers else None)
        for name, site in spec.sites.items()
    }
    fits = [one for one in spec.steps.values() if type(one).__name__ == "Fit"]
    featurizers = tuple(
        _spec_featurizer(name, one, spec, addresses, engine, fits)
        for name, one in spec.featurizers.items()
    )
    widths = {one.name: one.d for one in featurizers}

    def table(refs: dict[str, str]) -> dict[str, list[rows_module.Row]]:
        loaded = {role: rows_module.load(data_root, ref) for role, ref in refs.items()}
        counts = {role: len(one) for role, one in loaded.items()}
        if len(set(counts.values())) != 1:
            raise PlanError(f"roles must have the same row count, got {counts}")
        return loaded

    steps: dict[str, Step] = {}
    if featurizers:
        # Declaring a featurizer is what builds it; the document does not
        # spell out a step whose whole content would be the declaration.
        steps["featurizers"] = Featurizers(specs=featurizers)
    #: An unreduced output is (rows, width), and a write that swaps it in
    #: needs the same rows. The compiler knows both row counts; the block
    #: would only find out from a shape error.
    output_rows: dict[str, int | None] = {}
    for name, step in spec.steps.items():
        kind = type(step).__name__
        experiment = (
            _Experiment.of_spec(spec, spec.intervention_of(step))
            if kind in ("Observe", "Fit")
            else None
        )
        if experiment is not None:
            count = len(table(step.rows)["base"])
            for write_name, write in spec.intervention_of(step).writes.items():
                if type(write.operand).__name__ == "Reference":
                    have = output_rows[write.operand.ref]
                    if have is not None and have != count:
                        raise PlanError(
                            f"step {name!r}: write {write_name!r} swaps in "
                            f"{write.operand.ref!r}, which has {have} rows, over {count} "
                            "rows; reduce the output or run over the same rows"
                        )
        if kind == "Observe":
            assert experiment is not None
            rows = table(step.rows)
            outputs = tuple(
                OutputOp(name=out_name, read=out.read, reduce=out.reduce)
                for out_name, out in step.outputs.items()
            )
            for out in outputs:
                output_rows[out.name] = None if out.reduce == "mean" else len(rows["base"])
            steps[name] = replace(
                _pass(experiment, rows, addresses, engine.tokenizer),
                outputs=outputs,
                saves=_spec_saves(step.saves, spec, rows["base"], widths, outputs={o.name for o in outputs}),
            )
        elif kind == "Fit":
            assert experiment is not None
            steps[name] = _spec_fit(step, spec, experiment, table, addresses, engine, widths)
        elif kind == "Weights":
            steps[name] = Weights(
                names=tuple(step.names),
                saves=_spec_saves(step.saves, spec, [], widths, sites=_sites_of(spec)),
            )
        else:  # pragma: no cover — the discriminated union has no other arm
            raise PlanError(f"step {name!r}: {kind} is not a step this compiler knows")
    return Plan(steps=steps, source=spec.model_dump(mode="json"))


def _spec_fit(
    step: Any,
    spec: Any,
    experiment: _Experiment,
    table: Any,
    addresses: dict[str, Address],
    engine: Any,
    widths: dict[str, int],
) -> Fit:
    rows = table(step.rows)
    evaluation_rows = table(step.eval.rows)
    if step.rows != step.eval.rows:
        fitted = {json.dumps(row, sort_keys=True) for row in rows["base"]}
        shared = [one for one in evaluation_rows["base"] if json.dumps(one, sort_keys=True) in fitted]
        if shared:
            raise PlanError(
                f"the eval rows share {len(shared)} row(s) with the fitted rows; "
                "the two must be endpoint-disjoint"
            )
    count = len(rows["base"])
    order = random.Random(step.seed)
    epochs = tuple(
        tuple(
            _pass(experiment, _take(rows, draw[start : start + step.pairs]), addresses, engine.tokenizer)
            for start in range(0, count, step.pairs)
        )
        for draw in (order.sample(range(count), count) for _ in range(step.epochs))
    )
    return Fit(
        epochs=epochs,
        evaluation=replace(
            _pass(experiment, evaluation_rows, addresses, engine.tokenizer),
            saves=_spec_saves(step.eval.saves, spec, evaluation_rows["base"], widths),
        ),
        objective=tuple((weight, term) for weight, term in step.objective),
        params=tuple(step.params),
        lr=step.optimizer.lr,
        weight_decay=step.optimizer.weight_decay,
        eval_metrics=(step.early_stop.metric,),
        early_stop=step.early_stop.metric,
        patience=step.early_stop.patience,
        mode=step.early_stop.mode,
        saves=_spec_saves(step.saves, spec, [], widths),
    )


def _spec_featurizer(
    name: str, one: Any, spec: Any, addresses: dict[str, Address], engine: Any, fits: list[Any]
) -> FeaturizerOp:
    site = _sites_of(spec)[name]
    d = engine.width(addresses[site])
    if not 0 < one.k <= d:
        raise PlanError(
            f"featurizer {name!r}: k={one.k} is not a subspace of the {d}-wide site {site!r}"
        )
    trained = any(name in fit.params for fit in fits)
    seed = one.seed
    if seed is None:
        seed = next((fit.seed for fit in fits if name in fit.params), 0)
    return FeaturizerOp(
        name=name,
        kind=one.kind,
        k=one.k,
        d=d,
        parametrization=one.parametrization,
        seed=seed,
        trained=trained,
    )


def _sites_of(spec: Any) -> dict[str, str]:
    """Which site each featurizer acts at — derived, never authored, because
    a featurizer name is one parameter set and the document already says
    where it is used."""
    found = {}
    for name in spec.featurizers:
        at = {
            read.site
            for one in spec.interventions.values()
            for read in one.reads.values()
            if read.featurizer == name
        }
        at |= {
            w.site
            for one in spec.interventions.values()
            for w in one.writes.values()
            if w.featurizer == name
        }
        (found[name],) = at
    return found


def _spec_saves(
    saves: list[Any],
    spec: Any,
    base_rows: list[rows_module.Row],
    widths: dict[str, int],
    sites: dict[str, str] | None = None,
    outputs: frozenset[str] | set[str] = frozenset(),
) -> tuple[SaveFile, ...]:
    """A step's saves. A metric table carries its rows' labels; a fitted
    parameter carries the identity stamp a later run would check; a
    published output is a tensor and goes to a safetensors file as it is."""
    built = []
    for save in saves:
        if save.value in outputs:
            if not save.file_path.endswith(".safetensors"):
                raise PlanError(
                    f"save {save.value!r}: an output is a tensor, not a table; give it "
                    f"a .safetensors path, not {save.file_path!r}"
                )
            built.append(SaveFile(file_path=save.file_path, value=save.value))
            continue
        if sites is not None and save.value in sites:
            site = sites[save.value]
            one = spec.featurizers[save.value]
            built.append(
                SaveFile(
                    file_path=save.file_path,
                    value=save.value,
                    identity={
                        "model_key": spec.model.key,
                        "model_revision": spec.model.revision,
                        "model_dtype": spec.model.dtype,
                        "site": site,
                        "component": spec.sites[site].component,
                        "layer": str(spec.sites[site].layers[0] if spec.sites[site].layers else None),
                        "k": str(one.k),
                        "d": str(widths[save.value]),
                        "parametrization": one.parametrization,
                        "featurizer_dtype": "fp32",
                        "engine": "causalab-mini",
                    },
                )
            )
            continue
        kind = next(
            one.metrics[save.value].kind
            for one in spec.interventions.values()
            if save.value in one.metrics
        )
        built.append(
            SaveFile(
                file_path=save.file_path,
                value=save.value,
                example_ids=rows_module.example_ids(base_rows),
                unit=metrics_module.UNITS[kind][0],
                estimand_version=metrics_module.UNITS[kind][1],
            )
        )
    return tuple(built)


@dataclass(frozen=True)
class _Experiment:
    """The intervention, in the one shape the compiler works from.

    Both front ends reduce to this: the protocol's flat document
    (`document.py`) and the plan-shaped one (`spec.py`). Everything below it
    — the schedule, the taps, the metrics — is written once and shared, so
    the two formats cannot drift into producing different plans.
    """

    fields: dict[str, str]  # role -> which column of a row its text is
    reads: dict[str, Any]
    writes: dict[str, Any]
    models: dict[str, Any]
    metrics: dict[str, Any]

    @classmethod
    def of_document(cls, document: Document) -> "_Experiment":
        return cls(
            fields={role: one.field for role, one in document.roles.items()},
            reads=dict(document.reads),
            writes=dict(document.writes),
            models=dict(document.intervened_models),
            metrics=dict(document.metrics),
        )

    @classmethod
    def of_spec(cls, spec: Any, intervention: Any) -> "_Experiment":
        return cls(
            fields={role: one.field for role, one in spec.roles.items()},
            reads=dict(intervention.reads),
            writes={
                # a write's operand is a name whichever way it was written: a
                # read of this pass, or an output published before it
                name: _WriteSpec(
                    site=w.site, pos=w.pos, mechanism=w.mechanism,
                    operand=w.operand_name, featurizer=w.featurizer,
                )
                for name, w in intervention.writes.items()
            },
            models=dict(intervention.models),
            metrics=dict(intervention.metrics),
        )


@dataclass(frozen=True)
class _WriteSpec:
    site: str
    pos: int
    mechanism: str
    operand: str
    featurizer: str


def build_request(raw: dict[str, Any], data_root: str | Path, engine: Any) -> Plan:
    """Compile a document — either format, whatever number of experiments.

    A plain document compiles to one plan. A document with `{"sweep": […]}`
    wrappers is several **points**, and compiles to a root plan holding one
    child per point, named for the values it took — which is also the
    directory its results are written to. Nothing else in the project knows
    the difference: a point is an ordinary plan, and the engine that runs the
    root is walking the same tree it always walks.

    This is the one entry point. It tells the two formats apart by shape (a
    plan-shaped document has `steps`), lowers sweeps on the raw JSON — which
    neither format has to know about — and hands each point to its compiler.
    """
    compile_point = _compile_spec if "steps" in raw else _compile_document
    points = sweep.points(raw)
    if len(points) == 1 and not points[0][0]:
        return replace(compile_point(raw, data_root, engine), source=raw)
    return Plan(
        steps={
            label: replace(compile_point(point, data_root, engine), source=point)
            for label, point in points
        },
        source=raw,
    )


def _compile_document(raw: dict[str, Any], data_root: str | Path, engine: Any) -> Plan:
    return build(Document.from_json(raw), data_root, engine)


def _compile_spec(raw: dict[str, Any], data_root: str | Path, engine: Any) -> Plan:
    from .spec import Spec  # here, not at the top: spec.py is the front end and this is below it

    return build_spec(Spec.model_validate(raw), data_root, engine)


def build(document: Document, data_root: str | Path, engine: Any) -> Plan:
    """Compile a document into a plan. Takes the **engine**, because four
    things have to be decided against the loaded model here on the client: the
    tokenizer resolves prompts and answer columns, `num_layers` bounds the
    layer band, every site is resolved to an address, and every featurizer's
    width comes from the site it acts at. A document that cannot be compiled
    should fail here, not with an IndexError inside someone else's process."""
    tokenizer = engine.tokenizer
    for name, site in document.sites.items():
        if site.layer is not None and not 0 <= site.layer < engine.num_layers:
            raise PlanError(
                f"site {name!r}: layer {site.layer} is outside the model's "
                f"{engine.num_layers} layers"
            )
    # One address per site, resolved by the engine now: an interior's
    # operation is named by the loaded checkpoint's forward, so a document
    # that cannot be addressed is a load error here, on the client.
    addresses = {
        name: engine.locate(site.component, site.layer)
        for name, site in document.sites.items()
    }
    rows = {role: rows_module.load(data_root, spec.dataset) for role, spec in document.roles.items()}
    counts = {role: len(table) for role, table in rows.items()}
    if len(set(counts.values())) != 1:
        raise PlanError(f"roles must have the same row count, got {counts}")

    featurizers = tuple(
        _featurizer(name, document, addresses, engine) for name in document.featurizers
    )
    widths = {one.name: one.d for one in featurizers}

    # The steps, in the order they run. A step that has nothing to do is not
    # there at all: a document with no featurizers has no `featurizers` step,
    # rather than one holding an empty tuple.
    # Each save goes on the step that produces its value: a metric on the
    # scored pass, a fitted parameter on the step that publishes it. The
    # protocol's `save` is one flat list, so this is where the flat list
    # becomes a tree again.
    saves = [_save(entry, document, rows["base"], widths) for entry in document.saves]
    weight_names = {one.name for one in featurizers}
    metric_saves = tuple(save for save in saves if save.value not in weight_names)
    weight_saves = tuple(save for save in saves if save.value in weight_names)

    steps: dict[str, Step] = {}
    if featurizers:
        steps["featurizers"] = Featurizers(specs=featurizers)
    fit = _fit(document, data_root, rows, addresses, tokenizer)
    if fit is not None:
        steps["fit"] = fit
    steps["observe"] = replace(
        _pass(_Experiment.of_document(document), rows, addresses, tokenizer),
        saves=metric_saves,
    )
    if featurizers:
        steps["weights"] = Weights(
            names=tuple(one.name for one in featurizers), saves=weight_saves
        )
    return Plan(steps=steps)


def _pass(
    experiment: _Experiment,
    rows: dict[str, list[rows_module.Row]],
    addresses: dict[str, Address],
    tokenizer: Any,
) -> Observe:
    """One execution of the document's forwards over one set of rows. A
    training update, an eval pass and the scored run are all this step, over
    different rows."""
    batches = {
        role: encoding.encode(
            tokenizer,
            [rows_module.field_text(row, experiment.fields[role]) for row in table],
        )
        for role, table in rows.items()
    }
    forwards = tuple(
        _forward(name, role, experiment, batches[role], addresses)
        for name, role in _schedule(experiment)
    )
    base_rows = rows["base"]  # base is the schema of the pair: metrics read its columns
    metrics = tuple(
        MetricOp(
            name=name,
            kind=spec.kind,
            of=spec.of,
            ids=tuple(
                tuple(encoding.token_id(tokenizer, value, spec.token_form) for value in rows_module.column(base_rows, column))
                for column in spec.columns
            ),
        )
        for name, spec in experiment.metrics.items()
    )
    return Observe(forwards=forwards, metrics=metrics)


def _featurizer(
    name: str, document: Document, addresses: dict[str, Address], engine: Any
) -> FeaturizerOp:
    """One declared featurizer, with its width filled in from the model.

    `k` is the only width a document authors. `d` is the site's, and a `k` wider
    than the site it acts in is a load error here rather than a shape error
    inside someone else's process.
    """
    spec = document.featurizers[name]
    at = {read.site for read in document.reads.values() if read.featurizer == name}
    at |= {write.site for write in document.writes.values() if write.featurizer == name}
    (site,) = at  # the document refuses one name at two sites
    d = engine.width(addresses[site])
    if not 0 < spec.k <= d:
        raise PlanError(
            f"featurizer {name!r}: k={spec.k} is not a subspace of the {d}-wide "
            f"site {site!r}"
        )
    return FeaturizerOp(
        name=name,
        kind=spec.kind,
        k=spec.k,
        d=d,
        parametrization=spec.parametrization,
        # A subspace with no seed of its own takes the fit's, or 0 when there
        # is no fit at all — which is what makes an untrained subspace a
        # reproducible random rank-k basis. Authoring one is how a document
        # sweeps the draw.
        seed=(
            spec.seed
            if spec.seed is not None
            else (document.train.seed if document.train is not None else 0)
        ),
        trained=document.train is not None and name in document.train.params,
    )


def _save(
    entry: SaveSpec, document: Document, base_rows: list[rows_module.Row], widths: dict[str, int]
) -> SaveFile:
    if entry.site is not None:
        spec = document.featurizers[entry.value]
        return SaveFile(
            file_path=entry.file_path,
            value=entry.value,
            produced_by=document.digest,
            # "A rotation fitted against bf16 weights is not the same artifact as
            # one fitted against fp32 weights, and the stamp is what says so."
            identity={
                "produced_by": document.digest,
                "model_key": document.model.key,
                "model_revision": document.model.revision,
                "model_dtype": document.model.dtype,
                "site": entry.site,
                "component": document.sites[entry.site].component,
                "layer": str(document.sites[entry.site].layer),
                "k": str(spec.k),
                "d": str(widths[entry.value]),
                "parametrization": spec.parametrization,
                "featurizer_dtype": "fp32",
                "trained_on": document.roles["base"].dataset,
                "trained_on_digest": rows_module.digest(base_rows),
                "engine": "causalab-mini",
            },
        )
    kind = document.metrics[entry.value].kind
    return SaveFile(
        file_path=entry.file_path,
        value=entry.value,
        example_ids=rows_module.example_ids(base_rows),
        unit=metrics_module.UNITS[kind][0],
        estimand_version=metrics_module.UNITS[kind][1],
        produced_by=document.digest,
    )


def _fit(
    document: Document,
    data_root: str | Path,
    rows: dict[str, list[rows_module.Row]],
    addresses: dict[str, Address],
    tokenizer: Any,
) -> Fit | None:
    spec = document.train
    if spec is None:
        return None
    # The eval split is a dataset ref exactly like a `data` entry's, and each
    # role reads its own field off it.
    evaluation = {role: rows_module.load(data_root, spec.eval_split) for role in rows}
    if spec.eval_split != document.roles["base"].dataset:
        # Two different refs must be endpoint-disjoint. The same ref for both is
        # the visible train-equals-test ablation, and is allowed.
        fitted = {json.dumps(row, sort_keys=True) for row in rows["base"]}
        shared = [row for row in evaluation["base"] if json.dumps(row, sort_keys=True) in fitted]
        if shared:
            raise PlanError(
                f"method.train.eval.split {spec.eval_split!r} shares {len(shared)} row(s) "
                f"with the fitted rows {document.roles['base'].dataset!r}; the two must "
                "be endpoint-disjoint"
            )

    count = len(rows["base"])
    order = random.Random(spec.seed)
    epochs = tuple(
        tuple(
            _pass(
                _Experiment.of_document(document),
                _take(rows, draw[start : start + spec.pairs]),
                addresses,
                tokenizer,
            )
            for start in range(0, count, spec.pairs)
        )
        for draw in (order.sample(range(count), count) for _ in range(spec.epochs))
    )
    return Fit(
        epochs=epochs,
        evaluation=_pass(_Experiment.of_document(document), evaluation, addresses, tokenizer),
        objective=spec.objective,
        params=spec.params,
        lr=spec.lr,
        weight_decay=spec.weight_decay,
        eval_metrics=spec.eval_metrics,
        early_stop=spec.early_stop,
        patience=spec.patience,
        mode=spec.mode,
    )


def _take(rows: dict[str, list[rows_module.Row]], picked: list[int]) -> dict[str, list[rows_module.Row]]:
    """One minibatch. Every role is indexed the same way, because rows are
    paired by index and a shuffle that broke the pairing would silently fit a
    rotation against mismatched counterfactuals."""
    return {role: [table[index] for index in picked] for role, table in rows.items()}


def _schedule(experiment: _Experiment) -> list[tuple[str, str]]:
    """The forwards, as (model, input) pairs, in execution order.

    Cross-model data flow has one channel: a read in model A may be the operand
    of a write in force in model B, so B runs after A. The graph must be
    acyclic; it *is* the schedule.
    """
    units: dict[tuple[str, str], set[str]] = {}
    read_home: dict[str, tuple[str, str]] = {}
    for name, spec in experiment.reads.items():
        unit = (spec.model, spec.input)
        units.setdefault(unit, set())
        read_home[name] = unit
    for name, spec in experiment.models.items():
        units.setdefault((name, spec.input), set())
        for write in spec.writes:
            units[(name, spec.input)].add(experiment.writes[write].operand)

    ordered: list[tuple[str, str]] = []
    remaining = dict(units)
    while remaining:
        ready = [
            unit
            for unit, operands in remaining.items()
            # an operand that is not a read of this pass was published by an
            # earlier step; it is already there and orders nothing
            if all(read_home[operand] in ordered for operand in operands if operand in read_home)
        ]
        if not ready:
            raise PlanError(f"the model/read graph has a cycle: {sorted(remaining)}")
        ready.sort()
        ordered.extend(ready)
        for unit in ready:
            del remaining[unit]
    return ordered


def _forward(
    name: str,
    role: str,
    experiment: _Experiment,
    batch: encoding.Batch,
    addresses: dict[str, Address],
) -> Forward:
    """One model pass: its taps, grouped by address and put in forward order."""
    writes: dict[Address, list[WriteOp]] = {}
    if name in experiment.models:
        for write_name in experiment.models[name].writes:
            spec = experiment.writes[write_name]
            writes.setdefault(addresses[spec.site], []).append(
                WriteOp(
                    name=write_name,
                    positions=encoding.positions(batch, spec.pos),
                    operand=spec.operand,
                    mechanism=spec.mechanism,
                    featurizer=spec.featurizer,
                )
            )
    reads: dict[Address, list[ReadOp]] = {}
    for read_name, spec in experiment.reads.items():
        if (spec.model, spec.input) != (name, role):
            continue
        reads.setdefault(addresses[spec.site], []).append(
            ReadOp(
                name=read_name,
                positions=encoding.positions(batch, spec.pos),
                featurizer=spec.featurizer,
            )
        )

    taps = []
    for tap_address in sorted(set(writes) | set(reads), key=lambda one: one.key):
        taps.append(
            Tap(
                address=tap_address,
                # A read in model M sees M's writes applied, upstream and at the
                # same address — so at one address the writes go first.
                writes=tuple(writes.get(tap_address, ())),
                reads=tuple(reads.get(tap_address, ())),
            )
        )
    return Forward(
        name=name,
        input=role,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        taps=tuple(taps),
    )

