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
    Plan,
    PlanError,
    ReadOp,
    SaveFile,
    Step,
    Tap,
    Weights,
    WriteOp,
)


def build_request(raw: dict[str, Any], data_root: str | Path, engine: Any) -> Plan:
    """Compile a document, whatever number of experiments it is.

    A plain document compiles to one plan. A document with a `{"sweep": […]}`
    wrapper is several **points**, and compiles to a root plan holding one
    child per point, named for the value it took — which is also the directory
    its results are written to. Nothing else in the project knows the
    difference: a point is an ordinary plan, and the engine that runs the root
    is walking the same tree it always walks.
    """
    points = sweep.points(raw)
    if len(points) == 1 and not points[0][0]:
        return build(Document.from_json(raw), data_root, engine)
    return Plan(
        steps={
            label: build(Document.from_json(point), data_root, engine)
            for label, point in points
        }
    )


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
    steps: dict[str, Step] = {}
    if featurizers:
        steps["featurizers"] = Featurizers(specs=featurizers)
    fit = _fit(document, data_root, rows, addresses, tokenizer)
    if fit is not None:
        steps["fit"] = fit
    steps["observe"] = _pass(document, rows, addresses, tokenizer)
    if featurizers:
        steps["weights"] = Weights(names=tuple(one.name for one in featurizers))
    return Plan(
        steps=steps,
        saves=tuple(
            _save(entry, document, rows["base"], widths) for entry in document.saves
        ),
    )


def _pass(
    document: Document,
    rows: dict[str, list[rows_module.Row]],
    addresses: dict[str, Address],
    tokenizer: Any,
) -> Observe:
    """One execution of the document's forwards over one set of rows. A
    training update, an eval pass and the scored run are all this step, over
    different rows."""
    batches = {
        role: encoding.encode(tokenizer, [rows_module.field_text(row, document.roles[role].field) for row in table])
        for role, table in rows.items()
    }
    forwards = tuple(
        _forward(name, role, document, batches[role], addresses)
        for name, role in _schedule(document)
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
        for name, spec in document.metrics.items()
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
        # A subspace with no seed of its own takes the fit's, or 0 when there is
        # no fit at all — which is what makes an untrained subspace a
        # reproducible random rank-k basis.
        seed=document.train.seed if document.train is not None else 0,
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
            _pass(document, _take(rows, draw[start : start + spec.pairs]), addresses, tokenizer)
            for start in range(0, count, spec.pairs)
        )
        for draw in (order.sample(range(count), count) for _ in range(spec.epochs))
    )
    return Fit(
        epochs=epochs,
        evaluation=_pass(document, evaluation, addresses, tokenizer),
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


def _schedule(document: Document) -> list[tuple[str, str]]:
    """The forwards, as (model, input) pairs, in execution order.

    Cross-model data flow has one channel: a read in model A may be the operand
    of a write in force in model B, so B runs after A. The graph must be
    acyclic; it *is* the schedule.
    """
    units: dict[tuple[str, str], set[str]] = {}
    read_home: dict[str, tuple[str, str]] = {}
    for name, spec in document.reads.items():
        unit = (spec.model, spec.input)
        units.setdefault(unit, set())
        read_home[name] = unit
    for name, spec in document.intervened_models.items():
        units.setdefault((name, spec.input), set())
        for write in spec.writes:
            units[(name, spec.input)].add(document.writes[write].operand)

    ordered: list[tuple[str, str]] = []
    remaining = dict(units)
    while remaining:
        ready = [
            unit
            for unit, operands in remaining.items()
            if all(read_home[operand] in ordered for operand in operands)
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
    document: Document,
    batch: encoding.Batch,
    addresses: dict[str, Address],
) -> Forward:
    """One model pass: its taps, grouped by address and put in forward order."""
    writes: dict[Address, list[WriteOp]] = {}
    if name in document.intervened_models:
        for write_name in document.intervened_models[name].writes:
            spec = document.writes[write_name]
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
    for read_name, spec in document.reads.items():
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

