"""The frozen plan: ops in forward order, and the compiler that produces it.

A plan holds strings, integers and nothing else — no envoys, no tensors, no
tokenizer, no document. It pickles with plain `pickle`, which is the test that
keeps it honest. Everything that needed a decision (which rows, which tokens,
which position, which module path, which order) was decided here, on the
client, before the session opened.

The shape, top down:

    Plan
      forwards: one per (model, input), already in execution order
        Forward(name, input, input_ids, attention_mask, taps)
          taps: one per address, in forward order
            Tap(path, side, writes, reads)     writes run before reads at the
              WriteOp(name, positions, operand, mechanism, featurizer)  same
              ReadOp(name, positions)                                   address
      metrics: MetricOp(name, kind, of, ids)
      saves:   SaveFile(file_path, value, example_ids, unit, ...)
"""

from __future__ import annotations

from dataclasses import dataclass

from . import address, data, encoding, metrics as metrics_module


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class ReadOp:
    name: str
    positions: tuple[int, ...]


@dataclass(frozen=True)
class WriteOp:
    name: str
    positions: tuple[int, ...]
    operand: str  # the name of a read, produced by an earlier forward
    mechanism: str
    featurizer: str


@dataclass(frozen=True)
class Tap:
    path: str
    side: str
    writes: tuple[WriteOp, ...]
    reads: tuple[ReadOp, ...]


@dataclass(frozen=True)
class Forward:
    name: str  # "original" or an intervened model's name
    input: str  # the data role its rows come from
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    taps: tuple[Tap, ...]


@dataclass(frozen=True)
class MetricOp:
    name: str
    kind: str
    of: str  # the read it binds to
    ids: tuple[tuple[int, ...], ...]  # one vocabulary id per row, per operand


@dataclass(frozen=True)
class SaveFile:
    file_path: str
    value: str
    example_ids: tuple[str, ...]
    unit: str
    estimand_version: str
    produced_by: str


@dataclass(frozen=True)
class Plan:
    forwards: tuple[Forward, ...]
    metrics: tuple[MetricOp, ...]
    saves: tuple[SaveFile, ...]


def build(document, data_root, model) -> Plan:
    """Compile a document into a plan. Takes the loaded model because two
    things have to be decided against it on the client: the tokenizer resolves
    prompts and answer columns, and `num_layers` bounds the layer band — a site
    naming a layer the model does not have should be a load error here, not an
    IndexError inside someone else's process."""
    tokenizer = model.tokenizer
    for name, site in document.sites.items():
        if site.layer is not None and not 0 <= site.layer < model.num_layers:
            raise PlanError(
                f"site {name!r}: layer {site.layer} is outside the model's "
                f"{model.num_layers} layers"
            )
    rows = {role: data.load_rows(data_root, spec.dataset) for role, spec in document.roles.items()}
    counts = {role: len(table) for role, table in rows.items()}
    if len(set(counts.values())) != 1:
        raise PlanError(f"roles must have the same row count, got {counts}")
    batches = {
        role: encoding.encode(tokenizer, [data.field_text(row, document.roles[role].field) for row in table])
        for role, table in rows.items()
    }

    forwards = tuple(
        _forward(name, role, document, batches[role]) for name, role in _schedule(document)
    )
    base_rows = rows["base"]  # base is the schema of the pair: metrics read its columns
    metrics = tuple(
        MetricOp(
            name=name,
            kind=spec.kind,
            of=spec.of,
            ids=tuple(
                tuple(encoding.token_id(tokenizer, value, spec.token_form) for value in data.column(base_rows, column))
                for column in spec.columns
            ),
        )
        for name, spec in document.metrics.items()
    )
    ids = data.example_ids(base_rows)
    saves = tuple(
        SaveFile(
            file_path=entry.file_path,
            value=entry.value,
            example_ids=ids,
            unit=metrics_module.UNITS[document.metrics[entry.value].kind][0],
            estimand_version=metrics_module.UNITS[document.metrics[entry.value].kind][1],
            produced_by=document.digest,
        )
        for entry in document.saves
    )
    return Plan(forwards=forwards, metrics=metrics, saves=saves)


def _schedule(document) -> list[tuple[str, str]]:
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


def _forward(name, role, document, batch) -> Forward:
    """One model pass: its taps, grouped by address and put in forward order."""
    writes = {}
    if name in document.intervened_models:
        for write_name in document.intervened_models[name].writes:
            spec = document.writes[write_name]
            writes.setdefault(_key(document, spec.site), []).append(
                WriteOp(
                    name=write_name,
                    positions=encoding.positions(batch, spec.pos),
                    operand=spec.operand,
                    mechanism=spec.mechanism,
                    # The identity featurizer is the whole featurizer table for
                    # now; see ops.FEATURIZERS.
                    featurizer="identity",
                )
            )
    reads = {}
    for read_name, spec in document.reads.items():
        if (spec.model, spec.input) != (name, role):
            continue
        reads.setdefault(_key(document, spec.site), []).append(
            ReadOp(name=read_name, positions=encoding.positions(batch, spec.pos))
        )

    taps = []
    for key in sorted(set(writes) | set(reads)):
        _order, path, side = key
        taps.append(
            Tap(
                path=path,
                side=side,
                # A read in model M sees M's writes applied, upstream and at the
                # same address — so at one address the writes go first.
                writes=tuple(writes.get(key, ())),
                reads=tuple(reads.get(key, ())),
            )
        )
    return Forward(
        name=name,
        input=role,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        taps=tuple(taps),
    )


def _key(document, site_name):
    """(forward-order key, path, side) for a site — the only call into the one
    file that knows about models."""
    site = document.sites[site_name]
    path, side = address.locate(site.component, site.layer)
    return (address.order(site.component, site.layer), path, side)
