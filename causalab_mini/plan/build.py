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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from ..address import Address
from ..data import rows as rows_module, tokens
from ..ops import featurizer as featurizer_module
from ..ops import intervene as intervene_module
from ..ops import metrics as metrics_module
from . import sweep
from ..shapes import Selection, TokenRows, Where
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
    # site -> (groups, take): the feature half of every selection made there
    features: dict[str, tuple[int, tuple[int, ...] | None]] = {}
    for name, site in spec.sites.items():
        if site.units is not None:
            count, take, what = engine.width(addresses[name]), site.units, "unit"
        elif addresses[name].heads_kind is not None:
            count, take, what = engine.heads(addresses[name]), site.heads, "head"
        else:
            continue
        if take is not None and max(take) >= count:
            raise PlanError(f"site {name!r}: {what} {max(take)} of a {count}-{what} tensor")
        features[name] = (count, None if take is None else tuple(take))
    stacking = {
        read.site
        for one in spec.interventions.values()
        for read in one.reads.values()
        if _stacks(read.pos)
    }
    # `widths` below is the featurizers'; this is the sites' own, and only
    # for the ones a continuation read buffers at
    stack_widths = {name: engine.width(addresses[name]) for name in sorted(stacking)}
    fit_steps = [one for one in spec.steps.values() if type(one).__name__ == "Fit"]
    featurizers = tuple(
        _spec_featurizer(name, one, spec, addresses, engine, fit_steps)
        for name, one in spec.featurizers.items()
    )
    widths = {one.name: one.d for one in featurizers}
    spaces = {one.name: one.k for one in featurizers}
    for label, one in spec.interventions.items():
        for write_name, write in one.writes.items():
            if write.features is None:
                continue
            k = spaces.get(write.featurizer, engine.width(addresses[write.site]) if write.featurizer == "identity" else 0)
            if not write.features or len(set(write.features)) != len(write.features) or not 0 <= min(write.features) <= max(write.features) < k:
                raise PlanError(
                    f"interventions.{label}: write {write_name!r}: features {write.features} of a "
                    f"{k}-dimensional feature space; they are distinct indices below {k}"
                )

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
    #: An unreduced output is (rows, w, width), and a write that swaps it in
    #: needs the same rows and the same window. The compiler knows all of it;
    #: the block would only find out from a shape error.
    output_rows: dict[str, int | None] = {}
    output_widths: dict[str, tuple[int | None, Any]] = {}  # the read's width, how reduced
    for name, step in spec.steps.items():
        kind = type(step).__name__
        experiment = (
            replace(_Experiment.of_spec(spec, spec.intervention_of(step)), features=features, widths=stack_widths)
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
                    per_row, reduced = output_widths[write.operand.ref]  # per_row: None is ragged
                    if reduced == "pca":
                        raise PlanError(
                            f"step {name!r}: write {write_name!r} names {write.operand.ref!r}, a "
                            "pca basis, as its operand; a basis is loaded as a featurizer, "
                            "not written at a site"
                        )
                    ragged_source = per_row is None
                    want = write.pos.width  # None: the write is ragged
                    # What an output may land in. A mean over a ragged read is
                    # one vector and broadcasts anywhere; a mean over a
                    # rectangle keeps its window and must match; an unreduced
                    # rectangle must match; an unreduced ragged read must be
                    # reduced first, or read in the same pass.
                    if reduced and ragged_source:
                        fits = True
                    elif reduced:
                        fits = (want == per_row) or (want is None and per_row == 1)
                    elif not ragged_source:
                        fits = want == per_row
                    else:
                        fits = False
                    if not fits:
                        raise PlanError(
                            f"step {name!r}: write {write_name!r} covers "
                            f"{want if want is not None else 'a varying number of'} "
                            f"position(s) but {write.operand.ref!r} was read over "
                            f"{[per_row] if per_row is not None else 'a varying number of'}"
                            f"{'' if reduced else ', unreduced'}; "
                            "the windows must match, or reduce the output"
                        )
        if kind == "Observe":
            assert experiment is not None
            rows = table(step.rows)
            outputs = tuple(
                OutputOp(name=out_name, read=out.read, reduce=out.reduce, k=out.k)
                for out_name, out in step.outputs.items()
            )
            observe = _pass(experiment, rows, addresses, engine.tokenizer)
            read_widths = {
                read.name: read.at.where.width if read.at.where is not None else None
                for forward in observe.forwards for tap in forward.taps for read in tap.reads
            }
            for out in outputs:
                width = read_widths[out.read]
                if out.reduce == "pca" and width is not None:
                    # k directions need more than k vectors: the rows are
                    # centered first, which costs one rank. Known here, from
                    # the form and the row count, before any forward — a
                    # text-anchored read has neither until it runs.
                    vectors = width * len(rows["base"])
                    assert out.k is not None
                    if out.k > vectors - 1:
                        raise PlanError(
                            f"step {name!r}: output {out.name!r} asks for {out.k} principal "
                            f"directions of {vectors} vector(s); centered, they span at most "
                            f"{max(vectors - 1, 0)}. Harvest more rows, or more positions per row"
                        )
                output_rows[out.name] = None if out.reduce == "mean" else len(rows["base"])
                output_widths[out.name] = (
                    width,
                    out.reduce if out.reduce == "pca" else out.reduce == "mean",
                )
            steps[name] = replace(
                observe,
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
        # the watched metric, then the fraction each trained gate keeps
        eval_metrics=(
            step.early_stop.metric,
            *(f"{p}.mask" for p in step.params if spec.featurizers[p].kind == "gate"),
        ),
        early_stop=step.early_stop.metric,
        patience=step.early_stop.patience,
        mode=step.early_stop.mode,
        anneal=tuple((gate, one.start, one.end) for gate, one in step.anneal.items()),
        saves=_spec_saves(step.saves, spec, [], widths),
    )


def _spec_featurizer(
    name: str, one: Any, spec: Any, addresses: dict[str, Address], engine: Any, fits: list[Any]
) -> FeaturizerOp:
    site = _sites_of(spec)[name]
    d = engine.width(addresses[site])
    if spec.sites[site].heads is not None:
        d = d // engine.heads(addresses[site]) * len(spec.sites[site].heads)
    if spec.sites[site].units is not None:
        d = len(spec.sites[site].units)
    # a gate's features are the site's units, so its k is d; an encoder's
    # comes from its bundle and may exceed d — a dictionary is overcomplete
    k = d if one.k is None else one.k
    if one.kind in ("subspace", "pca") and not 0 < k <= d:
        raise PlanError(
            f"featurizer {name!r}: k={one.k} is not a subspace of the {d}-wide site {site!r}"
        )
    trained = any(name in fit.params for fit in fits)
    seed = one.seed
    if seed is None:
        seed = next((fit.seed for fit in fits if name in fit.params), 0)
    weight = None
    if one.file_path is not None:
        weight, k = _load_featurizer(name, one, spec, site, d)
    return FeaturizerOp(
        name=name,
        kind=one.kind,
        k=k,
        d=d,
        parametrization=one.parametrization,
        seed=seed,
        trained=trained,
        weight=weight,
        source=one.file_path or "",
    )


def _identity(spec: Any, name: str, site: str, d: int) -> dict[str, str]:
    """What a saved featurizer is *of*: the stamp a bundle carries, and the
    expectation a load is checked against. One function, so the two cannot
    disagree about which keys matter."""
    one = spec.featurizers[name]
    return {
        "model_key": spec.model.key,
        "model_revision": spec.model.revision,
        "model_dtype": spec.model.dtype,
        "site": site,
        "component": spec.sites[site].component,
        "layer": str(spec.sites[site].layers[0] if spec.sites[site].layers else None),
        "kind": one.kind,
        "k": str(one.k),
        "d": str(d),
        "parametrization": one.parametrization,
        "featurizer_dtype": "fp32",
        "engine": "causalab-mini",
    }


#: Stamp keys that may differ between the run that wrote a bundle and the
#: one that loads it without making the bundle a different thing: the site's
#: *name* is the document's, not the tensor's (component and layer are
#: checked), and the parametrization of a pca basis is not a thing.
_FREE = {"site", "engine"}


def _load_featurizer(name: str, one: Any, spec: Any, site: str, d: int) -> tuple[bytes, int]:
    """A saved bundle, checked against this document, as `(bytes, k)`.

    A rotation is a grid of numbers; nothing in the numbers says which model
    or which layer it came from. The header does, and this is where it is
    read. A mismatch on any key is refused naming the key — a rotation
    fitted at layer 13 loaded at layer 14 would run, and would be nonsense.
    A bundle from elsewhere (a published SAE) has no stamp of ours, and then
    the shapes are all there is to check: its width must be this site's.
    """
    import safetensors.torch
    from safetensors import safe_open

    path = Path(one.file_path)
    if not path.exists():
        raise PlanError(f"featurizer {name!r}: no bundle at {one.file_path!r}")
    required, optional = featurizer_module.TENSORS[one.kind]
    with safe_open(str(path), "pt") as bundle:
        stamp = dict(bundle.metadata() or {})
        keys = set(bundle.keys())
        if not set(required) <= keys or not keys <= set(required) | set(optional):
            raise PlanError(
                f"featurizer {name!r}: {one.file_path!r} holds {sorted(keys)}; a {one.kind} "
                f"bundle holds {list(required)}" + (f" and optionally {list(optional)}" if optional else "")
            )
        tensors = {key: bundle.get_tensor(key).to(torch.float32).contiguous() for key in sorted(keys)}
    expected = _identity(spec, name, site, d)
    checked = {key for key in expected if key in stamp} - _FREE
    if one.kind != "subspace":
        checked.discard("parametrization")
    if one.k is None:
        checked.discard("k")
    mismatched = {key: (stamp[key], expected[key]) for key in sorted(checked) if stamp[key] != expected[key]}
    if mismatched:
        detail = "; ".join(f"{key}: bundle says {got!r}, document says {want!r}" for key, (got, want) in mismatched.items())
        raise PlanError(f"featurizer {name!r}: {one.file_path!r} is not this featurizer — {detail}")

    k = d if one.kind == "gate" else (one.k or int(tensors[required[0]].shape[-1]))
    shapes = {"weight": (d,) if one.kind == "gate" else (d, k),
              "W_enc": (d, k), "W_dec": (k, d), "b_enc": (k,), "b_dec": (d,)}
    for key, tensor in tensors.items():
        if tuple(tensor.shape) != shapes[key]:
            raise PlanError(
                f"featurizer {name!r}: {one.file_path!r} has {key} {tuple(tensor.shape)}, not the "
                f"{shapes[key]} this document and its {d}-wide site declare"
            )
    return safetensors.torch.save(tensors), k


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
        if save.value in outputs or save.value.endswith(".generated"):
            # an output, or a decoding forward's generated ids: a tensor
            if not save.file_path.endswith(".safetensors"):
                raise PlanError(
                    f"save {save.value!r}: a tensor, not a table; give it a "
                    f".safetensors path, not {save.file_path!r}"
                )
            built.append(SaveFile(file_path=save.file_path, value=save.value))
            continue
        if sites is not None and save.value in sites:
            built.append(
                SaveFile(
                    file_path=save.file_path,
                    value=save.value,
                    produced_by=spec.digest,
                    identity={
                        "produced_by": spec.digest,
                        **_identity(spec, save.value, sites[save.value], widths[save.value]),
                    },
                )
            )
            continue
        metric = next(
            one.metrics[save.value]
            for one in spec.interventions.values()
            if save.value in one.metrics
        )
        built.append(
            SaveFile(
                file_path=save.file_path,
                value=save.value,
                example_ids=rows_module.example_ids(base_rows),
                eligible=_eligible(base_rows, metric),
                of=metric.of,
                unit=metrics_module.UNITS[metric.kind][0],
                estimand_version=metrics_module.UNITS[metric.kind][1],
                produced_by=spec.digest,
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
    decode: int = 0
    #: site -> `(groups, take)`: which part of the feature axis the site is.
    features: dict[str, Any] = field(default_factory=dict)
    #: site -> how wide the tensor there is, for the sites a continuation
    #: read stacks at. Only those: `engine.width` is a question about the
    #: checkpoint and there is no reason to ask it where nothing buffers.
    widths: dict[str, int] = field(default_factory=dict)

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
            writes=dict(intervention.writes),
            models=dict(intervention.models),
            metrics=dict(intervention.metrics),
            decode=intervention.decode,
        )


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
    batches = {role: _batch(tokenizer, role, experiment, table) for role, table in rows.items()}
    forwards = tuple(
        _forward(name, role, experiment, batches[role], addresses, rows[role])
        for name, role in _schedule(experiment)
    )
    _check_patterns(forwards)
    base_rows = rows["base"]  # base is the schema of the pair: metrics read its columns
    metrics = tuple(
        _metric(name, spec, base_rows, tokenizer) for name, spec in experiment.metrics.items()
    )
    return Observe(forwards=forwards, metrics=metrics)


def _batch(
    tokenizer: Any, role: str, experiment: _Experiment, table: list[rows_module.Row]
) -> tuple[TokenRows, TokenRows, str, tuple[dict[str, tuple[int, int]], ...]]:
    """One role's padded batch, and the runs the frame located in it.

    A role's field may hold a string or a conversation, and which it is, is
    the row's to say — so a chat prompt costs the document nothing and the
    dataset one column. What the template rendered has the family's opening
    token in it already, which is why it is encoded without another.
    """
    field = experiment.fields[role]
    values = [rows_module.field_value(row, field) for row in table]
    texts = [
        tokens.rendered(tokenizer, value, f"role {role!r} field {field!r}, row {row}")
        for row, value in enumerate(values)
    ]
    chat = any(not isinstance(value, str) for value in values)
    ids, mask, sample = tokens.encode(tokenizer, texts, add_special=not chat)
    spans = tokens.turns(tokenizer, ids, mask, values) if chat else ()
    return ids, mask, sample, spans


def _metric(name: str, spec: Any, base_rows: list[rows_module.Row], tokenizer: Any) -> MetricOp:
    """One metric over one batch of rows, with the rows it cannot be computed
    for taken out here, where the data is."""
    keep = rows_module.eligible(base_rows, tuple(spec.columns))
    if not any(keep):
        raise PlanError(
            f"metric {name!r}: none of these {len(base_rows)} row(s) has a value in "
            f"{list(spec.columns)}; a metric of nothing has no mean"
        )
    return MetricOp(
        name=name,
        kind=spec.kind,
        of=spec.of,
        ids=_ids(spec, base_rows, keep, tokenizer),
        rows=None if all(keep) else tuple(index for index, one in enumerate(keep) if one),
    )


def _eligible(base_rows: list[rows_module.Row], metric: Any) -> tuple[bool, ...]:
    """A table's eligibility column; empty when every row is in, which is
    what a plan compiled before this existed says too. (A fit's own saves
    have no rows, and so nothing to be eligible.)"""
    if not base_rows:
        return ()
    keep = rows_module.eligible(base_rows, tuple(metric.columns))
    return () if all(keep) else keep


def _ids(spec: Any, base_rows: list[rows_module.Row], keep: tuple[bool, ...], tokenizer: Any) -> tuple[Any, ...]:
    return tuple(
        tuple(
            tokens.token_id(tokenizer, value, spec.token_form)
            for value, kept in zip(rows_module.column(base_rows, column), keep)
            if kept and value is not None
        )
        for column in spec.columns
    )


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
        eligible=_eligible(base_rows, document.metrics[entry.value]),
        of=document.metrics[entry.value].of,
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


def _check_patterns(forwards: tuple[Forward, ...]) -> None:
    """The one layout question that is about masks rather than positions, and
    is therefore still the client's: an attention pattern swapped in from
    another prompt has to line up key for key.

    The two refusals that used to sit beside this — a write with nothing to
    write on a row, and a ragged write whose operand is a different width —
    are about *where* a position lands, so they moved to where positions are
    resolved (`engine/steps.py`).
    """
    read_in = {read.name: forward for forward in forwards for tap in forward.taps for read in tap.reads}
    for forward in forwards:
        for tap in forward.taps:
            for write in tap.writes:
                if tap.address.key_axis and isinstance(write.operand, str):
                    _check_keys(write, forward, read_in.get(write.operand))


def _check_keys(write: Any, forward: Forward, source: Forward | None) -> None:
    """A pattern swapped in from another prompt must line up key for key.

    The last axis of `attention_probs` is the keys of the padded batch: key
    `j` of the operand lands on key `j` here. That is only the same *token*
    if the two prompts are laid out identically — the same padded width and,
    row by row, the same number of real tokens (the rows are left-padded, so
    equal counts means equal masks). Otherwise attention mass meant for a
    token would land on a pad, or on a different word, with every shape
    correct. Both masks are in hand before any forward, so this is refused
    here rather than discovered — or not discovered — later.

    The arithmetic is `data/tokens.same_layout`; the refusal is this
    document's, and stays here.
    """
    if source is None:
        raise PlanError(
            f"write {write.name!r}: an attention pattern from an earlier step cannot be checked "
            "against these prompts' layout; swap one read in the same pass"
        )
    rows = tokens.same_layout(source.attention_mask, forward.attention_mask)
    if rows is None:
        return
    ours = [sum(row) for row in forward.attention_mask]
    theirs = [sum(row) for row in source.attention_mask]
    raise PlanError(
        f"write {write.name!r} swaps in the attention pattern {write.operand!r}, read from "
        f"prompts laid out differently: " + (
            f"rows {rows} have {[theirs[r] for r in rows]} tokens there and {[ours[r] for r in rows]} here"
            if rows else f"padded to {len(source.attention_mask[0])} there and {len(forward.attention_mask[0])} here"
        ) + ". A pattern is over keys, so the two prompts must tokenize to the same length, row by row"
    )


def _take(rows: dict[str, list[rows_module.Row]], picked: list[int]) -> dict[str, list[rows_module.Row]]:
    """One minibatch. Every role is indexed the same way, because rows are
    paired by index and a shuffle that broke the pairing would silently fit a
    rotation against mismatched counterfactuals."""
    return {role: [table[index] for index in picked] for role, table in rows.items()}


def _operand(write: Any) -> str | float | None:
    """A write's operand as the compiler carries it: a name for a read of
    this pass or for an output published before it, the number itself for a
    literal. The plan-shaped format spells a reference as an object and the
    protocol's as a bare name, and this is the one place that differs."""
    return getattr(write, "operand_name", None) or write.operand


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
            operand = _operand(experiment.writes[write])
            if isinstance(operand, str):  # a literal or nothing orders no forward
                units[(name, spec.input)].add(operand)

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


def _selection(pos: Where, anchors: tuple[str, ...], features: Any) -> Selection:
    """Where an op is: the spec, the per-row text it anchors to, and which
    part of the feature axis. Whether it gathers flat is decided *by the
    form*, over every row of the pass, so a window of those rows cannot
    decide differently.

    A continuation-frame tap never gathers flat: a decode step processes one
    position whatever the spec names, and the cut over the steps happens
    afterwards, in `engine/steps.py`.
    """
    groups, take = features or (None, None)
    return Selection(
        groups=groups,
        take=take,
        flat=pos.frame == "prompt" and pos.ragged,
        where=pos,
        anchors=anchors,
    )


#: How much a continuation-frame read may buffer before it is refused, in
#: bytes. A read that cannot say which step it wants until the decode has
#: finished keeps every step: `rows x decode x width` numbers. That is
#: nothing at `decode = 8` and a gigabyte at `decode = 256` over a wide
#: site, so there is a line — and all three numbers are known here, before
#: a model is loaded, so it is drawn here rather than after the memory has
#: been held.
STACK_LIMIT = 256 * 1024 * 1024


def _stacks(pos: Where) -> bool:
    """Whether a read has to see the whole continuation before it can say
    which of it it wants. `{"index": 2}` is step 2 and the tap fires there;
    every other cut of the continuation — the last real token, the stop
    token, where the model said the row's answer — is of text that does not
    exist until the decode has run, so the read fires at every step and is
    selected out of the stack afterwards."""
    return pos.frame == "generated" and not (pos.index is not None and pos.index >= 0)


def _fits(name: str, site: str, experiment: _Experiment, rows: int) -> None:
    """What a stacked read will hold, before anything holds it."""
    groups, take = experiment.features.get(site) or (None, None)
    wide = experiment.widths.get(site, 0)
    if take is not None and groups:
        wide = len(take) * (wide // groups)
    held = experiment.decode * rows * wide * 4
    if held > STACK_LIMIT:
        raise PlanError(
            f"read {name!r} keeps every decode step to cut against the continuation: "
            f"{experiment.decode} steps x {rows} rows x {wide} wide is {held / 2**20:.0f} MiB, "
            f"over the {STACK_LIMIT / 2**20:.0f} MiB a read may hold. The count is this pass's "
            "rows, whatever --batch-size the run uses: name the step ({'frame': 'generated', "
            "'index': k}), decode fewer tokens, or score fewer rows in one pass"
        )


def _forward(
    name: str,
    role: str,
    experiment: _Experiment,
    batch: tuple[TokenRows, TokenRows, str, tuple[dict[str, tuple[int, int]], ...]],
    addresses: dict[str, Address],
    rows: list[rows_module.Row],
) -> Forward:
    """One model pass: its taps, grouped by address and put in forward order.

    Nothing here resolves a position. What it does compile is the *anchor*: a
    text-anchored spec names a variable, and which text that is on this row
    is a fact about the dataset, which only the client has.
    """

    def anchors(pos: Where) -> tuple[str, ...]:
        if pos.scope is None or pos.scope.variable is None:
            return ()
        field = experiment.fields[role]
        return tuple(rows_module.variable_text(row, field, pos.scope.variable) for row in rows)

    # a tap is one place: an address, and — when the forward decodes — a step
    writes: dict[tuple[Address, Any], list[WriteOp]] = {}
    if name in experiment.models:
        for write_name in experiment.models[name].writes:
            spec = experiment.writes[write_name]
            # which decode step the tap acts at: none in the prompt frame,
            # every one of them for a steering write, else the step it names
            step = None if spec.pos.frame == "prompt" else ("all" if spec.pos.all else spec.pos.index)
            writes.setdefault((addresses[spec.site], step), []).append(
                WriteOp(
                    name=write_name,
                    at=_selection(spec.pos, anchors(spec.pos), experiment.features.get(spec.site)),
                    operand=_operand(spec),
                    mechanism=spec.mechanism,
                    featurizer=spec.featurizer,
                    params=dict(getattr(spec, "params", {})),
                    features=None if getattr(spec, "features", None) is None else tuple(spec.features),
                )
            )
    reads: dict[tuple[Address, Any], list[ReadOp]] = {}
    for read_name, spec in experiment.reads.items():
        if (spec.model, spec.input) != (name, role):
            continue
        at = _selection(spec.pos, anchors(spec.pos), experiment.features.get(spec.site))
        if not _stacks(spec.pos):
            step = None if spec.pos.frame == "prompt" else ("all" if spec.pos.all else spec.pos.index)
            reads.setdefault((addresses[spec.site], step), []).append(
                ReadOp(
                    name=read_name,
                    at=at,
                    featurizer=spec.featurizer,
                    view=getattr(spec, "view", "raw"),
                )
            )
            continue
        # one ordinary read per decode step, which the run stacks and cuts
        _fits(read_name, spec.site, experiment, len(batch[0]))
        for step in range(experiment.decode):
            reads.setdefault((addresses[spec.site], step), []).append(
                ReadOp(
                    name=f"{read_name}@{step}",
                    at=at,
                    featurizer=spec.featurizer,
                    view=getattr(spec, "view", "raw"),
                    stack=read_name,
                )
            )

    taps = []
    # forward order within a step; the prompt frame (None) before any step
    def order(place: tuple[Address, Any]) -> tuple[int, int, tuple[int, int]]:
        address, step = place
        return (0 if step is None else 1, -1 if step == "all" else (step if step is not None else -1), address.key)

    for place in sorted(set(writes) | set(reads), key=order):
        taps.append(
            Tap(
                address=place[0],
                # A read in model M sees M's writes applied, upstream and at the
                # same address — so at one address the writes go first.
                writes=tuple(writes.get(place, ())),
                reads=tuple(reads.get(place, ())),
                step=place[1],
            )
        )
    return Forward(
        name=name,
        input=role,
        input_ids=batch[0],
        attention_mask=batch[1],
        sample=batch[2],
        segments=batch[3],
        taps=tuple(taps),
        decode=experiment.decode,
    )

