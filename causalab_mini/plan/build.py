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

import collections
import contextvars
import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from ..address import Address, AddressError
from ..data import encoding, rows as rows_module
from ..ops import featurizer as featurizer_module
from ..ops import intervene as intervene_module
from ..ops import metrics as metrics_module
from . import sweep
from ..shapes import Positions, Selection
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
        elif addresses[name].heads_attribute is not None:
            count, take, what = engine.heads(addresses[name]), site.heads, "head"
        else:
            continue
        if take is not None and max(take) >= count:
            raise PlanError(f"site {name!r}: {what} {max(take)} of a {count}-{what} tensor")
        features[name] = (count, None if take is None else tuple(take))
    fits = [one for one in spec.steps.values() if type(one).__name__ == "Fit"]
    def site_width(site: str) -> int | None:
        """How wide a value read at `site` is: the component's width, cut
        down to the heads or units the site names. None where the width is
        not a fact about the model (the attention pattern's key axis)."""
        try:
            return _site_width(spec, site, addresses, engine)
        except AddressError:
            return None

    featurizers = tuple(
        _spec_featurizer(name, one, spec, addresses, engine, fits)
        for name, one in spec.featurizers.items()
    )
    widths = {one.name: one.d for one in featurizers}
    spaces = {one.name: one.k for one in featurizers}
    for label, one in spec.interventions.items():
        for write_name, write in one.writes.items():
            if write.features is None:
                continue
            # with no featurizer the feature space is the site itself — as
            # narrow as its heads or units make it, not the whole component
            k = spaces.get(write.featurizer) or site_width(write.site) or 0
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
    output_widths: dict[str, tuple[tuple[int, ...], Any]] = {}  # per-row widths, how reduced
    for name, step in spec.steps.items():
        kind = type(step).__name__
        experiment = (
            replace(_Experiment.of_spec(spec, spec.intervention_of(step)), features=features)
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
                    per_row, reduced = output_widths[write.operand.ref]
                    if reduced == "pca":
                        raise PlanError(
                            f"step {name!r}: write {write_name!r} names {write.operand.ref!r}, a "
                            "pca basis, as its operand; a basis is loaded as a featurizer, "
                            "not written at a site"
                        )
                    ragged_source = len(set(per_row)) > 1
                    want = encoding.width_of(write.pos)  # None: the write is ragged
                    # What an output may land in. A mean over a ragged read is
                    # one vector and broadcasts anywhere; a mean over a
                    # rectangle keeps its window and must match; an unreduced
                    # rectangle must match; an unreduced ragged read must be
                    # reduced first, or read in the same pass.
                    if reduced and ragged_source:
                        fits = True
                    elif reduced:
                        fits = (want == per_row[0]) or (want is None and per_row[0] == 1)
                    elif not ragged_source:
                        fits = want == per_row[0]
                    else:
                        fits = False
                    if not fits:
                        raise PlanError(
                            f"step {name!r}: write {write_name!r} covers "
                            f"{want if want is not None else 'a varying number of'} "
                            f"position(s) but {write.operand.ref!r} was read over "
                            f"{sorted(set(per_row))}{'' if reduced else ', unreduced'}; "
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
            read_positions = {
                read.name: read.at.positions
                for forward in observe.forwards for tap in forward.taps for read in tap.reads
            }
            for out in outputs:
                if out.reduce == "pca":
                    # k directions need more than k vectors: the rows are
                    # centered first, which costs one rank. Known here, from
                    # the positions, before any forward.
                    vectors = sum(len(window) for window in read_positions[out.read])
                    assert out.k is not None
                    if out.k > vectors - 1:
                        raise PlanError(
                            f"step {name!r}: output {out.name!r} asks for {out.k} principal "
                            f"directions of {vectors} vector(s); centered, they span at most "
                            f"{max(vectors - 1, 0)}. Harvest more rows, or more positions per row"
                        )
                output_rows[out.name] = None if out.reduce == "mean" else len(rows["base"])
                output_widths[out.name] = (
                    tuple(len(window) for window in read_positions[out.read]),
                    out.reduce if out.reduce == "pca" else out.reduce == "mean",
                )
            steps[name] = replace(
                observe,
                outputs=outputs,
                saves=_spec_saves(
                    step.saves, rows["base"],
                    metrics=dict(spec.intervention_of(step).metrics),
                    tensors={
                        **{name: {} for name in spec.intervention_of(step).generated().values()},
                        **{
                            out.name: _stamp(
                                spec, site_of_read[out.read], site_width(site_of_read[out.read]),
                                "pca" if out.reduce == "pca" else f"output/{out.reduce}", out.k,
                            )
                            for out in outputs
                            for site_of_read in [{n: r.site for n, r in spec.intervention_of(step).reads.items()}]
                        },
                    },
                ),
            )
        elif kind == "Fit":
            assert experiment is not None
            steps[name] = _spec_fit(step, spec, experiment, table, addresses, engine, widths)
        elif kind == "Weights":
            steps[name] = Weights(
                names=tuple(step.names),
                saves=_spec_saves(
                    step.saves, [],
                    tensors={
                        one: _identity(spec, one, _sites_of(spec)[one], widths[one]) for one in step.names
                    },
                ),
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
            saves=_spec_saves(step.eval.saves, evaluation_rows["base"], metrics=dict(spec.intervention_of(step).metrics)),
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
        # a fit's own results are its record: the loss per update, the watched
        # metrics per epoch. Tensors of no site.
        saves=_spec_saves(step.saves, [], tensors={"train/loss": {}, "train/eval": {}}),
    )


def _spec_featurizer(
    name: str, one: Any, spec: Any, addresses: dict[str, Address], engine: Any, fits: list[Any]
) -> FeaturizerOp:
    site = _sites_of(spec)[name]
    d = _site_width(spec, site, addresses, engine)
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


def _site_width(spec: Any, site: str, addresses: dict[str, Address], engine: Any) -> int:
    d = engine.width(addresses[site])
    if spec.sites[site].heads is not None:
        d = d // engine.heads(addresses[site]) * len(spec.sites[site].heads)
    if spec.sites[site].units is not None:
        d = len(spec.sites[site].units)
    return d


def _identity(spec: Any, name: str, site: str, d: int) -> dict[str, str]:
    """A declared featurizer's stamp."""
    one = spec.featurizers[name]
    return _stamp(spec, site, d, one.kind, one.k, one.parametrization)


def _stamp(spec: Any, site: str, d: int | None, kind: str, k: Any, parametrization: str = "") -> dict[str, str]:
    """What a saved tensor is *of*: the stamp a bundle carries, and the
    expectation a load is checked against. One function, so the two cannot
    disagree about which keys matter. The site's slice is part of it: a
    rotation fitted inside head 1 is not a rotation of head 2, though both
    are the same component, layer and width."""
    place = spec.sites[site]
    return {
        "model_key": spec.model.key,
        "model_revision": spec.model.revision,
        "model_dtype": spec.model.dtype,
        "site": site,
        "component": place.component,
        "layer": str(place.layers[0] if place.layers else None),
        "heads": json.dumps(place.heads),
        "units": json.dumps(place.units),
        "kind": kind,
        "k": str(k),
        "d": str(d),
        "parametrization": parametrization,
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
    ours = "model_key" in stamp
    if not ours and one.kind not in ("sae", "linear") and not one.trust_unstamped:
        # Checking "the keys the stamp has" against an empty stamp checks
        # nothing, and the shapes that remain collide across every layer of
        # a model. A dictionary trained elsewhere has no stamp of ours by
        # nature; a rotation, a basis or a gate was written by this library.
        raise PlanError(
            f"featurizer {name!r}: {one.file_path!r} carries no identity stamp, so nothing says "
            "which model, component or layer it is of and it cannot be checked against this "
            "document. Re-save it with this version — or, if you have verified its site by hand, "
            'say "trust_unstamped": true on the featurizer'
        )
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
    base_rows: list[rows_module.Row],
    *,
    metrics: dict[str, Any] | None = None,
    tensors: dict[str, dict[str, str]] | None = None,
) -> tuple[SaveFile, ...]:
    """One step's saves, resolved against what *that step* produces.

    `metrics` are the metrics of the step's own intervention — never another
    intervention's that happens to share a name, whose unit and eligibility
    would be stamped on these rows. `tensors` is everything else the step
    produces, by name, with the identity stamp it should carry (empty for a
    value that is of no site: generated ids, a fit's loss curve).

    The file's extension is a contract, both ways: a table is `.json`, a
    tensor is `.safetensors`. Either one written as the other loses what
    makes it readable — a rotation saved as `rot.pt` used to write `[]`.
    """
    metrics, tensors = metrics or {}, tensors or {}
    built = []
    for save in saves:
        if save.value in tensors:
            if not save.file_path.endswith(".safetensors"):
                raise PlanError(
                    f"save {save.value!r}: a tensor, not a table; give it a "
                    f".safetensors path, not {save.file_path!r}"
                )
            built.append(SaveFile(file_path=save.file_path, value=save.value, identity=tensors[save.value]))
            continue
        if save.value not in metrics:  # the validator names what a step produces; this is the backstop
            raise PlanError(
                f"save {save.value!r} is not something this step produces "
                f"(it produces {sorted(metrics) + sorted(tensors)})"
            )
        if not save.file_path.endswith(".json"):
            raise PlanError(
                f"save {save.value!r}: a per-row metric table; give it a .json path, not "
                f"{save.file_path!r} (a .safetensors file would drop the example ids, "
                "eligibility and units)"
            )
        metric = metrics[save.value]
        built.append(
            SaveFile(
                file_path=save.file_path,
                value=save.value,
                example_ids=rows_module.example_ids(base_rows),
                eligible=_eligible(base_rows, metric),
                unit=metrics_module.UNITS[metric.kind][0],
                estimand_version=metrics_module.UNITS[metric.kind][1],
                produced_by=_PRODUCED_BY.get(),
            )
        )
    return tuple(built)


#: The digest of the document being compiled, for `produced_by` on its tables.
_PRODUCED_BY: contextvars.ContextVar[str] = contextvars.ContextVar("produced_by", default="")


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
                    operand=w.operand_name, featurizer=w.featurizer, params=dict(w.params),
                    features=None if w.features is None else tuple(w.features),
                )
                for name, w in intervention.writes.items()
            },
            models=dict(intervention.models),
            metrics=dict(intervention.metrics),
            decode=intervention.decode,
        )


@dataclass(frozen=True)
class _WriteSpec:
    site: str
    pos: Any
    mechanism: str
    operand: str | float | None
    featurizer: str
    params: dict[str, float]
    features: tuple[int, ...] | None = None


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

    from ..engine import provenance

    token = _PRODUCED_BY.set(provenance.document_digest(raw))
    try:
        return build_spec(Spec.model_validate(raw), data_root, engine)
    finally:
        _PRODUCED_BY.reset(token)


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
        _forward(name, role, experiment, batches[role], addresses, rows[role])
        for name, role in _schedule(experiment)
    )
    # A forward is a model on an input; one model may run on two. Named by the
    # model alone their generated ids would collide, and the later one win.
    per_model = collections.Counter(forward.name for forward in forwards)
    forwards = tuple(
        replace(
            forward,
            generated=f"{forward.name}.generated"
            if per_model[forward.name] == 1
            else f"{forward.name}.{forward.input}.generated",
        )
        for forward in forwards
    )
    _check_ragged(forwards, experiment)
    base_rows = rows["base"]  # base is the schema of the pair: metrics read its columns
    metrics = tuple(
        _metric(name, spec, base_rows, tokenizer) for name, spec in experiment.metrics.items()
    )
    return Observe(forwards=forwards, metrics=metrics)


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
            encoding.token_id(tokenizer, value, spec.token_form)
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


def _check_ragged(forwards: tuple[Forward, ...], experiment: _Experiment) -> None:
    """The ragged write policy, and it is `refuse`.

    A read may have an empty window on a row — the column's text was not in
    that prompt — and that row is simply an excluded measurement. A *write*
    may not: writing nothing somewhere is not an intervention, and the row
    would score as if it were. And a ragged write's operand must have, row
    by row, exactly the width the write covers; the protocol's other
    landing policies are not implemented. All of it is knowable here, before
    any forward, because the positions are.
    """
    reads = {
        read.name: read.at.positions for forward in forwards for tap in forward.taps for read in tap.reads
    }
    read_in = {read.name: forward for forward in forwards for tap in forward.taps for read in tap.reads}
    for forward in forwards:
        for tap in forward.taps:
            for write in tap.writes:
                if tap.address.key_axis and isinstance(write.operand, str):
                    _check_keys(write, forward, read_in.get(write.operand))
                empty = [row for row, window in enumerate(write.at.positions) if not window]
                if empty:
                    raise PlanError(
                        f"write {write.name!r} has nothing to write on row(s) {empty}: its "
                        "position's text is not in those prompts. A read may skip a row; a "
                        "write may not"
                    )
                if isinstance(write.operand, str) and write.operand in reads:
                    have = [len(window) for window in reads[write.operand]]
                    want = [len(window) for window in write.at.positions]
                    mismatched = [row for row, (a, b) in enumerate(zip(have, want)) if a != b]
                    if mismatched:
                        raise PlanError(
                            f"write {write.name!r} covers {want} positions per row but its "
                            f"operand {write.operand!r} was read over {have}; rows "
                            f"{mismatched} differ. Landing a window of one width in another "
                            "is a policy this slice does not implement — the protocol's "
                            "`exact_length_buckets` and `padded_masked` — so it refuses"
                        )


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
    """
    if source is None:
        raise PlanError(
            f"write {write.name!r}: an attention pattern from an earlier step cannot be checked "
            "against these prompts' layout; swap one read in the same pass"
        )
    if source.attention_mask == forward.attention_mask:
        return
    ours = [sum(row) for row in forward.attention_mask]
    theirs = [sum(row) for row in source.attention_mask]
    rows = [row for row, (a, b) in enumerate(zip(ours, theirs)) if a != b]
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
            operand = experiment.writes[write].operand
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


def _selection(positions: Positions, features: Any) -> Selection:
    """Where an op is. Whether it gathers flat is decided here, over every
    row of the pass, so a window of those rows cannot decide differently."""
    groups, take = features or (None, None)
    return Selection(positions, groups, take, flat=intervene_module.is_ragged(positions))


def _forward(
    name: str,
    role: str,
    experiment: _Experiment,
    batch: encoding.Batch,
    addresses: dict[str, Address],
    rows: list[rows_module.Row],
) -> Forward:
    """One model pass: its taps, grouped by address and put in forward order."""

    def resolve(pos: Any) -> Positions:
        texts = None
        if isinstance(pos, dict) and set(pos) == {"column"}:
            texts = [rows_module.field_text(row, pos["column"]) for row in rows]
        return encoding.positions(batch, pos, texts)

    # a tap is one place: an address, and — when the forward decodes — a step
    writes: dict[tuple[Address, Any], list[WriteOp]] = {}
    if name in experiment.models:
        for order, write_name in enumerate(experiment.models[name].writes):
            spec = experiment.writes[write_name]
            writes.setdefault((addresses[spec.site], encoding.step_of(spec.pos)), []).append(
                WriteOp(
                    name=write_name,
                    at=_selection(resolve(spec.pos), experiment.features.get(spec.site)),
                    operand=spec.operand,
                    mechanism=spec.mechanism,
                    featurizer=spec.featurizer,
                    params=dict(getattr(spec, "params", {})),
                    features=getattr(spec, "features", None),
                    order=order,
                )
            )
    reads: dict[tuple[Address, Any], list[ReadOp]] = {}
    for read_name, spec in experiment.reads.items():
        if (spec.model, spec.input) != (name, role):
            continue
        reads.setdefault((addresses[spec.site], encoding.step_of(spec.pos)), []).append(
            ReadOp(
                name=read_name,
                at=_selection(resolve(spec.pos), experiment.features.get(spec.site)),
                featurizer=spec.featurizer,
                view=getattr(spec, "view", "raw"),
            )
        )

    taps = []
    # forward order within a step; the prompt frame (None) before any step
    def order(place: tuple[Address, Any]) -> tuple[int, int, tuple[int, int, int]]:
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
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        taps=tuple(taps),
        decode=experiment.decode,
    )

