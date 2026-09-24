"""The compiler: a document becomes a plan.

Everything that has to be decided against something the block will not have —
the tokenizer, the rows on disk, the model's layer count and hidden width, the
shuffle a seed defines — is decided here, once, on the client. What comes out
the other side is `plan.Plan`: strings and integers.

The compiler resolves every site to an `Address`, loads the rows, derives
each featurizer's width and compiles its document's steps — once for the
scored run, and once per training update and evaluation of a fit — and hands
every model call to `_forward`. `build_request` is the entry point.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from ..address import Address
from ..data import rows as rows_module, tokens
from ..ops import featurizer as featurizer_module
from ..ops import intervene as intervene_module
from ..ops import locate as locate_module
from . import sweep
from ..shapes import Selection, TokenRows, Where
from .plan import (
    FeaturizerOp,
    Featurizers,
    Fit,
    Forward,
    Generate,
    Metric,
    Plan,
    PlanError,
    ReadOp,
    Reduce,
    SaveFile,
    Step,
    Tap,
    WriteOp,
)

if TYPE_CHECKING:
    from .spec import Spec


#: One forward's padded batch: its ids, its mask, row 0 as the client's
#: tokenizer decodes it, and per row the runs the frame located in it.
_Batch = tuple[TokenRows, TokenRows, str, tuple[dict[str, tuple[int, int]], ...]]

#: Every site's three compiled facts, by site: its address — one per layer,
#: for a site at every layer — which part of the feature axis it is
#: (`(groups, take)`, where it names heads or units), and how wide the tensor
#: is where a continuation read buffers.
_Sites = tuple[dict[str, Any], dict[str, Any], dict[str, int]]


def build_spec(spec: Spec, data_root: str | Path, engine: Any) -> Plan:
    """Compile a document (`spec.py`) into a plan.

    The document's steps are the plan's, one for one: a forward or a
    generate is one model call, a metric and a reduce are their own steps,
    and a fit is a plan of its body's steps per minibatch and one over the
    held-out rows — whose results are its record and what it trained, when a save
    names it. What this adds is what needs a model and rows: the addresses,
    the widths, the tokens, the row counts.
    """
    calls = [spec.ops(step) for _, step in spec.forwards()]
    reads = [read for one, _ in calls for read in one.values()]
    ops = reads + [write for _, one in calls for write in one.values()]
    sites = _resolve_sites(
        # every site the document names: declared, or written in place
        {**spec.sites, **dict(spec.site(op.site) for op in ops)},
        {spec.site(read.site)[0] for read in reads if _stacks(read.pos)},
        engine,
    )
    # where each featurizer acts: the document refused one at two sites
    at = {op.featurizer.rpartition(".")[2]: spec.site(op.site) for op in ops if op.featurizer is not None}
    featurizers = _spec_featurizers(spec, at, sites, engine)

    loaded: dict[str, list[rows_module.Row]] = {}

    def table(key: str) -> list[rows_module.Row]:
        path = spec.path(key)
        if path not in loaded:
            loaded[path] = rows_module.load(data_root, path)
        return loaded[path]

    steps: dict[str, Step] = {}
    if featurizers:
        # Declaring a featurizer is what builds it; the document does not
        # spell out a step whose whole content would be the declaration.
        steps["featurizers"] = Featurizers(specs=featurizers)
    scope: dict[str, Step] = {}
    for name, step in spec.steps.items():
        if step.kind != "fit":
            scope[name] = _spec_step(spec, name, step, scope, table, sites, engine.tokenizer)
            continue
        scope[name] = replace(
            _spec_fit(spec, name, step, table, sites, engine.tokenizer),
            saves=_fit_saves(spec, name, step, featurizers, at),
        )
    _check_layouts(tuple(one for one in scope.values() if isinstance(one, Forward)))
    for ref, file in spec.steps.saves.items():
        head = ref.partition(".")[0]
        if spec.steps[head].kind != "fit":
            _spec_save(spec, spec.steps, scope, ref, file, table)
    return Plan(steps={**steps, **scope}, source=spec.model_dump(mode="json"))


def _spec_step(
    spec: Spec,
    name: str,
    step: Any,
    scope: dict[str, Step],
    table: Callable[[str], list[rows_module.Row]],
    sites: _Sites,
    tokenizer: Any,
) -> Step:
    """One step of a `steps` dict, compiled against the ones before it in
    `scope`. `table` gives a dataset's rows by its key — all of them, a
    minibatch, or the held-out ones, which is the whole difference between
    the scoring, a training update and an evaluation."""
    if step.kind in ("forward", "generate"):
        rows = table(spec.dataset(step.data))
        reads, writes = spec.ops(step)
        for write_name, write in writes.items():
            source = scope.get(str(write.operand).partition(".")[0])
            # a read taken as it is meets row i with row i: as many rows
            if isinstance(source, Forward) and len(source.input_ids) != len(rows):
                raise PlanError(
                    f"step {name!r}: write {write_name!r} swaps in {write.operand!r}, which has "
                    f"{len(source.input_ids)} rows, over {len(rows)} rows; reduce it to a mean, or "
                    "run over the same rows"
                )
        return _forward(
            spec.dataset(step.data),
            step.field,
            writes=[(f"{name}.{one}", _lowered(spec, op), op.operand) for one, op in writes.items()],
            reads=[(f"{name}.{one}", _lowered(spec, op)) for one, op in reads.items()],
            batch=_batch(tokenizer, f"step {name!r}", step.field, rows),
            rows=rows,
            sites=sites,
            tokenizer=tokenizer,
            max_new_tokens=step.max_new_tokens if step.kind == "generate" else 0,
            generation=step.generation if step.kind == "generate" else None,
        )
    source = scope[step.of.partition(".")[0]]
    assert isinstance(source, Forward)
    count = len(source.input_ids)
    if step.kind == "reduce":
        read = next(op for tap in source.taps for op in tap.reads if step.of in (op.name, op.stack))
        width = read.at.where.width if read.at.where is not None else None
        if step.reduce == "pca" and width is not None and step.k > width * count - 1:
            # k directions need more than k vectors: the rows are centered
            # first, which costs one rank. Known here, from the form and the
            # row count, before any forward — a text-anchored read has
            # neither until it runs.
            raise PlanError(
                f"step {name!r} asks for {step.k} principal directions of {width * count} "
                f"vector(s); centered, they span at most {max(width * count - 1, 0)}. Harvest "
                "more rows, or more positions per row"
            )
        return Reduce(of=step.of, reduce=step.reduce, k=step.k)
    rows = table(step.dataset)
    if len(rows) != count:
        raise PlanError(
            f"step {name!r}: its columns are {len(rows)} rows of {step.dataset!r}, and "
            f"{step.of!r} was read over {count}; a metric scores row i against row i"
        )
    return _metric(name, step.metric, step, rows, tokenizer, step.of, (source,))


def _lowered(spec: Spec, op: Any) -> Any:
    """A read or a write as `_forward` takes it: its site by the label its
    address is held under, and its featurizer by the parameter set's own
    name — `fit.rot` is `rot`."""
    featurizer = None if op.featurizer is None else op.featurizer.rpartition(".")[2]
    return op.model_copy(update={"site": spec.site(op.site)[0], "featurizer": featurizer})


def _spec_save(
    spec: Spec,
    steps: Any,
    scope: dict[str, Step],
    ref: str,
    file: str,
    table: Callable[[str], list[rows_module.Row]],
) -> None:
    """Put one save on the step that produces its value: a metric's table on
    the metric, carrying its rows' labels; a reduction's tensor on the
    reduction; a read, or a model call's own result, on the call that makes
    it — which then brings it home, and only then."""
    head = ref.partition(".")[0]
    step, target = steps[head], scope[head]
    save = SaveFile(file_path=file, value=ref)
    if step.kind == "metric":
        save = replace(save, example_ids=rows_module.example_ids(table(step.dataset)), produced_by=spec.digest)
    scope[head] = replace(target, saves=(*target.saves, save))


def _spec_fit(spec: Spec, name: str, fit: Any, table: Any, sites: _Sites, tokenizer: Any) -> Fit:
    """A fit: its body's steps over every minibatch, epoch by epoch, and over
    the held-out rows once an epoch.

    The shuffle is `random.Random(seed)` over the body's row count, and one
    draw indexes every dataset the body names: rows are paired by index, and
    a shuffle that broke the pairing would fit a rotation against mismatched
    counterfactuals.
    """
    counts = {key: len(table(key)) for key in sorted(fit.datasets)}
    if len(set(counts.values())) != 1:
        raise PlanError(
            f"step {name!r}: its body's datasets have {counts} rows; one draw indexes them all, "
            "so they have one count"
        )
    (count,) = set(counts.values())
    for trained, held_out in fit.eval.data.items():
        # the same dataset on both sides is the train-equals-test ablation,
        # and is allowed because it says so
        fitted = {json.dumps(row, sort_keys=True) for row in table(trained)}
        shared = [row for row in table(held_out) if json.dumps(row, sort_keys=True) in fitted]
        if trained != held_out and shared:
            raise PlanError(
                f"step {name!r}: eval puts {held_out!r} in place of {trained!r}, and the two share "
                f"{len(shared)} row(s); held-out rows are ones the fit never saw"
            )

    def body(rows: Callable[[str], list[rows_module.Row]]) -> dict[str, Step]:
        scope: dict[str, Step] = {}
        for inner, step in fit.steps.items():
            scope[inner] = _spec_step(spec, inner, step, scope, rows, sites, tokenizer)
        _check_layouts(tuple(one for one in scope.values() if isinstance(one, Forward)))
        return scope

    def minibatch(picked: list[int]) -> Callable[[str], list[rows_module.Row]]:
        return lambda key: [table(key)[index] for index in picked]

    order = random.Random(fit.seed)
    epochs = tuple(
        tuple(
            Plan(steps=body(minibatch(draw[start : start + fit.batch_size])))
            for start in range(0, count, fit.batch_size)
        )
        for draw in (order.sample(range(count), count) for _ in range(fit.epochs))
    )

    def held(key: str) -> list[rows_module.Row]:
        return table(fit.eval.data[key])

    evaluation = body(held)
    # a body's value a save names, `<fit>.<ref>`, is its held-out run's;
    # `<fit>` and `<fit>.<featurizer>` are the fit's own
    for ref, file in spec.steps.saves.items():
        head, _, inner = ref.partition(".")
        if head == name and inner and inner not in fit.train:
            _spec_save(spec, fit.steps, evaluation, inner, file, held)
    gates = [one for one in fit.train if spec.featurizers[one].kind == "gate"]
    return Fit(
        epochs=epochs,
        evaluation=Plan(steps=evaluation),
        objective=tuple((weight, term) for weight, term in fit.objective),
        params=tuple(fit.train),
        lr=fit.optimizer.lr,
        weight_decay=fit.optimizer.weight_decay,
        # the watched metric, then the fraction each trained gate keeps
        eval_metrics=(fit.early_stop.metric, *(f"{one}.mask" for one in gates)),
        early_stop=fit.early_stop.metric,
        patience=fit.early_stop.patience,
        mode=fit.early_stop.mode,
        anneal=tuple((gate, one.start, one.end) for gate, one in fit.anneal.items()),
    )


def _spec_featurizers(spec: Spec, at: dict[str, tuple[str, Any]], sites: _Sites, engine: Any) -> tuple[FeaturizerOp, ...]:
    """Every declared featurizer, built at the site it acts at, and every
    write's `features` checked against the space it names them in."""
    trainers = {one: step for _, step in spec.steps.items() if step.kind == "fit" for one in step.train}
    featurizers = tuple(
        _featurizer_op(
            name,
            one,
            spec,
            *at[name],
            sites[0][at[name][0]],
            engine,
            trained=name in trainers,
            # a draw with no seed of its own takes the seed of the fit that
            # trains it, or 0 — so an untrained one is a reproducible basis
            seed=one.seed if one.seed is not None else (trainers[name].seed if name in trainers else 0),
        )
        for name, one in spec.featurizers.items()
    )
    spaces = {one.name: one.k for one in featurizers}
    for path, step in spec.forwards():
        for name, write in spec.ops(step)[1].items():
            if write.features is None:
                continue
            base = None if write.featurizer is None else write.featurizer.rpartition(".")[2]
            k = spaces[base] if base in spaces else engine.width(sites[0][spec.site(write.site)[0]])
            if len(set(write.features)) != len(write.features) or not 0 <= min(write.features) <= max(write.features) < k:
                raise PlanError(
                    f"step {path!r}: write {name!r}: features {write.features} of a {k}-dimensional "
                    f"feature space; they are distinct indices below {k}"
                )
    return featurizers


def _fit_saves(
    spec: Spec, name: str, fit: Any, featurizers: tuple[FeaturizerOp, ...], at: dict[str, tuple[str, Any]]
) -> tuple[SaveFile, ...]:
    """A fit's own results a save names: `<fit>`, its record, as a bundle of
    `loss` and `eval`; and `<fit>.<name>`, a parameter it trained, stamped
    with what it is of, so a later document loading it is checked."""
    saves = spec.steps.saves
    d = {one.name: one.d for one in featurizers}
    record = (SaveFile(file_path=saves[name], value="train"),) if name in saves else ()
    return record + tuple(
        SaveFile(
            file_path=saves[f"{name}.{one}"],
            value=one,
            produced_by=spec.digest,
            identity={"produced_by": spec.digest, **_identity(spec, one, *at[one], d[one])},
        )
        for one in fit.train
        if f"{name}.{one}" in saves
    )


def _resolve_sites(places: dict[str, Any], stacking: set[str], engine: Any) -> _Sites:
    """Every site, by its label, resolved against the model once: its
    address, which part of the feature axis it is where it names heads or
    units, and — for the sites a continuation read buffers at, and only
    those — how wide it is."""
    for name, site in places.items():
        if isinstance(site.layers, list) and not 0 <= site.layers[0] < engine.num_layers:
            raise PlanError(
                f"site {name!r}: layer {site.layers[0]} is outside the model's "
                f"{engine.num_layers} layers"
            )
    addresses: dict[str, Any] = {
        name: tuple(engine.locate(site.component, layer) for layer in range(engine.num_layers))
        if site.layers == "all"
        else engine.locate(site.component, site.layers[0] if site.layers else None)
        for name, site in places.items()
    }
    # site -> (groups, take): the feature half of every selection made there
    features: dict[str, tuple[int, tuple[int, ...] | None]] = {}
    for name, site in places.items():
        address = addresses[name][0] if site.layers == "all" else addresses[name]
        if site.units is not None:
            count, take, what = engine.width(address), site.units, "unit"
        elif address.heads_kind is not None:
            count, take, what = engine.heads(address), site.heads, "head"
        else:
            continue
        if take is not None and max(take) >= count:
            raise PlanError(f"site {name!r}: {what} {max(take)} of a {count}-{what} tensor")
        features[name] = (count, None if take is None else tuple(take))
    return addresses, features, {name: engine.width(addresses[name]) for name in sorted(stacking)}


def _featurizer_op(
    name: str, one: Any, spec: Any, label: str, site: Any, address: Address, engine: Any, trained: bool, seed: int
) -> FeaturizerOp:
    """One declared featurizer, with its width filled in from the site it
    acts at — `site` is what the document wrote, `label` its name there —
    and its bundle loaded and checked when it names one."""
    d = engine.width(address)
    if site.heads is not None:
        d = d // engine.heads(address) * len(site.heads)
    if site.units is not None:
        d = len(site.units)
    # a gate's features are the site's units, so its k is d; an encoder's
    # comes from its bundle and may exceed d — a dictionary is overcomplete
    k = d if one.k is None else one.k
    if one.kind in ("subspace", "pca") and not 0 < k <= d:
        raise PlanError(
            f"featurizer {name!r}: k={one.k} is not a subspace of the {d}-wide site {label!r}"
        )
    weight = None
    if one.file_path is not None:
        weight, k = _load_featurizer(name, one, spec, label, site, d)
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


def _identity(spec: Any, name: str, label: str, site: Any, d: int) -> dict[str, str]:
    """What a saved featurizer is *of*: the stamp a bundle carries, and the
    expectation a load is checked against. One function, so the two cannot
    disagree about which keys matter."""
    one = spec.featurizers[name]
    return {
        "model_key": spec.model.key,
        "model_revision": spec.model.revision,
        "model_dtype": spec.model.dtype,
        "site": label,
        "component": site.component,
        "layer": str(site.layers[0] if site.layers else None),
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


def _load_featurizer(name: str, one: Any, spec: Any, label: str, site: Any, d: int) -> tuple[bytes, int]:
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
    expected = _identity(spec, name, label, site, d)
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


def build_request(raw: dict[str, Any], data_root: str | Path, engine: Any) -> Plan:
    """Compile a document, whatever number of experiments it holds.

    A plain document compiles to one plan. A document with `{"sweep": […]}`
    wrappers is several **points**, and compiles to a root plan holding one
    child per point, named for the values it took — which is also the
    directory its results are written to. Nothing else in the project knows
    the difference: a point is an ordinary plan, and the engine that runs the
    root is walking the same tree it always walks. Sweeps are lowered on the
    raw JSON, before a point is validated, so the document's model never
    sees one.
    """
    from .spec import Spec  # here, not at the top: spec.py is the front end and this is below it

    def compile_point(point: dict[str, Any]) -> Plan:
        return replace(build_spec(Spec.model_validate(point), data_root, engine), source=point)

    points = sweep.points(raw)
    if len(points) == 1 and not points[0][0]:
        return compile_point(raw)
    return Plan(steps={label: compile_point(point) for label, point in points}, source=raw)


def _batch(tokenizer: Any, what: str, field: str, table: list[rows_module.Row]) -> _Batch:
    """One forward's padded batch, and the runs the frame located in it.
    `what` is the forward as a refusal names it.

    A field may hold a string or a conversation, and which it is, is the
    row's to say — so a chat prompt costs the document nothing and the
    dataset one column. What the template rendered has the family's opening
    token in it already, which is why it is encoded without another.
    """
    values = [rows_module.field_value(row, field) for row in table]
    texts = [
        tokens.rendered(tokenizer, value, f"{what} field {field!r}, row {row}")
        for row, value in enumerate(values)
    ]
    chat = any(not isinstance(value, str) for value in values)
    ids, mask, sample = tokens.encode(tokenizer, texts, add_special=not chat)
    spans = tokens.turns(tokenizer, ids, mask, values) if chat else ()
    return ids, mask, sample, spans


def _metric(
    name: str, kind: str, spec: Any, rows: list[rows_module.Row], tokenizer: Any, of: str, forwards: tuple[Forward, ...]
) -> Metric:
    """One metric over one batch of rows, with the rows it cannot be computed
    for taken out here, where the data is. `of` is its read, as the plan
    names it, and one of `forwards` reads it."""
    try:
        keep = rows_module.eligible(rows, tuple(spec.columns))
    except rows_module.DataError as refusal:
        raise PlanError(f"metric {name!r}: {refusal}") from None
    if not any(keep):
        raise PlanError(
            f"metric {name!r}: none of these {len(rows)} row(s) has a value in "
            f"{list(spec.columns)}; a metric of nothing has no mean"
        )
    ops = [(tap.address, op) for one in forwards for tap in one.taps for op in tap.reads if of in (op.name, op.stack, op.layered)]
    op = ops[0][1]
    return Metric(
        kind=kind,
        of=of,
        ids=_ids(spec, rows, keep, tokenizer),
        rows=None if all(keep) else tuple(index for index, one in enumerate(keep) if one),
        flat=op.flat,
        layers=tuple(address.layer for address, one in ops if one.layered and address.layer is not None),
    )


def _ids(spec: Any, base_rows: list[rows_module.Row], keep: tuple[bool, ...], tokenizer: Any) -> tuple[Any, ...]:
    return tuple(
        tuple(
            tokens.token_id(tokenizer, value, spec.token_form)
            for value, kept in zip(rows_module.column(base_rows, column), keep)
            if kept and value is not None
        )
        for column in spec.columns
    )


def _check_layouts(forwards: tuple[Forward, ...]) -> None:
    """The layout questions a write and its operand answer before any
    forward, which are therefore the client's.

    Two, both about the operand read here: a value read flat — one entry per
    position found — cannot land in a write that is a rectangle, a window a
    row, nor a rectangle in a flat write; and an attention pattern swapped
    in from another prompt has to line up key for key.

    The two refusals that used to sit beside these — a write with nothing to
    write on a row, and a ragged write whose operand is a different width —
    are about *where* a position lands, so they moved to where positions are
    resolved (`engine/steps.py`).
    """
    read_in = {
        read.stack or read.name: (forward, read)
        for forward in forwards
        for tap in forward.taps
        for read in tap.reads
    }
    for forward in forwards:
        for tap in forward.taps:
            for write in tap.writes:
                if not isinstance(write.operand, str):
                    continue
                source, read = read_in.get(write.operand, (None, None))
                if read is not None and read.flat != write.at.flat:
                    raise PlanError(
                        f"write {write.name!r} at {_spelled(write.at)} is {_layout(write.at.flat)}, "
                        f"and its operand {write.operand!r}, read at {_spelled(read.at)}, is "
                        f"{_layout(read.flat)}: a value lands in a write of its own layout. "
                        "Read and write at the same position, or at two of one form"
                    )
                if tap.address.key_axis:
                    _check_keys(write, forward, source)


def _layout(flat: bool) -> str:
    return "flat (one entry per position found)" if flat else "a rectangle (one window a row)"


def _spelled(at: Any) -> str:
    return at.where.spelling() if at.where is not None else "its positions"


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
            f"write {write.name!r}: an attention pattern that is not a forward's read here — a "
            "reduction, or a value from outside these steps — has no prompts to check its layout "
            "against; swap in a read of the pattern itself"
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


def _selection(pos: Where, anchors: tuple[str, ...], features: Any) -> Selection:
    """Where an op is: the spec, the per-row text it anchors to, and which
    part of the feature axis. Whether it gathers flat is decided *by the
    form*, over every row of the step, so a window of those rows cannot
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


def _decode_step(pos: Where) -> int | str | None:
    """Which decode step a tap acts at: none for the prompt frame, every one
    of them for `{"all": true}` — a steering write — else the one it names."""
    return None if pos.frame == "prompt" else ("all" if pos.all else pos.index)


def _fits(name: str, site: str, sites: _Sites, decode: int, rows: int) -> None:
    """What a stacked read will hold, before anything holds it."""
    _, features, widths = sites
    groups, take = features.get(site) or (None, None)
    wide = widths.get(site, 0)
    if take is not None and groups:
        wide = len(take) * (wide // groups)
    held = decode * rows * wide * 4
    if held > STACK_LIMIT:
        raise PlanError(
            f"read {name!r} keeps every decode step to cut against the continuation: "
            f"{decode} steps x {rows} rows x {wide} wide is {held / 2**20:.0f} MiB, "
            f"over the {STACK_LIMIT / 2**20:.0f} MiB a read may hold. The count is the step's "
            "rows, whatever --batch-size the run uses: name the decode step ({'frame': "
            "'generated', 'index': k}), decode fewer tokens, or run over fewer rows"
        )


def _forward(
    dataset: str,
    field: str,
    writes: list[tuple[str, Any, Any]],
    reads: list[tuple[str, Any]],
    batch: _Batch,
    rows: list[rows_module.Row],
    sites: _Sites,
    tokenizer: Any,
    max_new_tokens: int = 0,
    generation: dict[str, Any] | None = None,
) -> Forward:
    """One model call over the rows of `dataset`: the writes in force, in the
    order they apply, as `(op name, spec, operand)` with the operand as the
    plan names its value, and the reads taken, as `(op name, spec)` —
    grouped into taps by address and put in forward order. With
    `max_new_tokens`, a `Generate`.

    Nothing here resolves a position. What it does compile is the *anchor*: a
    text-anchored spec names a variable, and which text that is on this row
    is a fact about the dataset, which only the client has.
    """
    addresses, features, _ = sites

    def anchors(pos: Where) -> tuple[str, ...]:
        if pos.scope is None or pos.scope.variable is None:
            return ()
        return tuple(rows_module.variable_text(row, field, pos.scope.variable) for row in rows)

    # a tap is one place: an address, and — when the call decodes — a step
    written: dict[tuple[Address, Any], list[WriteOp]] = {}
    for write_name, spec, operand in writes:
        written.setdefault((addresses[spec.site], _decode_step(spec.pos)), []).append(
            WriteOp(
                name=write_name,
                at=_selection(spec.pos, anchors(spec.pos), features.get(spec.site)),
                operand=operand,
                mechanism=spec.mechanism,
                featurizer=spec.featurizer,
                params=dict(getattr(spec, "params", {})),
                features=None if getattr(spec, "features", None) is None else tuple(spec.features),
            )
        )
    read: dict[tuple[Address, Any], list[ReadOp]] = {}
    for read_name, spec in reads:
        at = _selection(spec.pos, anchors(spec.pos), features.get(spec.site))
        step = _decode_step(spec.pos)
        if isinstance(addresses[spec.site], tuple):
            # a read at every layer: one per layer, which the run stacks
            for address in addresses[spec.site]:
                read.setdefault((address, step), []).append(
                    ReadOp(
                        name=f"{read_name}@{address.layer}",
                        at=at,
                        featurizer=spec.featurizer,
                        view=getattr(spec, "view", "raw"),
                        layered=read_name,
                    )
                )
            continue
        if not _stacks(spec.pos):
            read.setdefault((addresses[spec.site], step), []).append(
                ReadOp(
                    name=read_name,
                    at=at,
                    featurizer=spec.featurizer,
                    view=getattr(spec, "view", "raw"),
                )
            )
            continue
        # one ordinary read per decode step, which the run stacks and cuts
        _fits(read_name, spec.site, sites, max_new_tokens, len(batch[0]))
        for step in range(max_new_tokens):
            read.setdefault((addresses[spec.site], step), []).append(
                ReadOp(
                    name=f"{read_name}@{step}",
                    at=at,
                    featurizer=spec.featurizer,
                    view=getattr(spec, "view", "raw"),
                    stack=read_name,
                )
            )

    taps = []
    # Forward order: the prompt frame (None) first, then the decode steps in
    # order with `"all"` before them, and within each by the address's rank.
    # nnsight requires it, and the run relies on it too: a read cut from the
    # decode comes as its steps' parts in step order, and a read at every
    # layer as its layers' parts in layer order, with no sort of their own.
    def order(place: tuple[Address, Any]) -> tuple[int, int, tuple[int, int]]:
        address, step = place
        return (0 if step is None else 1, -1 if step == "all" else (step if step is not None else -1), address.key)

    for place in sorted(set(written) | set(read), key=order):
        taps.append(
            Tap(
                address=place[0],
                # A read in model M sees M's writes applied, upstream and at the
                # same address — so at one address the writes go first.
                writes=tuple(written.get(place, ())),
                reads=tuple(read.get(place, ())),
                step=place[1],
            )
        )
    _fits_every_row(taps, batch, tokenizer)
    decodes: dict[str, Any] = {"max_new_tokens": max_new_tokens, "generation": dict(generation or {})} if max_new_tokens else {}
    return (Generate if max_new_tokens else Forward)(
        input=dataset,
        input_ids=batch[0],
        attention_mask=batch[1],
        sample=batch[2],
        segments=batch[3],
        taps=tuple(taps),
        **decodes,
    )


def _fits_every_row(taps: list[Tap], batch: _Batch, tokenizer: Any) -> None:
    """A fixed-width cut that does not fit a row is an authoring error.

    `{"last": 12}` on a nine-token row, `{"index": 40}` on any of these —
    the form names the same number of tokens on every row, so a row it does
    not fit is a document that is wrong about its own prompts, not a row
    with nothing to say. The client tokenized the rows, so it is decided
    here, before any model runs. An anchored cut is the other case and is
    reported per row by the run instead: which rows carry a word is data.
    """
    fixed = [
        (kind, op.name, op.at.where)
        for tap in taps
        for kind, ops in (("write", tap.writes), ("read", tap.reads))
        for op in ops
        if op.at.where is not None and op.at.where.frame == "prompt" and not op.at.where.ragged
    ]
    if not fixed:
        return
    frame = locate_module.frame_of(tokenizer, batch[0], batch[1], text=False)
    for kind, name, where in fixed:
        windows, reasons = locate_module.locate(frame, where)
        missed = {row: reasons[row] for row, window in enumerate(windows) if not window}
        if missed:
            raise PlanError(
                f"{kind} {name!r} at {where.spelling()} has no position on row(s) "
                f"{missed}. A position of a fixed width names the same number of tokens on "
                "every row, so a row it does not fit is refused rather than skipped; a "
                "position anchored to the row's own text may skip a row, and says why"
            )
