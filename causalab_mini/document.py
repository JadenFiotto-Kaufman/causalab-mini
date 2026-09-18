"""The protocol document: JSON in, frozen dataclasses out.

Only the subset of `protocol_version` 3 that this slice runs is accepted. Every
other real protocol feature is refused **by name** at load, so a document we
cannot run fails loudly instead of running as something else. The rules below
are the load-time checks from NOTES.md §2 and §6; the cross-checks (a read's
`input` against its model's, a save's restated binding) are kept because they
are two lines each and they are what catches a hand-edited JSON.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROTOCOL_VERSION = "3"

DTYPES = ("fp32", "bf16")
COMPONENTS = ("block_output", "lm_head")
LAYERLESS = ("lm_head",)
MECHANISMS = ("swap",)
TOKEN_FORMS = ("space_prefixed",)

# A metric kind's operands, in the order metrics.compute() takes them. The
# values are *column names*: the answer is per row, so the document names a
# column and the table carries the string.
METRIC_COLUMNS = {
    "match": ("expected",),
    "logit_diff": ("a", "b"),
    "cross_entropy": ("target",),
}

# Sections that exist in the protocol and that this slice does not implement.
# Named here so the refusal can say which one.
UNSUPPORTED_METHOD_SECTIONS = (
    "segments",
    "positions",
    "featurizers",
    "params",
    "code",
    "train",
)


class DocumentError(ValueError):
    """A load error: the document is refused, nothing runs."""


def _check(condition, message):
    if not condition:
        raise DocumentError(message)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    revision: str
    dtype: str


@dataclass(frozen=True)
class RoleSpec:
    dataset: str  # "weekdays/train" or "weekdays/data#train"
    field: str  # "input" or "counterfactual_inputs[0]"


@dataclass(frozen=True)
class SiteSpec:
    component: str
    layer: int | None


@dataclass(frozen=True)
class ReadSpec:
    site: str
    pos: int
    model: str  # "original" or an intervened model name
    input: str  # "base" | "counterfactual"


@dataclass(frozen=True)
class WriteSpec:
    site: str
    pos: int
    mechanism: str  # "swap"
    operand: str  # a read name


@dataclass(frozen=True)
class IntervenedModelSpec:
    input: str
    writes: tuple[str, ...]


@dataclass(frozen=True)
class MetricSpec:
    kind: str
    of: str  # a read name
    token_form: str
    columns: tuple[str, ...]  # dataset columns, in the kind's operand order


@dataclass(frozen=True)
class SaveSpec:
    value: str
    file_path: str
    model: str | None
    input: str | None


@dataclass(frozen=True)
class Document:
    model: ModelSpec
    roles: dict[str, RoleSpec]
    sites: dict[str, SiteSpec]
    reads: dict[str, ReadSpec]
    writes: dict[str, WriteSpec]
    intervened_models: dict[str, IntervenedModelSpec]
    metrics: dict[str, MetricSpec]
    saves: tuple[SaveSpec, ...]
    digest: str
    """sha256 of the document with authoring metadata dropped. NOT causalab's
    point digest (that is a canonical re-emission, §7 of the spec); it is a
    stable identifier for `produced_by` and nothing more."""


def load(path) -> Document:
    return parse(json.loads(Path(path).read_text()))


def parse(raw: dict) -> Document:
    for group in ("header", "model", "data", "method"):
        _check(group in raw, f"missing top-level group {group!r}")
    _check(
        raw["header"].get("protocol_version") == PROTOCOL_VERSION,
        f"protocol_version must be {PROTOCOL_VERSION!r}",
    )

    model = _model(raw["model"])
    roles = _roles(raw["data"])
    method = raw["method"]
    for section in UNSUPPORTED_METHOD_SECTIONS:
        _check(
            section not in method,
            f"method.{section} is real protocol surface that causalab-mini does "
            "not implement yet; this slice is activation patching only",
        )
    for required in ("sites", "reads", "save"):
        _check(required in method, f"method.{required} is required")

    sites = _sites(method["sites"])
    intervened_models = _intervened_models(method.get("intervened_models", {}))
    reads = _reads(method["reads"], sites, intervened_models)
    writes = _writes(method.get("writes", {}), sites, reads)
    metrics = _metrics(method.get("metrics", {}), reads, sites)
    saves = _saves(method["save"], metrics, reads, intervened_models)
    _cross_check(reads, writes, intervened_models, metrics, saves)

    return Document(
        model=model,
        roles=roles,
        sites=sites,
        reads=reads,
        writes=writes,
        intervened_models=intervened_models,
        metrics=metrics,
        saves=saves,
        digest=digest(raw),
    )


def digest(raw: dict) -> str:
    """Identity of the experiment. `header.title`/`description` are authoring
    metadata and do not enter it (NOTES.md §2.2)."""
    body = {k: v for k, v in raw.items() if k != "header"}
    body["header"] = {"protocol_version": raw["header"]["protocol_version"]}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _model(raw: dict) -> ModelSpec:
    for key in ("key", "revision", "dtype"):
        _check(key in raw, f"model.{key} is required")
    _check(raw["dtype"] in DTYPES, f"model.dtype must be one of {DTYPES}")
    unsupported = set(raw) - {"key", "revision", "dtype"}
    _check(not unsupported, f"model.{sorted(unsupported)} is not implemented")
    return ModelSpec(raw["key"], raw["revision"], raw["dtype"])


def _roles(raw: dict) -> dict[str, RoleSpec]:
    _check("base" in raw, "data.base is required")
    _check(set(raw) <= {"base", "counterfactual"}, "only base/counterfactual roles")
    return {name: RoleSpec(spec["dataset"], spec["field"]) for name, spec in raw.items()}


def _sites(raw: dict) -> dict[str, SiteSpec]:
    sites = {}
    for name, spec in raw.items():
        component = spec["component"]
        _check(
            component in COMPONENTS,
            f"site {name!r}: component {component!r} is not implemented "
            f"(this slice has {COMPONENTS})",
        )
        _check(
            not (set(spec) - {"component", "layers"}),
            f"site {name!r}: only component/layers are implemented",
        )
        layers = spec.get("layers")
        if component in LAYERLESS:
            _check(layers is None, f"site {name!r}: {component} takes no layers")
            sites[name] = SiteSpec(component, None)
            continue
        _check(
            isinstance(layers, list) and len(layers) == 1,
            f"site {name!r}: layers must be a one-element band; a band spanning "
            "several layers is one address and is not implemented",
        )
        sites[name] = SiteSpec(component, int(layers[0]))
    return sites


def _intervened_models(raw: dict) -> dict[str, IntervenedModelSpec]:
    models = {}
    for name, spec in raw.items():
        _check(name != "original", "'original' is reserved and never declared")
        _check("input" in spec, f"intervened model {name!r}: input is mandatory")
        models[name] = IntervenedModelSpec(spec["input"], tuple(spec.get("writes", ())))
    return models


def _pos(spec, where) -> int:
    """The only position form this slice runs: `pos: -1`, sugar for
    {"index": -1} — one token per row, counted from the end of the sequence."""
    if isinstance(spec, dict) and set(spec) == {"index"}:
        spec = spec["index"]
    _check(
        isinstance(spec, int) and not isinstance(spec, bool),
        f"{where}: only an integer position (or {{'index': i}}) is implemented",
    )
    return spec


def _reads(raw, sites, intervened_models) -> dict[str, ReadSpec]:
    reads = {}
    for name, spec in raw.items():
        _check(
            not (set(spec) - {"site", "pos", "model", "input"}),
            f"read {name!r}: only site/pos/model/input are implemented "
            "(no featurizer, no dims)",
        )
        _check(spec["site"] in sites, f"read {name!r}: undeclared site")
        model = spec["model"]
        _check(
            model == "original" or model in intervened_models,
            f"read {name!r}: model {model!r} is neither 'original' nor declared",
        )
        _check(
            spec["input"] in ("base", "counterfactual"),
            f"read {name!r}: input must be base or counterfactual",
        )
        if model in intervened_models:
            # The restated binding, cross-checked: a mismatch is a load error,
            # never a silent override.
            _check(
                intervened_models[model].input == spec["input"],
                f"read {name!r}: input {spec['input']!r} contradicts model "
                f"{model!r}'s input {intervened_models[model].input!r}",
            )
        reads[name] = ReadSpec(spec["site"], _pos(spec["pos"], f"read {name!r}"), model, spec["input"])
    return reads


def _writes(raw, sites, reads) -> dict[str, WriteSpec]:
    writes = {}
    for name, spec in raw.items():
        _check(
            not (set(spec) - {"site", "pos", "do"}),
            f"write {name!r}: only site/pos/do are implemented (no featurizer)",
        )
        _check(spec["site"] in sites, f"write {name!r}: undeclared site")
        do = spec["do"]
        _check(len(do) == 1, f"write {name!r}: do has exactly one key")
        ((mechanism, operand),) = do.items()
        _check(
            mechanism in MECHANISMS,
            f"write {name!r}: mechanism {mechanism!r} is not implemented "
            f"(this slice has {MECHANISMS})",
        )
        _check(
            isinstance(operand, str) and operand in reads,
            f"write {name!r}: the operand must be a read name; param and literal "
            "operands are not implemented",
        )
        writes[name] = WriteSpec(
            spec["site"], _pos(spec["pos"], f"write {name!r}"), mechanism, operand
        )
    return writes


def _metrics(raw, reads, sites) -> dict[str, MetricSpec]:
    metrics = {}
    for name, spec in raw.items():
        kind = spec["kind"]
        _check(kind in METRIC_COLUMNS, f"metric {name!r}: kind {kind!r} is not implemented")
        _check(spec["of"] in reads, f"metric {name!r}: 'of' must name a read")
        _check(
            sites[reads[spec["of"]].site].component == "lm_head",
            f"metric {name!r}: a token-space kind binds to an lm_head read",
        )
        _check(
            spec.get("token_form") in TOKEN_FORMS,
            f"metric {name!r}: token_form is required and only {TOKEN_FORMS} is implemented",
        )
        _check(
            spec.get("mode", "exact") == "exact",
            f"metric {name!r}: only mode 'exact' is implemented",
        )
        columns = tuple(spec[key] for key in METRIC_COLUMNS[kind])
        metrics[name] = MetricSpec(kind, spec["of"], spec["token_form"], columns)
    return metrics


def _saves(raw, metrics, reads, intervened_models) -> tuple[SaveSpec, ...]:
    _check(isinstance(raw, list) and raw, "save must be a non-empty list")
    saves = []
    for entry in raw:
        value = entry["value"]
        _check(value in metrics, f"save {value!r}: only metric values are implemented")
        _check(
            entry["file_path"].endswith(".json"),
            f"save {value!r}: a metric table is a .json file",
        )
        read = reads[metrics[value].of]
        # The restated binding is drift protection, never a second source of truth.
        _check(
            entry.get("model") == read.model and entry.get("input") == read.input,
            f"save {value!r}: restated binding {entry.get('model')!r}/"
            f"{entry.get('input')!r} contradicts the declarations "
            f"({read.model!r}/{read.input!r})",
        )
        saves.append(SaveSpec(value, entry["file_path"], entry.get("model"), entry.get("input")))
    return tuple(saves)


def _cross_check(reads, writes, intervened_models, metrics, saves):
    saved = {entry.value for entry in saves}
    for name in metrics:
        _check(name in saved, f"metric {name!r} is never saved; every metric must be")
    used = {name for im in intervened_models.values() for name in im.writes}
    for name in writes:
        _check(name in used, f"write {name!r} appears in no intervened model")
    for im_name, im in intervened_models.items():
        for name in im.writes:
            _check(name in writes, f"intervened model {im_name!r}: undeclared write {name!r}")
    # At most one absolute write per (site, position, model). Every mechanism in
    # this slice is absolute, so this is "at most one write per address".
    for im_name, im in intervened_models.items():
        seen = set()
        for name in im.writes:
            write = writes[name]
            key = (write.site, write.pos)
            _check(
                key not in seen,
                f"intervened model {im_name!r}: two absolute writes at {key}",
            )
            seen.add(key)
