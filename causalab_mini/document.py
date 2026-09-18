"""The protocol document: JSON in, frozen dataclasses out.

Only the subset of `protocol_version` 3 that this slice runs is accepted. Every
other real protocol feature is refused **by name** at load, so a document we
cannot run fails loudly instead of running as something else.

Two rules about where a refusal lives:

* a check about **one value** is a `__post_init__` on the piece that holds it,
  so a spec built in code is refused exactly like one built from JSON;
* a check about **the JSON shape** (unknown keys, a sugar spelling) or about
  **two pieces** (a read's model, a save's restated binding) is a function —
  `from_json` for the first, `_cross_check` for the second.

`Document` is a container, not a god-class: it carries the pieces and the
digest, and its only behaviour is the two constructors `Document.load` and
`Document.from_json`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Json = dict[str, Any]

PROTOCOL_VERSION = "3"

DTYPES = ("fp32", "bf16")
COMPONENTS = ("block_output", "lm_head")
LAYERLESS = ("lm_head",)
MECHANISMS = ("swap",)
TOKEN_FORMS = ("space_prefixed",)
INPUTS = ("base", "counterfactual")

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


def _check(condition: object, message: str) -> None:
    if not condition:
        raise DocumentError(message)


def _pos(spec: Any, where: str) -> int:
    """The only position form this slice runs: `pos: -1`, sugar for
    {"index": -1} — one token per row, counted from the end of the sequence."""
    if isinstance(spec, dict) and set(spec) == {"index"}:
        spec = spec["index"]
    _check(
        isinstance(spec, int) and not isinstance(spec, bool),
        f"{where}: only an integer position (or {{'index': i}}) is implemented",
    )
    return int(spec)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    revision: str
    dtype: str

    def __post_init__(self) -> None:
        _check(self.dtype in DTYPES, f"model.dtype must be one of {DTYPES}")

    @classmethod
    def from_json(cls, raw: Json) -> "ModelSpec":
        for key in ("key", "revision", "dtype"):
            _check(key in raw, f"model.{key} is required")
        unsupported = set(raw) - {"key", "revision", "dtype"}
        _check(not unsupported, f"model.{sorted(unsupported)} is not implemented")
        return cls(raw["key"], raw["revision"], raw["dtype"])


@dataclass(frozen=True)
class RoleSpec:
    dataset: str  # "weekdays/train" or "weekdays/data#train"
    field: str  # "input" or "counterfactual_inputs[0]"

    @classmethod
    def from_json(cls, raw: Json) -> "RoleSpec":
        return cls(raw["dataset"], raw["field"])


@dataclass(frozen=True)
class SiteSpec:
    component: str
    layer: int | None

    def __post_init__(self) -> None:
        _check(
            self.component in COMPONENTS,
            f"component {self.component!r} is not implemented "
            f"(this slice has {COMPONENTS})",
        )
        if self.component in LAYERLESS:
            _check(self.layer is None, f"{self.component} takes no layers")
        else:
            _check(
                isinstance(self.layer, int),
                f"{self.component} is addressed at one layer",
            )

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "SiteSpec":
        _check(
            not (set(raw) - {"component", "layers"}),
            f"site {name!r}: only component/layers are implemented",
        )
        layers = raw.get("layers")
        if raw.get("component") in LAYERLESS:
            _check(layers is None, f"site {name!r}: {raw['component']} takes no layers")
            return cls(raw["component"], None)
        _check(
            isinstance(layers, list) and len(layers) == 1,
            f"site {name!r}: layers must be a one-element band; a band spanning "
            "several layers is one address and is not implemented",
        )
        return cls(raw["component"], int(layers[0]))


@dataclass(frozen=True)
class ReadSpec:
    site: str
    pos: int
    model: str  # "original" or an intervened model name
    input: str  # "base" | "counterfactual"

    def __post_init__(self) -> None:
        _check(self.input in INPUTS, f"read input must be one of {INPUTS}")

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "ReadSpec":
        _check(
            not (set(raw) - {"site", "pos", "model", "input"}),
            f"read {name!r}: only site/pos/model/input are implemented "
            "(no featurizer, no dims)",
        )
        return cls(
            raw["site"], _pos(raw["pos"], f"read {name!r}"), raw["model"], raw["input"]
        )


@dataclass(frozen=True)
class WriteSpec:
    site: str
    pos: int
    mechanism: str  # "swap"
    operand: str  # a read name

    def __post_init__(self) -> None:
        _check(
            self.mechanism in MECHANISMS,
            f"mechanism {self.mechanism!r} is not implemented "
            f"(this slice has {MECHANISMS})",
        )

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "WriteSpec":
        _check(
            not (set(raw) - {"site", "pos", "do"}),
            f"write {name!r}: only site/pos/do are implemented (no featurizer)",
        )
        do = raw["do"]
        _check(len(do) == 1, f"write {name!r}: do has exactly one key")
        ((mechanism, operand),) = do.items()
        _check(
            isinstance(operand, str),
            f"write {name!r}: the operand must be a read name; param and literal "
            "operands are not implemented",
        )
        return cls(raw["site"], _pos(raw["pos"], f"write {name!r}"), mechanism, operand)


@dataclass(frozen=True)
class IntervenedModelSpec:
    input: str
    writes: tuple[str, ...]

    def __post_init__(self) -> None:
        _check(self.input in INPUTS, f"intervened model input must be one of {INPUTS}")

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "IntervenedModelSpec":
        _check(name != "original", "'original' is reserved and never declared")
        _check("input" in raw, f"intervened model {name!r}: input is mandatory")
        return cls(raw["input"], tuple(raw.get("writes", ())))


@dataclass(frozen=True)
class MetricSpec:
    kind: str
    of: str  # a read name
    token_form: str
    columns: tuple[str, ...]  # dataset columns, in the kind's operand order

    def __post_init__(self) -> None:
        _check(self.kind in METRIC_COLUMNS, f"metric kind {self.kind!r} is not implemented")
        _check(
            self.token_form in TOKEN_FORMS,
            f"token_form is required and only {TOKEN_FORMS} is implemented",
        )
        _check(
            len(self.columns) == len(METRIC_COLUMNS[self.kind]),
            f"metric kind {self.kind!r} takes the columns {METRIC_COLUMNS[self.kind]}",
        )

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "MetricSpec":
        kind = raw["kind"]
        _check(kind in METRIC_COLUMNS, f"metric {name!r}: kind {kind!r} is not implemented")
        _check(
            raw.get("token_form") in TOKEN_FORMS,
            f"metric {name!r}: token_form is required and only {TOKEN_FORMS} is implemented",
        )
        _check(
            raw.get("mode", "exact") == "exact",
            f"metric {name!r}: only mode 'exact' is implemented",
        )
        columns = tuple(raw[key] for key in METRIC_COLUMNS[kind])
        return cls(kind, raw["of"], raw["token_form"], columns)


@dataclass(frozen=True)
class SaveSpec:
    value: str
    file_path: str
    model: str | None
    input: str | None

    def __post_init__(self) -> None:
        _check(
            self.file_path.endswith(".json"),
            f"save {self.value!r}: a metric table is a .json file",
        )

    @classmethod
    def from_json(cls, raw: Json) -> "SaveSpec":
        return cls(raw["value"], raw["file_path"], raw.get("model"), raw.get("input"))


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

    @classmethod
    def load(cls, path: str | Path) -> "Document":
        """The document at `path`."""
        return cls.from_json(json.loads(Path(path).read_text()))

    @classmethod
    def from_json(cls, raw: Json) -> "Document":
        """The document a JSON object describes, or a DocumentError."""
        for group in ("header", "model", "data", "method"):
            _check(group in raw, f"missing top-level group {group!r}")
        _check(
            raw["header"].get("protocol_version") == PROTOCOL_VERSION,
            f"protocol_version must be {PROTOCOL_VERSION!r}",
        )

        method = raw["method"]
        for section in UNSUPPORTED_METHOD_SECTIONS:
            _check(
                section not in method,
                f"method.{section} is real protocol surface that causalab-mini does "
                "not implement yet; this slice is activation patching only",
            )
        for required in ("sites", "reads", "save"):
            _check(required in method, f"method.{required} is required")

        roles = {name: RoleSpec.from_json(spec) for name, spec in raw["data"].items()}
        _check("base" in roles, "data.base is required")
        _check(set(roles) <= set(INPUTS), "only base/counterfactual roles")

        sites = {
            name: SiteSpec.from_json(name, spec) for name, spec in method["sites"].items()
        }
        intervened_models = {
            name: IntervenedModelSpec.from_json(name, spec)
            for name, spec in method.get("intervened_models", {}).items()
        }
        reads = {
            name: ReadSpec.from_json(name, spec) for name, spec in method["reads"].items()
        }
        writes = {
            name: WriteSpec.from_json(name, spec)
            for name, spec in method.get("writes", {}).items()
        }
        metrics = {
            name: MetricSpec.from_json(name, spec)
            for name, spec in method.get("metrics", {}).items()
        }
        _check(isinstance(method["save"], list) and method["save"], "save must be a non-empty list")
        saves = tuple(SaveSpec.from_json(entry) for entry in method["save"])

        _cross_check(sites, reads, writes, intervened_models, metrics, saves)
        return cls(
            model=ModelSpec.from_json(raw["model"]),
            roles=roles,
            sites=sites,
            reads=reads,
            writes=writes,
            intervened_models=intervened_models,
            metrics=metrics,
            saves=saves,
            digest=digest(raw),
        )


def digest(raw: Json) -> str:
    """Identity of the experiment. `header.title`/`description` are authoring
    metadata and do not enter it (NOTES.md §2.2)."""
    body = {k: v for k, v in raw.items() if k != "header"}
    body["header"] = {"protocol_version": raw["header"]["protocol_version"]}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _cross_check(
    sites: dict[str, SiteSpec],
    reads: dict[str, ReadSpec],
    writes: dict[str, WriteSpec],
    intervened_models: dict[str, IntervenedModelSpec],
    metrics: dict[str, MetricSpec],
    saves: tuple[SaveSpec, ...],
) -> None:
    """Everything that is about two pieces at once. No single dataclass owns
    any of it, so none of it is a `__post_init__`."""
    for name, read in reads.items():
        _check(read.site in sites, f"read {name!r}: undeclared site")
        _check(
            read.model == "original" or read.model in intervened_models,
            f"read {name!r}: model {read.model!r} is neither 'original' nor declared",
        )
        if read.model in intervened_models:
            # The restated binding, cross-checked: a mismatch is a load error,
            # never a silent override.
            _check(
                intervened_models[read.model].input == read.input,
                f"read {name!r}: input {read.input!r} contradicts model "
                f"{read.model!r}'s input {intervened_models[read.model].input!r}",
            )
    for name, write in writes.items():
        _check(write.site in sites, f"write {name!r}: undeclared site")
        _check(
            write.operand in reads,
            f"write {name!r}: the operand must be a read name; param and literal "
            "operands are not implemented",
        )
    for name, metric in metrics.items():
        _check(metric.of in reads, f"metric {name!r}: 'of' must name a read")
        _check(
            sites[reads[metric.of].site].component == "lm_head",
            f"metric {name!r}: a token-space kind binds to an lm_head read",
        )

    saved = {entry.value for entry in saves}
    for entry in saves:
        _check(
            entry.value in metrics,
            f"save {entry.value!r}: only metric values are implemented",
        )
        read = reads[metrics[entry.value].of]
        # The restated binding is drift protection, never a second source of truth.
        _check(
            entry.model == read.model and entry.input == read.input,
            f"save {entry.value!r}: restated binding {entry.model!r}/"
            f"{entry.input!r} contradicts the declarations "
            f"({read.model!r}/{read.input!r})",
        )
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
            key = (writes[name].site, writes[name].pos)
            _check(
                key not in seen,
                f"intervened model {im_name!r}: two absolute writes at {key}",
            )
            seen.add(key)
