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
COMPONENTS = (
    "embeddings",
    "block_input",
    "attention_query",
    "attention_key",
    "attention_z",
    "attention_output",
    "mlp_input",
    "mlp_output",
    "block_output",
    "ln_final",
    "lm_head",
)
LAYERLESS = ("embeddings", "ln_final", "lm_head")
MECHANISMS = ("swap",)
TOKEN_FORMS = ("space_prefixed",)
INPUTS = ("base", "counterfactual")
FEATURIZER_KINDS = ("subspace",)
PARAMETRIZATIONS = ("cayley",)
OPTIMIZERS = ("adamw",)
EARLY_STOP_MODES = ("max",)

#: `identity` is what a read or a write with no `featurizer` key gets. It is
#: never declared, the way `original` is never declared.
IDENTITY = "identity"

# A metric kind's operands, in the order metrics.compute() takes them. The
# values are *column names*: the answer is per row, so the document names a
# column and the table carries the string.
METRIC_COLUMNS = {
    "match": ("expected",),
    "logit_diff": ("a", "b"),
    "cross_entropy": ("target",),
    "token_logit": ("token",),
}

# Sections that exist in the protocol and that this slice does not implement.
# Named here so the refusal can say which one.
UNSUPPORTED_METHOD_SECTIONS = ("segments", "positions", "params", "code")

#: The groups a document may have at the top level, and the sections `method`
#: may have. Anything else is refused — a named refusal list can only ever be
#: a snapshot of the protocol, and what it misses would otherwise load, do
#: nothing, and still change the digest.
TOP_LEVEL = ("header", "model", "data", "method")
METHOD_SECTIONS = (
    "sites",
    "reads",
    "writes",
    "intervened_models",
    "featurizers",
    "metrics",
    "train",
    "save",
)


class DocumentError(ValueError):
    """A load error: the document is refused, nothing runs."""


def _check(condition: object, message: str) -> None:
    if not condition:
        raise DocumentError(message)


def _featurizer(raw: Json, where: str) -> str:
    """The one featurizer a read or a write names. A *list* is a chain, which is
    real protocol surface and is not implemented."""
    name = raw.get("featurizer", IDENTITY)
    _check(
        isinstance(name, str),
        f"{where}: a featurizer chain is not implemented; name exactly one",
    )
    return name


def _pos(spec: Any, where: str) -> int:
    """The only position form this slice runs: `pos: -1`, sugar for
    {"index": -1} — one token per row, counted from the end of the sequence."""
    if isinstance(spec, dict) and set(spec) == {"index"}:
        spec = spec["index"]
    if isinstance(spec, bool) or not isinstance(spec, int):
        raise DocumentError(
            f"{where}: only an integer position (or {{'index': i}}) is implemented"
        )
    return spec


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
        for key in ("dataset", "field"):
            _check(key in raw, f"data role: {key} is required")
        # `shuffle` and `draw` live here in the protocol, and `shuffle` is the
        # shuffled-source *control*: a role that quietly dropped it would run
        # the target experiment while stamping the control's digest.
        unsupported = set(raw) - {"dataset", "field"}
        _check(
            not unsupported,
            f"data role: {sorted(unsupported)} is not implemented — and "
            "`shuffle` in particular is the shuffled-source control, so "
            "ignoring it would run a different experiment than the one the "
            "digest names",
        )
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
        if layers is None:  # __post_init__ refuses a component that needs one
            return cls(raw["component"], None)
        if not isinstance(layers, list) or len(layers) != 1:
            raise DocumentError(
                f"site {name!r}: layers must be a one-element band; a band spanning "
                "several layers is one address and is not implemented"
            )
        return cls(raw["component"], int(layers[0]))


@dataclass(frozen=True)
class FeaturizerSpec:
    """A pair of maps between an activation and the feature space a write acts
    in. A featurizer *name* is a parameter set: the same name at a read and at a
    write is one rotation, and there is no tying field — naming is the tying."""

    kind: str
    k: int
    parametrization: str
    #: The draw this featurizer's *initial* basis comes from. Absent, it is
    #: the document's seed (`train.seed`, or 0 with no fit). Authoring it is
    #: what makes an **untrained** subspace a first-class random rank-k basis
    #: rather than a fit implementation detail — sweep it and the document is
    #: the matched-k random-subspace control.
    seed: int | None = None

    def __post_init__(self) -> None:
        _check(
            self.kind in FEATURIZER_KINDS,
            f"featurizer kind {self.kind!r} is not implemented "
            f"(this slice has {FEATURIZER_KINDS})",
        )
        _check(
            self.parametrization in PARAMETRIZATIONS,
            f"parametrization {self.parametrization!r} is not implemented "
            f"(this slice has {PARAMETRIZATIONS})",
        )
        _check(isinstance(self.k, int) and self.k > 0, "featurizer k is a positive width")
        _check(
            self.seed is None or (isinstance(self.seed, int) and not isinstance(self.seed, bool)),
            "featurizer seed is an integer",
        )

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "FeaturizerSpec":
        _check(
            not (set(raw) - {"kind", "k", "parametrization", "seed"}),
            f"featurizer {name!r}: only kind/k/parametrization/seed are implemented — "
            "in particular `d` is derived from (model, site) and may never be authored",
        )
        return cls(raw["kind"], raw["k"], raw["parametrization"], raw.get("seed"))


@dataclass(frozen=True)
class ReadSpec:
    site: str
    pos: int
    model: str  # "original" or an intervened model name
    input: str  # "base" | "counterfactual"
    featurizer: str = IDENTITY

    def __post_init__(self) -> None:
        _check(self.input in INPUTS, f"read input must be one of {INPUTS}")

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "ReadSpec":
        _check(
            not (set(raw) - {"site", "pos", "model", "input", "featurizer"}),
            f"read {name!r}: only site/pos/model/input/featurizer are implemented "
            "(no dims)",
        )
        return cls(
            raw["site"],
            _pos(raw["pos"], f"read {name!r}"),
            raw["model"],
            raw["input"],
            _featurizer(raw, f"read {name!r}"),
        )


@dataclass(frozen=True)
class WriteSpec:
    site: str
    pos: int
    mechanism: str  # "swap"
    operand: str  # a read name
    featurizer: str = IDENTITY

    def __post_init__(self) -> None:
        _check(
            self.mechanism in MECHANISMS,
            f"mechanism {self.mechanism!r} is not implemented "
            f"(this slice has {MECHANISMS})",
        )

    @classmethod
    def from_json(cls, name: str, raw: Json) -> "WriteSpec":
        _check(
            not (set(raw) - {"site", "pos", "do", "featurizer"}),
            f"write {name!r}: only site/pos/featurizer/do are implemented",
        )
        do = raw["do"]
        _check(len(do) == 1, f"write {name!r}: do has exactly one key")
        ((mechanism, operand),) = do.items()
        _check(
            isinstance(operand, str),
            f"write {name!r}: the operand must be a read name; param and literal "
            "operands are not implemented",
        )
        return cls(
            raw["site"],
            _pos(raw["pos"], f"write {name!r}"),
            mechanism,
            operand,
            _featurizer(raw, f"write {name!r}"),
        )


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
        # The kind first: it decides which columns to look for. token_form and
        # the columns' arity are `__post_init__`'s.
        _check(kind in METRIC_COLUMNS, f"metric {name!r}: kind {kind!r} is not implemented")
        _check(
            "token_form" in raw,
            f"metric {name!r}: token_form is required and only {TOKEN_FORMS} is implemented",
        )
        _check(
            raw.get("mode", "exact") == "exact",
            f"metric {name!r}: only mode 'exact' is implemented",
        )
        columns = tuple(raw[key] for key in METRIC_COLUMNS[kind])
        return cls(kind, raw["of"], raw["token_form"], columns)


TRAIN_KEYS = (
    "objective",
    "params",
    "optimizer",
    "steps",
    "batch",
    "precision",
    "eval",
    "early_stop",
    "seed",
)


@dataclass(frozen=True)
class TrainSpec:
    """The fit. `params` is the protocol's only trainability declaration: the
    model is frozen and the featurizers named here are the whole of what a
    backward pass is allowed to reach."""

    objective: tuple[tuple[float, str], ...]  # [[weight, metric], …], minimized
    params: tuple[str, ...]  # featurizer names
    lr: float
    weight_decay: float
    epochs: int
    pairs: int  # base+counterfactual pairs per update, not rows
    eval_split: str  # a dataset ref, exactly like a `data` entry's
    eval_metrics: tuple[str, ...]
    early_stop: str  # the metric watched
    patience: int
    mode: str
    seed: int  # parameter initialization *and* data order

    def __post_init__(self) -> None:
        _check(self.objective, "method.train.objective is a non-empty sum of terms")
        _check(self.params, "method.train.params names at least one featurizer")
        _check(self.epochs > 0 and self.pairs > 0, "method.train: epochs and pairs are positive")
        _check(self.patience > 0, "method.train.early_stop.patience is positive")
        _check(
            self.mode in EARLY_STOP_MODES,
            f"method.train.early_stop.mode {self.mode!r} is not implemented "
            f"(this slice has {EARLY_STOP_MODES})",
        )

    @classmethod
    def from_json(cls, raw: Json) -> "TrainSpec":
        unsupported = set(raw) - set(TRAIN_KEYS)
        _check(not unsupported, f"method.train.{sorted(unsupported)} is not implemented")
        for key in TRAIN_KEYS:
            _check(key in raw, f"method.train.{key} is required")

        objective = []
        for term in raw["objective"]:
            _check(
                isinstance(term, list) and len(term) == 2 and isinstance(term[1], str),
                "method.train.objective: only the positional [[weight, metric], …] "
                "spelling is implemented, and only over a metric (no regularizers)",
            )
            objective.append((float(term[0]), term[1]))

        optimizer = raw["optimizer"]
        _check(
            optimizer.get("name") in OPTIMIZERS,
            f"method.train.optimizer: only {OPTIMIZERS} is implemented",
        )
        _check(
            not (set(optimizer) - {"name", "lr", "weight_decay"}),
            "method.train.optimizer: only name/lr/weight_decay are implemented "
            "(no schedule)",
        )
        for key in ("lr", "weight_decay"):
            _check(
                isinstance(optimizer[key], (int, float)),
                f"method.train.optimizer.{key}: a per-parameter-group mapping is not "
                "implemented",
            )
        _check(
            set(raw["steps"]) == {"epochs"},
            "method.train.steps: only an epoch budget is implemented",
        )
        _check(
            set(raw["batch"]) == {"pairs"},
            "method.train.batch: only a pair count is implemented",
        )
        _check(
            raw["precision"] == {"feature": "fp32", "loss": "fp32"},
            "method.train.precision: only fp32 features and an fp32 loss are "
            "implemented; an engine that cannot honour the declared loop precision "
            "must refuse the document rather than run another one",
        )
        evaluation = raw["eval"]
        _check(
            set(evaluation) == {"every", "metrics", "split"},
            "method.train.eval takes every/metrics/split",
        )
        _check(
            evaluation["every"] == {"epochs": 1},
            "method.train.eval.every: only one pass per epoch is implemented",
        )
        stop = raw["early_stop"]
        _check(
            set(stop) == {"metric", "patience", "mode"},
            "method.train.early_stop takes metric/patience/mode",
        )
        return cls(
            objective=tuple(objective),
            params=tuple(raw["params"]),
            lr=float(optimizer["lr"]),
            weight_decay=float(optimizer["weight_decay"]),
            epochs=int(raw["steps"]["epochs"]),
            pairs=int(raw["batch"]["pairs"]),
            eval_split=evaluation["split"],
            eval_metrics=tuple(evaluation["metrics"]),
            early_stop=stop["metric"],
            patience=int(stop["patience"]),
            mode=stop["mode"],
            seed=int(raw["seed"]),
        )


@dataclass(frozen=True)
class SaveSpec:
    value: str
    file_path: str
    model: str | None
    input: str | None
    site: str | None = None

    def __post_init__(self) -> None:
        if self.site is None:
            _check(
                self.file_path.endswith(".json"),
                f"save {self.value!r}: a metric table is a .json file",
            )
        else:
            _check(
                self.model is None and self.input is None,
                f"save {self.value!r}: a trained featurizer's entry restates its "
                "site, not a model/input",
            )
            _check(
                self.file_path.endswith(".safetensors"),
                f"save {self.value!r}: a trained featurizer is a .safetensors bundle",
            )

    @classmethod
    def from_json(cls, raw: Json) -> "SaveSpec":
        _check(
            not (set(raw) - {"value", "file_path", "model", "input", "site"}),
            f"save {raw.get('value')!r}: only value/file_path/model/input/site are "
            "implemented (no reduce, no non-value kinds)",
        )
        return cls(
            raw["value"],
            raw["file_path"],
            raw.get("model"),
            raw.get("input"),
            raw.get("site"),
        )


@dataclass(frozen=True)
class Document:
    model: ModelSpec
    roles: dict[str, RoleSpec]
    sites: dict[str, SiteSpec]
    featurizers: dict[str, FeaturizerSpec]
    train: TrainSpec | None
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
        for group in TOP_LEVEL:
            _check(group in raw, f"missing top-level group {group!r}")
        unknown = [key for key in raw if key not in TOP_LEVEL]
        _check(
            not unknown,
            f"unknown top-level group(s) {sorted(unknown)}. The protocol has more "
            f"than {list(TOP_LEVEL)} — `axes` among them — and causalab-mini "
            "implements none of them; a group it ignored would still change this "
            "document's digest",
        )
        _check(
            raw["header"].get("protocol_version") == PROTOCOL_VERSION,
            f"protocol_version must be {PROTOCOL_VERSION!r}",
        )

        _check(
            not _swept(raw),
            "this document has a {'sweep': …} wrapper in it. A sweep is lowered "
            "before a document is built — use plan.build_request, which compiles "
            "one plan per point",
        )

        method = raw["method"]
        for section in UNSUPPORTED_METHOD_SECTIONS:
            _check(
                section not in method,
                f"method.{section} is real protocol surface that causalab-mini does "
                "not implement yet; this slice is activation patching only",
            )
        unknown = [key for key in method if key not in METHOD_SECTIONS]
        _check(
            not unknown,
            f"unknown method section(s) {sorted(unknown)}; causalab-mini implements "
            f"{list(METHOD_SECTIONS)}. `path_patching` and `at_once` are real "
            "protocol surface that would otherwise load and do nothing",
        )
        for required in ("sites", "reads", "save"):
            _check(required in method, f"method.{required} is required")

        roles = {name: RoleSpec.from_json(spec) for name, spec in raw["data"].items()}
        _check("base" in roles, "data.base is required")
        _check(set(roles) <= set(INPUTS), "only base/counterfactual roles")

        sites = {
            name: SiteSpec.from_json(name, spec) for name, spec in method["sites"].items()
        }
        featurizers = {
            name: FeaturizerSpec.from_json(name, spec)
            for name, spec in method.get("featurizers", {}).items()
        }
        train = TrainSpec.from_json(method["train"]) if "train" in method else None
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

        _cross_check(
            sites, featurizers, train, reads, writes, intervened_models, metrics, saves
        )
        return cls(
            model=ModelSpec.from_json(raw["model"]),
            roles=roles,
            sites=sites,
            featurizers=featurizers,
            train=train,
            reads=reads,
            writes=writes,
            intervened_models=intervened_models,
            metrics=metrics,
            saves=saves,
            digest=digest(raw),
        )


def _swept(node: Any) -> bool:
    """Whether a sweep wrapper is anywhere in the raw document. A `Document`
    is one point, so one reaching here has not been lowered."""
    if isinstance(node, dict):
        return set(node) == {"sweep"} or any(_swept(value) for value in node.values())
    if isinstance(node, list):
        return any(_swept(value) for value in node)
    return False


def digest(raw: Json) -> str:
    """Identity of the experiment. `header.title`/`description` are authoring
    metadata and do not enter it (NOTES.md §2.2)."""
    body = {k: v for k, v in raw.items() if k != "header"}
    body["header"] = {"protocol_version": raw["header"]["protocol_version"]}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _cross_check(
    sites: dict[str, SiteSpec],
    featurizers: dict[str, FeaturizerSpec],
    train: TrainSpec | None,
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

    # A featurizer name is a parameter set, so the sites it appears at are what
    # its width `d` is derived from. One name at several sites would be one
    # rotation over two widths.
    used_at: dict[str, set[str]] = {}
    for name, read in reads.items():
        used_at.setdefault(read.featurizer, set()).add(read.site)
    for name, write in writes.items():
        used_at.setdefault(write.featurizer, set()).add(write.site)
    for name, at in used_at.items():
        if name == IDENTITY:
            continue
        _check(name in featurizers, f"undeclared featurizer {name!r}")
        _check(
            len(at) == 1,
            f"featurizer {name!r} is one parameter set but is used at the sites "
            f"{sorted(at)}; one name per site is what this slice implements",
        )
    for name in featurizers:
        _check(name in used_at, f"featurizer {name!r} is declared and never used")

    for name, metric in metrics.items():
        _check(metric.of in reads, f"metric {name!r}: 'of' must name a read")
        _check(
            sites[reads[metric.of].site].component == "lm_head",
            f"metric {name!r}: a token-space kind binds to an lm_head read",
        )
        _check(
            reads[metric.of].featurizer == IDENTITY,
            f"metric {name!r}: a token-space kind binds to a *plain* lm_head read, "
            "with no featurizer",
        )

    _check(
        not (set(metrics) & set(featurizers)),
        f"{sorted(set(metrics) & set(featurizers))} names both a metric and a "
        "featurizer; a save entry could not say which one it meant",
    )
    trained = set(train.params) if train is not None else set()
    saved = {entry.value for entry in saves}
    for entry in saves:
        if entry.site is not None:
            _check(
                entry.value in featurizers,
                f"save {entry.value!r}: a site-restated entry saves a featurizer",
            )
            _check(
                entry.value in trained,
                f"save {entry.value!r}: an untrained featurizer may not be saved",
            )
            _check(
                {entry.site} == used_at[entry.value],
                f"save {entry.value!r}: restated site {entry.site!r} contradicts the "
                f"declarations ({sorted(used_at[entry.value])})",
            )
            continue
        _check(
            entry.value in metrics,
            f"save {entry.value!r}: only metric and trained-featurizer values are "
            "implemented",
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
    for name in trained:
        _check(name in featurizers, f"method.train.params: undeclared featurizer {name!r}")
        _check(
            name in saved,
            f"method.train.params: trained featurizer {name!r} is never saved; every "
            "trained featurizer must be",
        )
    if train is not None:
        for _weight, term in train.objective:
            _check(term in metrics, f"method.train.objective: {term!r} is not a metric")
        for name in train.eval_metrics:
            _check(name in metrics, f"method.train.eval.metrics: {name!r} is not a metric")
        _check(
            train.early_stop in train.eval_metrics,
            f"method.train.early_stop: {train.early_stop!r} is not one of the metrics "
            "the eval pass computes",
        )
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
