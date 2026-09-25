"""The command line: a document in, and every question an author can ask.

Two kinds of verb. Those that need no weights — `schema`, `vocab`, `model`,
`tokens`, `data`, `validate`, `explain` — are the loop an author lives in,
human or agent: ask what exists, write a document, check it, read what it
means. They run in seconds on a laptop against a model that will only ever
run on NDIF. `run` is the one that needs the weights.

Every verb takes `--json`, so an agent parses instead of scraping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, get_args

from nnterp.rename_utils import RenamingError
from pydantic import ValidationError

from . import address, plan as plan_module
from .address import AddressError
from .data import rows as rows_module
from .data.rows import DataError
from .data.tokens import TOKEN_FORMS, TokenError
from .engine import NNterpEngine
from .engine.base import EngineError
from .engine.engines.hooks import HooksEngine
from .ops import featurizer, intervene
from .ops.locate import LocateError
from .plan import sweep
from .plan.explain import explain
from .plan.plan import PlanError
from .plan.spec import METRIC_SIGNATURES, Model, Optimizer, Reduce, Spec
from .shapes import Where

#: What `--engine` means: the class, how it is loaded to *run*, and where it
#: runs. `ndif` is the nnterp engine with no local weights — a meta shell —
#: executed remotely; the server has the weights. Compiling, for every
#: engine, is the same shell with nothing dispatched.
ENGINES: dict[str, tuple[type, dict[str, Any], bool | str]] = {
    "nnterp": (NNterpEngine, {}, False),
    "ndif": (NNterpEngine, {"dispatch": False}, True),
    "hooks": (HooksEngine, {}, False),
}
SHAPE_ONLY = {"dispatch": False}

#: What this package raises when it means "no". Every one carries a message
#: written for the person who wrote the document, so the entry point prints
#: that and nothing else. `ValidationError` is pydantic's and is how a
#: document refuses; `RenamingError` is nnterp's, which mini
#: forwards wherever a place is a family's to have or not have.
REFUSALS: tuple[type[Exception], ...] = (
    PlanError,          # the compiler, and the run's own refusals
    AddressError,       # a component, a layer, an attention implementation
    TokenError,         # a prompt, a conversation, an answer column
    LocateError,        # a frame the resolver cannot build
    DataError,          # a dataset ref, a column, a row
    EngineError,        # a runtime asked for something it does not have
    ValidationError,    # the document
    RenamingError,      # nnterp, where a place is not this family's
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalab-mini")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--traceback", action="store_true",
        help="on a refusal, print where it was raised as well as what it says",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    sub.add_parser("schema", help="the JSON Schema of a document")
    sub.add_parser("vocab", help="step kinds, components, mechanisms, featurizer kinds, metric kinds, position forms")

    one = sub.add_parser("model", help="what a model looks like: layers, widths, which components resolve")
    one.add_argument("key")
    # the model block's own fields, with the model block's own choices — a
    # document says which attention implementation it runs under, and two
    # components exist only under one of them
    one.add_argument("--revision", default="main")
    one.add_argument("--dtype", default="fp32", choices=_choices("dtype"))
    one.add_argument("--attn-implementation", default=None, choices=_choices("attn_implementation"))
    one.add_argument("--engine", default="nnterp", choices=list(ENGINES))

    one = sub.add_parser("tokens", help="whether each string is one token — a metric column must be")
    one.add_argument("key")
    one.add_argument("text", nargs="+")
    one.add_argument("--revision", default="main")

    one = sub.add_parser("data", help="a dataset ref: columns, rows, a sample")
    one.add_argument("ref")
    one.add_argument("--data-root", default="documents/data")

    for verb, help_text in (
        ("validate", "compile a document without weights; say whether it is valid"),
        ("explain", "compile a document without weights and print the plan"),
        ("run", "execute a document and write its outputs"),
    ):
        one = sub.add_parser(verb, help=help_text)
        one.add_argument("document", help="a document")
        one.add_argument("--data-root", default="documents/data")
        one.add_argument("--engine", default="nnterp", choices=list(ENGINES))
        if verb == "run":
            one.add_argument("--out", default="out")
            one.add_argument("--device-map", default="auto", help="ignored by ndif, whose weights are the server's")
            one.add_argument("--batch-size", type=int, default=None,
                             help="rows per model call; bounds memory, moves only the last bit (default: every row at once)")

    args = parser.parse_args(argv)
    try:
        result = VERBS[args.verb](args)
    except REFUSALS as refusal:
        # A refusal is this package saying no, and it says why in its own
        # message — every one of them is written to be read. The traceback
        # is the library's business and is one flag away.
        if args.traceback:
            raise
        print(f"{type(refusal).__name__}: {refusal}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=1, default=str))
    else:
        print(result["text"])
    return 0


# --------------------------------------------------------------------- #
# the verbs — each returns {"text": …, …} so --json and plain share one body
# --------------------------------------------------------------------- #


def schema(args: argparse.Namespace) -> dict[str, Any]:
    payload = Spec.model_json_schema()
    return {"text": json.dumps(payload, indent=1), "schema": payload}


def vocab(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "step_kinds": _step_kinds(),
        "reductions": list(get_args(Reduce.model_fields["reduce"].annotation)),
        "components": address.describe(),
        "mechanisms": sorted(intervene.MECHANISMS),
        "featurizer_kinds": sorted(featurizer.KINDS),
        "optimizers": list(get_args(Optimizer.model_fields["name"].annotation)),
        "token_forms": list(TOKEN_FORMS),
        # `of`, then the further reads, then the columns — the order they are scored in
        "metric_kinds": {
            kind: {
                "reads": ["of", *one.reads],
                "columns": list(one.columns),
                "params": dict(one.params),
                "logits": one.logits,
                "doc": one.doc,
            }
            for kind, one in METRIC_SIGNATURES.items()
        },
        "position_forms": Where.forms(),
        "layers": "a site's `layers`: an int is one layer; a list is those layers — a read there is one "
        "value stacked in the listed order, layer axis first, and a write writes at each; \"all\" is every layer",
        "saves": "{reference: file}, or a list of references — a listed one, or one mapped to null, "
        "is written as itself: a metric's table to <reference>.json, a tensor to <reference>.safetensors",
        "units": {kind: {"unit": one.unit, "estimand_version": one.version} for kind, one in METRIC_SIGNATURES.items()},
    }
    lines = [f"step kinds:       {', '.join(payload['step_kinds'])}; a reduce is {' or '.join(payload['reductions'])}"]
    lines.append("components:")
    for name, entry in payload["components"].items():
        lines.append(f"  {name:22s} nnterp {entry['accessor']}{'  (read-only)' if entry['read_only'] else ''}"
                     f"{'  [heads]' if entry['heads'] else ''}"
                     f"{'  needs ' + entry['needs'] + ' attention' if entry['needs'] else ''}")
    lines.append(f"mechanisms:       {', '.join(payload['mechanisms'])}")
    lines.append(f"layers:           {payload['layers']}")
    lines.append(f"saves:            {payload['saves']}")
    lines.append(f"featurizer kinds: {', '.join(payload['featurizer_kinds'])}")
    lines.append(f"optimizers:       {', '.join(payload['optimizers'])} (betas for the Adams, momentum for sgd and rmsprop)")
    lines.append("metric kinds:     a read is `<step>.<read>`, a column `<dataset>.<column>`, spelled as a token "
                 + " | ".join(payload["token_forms"]))
    for kind, one in payload["metric_kinds"].items():
        given = one["reads"] + one["columns"] + [f"{k}={v}" for k, v in one["params"].items()]
        unit = payload["units"][kind]["unit"]
        lines.append(f"  {kind + '(' + ', '.join(given) + ')':38s} {one['doc']} [{unit}]")
    forms = payload["position_forms"]
    lines.append("position forms:   exactly one cut: "
                 + ", ".join(f"{k}={v}" for k, v in forms["cut"].items()))
    lines.append("                  scope: " + ", ".join(f"{k}={v}" for k, v in forms["scope"].items()))
    lines.append(f"                  frame: {forms['frame']}; {forms['sugar']}")
    return {"text": "\n".join(lines), **payload}


def _step_kinds() -> list[str]:
    """The kinds a step may be, off the document's own schema — so the list
    and what validates cannot come to disagree."""
    steps = Spec.model_json_schema()["$defs"]["Steps"]["additionalProperties"]
    return list(steps["discriminator"]["mapping"])


def model(args: argparse.Namespace) -> dict[str, Any]:
    """Every component, at every layer that answers differently.

    A place is not a property of a checkpoint but of a checkpoint's layer:
    DeepSeek's first blocks are dense and the rest are mixtures of experts,
    and nnterp answers `unavailable_on(layer)` per layer for exactly that.
    Asking at layer 0 only reported the first block as if it were the model.
    Layers that answer alike are one band, so a model whose layers are all
    the same prints one line per component, as before.
    """
    spec = Model.model_validate(  # the document's own model block, validated the same way
        {
            "key": args.key,
            "revision": args.revision,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
        }
    )
    engine = ENGINES[args.engine][0].load(spec, **SHAPE_ONLY)
    components = {
        name: _bands(engine, name, range(engine.num_layers) if entry["layered"] else [None])
        for name, entry in address.describe().items()
    }
    payload = {
        "key": args.key,
        "engine": args.engine,
        "attn_implementation": getattr(engine.model.config, "_attn_implementation", None),
        "num_layers": engine.num_layers,
        "padding_side": getattr(engine.tokenizer, "padding_side", None),
        "components": components,
    }
    lines = [
        f"{args.key} via {args.engine}: {payload['num_layers']} layers, "
        f"{payload['attn_implementation']} attention, padding_side={payload['padding_side']}"
    ]
    for name, bands in components.items():
        for index, band in enumerate(bands):
            at = "" if band["layers"] is None or len(bands) == 1 else f"layers {band['layers']}  "
            if band["resolves"]:
                width = f"width={band['width']}" if band["width"] is not None else "no width (no featurizer here)"
                said = f"ok   {at}{width}" + ("  inside a forward" if band["inside"] else "")
            else:
                said = f"--   {at}{band['why']}"
            lines.append(f"  {name if index == 0 else '':18s} {said}")
    return {"text": "\n".join(lines), **payload}


def _bands(engine: Any, name: str, layers: Any) -> list[dict[str, Any]]:
    """One entry per run of layers that answer alike, in layer order."""
    found: list[dict[str, Any]] = []
    for layer in layers:
        answer = _resolves(engine, name, layer)
        if found and {k: v for k, v in found[-1].items() if k != "layers"} == answer:
            found[-1]["layers"] = f"{found[-1]['layers'].split('-')[0]}-{layer}"
            continue
        found.append({"layers": None if layer is None else str(layer), **answer})
    return found


def _resolves(engine: Any, name: str, layer: int | None) -> dict[str, Any]:
    try:
        located = engine.locate(name, layer)
    except REFUSALS as refusal:
        return {"resolves": False, "why": str(refusal).splitlines()[0]}
    try:
        width: Any = engine.width(located)
    except REFUSALS:
        width = None
    return {"resolves": True, "width": width, "inside": located.inside}


def _choices(field: str) -> tuple[str, ...]:
    """A model-block field's own values, off the model block — so a flag and
    the document it stands in for cannot come to disagree."""
    annotation = Model.model_fields[field].annotation
    named = tuple(one for one in get_args(annotation) if isinstance(one, str))
    # `Literal[...] | None` on an optional field: the values are one level in
    return named or tuple(one for one in get_args(get_args(annotation)[0]) if isinstance(one, str))


def tokens(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.key, revision=args.revision)
    found = {}
    for text in args.text:
        ids = tokenizer.encode(text, add_special_tokens=False)
        found[text] = {"tokens": len(ids), "ids": ids, "one_token": len(ids) == 1}
    lines = [
        f"{text!r:24s} {entry['tokens']} token(s)" + ("" if entry["one_token"] else "  <- not usable as a metric column")
        for text, entry in found.items()
    ]
    return {"text": "\n".join(lines), "key": args.key, "texts": found}


def data(args: argparse.Namespace) -> dict[str, Any]:
    table = rows_module.load(args.data_root, args.ref)
    columns = sorted({key for row in table for key in row})
    splits = sorted({str(row.get("split")) for row in table})
    # A null or empty value is an excluded measurement for any metric naming
    # that column, so which columns have holes is worth knowing up front.
    empty = {
        name: count
        for name in columns
        if (count := sum(1 for row in table if row.get(name) in (None, "")))
    }
    payload = {"ref": args.ref, "rows": len(table), "columns": columns, "empty": empty,
               "splits": splits, "sample": table[:1]}
    lines = [
        f"{args.ref}: {len(table)} rows",
        f"  columns: {', '.join(columns)}",
        *([f"  empty:   {', '.join(f'{name} ({count} rows)' for name, count in empty.items())}"] if empty else []),
        f"  splits:  {', '.join(splits)}",
        f"  sample:  {json.dumps(table[0]) if table else '(empty)'}",
    ]
    return {"text": "\n".join(lines), **payload}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """The same work `explain` does, without printing the plan.

    A document is only valid against a model: which components exist, how
    wide each site is, whether a layer is in range and whether a metric
    column is one token are all questions about a checkpoint. `--engine
    nnterp` answers them from a meta shell, in under a second and with
    nothing downloaded but the config and the tokenizer, so there is no
    reason for a cheaper check that passes documents `explain` refuses.
    """
    _, built = _compile(args, **SHAPE_ONLY)
    return {
        "text": f"ok: {args.document} is a valid document, {_plural(len(built.steps), 'step')}",
        "ok": True,
        "steps": list(built.steps),
    }


def _compile(args: argparse.Namespace, **options: Any) -> tuple[Any, Any]:
    raw = _read(args.document)
    engine_class = ENGINES[args.engine][0]
    # The model is the same at every point of a sweep — a sweep may not touch
    # it — so the first point says what to load.
    engine = engine_class.load(Model.model_validate(sweep.points(raw)[0][1].get("model")), **options)
    return engine, plan_module.build_request(raw, args.data_root, engine)


def explain_verb(args: argparse.Namespace) -> dict[str, Any]:
    _, built = _compile(args, **SHAPE_ONLY)
    return {"text": explain(built), "steps": list(built.steps)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _, loading, remote = ENGINES[args.engine]
    if not remote:
        loading = {**loading, "device_map": args.device_map}
    engine, built = _compile(args, **loading)
    written = engine.execute(built, remote=remote, batch_size=args.batch_size).write(args.out)
    return {"text": "\n".join(str(path) for path in written), "written": [str(path) for path in written]}


def _plural(count: int, thing: str) -> str:
    return f"{count} {thing}" + ("" if count == 1 else "s")


def _read(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


VERBS = {
    "schema": schema,
    "vocab": vocab,
    "model": model,
    "tokens": tokens,
    "data": data,
    "validate": validate,
    "explain": explain_verb,
    "run": run,
}


if __name__ == "__main__":
    sys.exit(main())
