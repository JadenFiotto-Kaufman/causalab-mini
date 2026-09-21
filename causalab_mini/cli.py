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
from typing import Any

from . import address, plan as plan_module
from .data import rows as rows_module
from .engine import NNterpEngine
from .engine.engines.hooks import HooksEngine
from .ops import featurizer, intervene, metrics
from .plan import document, sweep
from .plan.explain import explain
from .plan.spec import METRIC_COLUMNS, Spec

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalab-mini")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="verb", required=True)

    sub.add_parser("schema", help="the JSON Schema of a plan-shaped document")
    sub.add_parser("vocab", help="components, mechanisms, featurizer kinds, metric kinds, position forms")

    one = sub.add_parser("model", help="what a model looks like: layers, widths, which components resolve")
    one.add_argument("key")
    one.add_argument("--revision", default="main")
    one.add_argument("--dtype", default="fp32", choices=("fp32", "bf16"))
    one.add_argument("--engine", default="nnterp", choices=list(ENGINES))

    one = sub.add_parser("tokens", help="whether each string is one token — a metric column must be")
    one.add_argument("key")
    one.add_argument("text", nargs="+")
    one.add_argument("--revision", default="main")

    one = sub.add_parser("data", help="a dataset ref: columns, rows, a sample")
    one.add_argument("ref")
    one.add_argument("--data-root", default="documents/data")

    for verb, help_text in (
        ("validate", "check a document; no model is loaded"),
        ("explain", "compile a document without weights and print the plan"),
        ("run", "execute a document and write its outputs"),
    ):
        one = sub.add_parser(verb, help=help_text)
        one.add_argument("document", help="a plan-shaped document (it has `steps`), or a protocol_version 3 one")
        one.add_argument("--data-root", default="documents/data")
        one.add_argument("--engine", default="nnterp", choices=list(ENGINES))
        if verb == "run":
            one.add_argument("--out", default="out")
            one.add_argument("--device-map", default="auto", help="ignored by ndif, whose weights are the server's")

    args = parser.parse_args(argv)
    result = VERBS[args.verb](args)
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
        "components": address.describe(),
        "mechanisms": sorted(intervene.MECHANISMS),
        "featurizer_kinds": sorted(featurizer.KINDS),
        "metric_kinds": {kind: list(columns) for kind, columns in METRIC_COLUMNS.items()},
        "position_forms": ["an integer (negative counts from the end)", "{\"index\": i}"],
        "units": {kind: {"unit": unit, "estimand_version": version} for kind, (unit, version) in metrics.UNITS.items()},
    }
    lines = ["components:"]
    for name, entry in payload["components"].items():
        kind = "interior" if entry["interior"] else f"{entry['side']} of {entry['path']}"
        lines.append(f"  {name:18s} {kind}")
    lines.append(f"mechanisms:       {', '.join(payload['mechanisms'])}")
    lines.append(f"featurizer kinds: {', '.join(payload['featurizer_kinds'])} (plus 'identity', never declared)")
    lines.append("metric kinds:     " + ", ".join(f"{k}({', '.join(v)})" for k, v in payload["metric_kinds"].items()))
    lines.append("position forms:   " + "; ".join(payload["position_forms"]))
    return {"text": "\n".join(lines), **payload}


def model(args: argparse.Namespace) -> dict[str, Any]:
    spec = document.ModelSpec(args.key, args.revision, args.dtype)
    engine = ENGINES[args.engine][0].load(spec, **SHAPE_ONLY)
    components: dict[str, Any] = {}
    for name, entry in address.describe().items():
        layer = 0 if entry["layered"] else None
        try:
            located = engine.locate(name, layer)
            width: Any
            try:
                width = engine.width(located)
            except Exception:
                width = None
            components[name] = {"resolves": True, "width": width, "op": located.op}
        except Exception as refusal:
            components[name] = {"resolves": False, "why": str(refusal).splitlines()[0]}
    payload = {
        "key": args.key,
        "engine": args.engine,
        "num_layers": engine.num_layers,
        "padding_side": getattr(engine.tokenizer, "padding_side", None),
        "components": components,
    }
    lines = [f"{args.key} via {args.engine}: {payload['num_layers']} layers, padding_side={payload['padding_side']}"]
    for name, entry in components.items():
        if entry["resolves"]:
            width = f"width={entry['width']}" if entry["width"] is not None else "no width (no featurizer here)"
            lines.append(f"  {name:18s} ok   {width}" + (f"  op={entry['op']}" if entry["op"] else ""))
        else:
            lines.append(f"  {name:18s} --   {entry['why']}")
    return {"text": "\n".join(lines), **payload}


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
    payload = {"ref": args.ref, "rows": len(table), "columns": columns, "splits": splits, "sample": table[:1]}
    lines = [
        f"{args.ref}: {len(table)} rows",
        f"  columns: {', '.join(columns)}",
        f"  splits:  {', '.join(splits)}",
        f"  sample:  {json.dumps(table[0]) if table else '(empty)'}",
    ]
    return {"text": "\n".join(lines), **payload}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    raw = _read(args.document)
    if "steps" in raw:
        Spec.model_validate(raw)
        shape = "plan-shaped"
    else:
        for _, point in sweep.points(raw):
            document.Document.from_json(point)
        shape = "protocol"
    return {"text": f"ok: {args.document} is a valid {shape} document", "ok": True, "format": shape}


def _compile(args: argparse.Namespace, **options: Any) -> tuple[Any, Any]:
    raw = _read(args.document)
    engine_class = ENGINES[args.engine][0]
    if "steps" in raw:
        spec = Spec.model_validate(raw)
        engine = engine_class.load(spec.model, **options)
        return engine, plan_module.build_spec(spec, args.data_root, engine)
    first = sweep.points(raw)[0][1]
    engine = engine_class.load(document.Document.from_json(first).model, **options)
    return engine, plan_module.build_request(raw, args.data_root, engine)


def explain_verb(args: argparse.Namespace) -> dict[str, Any]:
    _, built = _compile(args, **SHAPE_ONLY)
    return {"text": explain(built), "steps": list(built.steps)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _, loading, remote = ENGINES[args.engine]
    if not remote:
        loading = {**loading, "device_map": args.device_map}
    engine, built = _compile(args, **loading)
    written = engine.execute(built, remote=remote).write(args.out)
    return {"text": "\n".join(str(path) for path in written), "written": [str(path) for path in written]}


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
