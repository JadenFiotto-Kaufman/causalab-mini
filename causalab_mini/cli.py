"""Entry point: document in, output files out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import plan as plan_module
from .plan import sweep
from .engine import NNterpEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalab-mini")
    parser.add_argument("document", help="a protocol_version 3 document")
    parser.add_argument("--data-root", default="documents/data")
    parser.add_argument("--out", default="out")
    parser.add_argument(
        "--remote",
        default="",
        help="'local' for nnsight's serverless dry run of the remote path, "
        "or 'true' to ship the session to NDIF",
    )
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.document).read_text())
    # The model is loaded once and every point of a sweep runs against it,
    # which is why a sweep may not touch the `model` section.
    engine = NNterpEngine.load(plan_module.Document.from_json(_first_point(raw)).model,
                               device_map=args.device_map)
    plan = plan_module.build_request(raw, args.data_root, engine)
    remote = True if args.remote == "true" else (args.remote or False)
    executed = engine.execute(plan, remote=remote)
    for path in executed.write(args.out):
        print(path)
    return 0


def _first_point(raw):
    """A swept document is not a `Document` until it is lowered, and the model
    is the same at every point, so the first one answers what to load."""
    return sweep.points(raw)[0][1]


if __name__ == "__main__":
    raise SystemExit(main())
