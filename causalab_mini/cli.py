"""Entry point: document in, output files out."""

from __future__ import annotations

import argparse

from . import document, model as model_module, output, plan as plan_module, run


def main(argv=None) -> int:
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

    doc = document.Document.load(args.document)
    model = model_module.load(doc.model, device_map=args.device_map)
    plan = plan_module.build(doc, args.data_root, model)
    remote = True if args.remote == "true" else (args.remote or False)
    results = run.execute(model, plan, remote=remote)
    for path in output.write_results(args.out, plan, results):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
