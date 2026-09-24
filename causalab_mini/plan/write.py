"""Write what the run produced: the save manifest, and nothing else.

`Plan.saves` is the complete list of what leaves a run — nothing is written
that is not listed. A metric table is a JSON array of row objects, one file per
metric, with the labels repeated on every row so that `jq` and a human can both
read it. A trained featurizer is a safetensors bundle holding its one `weight`
slot, with its identity stamped into the header: the thing a later document's
`file_path` load would check before trusting the rotation.

**A nested plan writes below its parent**, in a directory named by its step
name, so a plan's path in the tree is its path on disk. A one-plan document has
its saves on the root and writes them straight into `out`, which is why nesting
cost the existing documents nothing.

A metric row carries where its number was read, why it was nowhere on a row
that has none, and what the window says — the three the run records per op
(`engine/steps.py`), for this metric's own read. A **write's** provenance has
no table to live in: it is in the returned plan, at
`step.results["positions"][<write>]`, and the case that matters on disk is the
one that never gets there, because a write that could not land refuses the run
and names the rows and the reason. Letting a save name `positions` would give
it a file through the mechanism that already exists; it is not built, because
a step with no dynamic position records nothing and the save would then be a
refusal the document could not have predicted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from safetensors.torch import save_file

from .plan import Plan, SaveFile, Step, children


def write(step: Step, out_dir: str | Path) -> list[Path]:
    """Every save in this subtree, written.

    A **plan** in a plan gets a directory of its own, so a swept point's
    files land under `pos=-1/`. Any other step writes into its enclosing
    plan's directory — a fit's evaluation too, though it is a plan of steps:
    it is where the fit's held-out numbers are, and giving it a folder would
    say they were somewhere else.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [_file(step, save, out) for save in step.saves]
    if isinstance(step, Plan):
        # A plan's directory carries the experiment that produced it and, on
        # the root, what ran it — so a result is never a file with no way back.
        if step.source is not None:
            written.append(_json(out / "document.json", step.source))
        if step.provenance:
            written.append(_json(out / "run.json", step.provenance))
    for name, child in children(step):
        written.extend(write(child, out / name if isinstance(child, Plan) and isinstance(step, Plan) else out))
    return written


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def _file(step: Step, save: SaveFile, out: Path) -> Path:
    path = out / save.file_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if save.value.endswith("/"):
        # A prefix: every result under it, one tensor each, keyed by the
        # rest of its name — a fit's record is `train/loss` and `train/eval`.
        tensors = {
            name[len(save.value) :]: one.contiguous()
            for name, one in step.results.items()
            if name.startswith(save.value)
        }
        save_file(tensors, str(path), metadata=save.identity)
        return path
    # A save names a result of the step it sits on. No search, so no
    # ambiguity: two steps may both produce `iia` and each saves its own.
    value = step.results[save.value]
    if save.file_path.endswith(".safetensors"):
        # One auto-declared slot per featurizer, named `<featurizer>.weight`.
        save_file({"weight": value.contiguous()}, str(path), metadata=save.identity)
        return path
    # The result holds one value per eligible row; an excluded measurement
    # is still a row of the table, with no value and `eligible: false` — so
    # it can never be read as a zero, or silently shorten a denominator.
    #
    # Which rows those are has two halves. The compiled `eligible` is the
    # column half — whether the data had an answer to score. A run that
    # anchored a position to text also reports which rows it could place,
    # and that list is already the intersection, so it wins where it exists.
    run = step.results.get("eligible", {}).get(save.value)
    eligible = run or save.eligible or (True,) * len(save.example_ids)
    where = step.results.get("positions", {}).get(save.of, {})
    rows = []
    # a metric of a read at every layer is a row of scores per layer, and a
    # table row per layer and example, which says its layer
    for layer, scores in zip(save.layers or (None,), value if save.layers else [value]):
        rows += _rows(save, scores, eligible, where, {} if layer is None else {"layer": layer})
    path.write_text(json.dumps(rows, indent=1) + "\n")
    return path


def _rows(save: SaveFile, scores: Any, eligible: tuple[bool, ...], where: dict[str, Any], layer: dict[str, int]) -> list[dict[str, Any]]:
    """One table row per example: its number when it was scored, and where."""
    numbers = iter(scores.tolist())
    rows = []
    for index, (example_id, included) in enumerate(zip(save.example_ids, eligible)):
        number = next(numbers) if included else None
        rows.append(
            {
                "example_id": example_id,
                "metric": save.value,
                **layer,
                # JSON has no NaN or Infinity: `json.dumps` would emit a bare
                # `NaN`, which Python reads back and a strict parser refuses.
                "value": float(number) if number is not None and math.isfinite(number) else None,
                "eligible": included,
                # where the number was read, why it was nowhere, and what
                # the window it came from actually says — the three the run
                # records per op, printed for this metric's own read
                "positions": list(where["rows"][index]) if where else None,
                "reason": where["reason"][index] if where else "",
                "tokens": where["tokens"][index] if where else "",
                "unit": save.unit,
                "estimand_version": save.estimand_version,
                "produced_by": save.produced_by,
            }
        )
    return rows
