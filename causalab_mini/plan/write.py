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
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors.torch import save_file

from .plan import Plan, SaveFile, Step, children


def write(step: Step, out_dir: str | Path) -> list[Path]:
    """Every save in this subtree, written.

    A **plan** gets a directory of its own, so a swept point's files land
    under `pos=-1/`. Any other step writes into its enclosing plan's
    directory: a fit's eval pass is a place, not a place*s*, and giving it a
    folder would say otherwise.
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
        written.extend(write(child, out / name if isinstance(child, Plan) else out))
    return written


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def _file(step: Step, save: SaveFile, out: Path) -> Path:
    path = out / save.file_path
    path.parent.mkdir(parents=True, exist_ok=True)
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
    numbers = iter(value.tolist())
    rows = []
    for index, (example_id, included) in enumerate(zip(save.example_ids, eligible)):
        number = next(numbers) if included else None
        rows.append(
            {
                "example_id": example_id,
                "metric": save.value,
                # JSON has no NaN or Infinity: `json.dumps` would emit a bare
                # `NaN`, which Python reads back and a strict parser refuses.
                "value": float(number) if number is not None and math.isfinite(number) else None,
                "eligible": included,
                # where the number was read, and why it was nowhere
                "positions": list(where["rows"][index]) if where else None,
                "reason": where["reason"][index] if where else "",
                "unit": save.unit,
                "estimand_version": save.estimand_version,
                "produced_by": save.produced_by,
            }
        )
    path.write_text(json.dumps(rows, indent=1) + "\n")
    return path
