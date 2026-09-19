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
    for name, child in children(step):
        written.extend(write(child, out / name if isinstance(child, Plan) else out))
    return written


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
    rows = [
        {
            "example_id": example_id,
            "metric": save.value,
            # JSON has no NaN or Infinity: `json.dumps` would emit a bare
            # `NaN`, which Python reads back and a strict parser refuses.
            "value": float(number) if math.isfinite(number) else None,
            "eligible": True,
            "unit": save.unit,
            "estimand_version": save.estimand_version,
            "produced_by": save.produced_by,
        }
        for example_id, number in zip(save.example_ids, value.tolist())
    ]
    path.write_text(json.dumps(rows, indent=1) + "\n")
    return path
