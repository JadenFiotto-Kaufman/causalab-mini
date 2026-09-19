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

from .plan import Plan, SaveFile


def write(plan: Plan, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [_file(plan, save, out) for save in plan.saves]
    for name, step in plan.steps.items():
        if isinstance(step, Plan):
            written.extend(write(step, out / name))
    return written


def _file(plan: Plan, save: SaveFile, out: Path) -> Path:
    path = out / save.file_path
    path.parent.mkdir(parents=True, exist_ok=True)
    value = plan.result(save.value)
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
