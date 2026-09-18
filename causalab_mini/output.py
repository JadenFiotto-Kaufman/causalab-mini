"""Write the result files.

`save` is the complete manifest of everything that leaves the run: nothing is
written that is not listed. A metric table is a JSON array of row objects, one
file per metric, with the labels repeated on every row so that `jq` and a human
can both read it. A trained featurizer is a safetensors bundle holding its one
`weight` slot, with its identity stamped into the header — the thing a later
document's `file_path` load would check before trusting the rotation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import save_file

from .plan import Plan


def write_results(out_dir: str | Path, plan: Plan, results: dict[str, Any]) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for save in plan.saves:
        path = out / save.file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if save.file_path.endswith(".safetensors"):
            # One auto-declared slot per featurizer, named `<featurizer>.weight`.
            save_file(
                {"weight": results[save.value].contiguous()},
                str(path),
                metadata=dict(save.identity),
            )
            written.append(path)
            continue
        values = results[save.value].tolist()
        rows = [
            {
                "example_id": example_id,
                "metric": save.value,
                "value": float(value),
                "eligible": True,
                "unit": save.unit,
                "estimand_version": save.estimand_version,
                "produced_by": save.produced_by,
            }
            for example_id, value in zip(save.example_ids, values)
        ]
        path.write_text(json.dumps(rows, indent=1) + "\n")
        written.append(path)
    return written
