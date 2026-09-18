"""Write the result files.

`save` is the complete manifest of everything that leaves the run: nothing is
written that is not listed. A metric table is a JSON array of row objects, one
file per metric, with the labels repeated on every row so that `jq` and a human
can both read it.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_results(out_dir, plan, results) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for save in plan.saves:
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
        path = out / save.file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1) + "\n")
        written.append(path)
    return written
