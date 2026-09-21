"""Datasets, rows, splits.

A table is a JSON array of flat row objects and every row carries `split`, so a
split is a *column of one table* selected by value, not a second file. A dataset
ref is a relative path under the data root plus an optional `#split` fragment.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..shapes import ExampleIds

Row = dict[str, Any]

_STEP = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")


class DataError(ValueError):
    pass


def load(data_root: str | Path, ref: str) -> list[Row]:
    """`weekdays/data#train` -> the rows of <root>/weekdays/data.json whose
    `split` column is "train"."""
    path, _, split = ref.partition("#")
    table = json.loads((Path(data_root) / (path + ".json")).read_text())
    if not split:
        return table
    rows = [row for row in table if row.get("split") == split]
    if not rows:
        raise DataError(f"dataset ref {ref!r} selects no rows")
    return rows


def digest(rows: list[Row]) -> str:
    """The content digest of a table's rows, for an artifact's identity stamp:
    a rotation fitted against these rows is not the same artifact as one fitted
    against others."""
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def field_text(row: Row, field: str) -> str:
    """`input` -> the column; `counterfactual_inputs[0]` -> one entry of a
    list-valued column; `counterfactual_inputs_variables[0].entity` -> a key
    of a dict inside one. Dots walk into dicts, brackets into lists."""
    value: Any = row
    for step in field.split("."):
        match = _STEP.match(step)
        if match is None:
            raise DataError(f"field {field!r}: `column`, `column[i]` and `a.b` are the forms")
        key, indices = match.group(1), re.findall(r"\[(\d+)\]", match.group(2))
        if not isinstance(value, dict) or key not in value:
            raise DataError(f"field {field!r}: no such column {key!r}")
        value = value[key]
        for index in indices:
            value = value[int(index)]
    if not isinstance(value, str):
        raise DataError(f"field {field!r}: expected a string, got {type(value).__name__}")
    return value


def column(rows: list[Row], name: str) -> list[str]:
    """A metric's column, off the *base* rows. A row whose value is null or
    empty is an excluded measurement in causalab; nothing in this corpus has
    one, so we refuse rather than pretend to have eligibility machinery."""
    values = []
    for index, row in enumerate(rows):
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise DataError(
                f"row {index}: column {name!r} is missing or empty; per-row "
                "eligibility is not implemented"
            )
        values.append(value)
    return values


def example_ids(rows: list[Row]) -> ExampleIds:
    """The row's label: the `example_id` column if the table has one, else the
    zero-based row index as a string."""
    return tuple(
        str(row["example_id"]) if "example_id" in row else str(index)
        for index, row in enumerate(rows)
    )
