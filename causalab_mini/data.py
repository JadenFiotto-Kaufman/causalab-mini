"""Datasets, rows, splits.

A table is a JSON array of flat row objects and every row carries `split`, so a
split is a *column of one table* selected by value, not a second file. A dataset
ref is a relative path under the data root plus an optional `#split` fragment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_INDEXED_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")


class DataError(ValueError):
    pass


def load_rows(data_root, ref: str) -> list[dict]:
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


def field_text(row: dict, field: str) -> str:
    """`input` -> the column; `counterfactual_inputs[0]` -> one entry of a
    list-valued column. No deeper indexing exists."""
    match = _INDEXED_FIELD.match(field)
    if match is None:
        raise DataError(f"field {field!r}: only `column` and `column[i]` are implemented")
    column, index = match.group(1), match.group(2)
    if column not in row:
        raise DataError(f"field {field!r}: no such column")
    value = row[column]
    if index is not None:
        value = value[int(index)]
    if not isinstance(value, str):
        raise DataError(f"field {field!r}: expected a string, got {type(value).__name__}")
    return value


def column(rows: list[dict], name: str) -> list[str]:
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


def example_ids(rows: list[dict]) -> tuple[str, ...]:
    """The row's label: the `example_id` column if the table has one, else the
    zero-based row index as a string."""
    return tuple(
        str(row["example_id"]) if "example_id" in row else str(index)
        for index, row in enumerate(rows)
    )
