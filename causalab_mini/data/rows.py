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
    root, file = Path(data_root).resolve(), (Path(data_root) / (path + ".json")).resolve()
    if not file.is_relative_to(root):
        raise DataError(f"dataset ref {ref!r} resolves to {file}, outside the data root {root}")
    if not file.is_file():
        near = sorted(str(one.relative_to(root).with_suffix("")) for one in root.glob("*/*.json"))
        raise DataError(
            f"dataset ref {ref!r}: no table at {file}. A ref is `<dir>/<file>` under the data root, "
            f"without `.json`, optionally `#<split>`. Here: {near}"
        )
    table = json.loads(file.read_text())
    if not isinstance(table, list) or not table:
        raise DataError(f"dataset ref {ref!r} has no rows")
    ids = [str(row["example_id"]) for row in table if isinstance(row, dict) and "example_id" in row]
    if len(set(ids)) != len(ids):
        raise DataError(f"dataset ref {ref!r}: `example_id` values repeat, so a row could not be told from another")
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
            if not isinstance(value, list) or int(index) >= len(value):
                raise DataError(
                    f"field {field!r}: index {index} of column {key!r}, which has "
                    f"{len(value) if isinstance(value, list) else 'no'} entries on this row"
                )
            value = value[int(index)]
    if not isinstance(value, str):
        raise DataError(f"field {field!r}: expected a string, got {type(value).__name__}")
    return value


def column(rows: list[Row], name: str) -> list[str | None]:
    """A metric's column, off the *base* rows. A row whose value is null or
    empty is an **excluded measurement** — None here — which is not a zero:
    the metric is not computed for it, and its row in the table says so. A
    column no row has at all is a misspelling, and is refused."""
    if not any(name in row for row in rows):
        raise DataError(f"no row has a column {name!r}")
    values: list[str | None] = []
    for index, row in enumerate(rows):
        value = row.get(name)
        if value is not None and not isinstance(value, str):
            raise DataError(f"row {index}: column {name!r}: expected a string, got {type(value).__name__}")
        values.append(value if value and value.strip() else None)
    return values


def eligible(rows: list[Row], names: tuple[str, ...]) -> tuple[bool, ...]:
    """Which rows a metric over these columns can be computed for: the ones
    where every column has a value."""
    columns = [column(rows, name) for name in names]
    return tuple(all(one[index] is not None for one in columns) for index in range(len(rows)))


def example_ids(rows: list[Row]) -> ExampleIds:
    """The row's label: the `example_id` column if the table has one, else the
    zero-based row index as a string."""
    return tuple(
        str(row["example_id"]) if "example_id" in row else str(index)
        for index, row in enumerate(rows)
    )
