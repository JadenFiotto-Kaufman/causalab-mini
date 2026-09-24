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
    """`field_value`, refused unless it is a string."""
    value = field_value(row, field)
    if not isinstance(value, str):
        raise DataError(f"field {field!r}: expected a string, got {type(value).__name__}")
    return value


def field_value(row: Row, field: str) -> Any:
    """`input` -> the column; `counterfactual_inputs[0]` -> one entry of a
    list-valued column; `counterfactual_inputs_variables[0].entity` -> a key
    of a dict inside one. Dots walk into dicts, brackets into lists.

    Whatever is there, not necessarily a string: a role whose field is a
    *list of messages* is a conversation, and what a row holds is how a
    document says so."""
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
    return value


def variable_text(row: Row, field: str, name: str) -> str:
    """This row's own value for a variable name, for the role whose text is
    `field`.

    The protocol's rule, kept: the `<column>_variables` sibling of the role's
    own field first — `counterfactual_inputs[0]` and `entity` give
    `counterfactual_inputs_variables[0].entity` — and a top-level column of
    that name otherwise. That is what lets one position spec mean "this row's
    entity" in both prompts of a pair while each role resolves its own text.

    A name that is neither is a misspelling and is refused here, on the
    client, naming both places it looked.
    """
    head, *rest = field.split(".")
    match = _STEP.match(head)
    if match is None:
        raise DataError(f"field {field!r}: `column`, `column[i]` and `a.b` are the forms")
    sibling = ".".join([f"{match.group(1)}_variables{match.group(2)}", *rest, name])
    for path in (sibling, name):
        try:
            return field_text(row, path)
        except (DataError, IndexError, KeyError):
            continue
    raise DataError(
        f"variable {name!r}: this row has neither {sibling!r} nor a column {name!r}"
    )


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
