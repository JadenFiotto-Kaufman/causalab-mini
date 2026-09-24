"""Sweeps: one document that is several experiments.

`{"sweep": [a, b, c]}` at a field means the document is three **points** —
the same experiment three times, differing in that one field. It is the
protocol's spelling for "three experiments in a row", and the reason this
project's plans nest: a swept document compiles to a root plan whose children
are one plan per point.

A sweep is **lowered on the client, before anything is compiled**: the wrapper
is replaced by each of its values in turn, and each resulting document is
compiled on its own. So a point is an ordinary document in every way — its
own addresses, its own tokenization, its own digest — and nothing downstream
of `build_request` knows a sweep ever happened. That is why the engine did not change
to support this.

Several swept fields are their **cross product**, one point per combination,
labelled `k=8,seed=0` in the order the fields appear.

Two spellings beyond the literal list:

    {"sweep": {"range": [0, 16]}}         0, 1, … 15  — `[start, stop]` or
                                          `[start, stop, step]`, as Python's
    {"sweep": [-4, -1], "as": "pos"}      a **named axis**: every wrapper
                                          with the same `as` moves together

The named axis exists because a cross product is sometimes the wrong thing.
A patch reads at a position and writes at the same one; swept separately
those are N² points, most of them reading one place and writing another.
Naming the axis makes them one coordinate, labelled by its name.

Everything else the protocol allows here (`at_once`, cohorts, swept bundles)
is refused by name.
"""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any

Json = dict[str, Any]

#: The path to a sweep wrapper, as the keys and indices that reach it.
Path = tuple[str | int, ...]


class SweepError(ValueError):
    pass


def points(raw: Json) -> tuple[tuple[str, Json], ...]:
    """The documents one document is, as (label, document) pairs.

    An unswept document is one point labelled `""`, which is how a caller
    tells the two apart without asking.
    """
    found = wrappers(raw)
    if not found:
        return (("", raw),)
    axes = []
    for path in found:
        if path and path[0] in ("model", "header"):
            # The engine loads the model once and every point runs against
            # it, so a point may not ask for a different one.
            raise SweepError(
                f"{_spell(path)} is swept; a sweep may not change the model or the "
                "header — those are the identity of the run, not a coordinate in it"
            )
        wrapper = _at(raw, path)
        values = _values(wrapper["sweep"], path)
        if not values:
            raise SweepError(f"{_spell(path)}: a sweep of nothing")
        name = wrapper.get("as")
        linked = next((axis for axis in axes if name is not None and axis.name == name), None)
        if linked is None:
            axes.append(_Axis(name, [path], [values]))
            continue
        if len(values) != len(linked.values[0]):
            raise SweepError(
                f"{_spell(path)}: axis {name!r} has {len(linked.values[0])} values at "
                f"{_spell(linked.paths[0])} and {len(values)} here; fields that move "
                "together need a value each per point"
            )
        linked.paths.append(path)
        linked.values.append(values)
    points = []
    for combination in itertools.product(*(range(len(axis.values[0])) for axis in axes)):
        point, labels = raw, []
        for axis, index in zip(axes, combination):
            for path, values in zip(axis.paths, axis.values):
                point = _substitute(point, path, values[index])
            labels.append(_label(axis.paths[0], axis.values[0][index], axis.name))
        points.append((",".join(labels), point))
    return tuple(points)


class _Axis:
    """One coordinate of a sweep: usually one field, or — when named —
    every field that shares the name, moving together."""

    def __init__(self, name: str | None, paths: list[Path], values: list[list[Any]]) -> None:
        self.name, self.paths, self.values = name, paths, values


def _values(spelled: Any, path: Path) -> list[Any]:
    """A sweep's values: the literal list, or the range it names."""
    if isinstance(spelled, list):
        return spelled
    if isinstance(spelled, dict) and set(spelled) == {"range"}:
        bounds = spelled["range"]
        if (
            isinstance(bounds, list)
            and len(bounds) in (2, 3)
            and all(isinstance(one, int) and not isinstance(one, bool) for one in bounds)
            and (len(bounds) == 2 or bounds[2] != 0)
        ):
            return list(range(*bounds))
        raise SweepError(
            f"{_spell(path)}: a range is [start, stop] or [start, stop, step] in integers, "
            f"got {bounds!r}"
        )
    raise SweepError(
        f"{_spell(path)}: a sweep is a list of values or {{\"range\": [start, stop]}}, "
        f"got {spelled!r}"
    )


def wrappers(node: Any, path: Path = ()) -> list[Path]:
    """Every `{"sweep": …}` in the document, by path, in reading order.

    The document calls this to refuse one that still has one: a `Spec` is one
    point, so a wrapper reaching it has not been lowered.
    """
    if isinstance(node, dict):
        if "sweep" in node and set(node) <= {"sweep", "as"}:
            return [path]
        return [one for key, value in node.items() for one in wrappers(value, (*path, key))]
    if isinstance(node, list):
        return [one for index, value in enumerate(node) for one in wrappers(value, (*path, index))]
    return []


def _at(node: Any, path: Path) -> Any:
    for step in path:
        node = node[step]
    return node


def _substitute(raw: Json, path: Path, value: Any) -> Json:
    """`raw` with the wrapper at `path` replaced by one of its values."""
    point = copy.deepcopy(raw)
    parent = _at(point, path[:-1])
    parent[path[-1]] = value
    return point


def _label(path: Path, value: Any, axis: str | None = None) -> str:
    """What this point is called — in the plan tree, and as a directory on
    disk. The swept field's own name and the value it took: `pos=-1`."""
    name = axis or next((step for step in reversed(path) if isinstance(step, str)), "point")
    if isinstance(value, list) and len(value) == 1:
        value = value[0]  # a one-layer band sweeps as its layer
    if isinstance(value, (str, int, float, bool)) or value is None:
        return f"{name}={value}"
    return f"{name}={json.dumps(value, separators=(',', ':'))}"


def _spell(path: Path) -> str:
    return ".".join(str(step) for step in path)
