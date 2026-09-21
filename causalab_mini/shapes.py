"""Names for the shapes that travel as plain tuples.

`tuple[int, ...]` means three different things in this project, and a reader of
a signature cannot tell which. These aliases are documentation with a type
checker attached; nothing here has a runtime effect.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A window of absolute indices into the padded sequence, per row of a batch.
#: `((10,), (10,))` is one position per row — the unit window, which is what a
#: metric reads. `((8, 9, 10), (8, 9, 10))` is a three-token window. Every row's
#: window has the same width today; a ragged one is the same type with that
#: rule dropped, which is why the type is already per row.
Positions = tuple[tuple[int, ...], ...]



@dataclass(frozen=True)
class Selection:
    """Where in a tensor: which positions along the sequence and, optionally,
    which part of the feature axis.

    The feature half is one rule — view the features as `(groups, -1)` and
    keep `take` of the groups — and what the groups *are* is the site's
    business: attention heads (`groups` = the head count), single units of an
    MLP (`groups` = the width), experts. A per-group tensor is handed on
    flat, `(…, len(take) · per_group)`, however the model holds it, so
    nothing downstream of `gather` and `scatter` knows groups exist.

    `groups` alone, with no `take`, is every group: it only says that a
    tensor the model holds as `(groups, per_group)` is handed on flat.
    A bare `Positions` is the selection of every feature at those positions,
    and the tensor functions accept it as such.
    """

    positions: Positions
    groups: int | None = None
    take: tuple[int, ...] | None = None
    #: Gathered flat, `(total, width)`, because the windows' widths vary by
    #: row. Decided once, over *every* row of the pass, and carried — not
    #: re-derived from the positions in hand, because a window of a ragged
    #: pass's rows can happen to be rectangular and must still come back flat.
    flat: bool = False


#: One integer per row: a row's content start, a row's content end.
Indices = tuple[int, ...]

#: One vocabulary id, per row of a batch — what a metric column resolved to.
TokenIds = tuple[int, ...]

#: One encoded prompt: the token ids of a single sequence, padded.
Tokens = tuple[int, ...]

#: A padded batch of encoded prompts, or its attention mask.
TokenRows = tuple[Tokens, ...]

#: One label, per row — the `example_id` column, or the row index as a string.
ExampleIds = tuple[str, ...]
