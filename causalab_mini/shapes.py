"""Names for the shapes that travel as plain tuples.

`tuple[int, ...]` means three different things in this project, and a reader of
a signature cannot tell which. These aliases are documentation with a type
checker attached; nothing here has a runtime effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: A window of absolute indices into the padded sequence, per row of a batch.
#: `((10,), (10,))` is one position per row — the unit window, which is what a
#: metric reads. `((8, 9, 10), (8, 9, 10))` is a three-token window. Every row's
#: window has the same width today; a ragged one is the same type with that
#: rule dropped, which is why the type is already per row.
Positions = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Anchor:
    """A run of tokens located in this row by its *text*.

    `variable` is the row's own value for a name: the `<column>_variables`
    sibling of the role's own field first, a top-level column of that name
    otherwise. That rule is why a pair-native document writes one spec and
    each role resolves its own text. `segment` is a run the *frame* located:
    a chat turn in the prompt frame, `eos` in the generated one.

    Both keys together is the composition — the variable's run searched
    *inside* the segment's — which is also how a value occurring twice
    becomes unique, and why there is no `occurrence` field.
    """

    #: pydantic validates this dataclass where a document names one, and
    #: reads its config off this attribute. A plain dict, so `shapes.py`
    #: stays a file the block can carry with nothing but the standard
    #: library behind it.
    __pydantic_config__ = {"extra": "forbid"}

    variable: str | None = None
    segment: Literal["system", "user", "assistant", "eos"] | None = None

    def __post_init__(self) -> None:
        if self.variable is None and self.segment is None:
            raise ValueError("an anchor names a variable, a segment, or both")


@dataclass(frozen=True)
class Where:
    """Where along the sequence, per row — a spec, never an integer.

    Three independent questions, one field each:

        frame                 which sequence: the prompt, or what was generated
        scope                 which run of it: the row's content, or an anchor's
        index/span/last/all   how much of that run

    Exactly one of index/span/last/all; negatives count from the end of the
    run. Resolution happens where the model is, per row, against its
    tokenizer, so `{"index": -1, "scope": {"variable": "entity"}}` is the
    last token of *this row's* entity — whatever text that is, and however
    many tokens it takes. That is the whole point: the spec is the same for
    every row and the integer is not.
    """

    __pydantic_config__ = {"extra": "forbid"}

    index: int | None = None
    span: tuple[int, int] | None = None
    last: int | None = None
    all: bool = False
    scope: Anchor | None = None
    frame: Literal["prompt", "generated"] = "prompt"

    def __post_init__(self) -> None:
        named = [key for key in ("index", "span", "last") if getattr(self, key) is not None]
        named += ["all"] if self.all else []
        if len(named) != 1:
            raise ValueError(f"a position names exactly one of index/span/last/all, got {named}")
        if self.last is not None and self.last <= 0:
            raise ValueError(f"last {self.last} is not a positive number of tokens")
        if self.span is not None:
            a, b = self.span
            if (a >= 0) == (b >= 0) and b <= a:
                raise ValueError(f"span {list(self.span)} is not a forward window")
        if self.frame == "prompt" and self.scope is not None and self.scope.segment == "eos":
            raise ValueError("segment 'eos' is a run of the generated frame, not the prompt")

    @property
    def width(self) -> int | None:
        """How many positions this names, knowable without a row — which is
        what lets the compiler check a write against its operand — or None
        when only the row can say. It is the width of the *cut*, so an
        anchored `{"index": -1}` is 1: whether a row has that one position
        at all is `ragged`'s question, not this one."""
        if self.index is not None:
            return 1
        if self.last is not None:
            return self.last
        if self.span is not None:
            a, b = self.span
            return b - a if (a >= 0) == (b >= 0) else None
        return None  # `all`

    @property
    def ragged(self) -> bool:
        """Whether a row may come back with a window the others do not have,
        so the gather is flat.

        A cut of varying width is one. So is any anchored run, *however
        narrow the cut*: a row whose anchor is not in its prompt has no
        window at all, and `{"index": -1, "scope": …}` is therefore one
        position on the rows that have it and none on the rows that do not.
        The two are different questions, which is why `width` answers the
        first and this the second.

        A continuation-frame spec is never ragged *at the address*: whatever
        it names, a decode step processes one position, and how much of the
        continuation the spec keeps is decided afterwards, over the steps.
        """
        return self.frame == "prompt" and (self.width is None or self.scope is not None)


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

    #: Where along the sequence, once a run has resolved `where` against its
    #: own tokenizer. **Empty in a fresh plan**: a spec is what a document
    #: carries and integers are what a row produces.
    positions: Positions = ()
    groups: int | None = None
    take: tuple[int, ...] | None = None
    #: Gathered flat, `(total, width)`, because the windows' widths vary by
    #: row — or because a row may have none. Decided by the *form*, over
    #: every row of the pass, and carried: a window of a ragged pass's rows
    #: can happen to be rectangular and must still come back flat.
    flat: bool = False
    #: The spec the positions come from. `None` only where a caller hands
    #: the tensor functions a bare `Positions`.
    where: Where | None = None
    #: Per row, the text this row's `where.scope.variable` binds to — the
    #: one input the resolver cannot find for itself, because it has no
    #: dataset. Empty for a spec with no variable anchor.
    anchors: tuple[str, ...] = ()


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
