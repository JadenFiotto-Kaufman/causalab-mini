"""Names for the shapes that travel as plain tuples.

`tuple[int, ...]` means three different things in this project, and a reader of
a signature cannot tell which. These aliases are documentation with a type
checker attached; nothing here has a runtime effect.
"""

from __future__ import annotations

#: One absolute index into the padded sequence, per row of a batch.
Positions = tuple[int, ...]

#: One vocabulary id, per row of a batch — what a metric column resolved to.
TokenIds = tuple[int, ...]

#: One encoded prompt: the token ids of a single sequence, padded.
Tokens = tuple[int, ...]

#: A padded batch of encoded prompts, or its attention mask.
TokenRows = tuple[Tokens, ...]

#: One label, per row — the `example_id` column, or the row index as a string.
ExampleIds = tuple[str, ...]
