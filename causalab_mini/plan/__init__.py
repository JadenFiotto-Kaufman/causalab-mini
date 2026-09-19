"""The request, as pure data: a document, and the plan it compiles to.

Two frozen descriptions and one compiler between them.

* `document.py` is what was *written* — the protocol JSON, validated, with
  every feature this slice does not run refused by name at load.
* `plan.py` is what will *run* — a tree of steps, strings and integers, with
  every decision already taken.
* `build.py` is the only thing that sits between them, and it is the only place
  in the project where the tokenizer, the rows on disk and the model's own
  shapes are consulted.
* `write.py` puts what a run produced on disk.

Nothing in here knows how to execute anything: that is an engine's job, and a
plan that knew would only work on one engine.
"""

from .build import build
from .document import Document, DocumentError
from .plan import (
    FeaturizerOp,
    Featurizers,
    Fit,
    Forward,
    MetricOp,
    Observe,
    Plan,
    PlanError,
    ReadOp,
    SaveFile,
    Step,
    Tap,
    Weights,
    WriteOp,
    steps_of,
)

__all__ = [
    "Document",
    "DocumentError",
    "FeaturizerOp",
    "Featurizers",
    "Fit",
    "Forward",
    "MetricOp",
    "Observe",
    "Plan",
    "PlanError",
    "ReadOp",
    "SaveFile",
    "Step",
    "Tap",
    "Weights",
    "WriteOp",
    "build",
    "steps_of",
]
