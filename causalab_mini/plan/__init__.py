"""The request, as pure data: a document, and the plan it compiles to.

Two frozen descriptions and one compiler between them.

* `document.py` is what was *written* — the protocol JSON, validated, with
  every feature this slice does not run refused by name at load.
* `plan.py` is what will *run* — strings and integers in forward order, with
  every decision already taken.
* `build.py` is the only thing that sits between them, and it is the only place
  in the project where the tokenizer, the rows on disk and the model's own
  shapes are consulted.

Nothing in here imports torch or nnsight. A plan is data; the session is what
turns it into tensors.
"""

from .build import build
from .document import Document, DocumentError
from .plan import (
    FeaturizerOp,
    Forward,
    MetricOp,
    Plan,
    PlanError,
    ReadOp,
    SaveFile,
    Tap,
    TrainPlan,
    WriteOp,
)

__all__ = [
    "Document",
    "DocumentError",
    "FeaturizerOp",
    "Forward",
    "MetricOp",
    "Plan",
    "PlanError",
    "ReadOp",
    "SaveFile",
    "Tap",
    "TrainPlan",
    "WriteOp",
    "build",
]
