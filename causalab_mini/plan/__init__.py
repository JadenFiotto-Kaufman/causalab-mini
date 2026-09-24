"""The request, as pure data: a document, and the plan it compiles to.

Two frozen descriptions and one compiler between them.

* `spec.py` is what was *written* — the document, validated: its
  `steps` are what runs, in order, one plan step each, and a step's name is
  how everything after it reaches what it produced. Everything it does not
  run is refused by name.
* `plan.py` is what will *run* — a tree of steps, strings and integers, with
  every decision already taken.
* `build.py` is the only thing that sits between them, and it is the only place
  in the project where the tokenizer, the rows on disk and the model's own
  shapes are consulted. `build_request` is its entry point: it lowers any
  sweep first, so a swept document becomes a root plan with one child plan
  per point.
* `sweep.py` does that lowering, on the raw JSON, before anything is compiled.
* `write.py` puts what a run produced on disk.

Nothing in here knows how to execute anything: that is an engine's job, and a
plan that knew would only work on one engine.
"""

from .build import build_request, build_spec
from .plan import (
    FeaturizerOp,
    Featurizers,
    Fit,
    Forward,
    Generate,
    Metric,
    Plan,
    PlanError,
    ReadOp,
    Reduce,
    SaveFile,
    Step,
    Tap,
    WriteOp,
    children,
    steps_of,
    window,
)

__all__ = [
    "FeaturizerOp",
    "Featurizers",
    "Fit",
    "Forward",
    "Generate",
    "Metric",
    "Plan",
    "PlanError",
    "ReadOp",
    "Reduce",
    "SaveFile",
    "Step",
    "Tap",
    "WriteOp",
    "build_request",
    "build_spec",
    "children",
    "steps_of",
]
