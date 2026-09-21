"""The request, as pure data: a document, and the plan it compiles to.

Two frozen descriptions and one compiler between them.

* `document.py` is what was *written* — the protocol JSON, validated, with
  every feature this slice does not run refused by name at load.
* `plan.py` is what will *run* — a tree of steps, strings and integers, with
  every decision already taken.
* `build.py` is the only thing that sits between them, and it is the only place
  in the project where the tokenizer, the rows on disk and the model's own
  shapes are consulted. `build_request` is its entry point: it lowers any
  sweep first, so a swept document becomes a root plan with one child plan
  per point.
* `sweep.py` does that lowering, on the raw JSON, before anything is compiled.
* `write.py` puts what a run produced on disk.
* `spec.py` is a **second** authoring format, shaped like the plan itself:
  its `steps` are the plan's steps and a save sits on the step whose result
  it names. Both formats compile through the same helpers, so they cannot
  drift into producing different plans.

Nothing in here knows how to execute anything: that is an engine's job, and a
plan that knew would only work on one engine.
"""

from .build import build, build_request, build_spec
from .document import Document, DocumentError
from .plan import (
    FeaturizerOp,
    Featurizers,
    Fit,
    Forward,
    MetricOp,
    Observe,
    OutputOp,
    Plan,
    PlanError,
    ReadOp,
    SaveFile,
    Step,
    Tap,
    Weights,
    WriteOp,
    children,
    steps_of,
    window,
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
    "OutputOp",
    "Plan",
    "PlanError",
    "ReadOp",
    "SaveFile",
    "Step",
    "Tap",
    "Weights",
    "WriteOp",
    "build",
    "build_request",
    "build_spec",
    "children",
    "steps_of",
]
