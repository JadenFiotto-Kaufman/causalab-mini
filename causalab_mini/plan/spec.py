"""The authored document, shaped like the plan it compiles to.

The protocol's own JSON (`document.py`) is a flat description of one
experiment: sites, reads, writes, metrics and one `save` list, with the order
of execution implied. This is the other way round — **the document is the
tree**. Its `steps` are the plan's steps, in order, and a `save` sits on the
step whose result it names. Nothing needs a naming convention, because two
steps that both produce `iia` are two different places.

What is *not* here, deliberately: the resolved values. A plan holds
`pos=(10, 10)` and token ids, and this holds `pos: -1` and a column name,
because resolving them is what a compiler does against a loaded model. So
there are still two types, and the honest claim is that their *shapes* now
match — not that a document is a plan.

The three sections above `steps` are the vocabulary the steps share:

    roles         which column of a row each input role reads
    sites         the places a read or a write may name
    featurizers   the parameter sets, by name
    intervention  the experiment itself: reads, writes, models, metrics

and each step says which rows it runs over and what it writes out.

pydantic does two things a hand-written checker was doing badly: `extra=
"forbid"` refuses an unknown key *anywhere* with the path to it, which is
the catch-all `document.py` needed four bugs to learn it wanted, and the
model dump is a JSON Schema — which the protocol itself does not have.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

class Node(BaseModel):
    """Every node refuses a key it does not know, and says where it was."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------- #
# the shared vocabulary
# --------------------------------------------------------------------- #


class Header(Node):
    description: str = ""


class Model(Node):
    key: str
    revision: str
    dtype: Literal["fp32", "bf16"]


class Role(Node):
    #: The column a row's text comes from, e.g. `counterfactual_inputs[0]`.
    field: str


class Site(Node):
    component: str
    layers: list[int] | None = None

    @model_validator(mode="after")
    def _one_layer(self) -> "Site":
        if self.layers is not None and len(self.layers) != 1:
            raise ValueError(
                "layers must be a one-element band; a band spanning several "
                "layers is one address and is not implemented"
            )
        return self


class Featurizer(Node):
    kind: Literal["subspace"]
    k: int = Field(gt=0)
    parametrization: Literal["cayley"]
    #: The draw the initial basis comes from. Absent, the fit's seed, or 0.
    seed: int | None = None


class Read(Node):
    site: str
    pos: int
    model: str = "original"
    input: str
    featurizer: str = "identity"


class Write(Node):
    site: str
    pos: int
    #: `{"swap": "<read name>"}` — the mechanism and its operand.
    do: dict[str, str]
    featurizer: str = "identity"

    @model_validator(mode="after")
    def _one_mechanism(self) -> "Write":
        if len(self.do) != 1:
            raise ValueError(f"a write does exactly one thing, got {sorted(self.do)}")
        (mechanism,) = self.do
        if mechanism != "swap":
            raise ValueError(f"mechanism {mechanism!r} is not implemented (this slice has swap)")
        return self

    @property
    def mechanism(self) -> str:
        return next(iter(self.do))

    @property
    def operand(self) -> str:
        return next(iter(self.do.values()))


class IntervenedModel(Node):
    input: str
    writes: list[str]


class Match(Node):
    kind: Literal["match"]
    of: str
    expected: str
    token_form: Literal["space_prefixed"] = "space_prefixed"


class LogitDiff(Node):
    kind: Literal["logit_diff"]
    of: str
    a: str
    b: str
    token_form: Literal["space_prefixed"] = "space_prefixed"


class CrossEntropy(Node):
    kind: Literal["cross_entropy"]
    of: str
    target: str
    token_form: Literal["space_prefixed"] = "space_prefixed"


class TokenLogit(Node):
    kind: Literal["token_logit"]
    of: str
    token: str
    token_form: Literal["space_prefixed"] = "space_prefixed"


for _cls, _columns in (
    (Match, ("expected",)),
    (LogitDiff, ("a", "b")),
    (CrossEntropy, ("target",)),
    (TokenLogit, ("token",)),
):
    # The data columns this kind names, in the order `ops.metrics.compute`
    # takes them — the one thing the compiler needs and the shape of the
    # class already says.
    _cls.columns = property(  # type: ignore[attr-defined]
        lambda self, _columns=_columns: tuple(getattr(self, one) for one in _columns)
    )


Metric = Annotated[
    Union[Match, LogitDiff, CrossEntropy, TokenLogit], Field(discriminator="kind")
]

#: Which of a metric's own fields name data columns, in the order
#: `ops.metrics.compute` takes them.
METRIC_COLUMNS = {
    "match": ("expected",),
    "logit_diff": ("a", "b"),
    "cross_entropy": ("target",),
    "token_logit": ("token",),
}


class Intervention(Node):
    """The experiment, declared once. Every step runs *this*, over its own
    rows — which is why a fit and the run it scores cannot drift apart."""

    reads: dict[str, Read]
    writes: dict[str, Write] = Field(default_factory=dict)
    models: dict[str, IntervenedModel] = Field(default_factory=dict)
    metrics: dict[str, Metric] = Field(default_factory=dict)


# --------------------------------------------------------------------- #
# the steps
# --------------------------------------------------------------------- #


class Save(Node):
    """One file. `value` names a result **of the step this sits on**.

    The field that holds these is `saves`, not `save`, for a reason worth
    knowing: nnsight mounts a `.save()` method onto `object`, so every
    pydantic model in a process that imports nnsight inherits one, and a
    field called `save` shadows it. FINDINGS §9.
    """

    value: str
    file_path: str


class Optimizer(Node):
    name: Literal["adamw"] = "adamw"
    lr: float
    weight_decay: float = 0.0


class EarlyStop(Node):
    metric: str
    mode: Literal["max"] = "max"
    patience: int = Field(gt=0)


class Evaluation(Node):
    """The fit's held-out pass: its own rows, and its own saves. This is the
    whole reason the format changed — the held-out score is a result of a
    place, not a name that has to avoid colliding with another."""

    rows: dict[str, str]
    saves: list[Save] = Field(default_factory=list)


class Observe(Node):
    kind: Literal["observe"]
    rows: dict[str, str]
    saves: list[Save] = Field(default_factory=list)


class Fit(Node):
    kind: Literal["fit"]
    rows: dict[str, str]
    params: list[str]
    #: Σ wᵢ·metricᵢ, minimized.
    objective: list[tuple[float, str]]
    epochs: int = Field(gt=0)
    pairs: int = Field(gt=0)
    seed: int = 0
    optimizer: Optimizer
    early_stop: EarlyStop
    eval: Evaluation
    saves: list[Save] = Field(default_factory=list)


class Weights(Node):
    kind: Literal["weights"]
    names: list[str]
    saves: list[Save] = Field(default_factory=list)


Step = Annotated[Union[Observe, Fit, Weights], Field(discriminator="kind")]


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


class Spec(Node):
    header: Header = Field(default_factory=Header)
    model: Model
    roles: dict[str, Role]
    sites: dict[str, Site]
    intervention: Intervention
    steps: dict[str, Step]
    featurizers: dict[str, Featurizer] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_check(self) -> "Spec":
        """Everything that is about two pieces at once. No single node owns
        any of it, so none of it is a field validator."""
        known = set(self.featurizers) | {"identity"}
        for name, read in self.intervention.reads.items():
            _refuse(read.site in self.sites, f"read {name!r}: undeclared site {read.site!r}")
            _refuse(read.input in self.roles, f"read {name!r}: undeclared role {read.input!r}")
            _refuse(
                read.model == "original" or read.model in self.intervention.models,
                f"read {name!r}: model {read.model!r} is neither 'original' nor declared",
            )
            _refuse(read.featurizer in known, f"read {name!r}: undeclared featurizer")
        for name, write in self.intervention.writes.items():
            _refuse(write.site in self.sites, f"write {name!r}: undeclared site {write.site!r}")
            _refuse(
                write.operand in self.intervention.reads,
                f"write {name!r}: the operand must be a read name; param and literal "
                "operands are not implemented",
            )
            _refuse(write.featurizer in known, f"write {name!r}: undeclared featurizer")
        for name, model in self.intervention.models.items():
            _refuse(model.input in self.roles, f"model {name!r}: undeclared role")
            for write in model.writes:
                _refuse(write in self.intervention.writes, f"model {name!r}: undeclared write {write!r}")
        for name, metric in self.intervention.metrics.items():
            _refuse(
                metric.of in self.intervention.reads,
                f"metric {name!r}: `of` must be a read name, got {metric.of!r}",
            )

        # one featurizer name is one parameter set, so it acts at one site
        for name in self.featurizers:
            at = {read.site for read in self.intervention.reads.values() if read.featurizer == name}
            at |= {w.site for w in self.intervention.writes.values() if w.featurizer == name}
            _refuse(len(at) == 1, f"featurizer {name!r} is used at {sorted(at)}; one name is one site")

        produced = set(self.intervention.metrics)
        for name, step in self.steps.items():
            # A step that runs forwards needs rows for every role; `weights`
            # runs none, and asking it for rows would be asking a question
            # about a step that does not touch the model.
            if isinstance(step, (Observe, Fit)):
                for role in self.roles:
                    _refuse(role in step.rows, f"step {name!r}: no rows for role {role!r}")
                for role, dataset in step.rows.items():
                    _refuse(role in self.roles, f"step {name!r}: undeclared role {role!r}")
                    _refuse(bool(dataset), f"step {name!r}: role {role!r} has no dataset")
            if isinstance(step, Fit):
                for role in self.roles:
                    _refuse(
                        role in step.eval.rows, f"step {name!r}.eval: no rows for role {role!r}"
                    )
            if isinstance(step, Fit):
                _refuse(
                    set(step.params) <= set(self.featurizers),
                    f"step {name!r}: trains {sorted(set(step.params) - set(self.featurizers))}, "
                    "which is not a declared featurizer",
                )
                _refuse(
                    step.early_stop.metric in produced,
                    f"step {name!r}: early_stop watches {step.early_stop.metric!r}, not a metric",
                )
                for _, term in step.objective:
                    _refuse(term in produced, f"step {name!r}: objective names {term!r}, not a metric")
                for save in step.eval.saves:
                    _refuse(
                        save.value in produced,
                        f"step {name!r}.eval: saves {save.value!r}, which that pass does not produce",
                    )
            if isinstance(step, Weights):
                _refuse(
                    set(step.names) <= set(self.featurizers),
                    f"step {name!r}: names {sorted(set(step.names) - set(self.featurizers))}, "
                    "which is not a declared featurizer",
                )
            for save in step.saves:
                available = produced if isinstance(step, Observe) else set(_publishes(step))
                _refuse(
                    save.value in available,
                    f"step {name!r}: saves {save.value!r}, which it does not produce "
                    f"(it produces {sorted(available)})",
                )
        return self


def _refuse(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _publishes(step: Any) -> tuple[str, ...]:
    """What a non-observing step puts in its own results."""
    if isinstance(step, Fit):
        return ("train/loss", "train/eval")
    if isinstance(step, Weights):
        return tuple(step.names)
    return ()
