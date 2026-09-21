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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .. import address
from ..data import encoding

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
    #: Which attention code the model runs. Absent, the library's default
    #: (sdpa). It is part of the experiment and not of the run: `eager` is
    #: the only one under which `attention_scores` and `attention_probs`
    #: exist, and the implementations differ in the last bits.
    attn_implementation: Literal["eager", "sdpa"] | None = None


class Role(Node):
    #: The column a row's text comes from, e.g. `counterfactual_inputs[0]`.
    field: str


class Site(Node):
    component: str
    layers: list[int] | None = None
    #: At a per-head tensor, the heads this site is. The site's width is then
    #: theirs — `len(heads) · head_dim` — so a featurizer, a swap or a harvest
    #: here is of those heads and leaves the others alone.
    heads: list[int] | None = None

    @model_validator(mode="after")
    def _one_layer(self) -> "Site":
        if self.heads is not None:
            if not address.describe().get(self.component, {}).get("heads", False):
                per_head = sorted(n for n, one in address.describe().items() if one["heads"])
                raise ValueError(
                    f"{self.component!r} is not a per-head tensor; `heads` applies at {per_head}"
                )
            if not self.heads or len(set(self.heads)) != len(self.heads) or min(self.heads) < 0:
                raise ValueError("heads is a non-empty list of distinct head indices")
        if self.layers is not None and len(self.layers) != 1:
            raise ValueError(
                "layers must be a one-element band; a band spanning several "
                "layers is one address and is not implemented"
            )
        return self


class Featurizer(Node):
    """A parameter set. `subspace` is a Cayley-parametrized rotation, drawn
    from a seed or loaded, and trainable; `pca` is a fixed basis, always
    loaded, never trained — the control a fit is compared against; `gate` is
    a learned binary mask over the site's units (DBM), soft while a fit
    updates it and hard whenever it is scored."""

    kind: Literal["subspace", "pca", "gate"]
    #: The rank of a basis. A gate has none: its features are the units.
    k: int | None = Field(default=None, gt=0)
    parametrization: Literal["cayley"] = "cayley"
    #: The draw the initial basis comes from. Absent, the fit's seed, or 0.
    seed: int | None = None
    #: Load the parameter from a safetensors bundle a previous run wrote,
    #: instead of drawing it. Its header is checked against this document —
    #: model, site, layer, k, d — and a mismatch is refused by key.
    file_path: str | None = None

    @model_validator(mode="after")
    def _drawn_or_loaded(self) -> "Featurizer":
        if (self.kind == "gate") != (self.k is None):
            raise ValueError(
                "a gate masks every unit of its site and takes no k"
                if self.kind == "gate"
                else f"a {self.kind} needs k, the rank of its basis"
            )
        if self.kind == "gate" and self.seed is not None:
            raise ValueError("a gate starts at θ = 0, not from a draw; it has no seed")
        if self.kind == "pca" and self.file_path is None:
            raise ValueError("a pca featurizer is loaded from a file; nothing draws or trains one")
        if self.file_path is not None and self.seed is not None:
            raise ValueError("a loaded featurizer has no seed: its weights are its bytes")
        return self


#: The components whose tensor is the residual stream — the only ones a
#: logits view makes sense of, because the final norm and head expect it.
RESIDUAL_STREAM = frozenset({"embeddings", "block_input", "block_output", "ln_final"})

#: A position form: one index, or a window of the same width on every row.
Position = int | dict[str, Any]


class Read(Node):
    site: str
    pos: Position
    model: str = "original"
    input: str
    featurizer: str = "identity"
    #: `"logits"` projects a residual-stream read through the model's final
    #: norm and head. With a layer sweep and a `token_prob` metric that is
    #: the logit lens, as one document.
    view: Literal["raw", "logits"] = "raw"

    @field_validator("pos")
    @classmethod
    def _a_known_form(cls, pos: Any) -> Any:
        encoding.width_of(pos)  # refuses an unknown or ragged form by name
        return pos


class Reference(Node):
    """A value an earlier sibling step published under `outputs`."""

    ref: str


class Write(Node):
    """`mechanism` and `operand` are two fields, not the protocol's
    `{"swap": "v_cf"}`, because a key that is itself the mechanism's name
    cannot be enumerated by a schema — and the schema is what an agent
    reads. The operand is a read of this intervention, or a reference to
    what an earlier step output."""

    site: str
    pos: Position
    mechanism: Literal["swap", "add_scaled", "lerp", "gaussian"]
    #: A read of this intervention, a `{"ref": …}` to an earlier step's
    #: output, a literal number (zero ablation is `0.0`), or nothing for a
    #: mechanism that takes none.
    operand: str | float | Reference | None = None
    featurizer: str = "identity"
    #: The mechanism's numbers: `scale` for add_scaled and gaussian, `t` for
    #: lerp, `seed` for gaussian.
    params: dict[str, float] = Field(default_factory=dict)

    @field_validator("pos")
    @classmethod
    def _a_known_form(cls, pos: Any) -> Any:
        encoding.width_of(pos)
        return pos

    @model_validator(mode="after")
    def _mechanism_and_its_numbers(self) -> "Write":
        needs, takes = MECHANISM_PARAMS[self.mechanism]
        missing = needs - set(self.params)
        extra = set(self.params) - needs - takes
        if missing:
            raise ValueError(f"mechanism {self.mechanism!r} needs params {sorted(missing)}")
        if extra:
            raise ValueError(f"mechanism {self.mechanism!r} takes no params {sorted(extra)}")
        if self.mechanism == "gaussian" and self.operand is not None:
            raise ValueError("mechanism 'gaussian' takes no operand; it draws its own noise")
        if self.mechanism != "gaussian" and self.operand is None:
            raise ValueError(f"mechanism {self.mechanism!r} needs an operand")
        return self

    @property
    def operand_name(self) -> str | float | None:
        """The operand as the compiler carries it: a name for a read or a
        reference, the number itself for a literal."""
        return self.operand.ref if isinstance(self.operand, Reference) else self.operand


#: (required, optional) params per mechanism.
MECHANISM_PARAMS: dict[str, tuple[set[str], set[str]]] = {
    "swap": (set(), set()),
    "add_scaled": (set(), {"scale"}),
    "lerp": ({"t"}, set()),
    "gaussian": ({"seed"}, {"scale"}),
}


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


class TokenProb(Node):
    kind: Literal["token_prob"]
    of: str
    token: str
    token_form: Literal["space_prefixed"] = "space_prefixed"


for _cls, _columns in (
    (Match, ("expected",)),
    (LogitDiff, ("a", "b")),
    (CrossEntropy, ("target",)),
    (TokenLogit, ("token",)),
    (TokenProb, ("token",)),
):
    # The data columns this kind names, in the order `ops.metrics.compute`
    # takes them — the one thing the compiler needs and the shape of the
    # class already says.
    _cls.columns = property(  # type: ignore[attr-defined]
        lambda self, _columns=_columns: tuple(getattr(self, one) for one in _columns)
    )


Metric = Annotated[
    Union[Match, LogitDiff, CrossEntropy, TokenLogit, TokenProb], Field(discriminator="kind")
]

#: Which of a metric's own fields name data columns, in the order
#: `ops.metrics.compute` takes them.
METRIC_COLUMNS = {
    "match": ("expected",),
    "logit_diff": ("a", "b"),
    "cross_entropy": ("target",),
    "token_logit": ("token",),
    "token_prob": ("token",),
}


class Intervention(Node):
    """The experiment, declared once. Every step runs *this*, over its own
    rows — which is why a fit and the run it scores cannot drift apart."""

    reads: dict[str, Read]
    writes: dict[str, Write] = Field(default_factory=dict)
    models: dict[str, IntervenedModel] = Field(default_factory=dict)
    metrics: dict[str, Metric] = Field(default_factory=dict)
    #: Generate this many tokens after the prompt, greedily, on every
    #: forward of this intervention. 0 is one forward pass. With it, a
    #: position may be `{"step": k}` — the continuation frame.
    decode: int = Field(default=0, ge=0)


# --------------------------------------------------------------------- #
# the steps
# --------------------------------------------------------------------- #


class Output(Node):
    """One value a pass publishes for the steps after it: a read, kept as it
    is (`rows x width`), averaged over rows to one vector — which is how a
    corpus mean gets made without the un-reduced activations ever leaving —
    or reduced to its top-k principal directions, a basis a later document
    loads as a `pca` featurizer."""

    read: str
    reduce: Literal["none", "mean", "pca"] = "none"
    k: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _k_iff_pca(self) -> "Output":
        if (self.reduce == "pca") != (self.k is not None):
            raise ValueError("`k` is for `reduce: pca`, and pca needs it")
        return self


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
    mode: Literal["max", "min"] = "max"
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
    #: Which experiment this pass runs. Optional when the document declares
    #: exactly one.
    intervention: str | None = None
    #: name -> what to publish. A bare read name is shorthand for keeping it
    #: unreduced.
    outputs: dict[str, Output] = Field(default_factory=dict)
    saves: list[Save] = Field(default_factory=list)

    @field_validator("outputs", mode="before")
    @classmethod
    def _shorthand(cls, raw: Any) -> Any:
        if isinstance(raw, dict):
            return {name: {"read": one} if isinstance(one, str) else one for name, one in raw.items()}
        return raw


class Anneal(Node):
    """A gate's temperature across a fit: `start` on the first update, `end`
    on the last, geometric between. Toward zero the soft mask the optimizer
    sees becomes the hard mask the score uses."""

    start: float = Field(gt=0)
    end: float = Field(gt=0)


class Fit(Node):
    kind: Literal["fit"]
    rows: dict[str, str]
    intervention: str | None = None
    params: list[str]
    #: Σ wᵢ·termᵢ, minimized. A term is a metric, or `<gate>.mask` for a gate
    #: this fit trains — the mean of its soft mask, which is its L1 penalty.
    objective: list[tuple[float, str]]
    #: gate name -> its temperature schedule. A gate without one stays at 1.
    anneal: dict[str, Anneal] = Field(default_factory=dict)
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
    """A document. `steps` run in the order written, and that order is
    checked: a step may not use a featurizer before the step that trains it
    has run, or it would score an untrained rotation without complaint."""

    header: Header = Field(default_factory=Header)
    model: Model
    roles: dict[str, Role]
    sites: dict[str, Site]
    #: The experiments, by name. A step names the one it runs; when there is
    #: exactly one, it need not.
    interventions: dict[str, Intervention]
    steps: dict[str, Step]
    featurizers: dict[str, Featurizer] = Field(default_factory=dict)

    def intervention_of(self, step: Any) -> Intervention:
        """The experiment a step runs, resolved."""
        name = getattr(step, "intervention", None)
        if name is None:
            if len(self.interventions) != 1:
                raise ValueError(
                    f"a step must name its intervention when the document declares "
                    f"{len(self.interventions)}: {sorted(self.interventions)}"
                )
            (name,) = self.interventions
        if name not in self.interventions:
            raise ValueError(f"undeclared intervention {name!r}; declared: {sorted(self.interventions)}")
        return self.interventions[name]

    @model_validator(mode="before")
    @classmethod
    def _not_swept(cls, raw: Any) -> Any:
        if _swept(raw):
            raise ValueError(
                "this document has a {'sweep': …} wrapper in it. A sweep is lowered "
                "before a document is validated — build_request does this, and "
                "compiles one plan per point"
            )
        return raw

    @model_validator(mode="after")
    def _cross_check(self) -> "Spec":
        """Everything that is about two pieces at once. No single node owns
        any of it, so none of it is a field validator."""
        _refuse(bool(self.interventions), "a document declares at least one intervention")
        known = set(self.featurizers) | {"identity"}
        for label, one in self.interventions.items():
            self._check_intervention(label, one, known)

        # one featurizer name is one parameter set, so it acts at one site
        for name in self.featurizers:
            at = {
                read.site
                for one in self.interventions.values()
                for read in one.reads.values()
                if read.featurizer == name
            }
            at |= {
                write.site
                for one in self.interventions.values()
                for write in one.writes.values()
                if write.featurizer == name
            }
            _refuse(len(at) == 1, f"featurizer {name!r} is used at {sorted(at)}; one name is one site")

        published: dict[str, str] = {}  # output name -> the step that publishes it
        all_reads = {name for one in self.interventions.values() for name in one.reads}
        for name, step in self.steps.items():
            if isinstance(step, (Observe, Fit)):
                intervention = self.intervention_of(step)
                for role in self.roles:
                    _refuse(role in step.rows, f"step {name!r}: no rows for role {role!r}")
                for role, dataset in step.rows.items():
                    _refuse(role in self.roles, f"step {name!r}: undeclared role {role!r}")
                    _refuse(bool(dataset), f"step {name!r}: role {role!r} has no dataset")
                # a reference must be to something an EARLIER sibling published
                for write_name, write in intervention.writes.items():
                    if isinstance(write.operand, Reference):
                        _refuse(
                            write.operand.ref in published,
                            f"step {name!r}: write {write_name!r} references "
                            f"{write.operand.ref!r}, which no earlier step outputs "
                            f"(published so far: {sorted(published)})",
                        )
            produced = set(self.intervention_of(step).metrics) if isinstance(step, (Observe, Fit)) else set()
            if isinstance(step, (Observe, Fit)) and self.intervention_of(step).decode:
                one = self.intervention_of(step)
                models = {"original"} | set(one.models)
                produced |= {f"{model}.generated" for model in models}
            if isinstance(step, Observe):
                for output_name, output in step.outputs.items():
                    _refuse(
                        output.read in self.intervention_of(step).reads,
                        f"step {name!r}: output {output_name!r} keeps read {output.read!r}, "
                        "which its intervention does not have",
                    )
                    _refuse(
                        output_name not in published,
                        f"step {name!r}: output {output_name!r} is already published by "
                        f"step {published.get(output_name)!r}",
                    )
                    _refuse(
                        output_name not in all_reads,
                        f"step {name!r}: output {output_name!r} shares its name with a "
                        "read; an operand names one or the other",
                    )
                    published[output_name] = name
                produced |= set(step.outputs)
            if isinstance(step, Fit):
                for role in self.roles:
                    _refuse(
                        role in step.eval.rows, f"step {name!r}.eval: no rows for role {role!r}"
                    )
                _refuse(
                    set(step.params) <= set(self.featurizers),
                    f"step {name!r}: trains {sorted(set(step.params) - set(self.featurizers))}, "
                    "which is not a declared featurizer",
                )
                for param in step.params:
                    _refuse(
                        self.featurizers[param].kind != "pca",
                        f"step {name!r}: trains {param!r}, a pca basis, which is fixed by "
                        "definition — declare a subspace loaded from it to fine-tune",
                    )
                gates = {p for p in step.params if self.featurizers[p].kind == "gate"}
                _refuse(
                    set(step.anneal) <= gates,
                    f"step {name!r}: anneals {sorted(set(step.anneal) - gates)}; only a gate "
                    f"this fit trains has a temperature (those are {sorted(gates)})",
                )
                terms = produced | {f"{gate}.mask" for gate in gates}
                _refuse(
                    step.early_stop.metric in produced,
                    f"step {name!r}: early_stop watches {step.early_stop.metric!r}, not a metric",
                )
                for _, term in step.objective:
                    _refuse(
                        term in terms,
                        f"step {name!r}: objective names {term!r}, which is neither a metric "
                        f"nor the mask of a gate this fit trains (one of {sorted(terms)})",
                    )
                for save in step.eval.saves:
                    _refuse(
                        save.value in produced,
                        f"step {name!r}.eval: saves {save.value!r}, which that pass does not produce",
                    )
                produced = set(_publishes(step))
            if isinstance(step, Weights):
                _refuse(
                    set(step.names) <= set(self.featurizers),
                    f"step {name!r}: names {sorted(set(step.names) - set(self.featurizers))}, "
                    "which is not a declared featurizer",
                )
                produced = set(step.names)
            for save in step.saves:
                _refuse(
                    save.value in produced,
                    f"step {name!r}: saves {save.value!r}, which it does not produce "
                    f"(it produces {sorted(produced)})",
                )
        self._check_order()
        return self

    def _check_intervention(self, label: str, one: Intervention, known: set[str]) -> None:
        where = f"interventions.{label}"
        for name, read in one.reads.items():
            _refuse(read.site in self.sites, f"{where}: read {name!r}: undeclared site {read.site!r}")
            if read.view == "logits":
                component = self.sites[read.site].component if read.site in self.sites else ""
                _refuse(
                    component in RESIDUAL_STREAM,
                    f"{where}: read {name!r}: view 'logits' projects the residual stream "
                    f"through the head, and {component!r} is not the residual stream "
                    f"(one of {sorted(RESIDUAL_STREAM)})",
                )
                _refuse(
                    read.featurizer == "identity",
                    f"{where}: read {name!r}: a featurized read cannot also be viewed as logits",
                )
            _refuse(read.input in self.roles, f"{where}: read {name!r}: undeclared role {read.input!r}")
            _refuse(
                read.model == "original" or read.model in one.models,
                f"{where}: read {name!r}: model {read.model!r} is neither 'original' nor declared",
            )
            _refuse(read.featurizer in known, f"{where}: read {name!r}: undeclared featurizer")
        for name, write in one.writes.items():
            _refuse(write.site in self.sites, f"{where}: write {name!r}: undeclared site {write.site!r}")
            component = self.sites[write.site].component if write.site in self.sites else ""
            _refuse(
                not address.describe().get(component, {}).get("read_only", False),
                f"{where}: write {name!r}: {component!r} is read-only — the model's input, "
                "not an activation",
            )
            if isinstance(write.operand, str):
                _refuse(
                    write.operand in one.reads,
                    f"{where}: write {name!r}: operand {write.operand!r} is not a read of "
                    'this intervention; a value from an earlier step is {"ref": …}',
                )
                _refuse(
                    write.operand not in one.reads or one.reads[write.operand].view == "raw",
                    f"{where}: write {name!r}: operand {write.operand!r} is a logits view, "
                    "which is vocabulary-wide and cannot be written back at a site",
                )
            _refuse(write.featurizer in known, f"{where}: write {name!r}: undeclared featurizer")
        for name, model in one.models.items():
            _refuse(model.input in self.roles, f"{where}: model {name!r}: undeclared role")
            for write in model.writes:
                _refuse(write in one.writes, f"{where}: model {name!r}: undeclared write {write!r}")
        for name, read in one.reads.items():
            step = encoding.step_of(read.pos)
            if step is not None:
                _refuse(step != "all", f"{where}: read {name!r}: a read is at one step; 'all' is for writes")
                _refuse(one.decode > 0, f"{where}: read {name!r}: a step position needs `decode` > 0")
                _refuse(
                    isinstance(step, int) and step < one.decode,
                    f"{where}: read {name!r}: step {step} of a {one.decode}-token decode",
                )
        for name, write in one.writes.items():
            step = encoding.step_of(write.pos)
            if step is not None:
                _refuse(one.decode > 0, f"{where}: write {name!r}: a step position needs `decode` > 0")
                _refuse(
                    step == "all" or (isinstance(step, int) and step < one.decode),
                    f"{where}: write {name!r}: step {step} of a {one.decode}-token decode",
                )
        for name, metric in one.metrics.items():
            _refuse(
                metric.of in one.reads,
                f"{where}: metric {name!r}: `of` must be a read name, got {metric.of!r}",
            )
            _refuse(
                encoding.width_of(one.reads[metric.of].pos) == 1,
                f"{where}: metric {name!r} reads {metric.of!r}, a window of "
                f"{encoding.width_of(one.reads[metric.of].pos) or 'varying'} positions; "
                "a metric scores one position per row",
            )
        for name, write in one.writes.items():
            if isinstance(write.operand, str):
                have, want = encoding.width_of(one.reads[write.operand].pos), encoding.width_of(write.pos)
                # a ragged side has no width until the rows are known; the
                # compiler checks those row by row
                _refuse(
                    have is None or want is None or have == want,
                    f"{where}: write {name!r} covers {want} position(s) but its operand "
                    f"{write.operand!r} was read over {have}; the windows must match",
                )

    def _check_order(self) -> None:
        """A trained featurizer may not be used before it is trained.

        A step that runs a forward uses every featurizer its intervention
        names. If a later fit trains one of them, an earlier step scored it
        untrained — a valid document with a silently wrong number. Refused,
        with the fix: a deliberate untrained baseline is a *second* featurizer
        that no fit names (see documents/random_subspace_cpu.json).
        """
        trained_at: dict[str, int] = {}
        for index, (name, step) in enumerate(self.steps.items()):
            if isinstance(step, Fit):
                for param in step.params:
                    trained_at.setdefault(param, index)
        for index, (name, step) in enumerate(self.steps.items()):
            if isinstance(step, (Observe, Fit)):
                one = self.intervention_of(step)
                touches = {read.featurizer for read in one.reads.values()}
                touches |= {write.featurizer for write in one.writes.values()}
            elif isinstance(step, Weights):
                touches = set(step.names)
            else:
                continue
            for featurizer in sorted(touches):
                first = trained_at.get(featurizer)
                _refuse(
                    first is None or first <= index,
                    f"step {name!r} uses featurizer {featurizer!r} before step "
                    f"{list(self.steps)[first or 0]!r} trains it, so it would run on "
                    "the untrained parameter. Move it after the fit — or, for a "
                    "deliberate untrained baseline, declare a second featurizer that "
                    "no fit names",
                )


def _swept(node: Any) -> bool:
    if isinstance(node, dict):
        return set(node) == {"sweep"} or any(_swept(value) for value in node.values())
    if isinstance(node, list):
        return any(_swept(value) for value in node)
    return False


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
