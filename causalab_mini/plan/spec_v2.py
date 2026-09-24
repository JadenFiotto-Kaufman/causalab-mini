"""The roles-and-interventions format, kept beside the steps-first one
(`spec.py`) until the corpus is rewritten in that, and deleted when it is.

A document here declares input `roles`, a site vocabulary, and experiments
under `interventions` — reads with a `model` and an `input`, writes bound to
intervened `models`, metrics and a `decode` — and its steps (`observe`,
`fit`, `weights`) name the experiment they run and the rows they run it
over. `build_request` tells the two apart by `roles`, and both compile to
the same plan. The vocabulary the two share — model, sites, featurizers,
positions, mechanisms — is `spec.py`'s.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Union

from pydantic import Field, field_validator, model_validator

from .. import address
from ..ops.metrics import COLUMNS as METRIC_COLUMNS
from . import sweep as sweep_module
from .spec import (
    LOADED_ONLY,
    MECHANISM_PARAMS,
    NO_OPERAND,
    RESIDUAL_STREAM,
    Anneal,
    EarlyStop,
    Featurizer,
    Header,
    Model,
    Node,
    Optimizer,
    Position,
    Site,
)


class Role(Node):
    #: The column a row's text comes from, e.g. `counterfactual_inputs[0]`.
    field: str


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
    mechanism: Literal["swap", "add_scaled", "lerp", "gaussian", "clamp", "renormalize"]
    #: A read of this intervention, a `{"ref": …}` to an earlier step's
    #: output, a literal number (zero ablation is `0.0`), or nothing for a
    #: mechanism that takes none.
    operand: str | float | Reference | None = None
    featurizer: str = "identity"
    #: Which coordinates of the featurizer's space the mechanism acts on —
    #: SAE latents, directions of a rotation. The others pass through, and
    #: so does whatever the featurizer does not explain (its error term).
    features: list[int] | None = None
    #: The mechanism's numbers: `scale` for add_scaled and gaussian, `t` for
    #: lerp, `seed` for gaussian.
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _mechanism_and_its_numbers(self) -> "Write":
        needs, takes = MECHANISM_PARAMS[self.mechanism]
        missing = needs - set(self.params)
        extra = set(self.params) - needs - takes
        if missing:
            raise ValueError(f"mechanism {self.mechanism!r} needs params {sorted(missing)}")
        if extra:
            raise ValueError(f"mechanism {self.mechanism!r} takes no params {sorted(extra)}")
        if self.mechanism in NO_OPERAND and self.operand is not None:
            raise ValueError(f"mechanism {self.mechanism!r} takes no operand")
        if self.mechanism not in NO_OPERAND and self.operand is None:
            raise ValueError(f"mechanism {self.mechanism!r} needs an operand")
        if self.mechanism == "clamp" and not self.params:
            raise ValueError("mechanism 'clamp' needs a bound: params lo, hi or both")
        return self

    @property
    def operand_name(self) -> str | float | None:
        """The operand as the compiler carries it: a name for a read or a
        reference, the number itself for a literal."""
        return self.operand.ref if isinstance(self.operand, Reference) else self.operand


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


for _cls in (Match, LogitDiff, CrossEntropy, TokenLogit, TokenProb):
    # The data columns this kind names, in the order `ops.metrics.compute`
    # takes them — read off the one table, so a class and the `compute` it
    # feeds cannot disagree about which column is which.
    _cls.columns = property(  # type: ignore[attr-defined]
        lambda self: tuple(getattr(self, one) for one in METRIC_COLUMNS[self.kind])
    )


Metric = Annotated[
    Union[Match, LogitDiff, CrossEntropy, TokenLogit, TokenProb], Field(discriminator="kind")
]

class Intervention(Node):
    """The experiment, declared once. Every step runs *this*, over its own
    rows — which is why a fit and the run it scores cannot drift apart."""

    reads: dict[str, Read]
    writes: dict[str, Write] = Field(default_factory=dict)
    models: dict[str, IntervenedModel] = Field(default_factory=dict)
    metrics: dict[str, Metric] = Field(default_factory=dict)
    #: Generate this many tokens after the prompt, greedily, on every
    #: forward of this intervention. 0 is one forward pass. With it, a
    #: position may name the continuation frame, `{"frame": "generated"}`.
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


class Evaluation(Node):
    """The fit's held-out pass: its own rows, and its own saves. This is the
    whole reason the format changed — the held-out score is a result of a
    place, not a name that has to avoid colliding with another."""

    rows: dict[str, str]
    saves: list[Save] = Field(default_factory=list)


class Observe(Node):
    kind: Literal["observe"]
    rows: dict[str, str]
    #: Which experiments this pass runs: the name of one declared under the
    #: document's `interventions`, one written here in place, or a list of
    #: either. Every step that runs a forward says which, and a list runs
    #: each over these rows — lowered by `Spec.lower` to one child per entry.
    interventions: "str | Intervention | list[str | Intervention]"
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


class Fit(Node):
    kind: Literal["fit"]
    rows: dict[str, str]
    #: The experiment the fit trains through, by name or in place — the
    #: same one the score after it names, which is what makes the score
    #: measure what was trained. A list fits through each in turn.
    interventions: "str | Intervention | list[str | Intervention]"
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
    #: The experiments, by name. Nothing here runs: a step names the ones it
    #: runs, and one a step writes in place is declared here by `_inline`.
    interventions: dict[str, Intervention]
    steps: dict[str, Step]
    featurizers: dict[str, Featurizer] = Field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Identity of the experiment, the same rule the protocol format
        uses: everything but `header`, which is authoring metadata. It is
        what a metric row is `produced_by` and what a saved featurizer is
        stamped with."""
        body = {key: value for key, value in self.model_dump(mode="json").items() if key != "header"}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def intervention_of(self, step: Any) -> Intervention:
        """The one experiment a step runs, resolved. An inline one was
        hoisted before anything else looked (`_inline`), so this is a lookup;
        a list step has no one experiment and is asked through `lower`."""
        name = step.interventions
        if isinstance(name, list):
            raise ValueError("a step that lists interventions runs each through `Spec.lower`")
        if name not in self.interventions:
            raise ValueError(f"undeclared intervention {name!r}; declared: {sorted(self.interventions)}")
        return self.interventions[name]

    def runs(self) -> list[tuple[str, Any, bool]]:
        """Every step as it will run, in order: `(path, step, nested)`.

        A step with one intervention is itself. A step with a list is the
        children `lower` makes, at `<step>/<intervention>` — the path the
        nested plan they compile to gives them, and so the name its checks
        refuse under. `nested` says the step runs inside its own plan, whose
        outputs are its own (`engine.steps.State.child`).
        """
        found: list[tuple[str, Any, bool]] = []
        for name, step in self.steps.items():
            if isinstance(getattr(step, "interventions", None), list):
                found.extend((f"{name}/{child}", one, True) for child, one in self.lower(name, step).items())
            else:
                found.append((name, step, False))
        return found

    def produces(self, step: Any) -> set[str]:
        """The value names a single-intervention observe or fit puts in its
        own results — what a save on it may name."""
        one = self.intervention_of(step)
        produced = set(one.metrics)
        if one.decode:
            produced |= {f"{model}.generated" for model in {"original"} | set(one.models)}
        if isinstance(step, Observe):
            produced |= set(step.outputs)
        if isinstance(step, Fit):
            produced = set(_publishes(step))
        return produced

    def lower(self, name: str, step: Any) -> dict[str, Any]:
        """A step that lists interventions, as one step per intervention.

        Each child is the step with that one intervention: the same rows and
        every other field, and the outputs and saves that belong to it. A
        value is qualified the way a nested plan's path is, `<child>/<value>`
        — `patching/logit_diff` — and an unqualified name goes to the one
        child that has it. A name that no child has, or several do, is
        refused with the qualified names it could have been.
        """
        children = list(step.interventions)

        def resolve(value: str, has: dict[str, set[str]], what: str) -> tuple[str, str]:
            head, _, rest = value.partition("/")
            if rest and head in has and rest in has[head]:
                return head, rest
            owners = [child for child in children if value in has[child]]
            if len(owners) == 1:
                return owners[0], value
            candidates = sorted(f"{child}/{one}" for child in children for one in has[child])
            if owners:
                raise ValueError(
                    f"step {name!r}: {what} {value!r} is produced by {len(owners)} of its "
                    f"interventions {owners}; qualify it as one of "
                    f"{[f'{child}/{value}' for child in owners]}"
                )
            raise ValueError(f"step {name!r}: {what} {value!r} names nothing it runs; one of {candidates}")

        outputs: dict[str, dict[str, Any]] = {child: {} for child in children}
        if isinstance(step, Observe):
            reads = {child: set(self.interventions[child].reads) for child in children}
            for out_name, out in step.outputs.items():
                child, read = resolve(out.read, reads, "output read")
                outputs[child][out_name] = out.model_copy(update={"read": read})
        single = {
            child: step.model_copy(update={"interventions": child, "saves": [], **({"outputs": outputs[child]} if isinstance(step, Observe) else {})})
            for child in children
        }
        saves: dict[str, list[Save]] = {child: [] for child in children}
        has = {child: self.produces(single[child]) for child in children}
        for save in step.saves:
            child, value = resolve(save.value, has, "save")
            saves[child].append(save.model_copy(update={"value": value}))
        lowered = {child: single[child].model_copy(update={"saves": saves[child]}) for child in children}
        if isinstance(step, Fit):
            evals: dict[str, list[Save]] = {child: [] for child in children}
            metrics = {child: set(self.interventions[child].metrics) for child in children}
            for save in step.eval.saves:
                child, value = resolve(save.value, metrics, "eval save")
                evals[child].append(save.model_copy(update={"value": value}))
            lowered = {
                child: one.model_copy(update={"eval": step.eval.model_copy(update={"saves": evals[child]})})
                for child, one in lowered.items()
            }
        return lowered

    @model_validator(mode="before")
    @classmethod
    def _not_swept(cls, raw: Any) -> Any:
        if sweep_module.wrappers(raw):
            raise ValueError(
                "this document has a {'sweep': …} wrapper in it. A sweep is lowered "
                "before a document is validated — build_request does this, and "
                "compiles one plan per point"
            )
        return raw

    @model_validator(mode="before")
    @classmethod
    def _inline(cls, raw: Any) -> Any:
        """Every step names its experiments, and one written in place is
        declared under a name derived from its step.

        That is the whole of what inline costs: after this, a step's
        `interventions` is a name or a list of names and every experiment is
        under the document's `interventions`, so every check, the compiler, a
        sweep and a fit's eval see one path and cannot tell the spellings
        apart. A lone inline intervention is named after its step — already
        unique in the document, and what an error will print. One at index
        `i` of a list is `<step>[i]`, the bracket `rows.field_text` already
        uses for a list.
        """
        if not isinstance(raw, dict) or not isinstance(raw.get("steps"), dict):
            return raw
        declared = dict(raw.get("interventions") or {})
        steps = {}
        for name, step in raw["steps"].items():
            if isinstance(step, dict) and step.get("kind") in ("observe", "fit"):
                if "interventions" not in step:
                    raise ValueError(
                        f"step {name!r} runs a forward and names no intervention; say which "
                        f"with `interventions` — declared: {sorted(declared)} — or write one "
                        "there in place"
                    )

                def hoist(one: Any, label: str) -> Any:
                    if not isinstance(one, dict):
                        return one
                    if label in declared:
                        raise ValueError(
                            f"step {name!r} writes an intervention in place that would be named "
                            f"{label!r}, and one is already declared under that name; rename one "
                            "of them"
                        )
                    declared[label] = one
                    return label

                given = step["interventions"]
                if isinstance(given, list):
                    if not given:
                        raise ValueError(f"step {name!r} lists no interventions")
                    named = [hoist(one, f"{name}[{index}]") for index, one in enumerate(given)]
                    repeated = sorted({one for one in named if isinstance(one, str) and named.count(one) > 1})
                    if repeated:
                        raise ValueError(f"step {name!r} lists {repeated} more than once")
                    step = {**step, "interventions": named}
                else:
                    step = {**step, "interventions": hoist(given, name)}
            steps[name] = step
        return {**raw, "interventions": declared, "steps": steps}

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

        for name, step in self.steps.items():
            listed = getattr(step, "interventions", None)
            for one in listed if isinstance(listed, list) else [listed] if listed else []:
                _refuse(
                    one in self.interventions,
                    f"step {name!r}: undeclared intervention {one!r}; declared: {sorted(self.interventions)}",
                )
        # output name -> the step that publishes it, to every later step
        # (`published`), or only inside its own nested plan (`taken`)
        published: dict[str, str] = {}
        taken: dict[str, str] = {}
        all_reads = {name for one in self.interventions.values() for name in one.reads}
        for name, step, nested in self.runs():
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
            produced = self.produces(step) if isinstance(step, Observe) else set()
            if isinstance(step, Fit):
                one = self.intervention_of(step)
                produced = set(one.metrics)
                if one.decode:
                    produced |= {f"{model}.generated" for model in {"original"} | set(one.models)}
            if isinstance(step, Observe):
                for output_name, output in step.outputs.items():
                    _refuse(
                        output.read in self.intervention_of(step).reads,
                        f"step {name!r}: output {output_name!r} keeps read {output.read!r}, "
                        "which its intervention does not have",
                    )
                    _refuse(
                        output_name not in taken,
                        f"step {name!r}: output {output_name!r} is already published by "
                        f"step {taken.get(output_name)!r}",
                    )
                    _refuse(
                        output_name not in all_reads,
                        f"step {name!r}: output {output_name!r} shares its name with a "
                        "read; an operand names one or the other",
                    )
                    taken[output_name] = name
                    if not nested:
                        published[output_name] = name
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
                        self.featurizers[param].kind not in LOADED_ONLY,
                        f"step {name!r}: trains {param!r}, a {self.featurizers[param].kind}, which is "
                        "fixed by definition — a pca basis can be fine-tuned as a subspace loaded from it",
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
                for one in step.names:
                    _refuse(
                        self.featurizers[one].kind not in ("sae", "linear"),
                        f"step {name!r}: {one!r} is a loaded {self.featurizers[one].kind}; its "
                        "tensors are the file it came from, and there is no fitted weight to publish",
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
            for index, write_name in enumerate(model.writes):
                write = one.writes.get(write_name)
                if write is None or write.mechanism != "renormalize":
                    continue
                here = [w for w in model.writes if w in one.writes and one.writes[w].site == write.site]
                _refuse(
                    len(here) > 1 and here[-1] == write_name,
                    f"{where}: model {name!r}: renormalize {write_name!r} restores the norm the "
                    f"other writes at site {write.site!r} changed, so it comes after at least one "
                    "of them and last among them; first, or alone, it is the identity",
                )
        for name, model in one.models.items():
            _refuse(model.input in self.roles, f"{where}: model {name!r}: undeclared role")
            for write in model.writes:
                _refuse(write in one.writes, f"{where}: model {name!r}: undeclared write {write!r}")
        for name, read in one.reads.items():
            if read.pos.frame == "generated":
                _refuse(one.decode > 0, f"{where}: read {name!r}: a generated position needs `decode` > 0")
                _refuse(
                    read.pos.index is None or read.pos.index < one.decode,
                    f"{where}: read {name!r}: step {read.pos.index} of a {one.decode}-token decode",
                )
        for name, write in one.writes.items():
            if write.pos.frame == "generated":
                _refuse(one.decode > 0, f"{where}: write {name!r}: a generated position needs `decode` > 0")
                # A write happens *during* the decode, so it may only name a
                # step the run has already reached. Every other form of the
                # continuation frame is a cut of the finished text — the last
                # real token, the row's stop token, where it said the answer
                # — and there is nothing to write into a step that has not
                # happened yet.
                _refuse(
                    write.pos.all or (write.pos.index is not None and write.pos.index >= 0),
                    f"{where}: write {name!r}: a write in the continuation frame is at a step "
                    "the decode has reached — {'index': k} with k >= 0, or {'all': true}. "
                    f"{write.pos.spelling()} names a cut of the finished continuation, which is "
                    "something to read and not something to write into",
                )
                _refuse(
                    write.pos.scope is None,
                    f"{where}: write {name!r}: a write in the continuation frame takes no scope; "
                    f"{write.pos.spelling()} is only known once the decode has finished",
                )
                _refuse(
                    write.pos.index is None or write.pos.index < one.decode,
                    f"{where}: write {name!r}: step {write.pos.index} of a {one.decode}-token decode",
                )
        for name, metric in one.metrics.items():
            _refuse(
                metric.of in one.reads,
                f"{where}: metric {name!r}: `of` must be a read name, got {metric.of!r}",
            )
            _refuse(
                one.reads[metric.of].pos.width == 1,
                f"{where}: metric {name!r} reads {metric.of!r}, a window of "
                f"{one.reads[metric.of].pos.width or 'varying'} positions; "
                "a metric scores one position per row",
            )
        for name, write in one.writes.items():
            if isinstance(write.operand, str):
                have, want = one.reads[write.operand].pos.width, write.pos.width
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
        runs = self.runs()
        trained_at: dict[str, int] = {}
        for index, (name, step, _) in enumerate(runs):
            if isinstance(step, Fit):
                for param in step.params:
                    trained_at.setdefault(param, index)
        for index, (name, step, _) in enumerate(runs):
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
                    f"{runs[first or 0][0]!r} trains it, so it would run on "
                    "the untrained parameter. Move it after the fit — or, for a "
                    "deliberate untrained baseline, declare a second featurizer that "
                    "no fit names",
                )


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
