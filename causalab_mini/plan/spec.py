"""The authored document: the data, the places, and the steps that run.

A document is seven keys. Four of them are vocabulary, and nothing in them
runs:

    data           the datasets, by name — a table of rows on disk
    sites          the places a read or a write may name
    featurizers    the parameter sets, by name
    interventions  bundles of reads and writes, by name

`steps` is what runs, in the order written, and a step is one of five kinds:

    forward    one model call over one dataset's column, with the writes of
               the interventions it lists in force, taking its reads; the
               step itself is the logits it produced
    generate   the same call, decoding: the step itself is the ids it said
    metric     a score of one read, per row
    reduce     a mean, or a principal basis, of one read over its rows
    fit        a body of steps, run once per minibatch with an optimizer
               between; it produces the parameters it trained

A step's name is how everything after it reaches what it produced, by a
dotted reference: `patched.logits` is the read `logits` of step `patched`,
`iia` the metric, `fit.rot` the rotation `fit` trained, `fit.iia` the metric
of the fit's body on its held-out pass. `steps.saves`, the one reserved key,
names the references that go to disk. A step's name is unique where it is
written and a reference says which step it means, so nothing needs a naming
convention and no two values can collide.

What is *not* here, deliberately: the resolved values. A plan holds
`pos=(10, 10)` and token ids, and this holds `pos: -1` and a column name,
because resolving them is what a compiler does against a loaded model — so
checks that need the rows or the model (how many rows, how wide a site)
are the compiler's, and everything that is about the document alone is here.

pydantic does two things a hand-written checker was doing badly: `extra=
"forbid"` refuses an unknown key *anywhere* with the path to it, which is
the catch-all `document.py` needed four bugs to learn it wanted, and the
model dump is a JSON Schema — which the protocol itself does not have.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from typing import Annotated, Any, Iterator, Literal, NamedTuple, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field, PrivateAttr, StringConstraints, Tag, model_validator

from .. import address
from ..ops.metrics import COLUMNS as METRIC_COLUMNS
from ..shapes import Where
from . import sweep as sweep_module


class Node(BaseModel):
    """Every node refuses a key it does not know, and says where it was."""

    model_config = ConfigDict(extra="forbid", frozen=True)


#: What a name is: it cannot hold a `.`, which is how a reference reaches
#: into a step, or a `/`, which is a directory — a step's results are found
#: by its path in the plan.
NAME = r"[A-Za-z_][A-Za-z0-9_-]*"
Name = Annotated[str, StringConstraints(pattern=f"^{NAME}$")]

#: Step names the plan already uses: `saves` is a key of `steps` itself, and
#: `featurizers` is the step the compiler adds to build the parameter sets.
RESERVED = frozenset({"saves", "featurizers"})


# --------------------------------------------------------------------- #
# the vocabulary
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


class Dataset(Node):
    """A table of rows: `<dir>/<table>[#split]` under the run's data root.
    The same shape declared under `data` and written in place on a step, so
    a string on a step is always a *name*."""

    path: str


class Site(Node):
    component: str
    #: The one layer this site is at, as a one-element band — or `"all"`: a
    #: read here is taken at every layer of the model, in one call, and is
    #: one value with the layer axis first.
    layers: list[int] | Literal["all"] | None = None
    #: At a per-head tensor, the heads this site is. The site's width is then
    #: theirs — `len(heads) · head_dim` — so a featurizer, a swap or a harvest
    #: here is of those heads and leaves the others alone.
    heads: list[int] | None = None
    #: Anywhere with a known width, the single features this site is — the
    #: neurons of `mlp_activation`, say. Heads and units are one mechanism
    #: (a group of the feature axis) at two grains, so a site names one.
    units: list[int] | None = None

    @model_validator(mode="after")
    def _one_layer(self) -> "Site":
        if self.heads is not None and self.units is not None:
            raise ValueError("a site names heads or units, not both: they slice the same axis")
        if self.units is not None and (
            not self.units or len(set(self.units)) != len(self.units) or min(self.units) < 0
        ):
            raise ValueError("units is a non-empty list of distinct feature indices")
        if self.heads is not None:
            if not address.describe().get(self.component, {}).get("heads", False):
                per_head = sorted(n for n, one in address.describe().items() if one["heads"])
                raise ValueError(
                    f"{self.component!r} is not a per-head tensor; `heads` applies at {per_head}"
                )
            if not self.heads or len(set(self.heads)) != len(self.heads) or min(self.heads) < 0:
                raise ValueError("heads is a non-empty list of distinct head indices")
        if isinstance(self.layers, list) and len(self.layers) != 1:
            raise ValueError(
                "layers must be a one-element band, or \"all\"; a band spanning several "
                "layers is one address and is not implemented"
            )
        # the same question the protocol format asks, of the same table
        wrong = address.layered(self.component, 0 if self.layers == "all" else self.layers[0] if self.layers else None)
        if wrong is not None:
            raise ValueError(wrong)
        return self

    @property
    def spelling(self) -> str:
        """The site as a document writes it, and so what names one written
        in place: two sites that spell the same are the same place."""
        return json.dumps(self.model_dump(exclude_none=True), separators=(",", ":"))


#: Kinds that exist only as a file: a basis someone computed, a dictionary
#: someone trained elsewhere. A fit may not name one. (`sae` and `linear` are
#: an encoder/decoder pair with an error term — `k` comes from the bundle.)
LOADED_ONLY = frozenset({"pca", "sae", "linear"})


class Featurizer(Node):
    """A parameter set. `subspace` is a Cayley-parametrized rotation, drawn
    from a seed or loaded, and trainable; `pca` is a fixed basis, always
    loaded, never trained — the control a fit is compared against; `gate` is
    a learned binary mask over the site's units (DBM), soft while a fit
    updates it and hard whenever it is scored."""

    kind: Literal["subspace", "pca", "gate", "sae", "linear"]
    #: The rank of a basis. A gate has none: its features are the units.
    k: int | None = Field(default=None, gt=0)
    parametrization: Literal["cayley"] = "cayley"
    #: The draw the initial basis comes from. Absent, the seed of the fit
    #: that trains it, or 0.
    seed: int | None = None
    #: Load the parameter from a safetensors bundle a previous run wrote,
    #: instead of drawing it. Its header is checked against this document —
    #: model, site, layer, k, d — and a mismatch is refused by key.
    file_path: str | None = None

    @model_validator(mode="after")
    def _drawn_or_loaded(self) -> "Featurizer":
        if self.kind == "gate" and self.k is not None:
            raise ValueError("a gate masks every unit of its site and takes no k")
        if self.kind in ("subspace", "pca") and self.k is None:
            raise ValueError(f"a {self.kind} needs k, the rank of its basis")
        if self.kind in LOADED_ONLY and self.file_path is None:
            raise ValueError(f"a {self.kind} featurizer is loaded from a file; nothing draws or trains one")
        if self.kind == "gate" and self.seed is not None:
            raise ValueError("a gate starts at θ = 0, not from a draw; it has no seed")
        if self.file_path is not None and self.seed is not None:
            raise ValueError("a loaded featurizer has no seed: its weights are its bytes")
        return self


#: The components whose tensor is the residual stream — the only ones a
#: logits view makes sense of, because the final norm and head expect it.
RESIDUAL_STREAM = frozenset({"embeddings", "block_input", "block_output", "ln_final"})


def _spelling(raw: Any) -> Any:
    """The one sugar a document may use: `-1` is `{"index": -1}`.

    Every shipped document is written that way and the meaning is
    unambiguous. There is no other spelling — a decode step is
    `{"frame": "generated", "index": k}`, in the same vocabulary as
    everything else.
    """
    return {"index": raw} if isinstance(raw, int) and not isinstance(raw, bool) else raw


#: A position, as a document may write it: a `Where`, or the bare index
#: above — which the schema says too, so a document the model accepts is one
#: the schema accepts.
Position = Annotated[Where, BeforeValidator(_spelling, json_schema_input_type=int | Where)]


class Read(Node):
    #: A site declared under `sites`, by name, or one written here in place.
    site: Name | Site
    pos: Position
    #: `identity`, a declared featurizer, or `<fit>.<name>` — the one a fit
    #: trains, which is how every read and write of it names it.
    featurizer: str = "identity"
    #: `"logits"` projects a residual-stream read through the model's final
    #: norm and head. With a layer sweep and a `token_prob` metric that is
    #: the logit lens, as one document.
    view: Literal["raw", "logits"] = "raw"


class Write(Node):
    """`mechanism` and `operand` are two fields, not the protocol's
    `{"swap": "v_cf"}`, because a key that is itself the mechanism's name
    cannot be enumerated by a schema — and the schema is what an agent
    reads."""

    site: Name | Site
    pos: Position
    mechanism: Literal["swap", "add_scaled", "lerp", "gaussian", "clamp", "renormalize"]
    #: A read of an earlier step, `<step>.<read>`; a mean, by its step's
    #: name; a literal number (zero ablation is `0.0`); or nothing, for a
    #: mechanism that takes none.
    operand: str | float | None = None
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


#: (required, optional) params per mechanism.
MECHANISM_PARAMS: dict[str, tuple[set[str], set[str]]] = {
    "swap": (set(), set()),
    "add_scaled": (set(), {"scale"}),
    "lerp": ({"t"}, set()),
    "gaussian": ({"seed"}, {"scale"}),
    "clamp": (set(), {"lo", "hi"}),
    "renormalize": (set(), set()),
}

#: Mechanisms a document gives no operand: one draws its own noise, one only
#: bounds, one measures against the pre-write value the seam supplies.
NO_OPERAND = frozenset({"gaussian", "clamp", "renormalize"})


def _spelled(value: Any) -> str:
    """Which spelling of an intervention this is, by its JSON type — so a
    mistake inside one is reported against that spelling alone, not against
    each of the three it could have been."""
    return "name" if isinstance(value, str) else "list" if isinstance(value, list) else "written"


class Intervention(Node):
    """Reads and writes, and nothing else: an experiment is what the steps
    that list it do with it. A reference in here is resolved where it is
    used, so one declared intervention serves a fit's body and the steps
    that score what it trained."""

    reads: dict[Name, Read] = Field(default_factory=dict)
    writes: dict[Name, Write] = Field(default_factory=dict)


#: One intervention a step lists: a declared name, or one written in place.
Listed = Annotated[
    Union[Annotated[Name, Tag("name")], Annotated[Intervention, Tag("written")]], Discriminator(_spelled)
]

#: What a step's `interventions` is: one, or a list of them, in order.
Interventions = Annotated[
    Union[Annotated[Name, Tag("name")], Annotated[Intervention, Tag("written")], Annotated[list[Listed], Tag("list")]],
    Discriminator(_spelled),
]


# --------------------------------------------------------------------- #
# the steps
# --------------------------------------------------------------------- #


class _Call(Node):
    """What a forward and a generate share: one model call over one
    dataset's column, with some writes in force and some reads taken."""

    #: A dataset declared under `data`, by name, or one written in place.
    #: Forwards over one dataset see its rows paired by index, as the
    #: counterfactual and the base of one example are.
    data: Name | Dataset
    #: Which column is the prompt: `input`, `counterfactual_inputs[0]`, or a
    #: column of chat messages, which renders through the chat template.
    field: str
    #: The interventions whose writes are in force, in the order they apply,
    #: and whose reads are taken: a declared name, one written in place, or
    #: a list of either.
    interventions: Interventions = Field(default_factory=list)
    #: Reads of this call that belong to no intervention.
    reads: dict[Name, Read] = Field(default_factory=dict)


class Forward(_Call):
    kind: Literal["forward"]


#: `model.generate` arguments a step may not pass, and why: each changes how
#: many rows come back, what comes back, or what bounds the decode — which
#: `max_new_tokens` does, alone.
REFUSED_GENERATION = {
    "max_length": "the bound is `max_new_tokens`, and a second one would disagree with it",
    "return_dict_in_generate": "the step's result is the generated ids",
    "output_attentions": "a read is how a generate step keeps a tensor",
    "output_hidden_states": "a read is how a generate step keeps a tensor",
    "output_scores": "a read of the head is how a generate step keeps the scores",
    "output_logits": "a read of the head is how a generate step keeps the logits",
    "max_time": "a stop by the clock ends the decode wherever it has got to, before the taps "
    "at the steps after it have run — and the bound is `max_new_tokens`",
    "stop_strings": "a stop at a string ends the decode before its bound, and needs the "
    "tokenizer handed to generate, which a step cannot pass",
    "_from_model_config": "it is the config's bookkeeping, not an argument",
    "transformers_version": "it is the config's bookkeeping, not an argument",
}

#: Arguments that would turn one row into several.
ONE_ROW_EACH = ("num_beams", "num_return_sequences")


class Generate(_Call):
    """A call that decodes. Every key beyond the ones above is an argument of
    `model.generate`, passed as written — `do_sample`, `temperature`,
    `min_new_tokens` — and checked against transformers' `GenerationConfig`,
    so a misspelling is refused with its path. Nothing is defaulted: what the
    step does not say comes from the checkpoint's own generation config."""

    model_config = ConfigDict(extra="allow", frozen=True)

    kind: Literal["generate"]
    #: The bound: at most this many tokens, and the continuation frame's
    #: steps are numbered below it.
    max_new_tokens: int = Field(gt=0)
    __pydantic_extra__: dict[str, int | float | bool | str | list[int] | None]  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def generation(self) -> dict[str, Any]:
        """The arguments beyond the bound, as written."""
        return dict(self.model_extra or {})

    @model_validator(mode="before")
    @classmethod
    def _generate_arguments(cls, raw: Any) -> Any:
        """Every key beyond the step's own, checked by name before its value
        is: a refused argument is refused for what it does, whatever its
        value would have been."""
        for key, value in raw.items() if isinstance(raw, dict) else ():
            if key in cls.model_fields:
                continue
            if key in REFUSED_GENERATION:
                raise ValueError(f"`{key}` is not a generate step's to set: {REFUSED_GENERATION[key]}")
            if key not in _generation_arguments():
                raise ValueError(
                    f"`{key}` is neither a field of a generate step nor an argument of "
                    "model.generate (transformers' GenerationConfig)"
                )
            if key in ONE_ROW_EACH and value != 1:
                raise ValueError(f"`{key}: {value}` makes several rows of one; a step keeps its rows")
        return raw


@functools.cache
def _generation_arguments() -> frozenset[str]:
    from transformers import GenerationConfig  # here: only a generate step needs it

    return frozenset(GenerationConfig().to_dict())


class _Metric(Node):
    """A score of one read, per row. Every other field names a column, as
    `<dataset>.<column>`: the one dataset whose row i is scored against row
    i of the read."""

    kind: Literal["metric"]
    #: The read it scores, `<step>.<read>`, at one position per row.
    of: str
    token_form: Literal["space_prefixed"] = "space_prefixed"

    @property
    def references(self) -> tuple[str, ...]:
        """Its column fields, as written, in the order `ops.metrics.compute`
        takes them — read off the one table, so a class and the function it
        feeds cannot disagree about which is which."""
        return tuple(getattr(self, one) for one in METRIC_COLUMNS[getattr(self, "metric")])

    @property
    def columns(self) -> tuple[str, ...]:
        """The column names, each without its dataset."""
        return tuple(one.partition(".")[2] for one in self.references)

    @property
    def dataset(self) -> str:
        """The one dataset the columns are of — `Spec` refuses two."""
        return self.references[0].partition(".")[0]


class Match(_Metric):
    metric: Literal["match"]
    expected: str


class LogitDiff(_Metric):
    metric: Literal["logit_diff"]
    a: str
    b: str


class CrossEntropy(_Metric):
    metric: Literal["cross_entropy"]
    target: str


class TokenLogit(_Metric):
    metric: Literal["token_logit"]
    token: str


class TokenProb(_Metric):
    metric: Literal["token_prob"]
    token: str


Metric = Annotated[
    Union[Match, LogitDiff, CrossEntropy, TokenLogit, TokenProb], Field(discriminator="metric")
]


class Reduce(Node):
    """One read, reduced over its rows: averaged to one vector — which is how
    a corpus mean gets made without the unreduced activations ever leaving —
    or to its top-k principal directions, a basis a later document loads as a
    `pca` featurizer. A rectangle keeps its window; a ragged read, which has
    none, is over every position it found."""

    kind: Literal["reduce"]
    reduce: Literal["mean", "pca"]
    of: str
    k: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _k_iff_pca(self) -> "Reduce":
        if (self.reduce == "pca") != (self.k is not None):
            raise ValueError("`k` is for `reduce: pca`, and pca needs it")
        return self


class Optimizer(Node):
    name: Literal["adamw"] = "adamw"
    lr: float
    weight_decay: float = 0.0


class EarlyStop(Node):
    #: A metric of the fit's body, watched on its held-out pass.
    metric: Name
    mode: Literal["max", "min"] = "max"
    patience: int = Field(gt=0)


class Anneal(Node):
    """A gate's temperature across a fit: `start` on the first update, `end`
    on the last, geometric between. Toward zero the soft mask the optimizer
    sees becomes the hard mask the score uses."""

    start: float = Field(gt=0)
    end: float = Field(gt=0)


class Evaluation(Node):
    """The held-out pass is the body again, with these datasets in place of
    the ones it trains on — the same experiment, so it cannot drift from the
    one it scores. A dataset mapped to itself is the train-equals-test
    ablation, and says so."""

    data: dict[Name, Name]


class Fit(Node):
    kind: Literal["fit"]
    #: The declared featurizers this fit trains. Everywhere they are used,
    #: including in the body, they are `<fit>.<name>`.
    train: list[Name]
    #: Σ wᵢ·termᵢ, minimized. A term is a metric of the body, or
    #: `<gate>.mask` for a gate this fit trains — the mean of its soft mask,
    #: which is its L1 penalty.
    objective: list[tuple[float, str]]
    #: gate name -> its temperature schedule. A gate without one stays at 1.
    anneal: dict[Name, Anneal] = Field(default_factory=dict)
    epochs: int = Field(gt=0)
    #: Rows per update, drawn from every dataset of the body by one shuffle.
    batch_size: int = Field(gt=0)
    seed: int = 0
    optimizer: Optimizer
    early_stop: EarlyStop
    eval: Evaluation
    #: The body: its steps, run once per minibatch. It saves nothing itself —
    #: what it produces is `<fit>.<ref>` at the root.
    steps: Steps

    @model_validator(mode="after")
    def _one_body(self) -> "Fit":
        if self.steps.saves:
            raise ValueError("a fit's body saves nothing; save `<fit>.<ref>` under the root's `saves`")
        if any(isinstance(one, Fit) for _, one in self.steps.items()):
            raise ValueError("a fit's body holds no fit")
        return self


Step = Annotated[Union[Forward, Generate, Metric, Reduce, Fit], Field(discriminator="kind")]


class Steps(Node):
    """The steps, by name, in the order they run — and `saves`, the one key
    that is not a step: `{reference: file}`, relative to the output
    directory. A table for a metric (`.json`), a tensor for anything else
    (`.safetensors`)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    saves: dict[str, str] = Field(default_factory=dict)
    __pydantic_extra__: dict[str, Step]  # pyright: ignore[reportIncompatibleVariableOverride]

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter((self.model_extra or {}).items())

    def __getitem__(self, name: str) -> Any:
        return (self.model_extra or {})[name]

    def __contains__(self, name: object) -> bool:
        return name in (self.model_extra or {})

    @model_validator(mode="after")
    def _names(self) -> "Steps":
        for name, _ in self.items():
            _refuse(
                re.fullmatch(NAME, name),
                f"step {name!r}: a name is a letter or `_`, then letters, digits, `_` or `-` — "
                "a `.` is how a reference reaches into a step",
            )
            _refuse(name not in RESERVED, f"step {name!r}: the name is reserved ({sorted(RESERVED)})")
        return self


Fit.model_rebuild()


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


#: What a reference resolves to: its kind (`read`, `layers` — a read at
#: every layer — `ids`, `logits`, `metric`, `mean`, `pca`, `fit`, `trained`), the node that says what it is (for a read, the
#: `Read`; for a mean or a basis, the `Read` it reduced), and the scope it
#: belongs to — `""` for the root, a fit's name for its body.
class Ref(NamedTuple):
    kind: str
    node: Any
    scope: str

#: A reference's kind, as a refusal says it.
SAID = {
    "read": "a read",
    "layers": "a read at every layer",
    "ids": "a decode's ids",
    "logits": "a forward's logits",
    "metric": "a metric",
    "mean": "a mean",
    "pca": "a pca basis",
    "fit": "a fit's training record",
    "trained": "a trained parameter set",
}


class Spec(Node):
    """A document. Its checks are the ones that need no model and no rows."""

    header: Header = Field(default_factory=Header)
    model: Model
    data: dict[Name, Dataset] = Field(default_factory=dict)
    sites: dict[Name, Site] = Field(default_factory=dict)
    featurizers: dict[Name, Featurizer] = Field(default_factory=dict)
    #: Bundles of reads and writes, by name. Nothing here runs: a step lists
    #: the ones it runs.
    interventions: dict[Name, Intervention] = Field(default_factory=dict)
    steps: Steps
    #: each forward's ops, by the step's identity, once resolved (`ops`)
    _ops: dict[int, tuple[dict[str, Read], dict[str, Write]]] = PrivateAttr(default_factory=dict)

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

    def site(self, ref: str | Site) -> tuple[str, Site]:
        """A read's or a write's site, as `(label, site)`: a declared one
        under its name, one written in place under its own spelling."""
        return (ref, self.sites[ref]) if isinstance(ref, str) else (ref.spelling, ref)

    def dataset(self, ref: str | Dataset) -> str:
        """A step's dataset, by the key its rows are held under: a declared
        one's name, or the path of one written in place — which nothing else
        can name."""
        return ref if isinstance(ref, str) else ref.path

    def path(self, key: str) -> str:
        """Where a dataset's rows are, by its key."""
        return self.data[key].path if key in self.data else key

    def ops(self, step: Any) -> tuple[dict[str, Read], dict[str, Write]]:
        """A forward's reads and the writes in force, each by name: its own
        reads, then each listed intervention's reads and writes, in the order
        the list gives. One name means one op of the step, so a name two of
        them share is refused, naming both. Resolved once per step: the
        checks and the compiler all ask."""
        if id(step) not in self._ops:
            self._ops[id(step)] = self._resolve(step)
        return self._ops[id(step)]

    def _resolve(self, step: Any) -> tuple[dict[str, Read], dict[str, Write]]:
        listed = step.interventions if isinstance(step.interventions, list) else [step.interventions]
        reads: dict[str, Read] = dict(step.reads)
        writes: dict[str, Write] = {}
        origin = dict.fromkeys(reads, "the step")
        for index, one in enumerate(listed):
            if isinstance(one, str):
                label = f"intervention {one!r}"
                _refuse(
                    one in self.interventions,
                    f"undeclared intervention {one!r}; declared: {sorted(self.interventions)}",
                )
                one = self.interventions[one]
            else:
                label = "the intervention written in place" + (f" at [{index}]" if len(listed) > 1 else "")
            for kind, names in (("read", one.reads), ("write", one.writes)):
                for name in names:
                    _refuse(
                        name not in origin,
                        f"{kind} {name!r} is named by both {origin.get(name)} and {label}; "
                        "a name is one op of the step",
                    )
                    origin[name] = label
            reads.update(one.reads)
            writes.update(one.writes)
        return reads, writes

    def forwards(self) -> Iterator[tuple[str, _Call]]:
        """Every forward and generate in the document, a fit's body included,
        by its reference: `patched`, `fit.patched`."""
        for name, step in self.steps.items():
            if isinstance(step, _Call):
                yield name, step
            if isinstance(step, Fit):
                yield from ((f"{name}.{inner}", one) for inner, one in step.steps.items() if isinstance(one, _Call))

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

    @model_validator(mode="after")
    def _cross_check(self) -> "Spec":
        """Everything that is about two pieces at once. No single node owns
        any of it, so none of it is a field validator."""
        # who trains what, before anything is resolved: a trained featurizer
        # is named `<fit>.<name>` everywhere, and the bare name nowhere
        # `pairs.cf_answer` is a column and `patched.logits` a read, told apart
        # by the field they sit in; one name that is both would read as either
        steps = {name for name, _ in self.steps.items()}
        steps |= {inner for _, step in self.steps.items() if isinstance(step, Fit) for inner, _ in step.steps.items()}
        shared = sorted(steps & set(self.data))
        _refuse(
            not shared,
            f"{shared} is the name of a step and of a dataset; a reference `<name>.<part>` could "
            "mean either — rename one",
        )
        trainers: dict[str, str] = {}
        for name, step in self.steps.items():
            if not isinstance(step, Fit):
                continue
            for one in step.train:
                _refuse(
                    one in self.featurizers,
                    f"step {name!r}: trains {one!r}, which is not a declared featurizer",
                )
                _refuse(
                    self.featurizers[one].kind not in LOADED_ONLY,
                    f"step {name!r}: trains {one!r}, a {self.featurizers[one].kind}, which is fixed "
                    "by definition — a pca basis can be fine-tuned as a subspace loaded from it",
                )
                _refuse(
                    one not in trainers,
                    f"featurizer {one!r} is trained by both {trainers.get(one)!r} and {name!r}; one "
                    "parameter set is trained once — declare a second featurizer",
                )
                trainers[one] = name
        produced = _scope(self, self.steps, {}, trainers)

        # one featurizer name is one parameter set, so it acts at one site
        at: dict[str, set[str]] = {}
        for _, step in self.forwards():
            reads, writes = self.ops(step)
            for op in (*reads.values(), *writes.values()):
                if op.featurizer != "identity":
                    at.setdefault(op.featurizer.rpartition(".")[2], set()).add(self.site(op.site)[1].spelling)
        for name in self.featurizers:
            _refuse(
                len(at.get(name, ())) == 1,
                f"featurizer {name!r} is used at {len(at.get(name, ()))} site(s) {sorted(at.get(name, ()))}; "
                "one name is one parameter set at one site",
            )

        files: dict[str, str] = {}
        for ref, file in self.steps.saves.items():
            found = produced.get(ref)
            _refuse(
                found is not None,
                f"saves: {ref!r} is nothing this document produces; it produces {sorted(produced)}",
            )
            assert found is not None
            _refuse(
                not file.startswith("/") and ".." not in file.split("/"),
                f"saves: {ref!r} -> {file!r}: a file is a path inside the output directory",
            )
            _refuse(file not in files, f"saves: {files.get(file)!r} and {ref!r} are both saved to {file!r}")
            files[file] = ref
            table = found.kind == "metric"
            _refuse(
                file.endswith(".json" if table else ".safetensors"),
                f"saves: {ref!r} is " + (
                    "a metric, one row per example: a table, saved to a .json file"
                    if table else f"{SAID[found.kind]}, a tensor: saved to a .safetensors file"
                ) + f", not {file!r}",
            )
        return self


def _refuse(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _scope(spec: Spec, steps: Steps, outer: dict[str, Ref], trainers: dict[str, str], fit: str = "") -> dict[str, Ref]:
    """Check one `steps`, in order, and return what it produces, by reference.

    A reference names an *earlier* step, of this `steps` or of the one around
    it — a fit's body sees the root's steps written before the fit — so
    walking in order and adding each step's references after checking it is
    the whole scoping rule. `fit` is the body's fit, or `""` at the root.
    """
    visible = dict(outer)
    produced: dict[str, Ref] = {}
    #: the fits a `<fit>.<name>` may name here: those already run, and the
    #: one whose body this is — inside it, the parameter it is training
    fits = {ref for ref, (kind, _, _) in outer.items() if kind == "fit"} | ({fit} if fit else set())

    def publish(ref: str, kind: str, node: Any, scope: str = fit) -> None:
        visible[ref] = produced[ref] = Ref(kind, node, scope)

    for name, step in steps.items():
        where = f"step {(f'{fit}.' if fit else '') + name!r}"
        if isinstance(step, _Call):
            reads, _ = _call(spec, where, name, step, visible, fit, trainers, fits)
            for read_name, read in reads.items():
                every = spec.site(read.site)[1].layers == "all"
                publish(f"{name}.{read_name}", "layers" if every else "read", read)
            # the call's own result: a decode's ids, a forward's logits
            publish(name, "ids" if isinstance(step, Generate) else "logits", step)
        elif isinstance(step, (_Metric, Reduce)):
            found = visible.get(step.of)
            _refuse(
                found is not None and found.kind in ("read", "layers") and found.scope == fit,
                f"{where}: `of` is {step.of!r}, which is not a read of a step before it in this "
                "`steps`; a metric or a reduce is of `<step>.<read>`",
            )
            assert found is not None
            _refuse(
                isinstance(step, _Metric) or found.kind == "read",
                f"{where}: {step.of!r} is read at every layer; a metric scores it layer by layer, "
                "and a reduction of it is not implemented",
            )
            if isinstance(step, Reduce):
                publish(name, step.reduce, found.node)
                continue
            _refuse(
                found.node.pos.width == 1,
                f"{where} scores {step.of!r}, a window of {found.node.pos.width or 'varying'} "
                "positions; a metric scores one position per row",
            )
            datasets = {one.partition(".")[0] for one in step.references}
            for one in step.references:
                _refuse(
                    "." in one and one.partition(".")[0] in spec.data,
                    f"{where}: {one!r} is not `<dataset>.<column>` of a dataset declared under `data` "
                    f"({sorted(spec.data)}); a dataset written in place on a step has no name to name it by",
                )
            _refuse(
                len(datasets) == 1,
                f"{where}: its columns come from {sorted(datasets)}; a metric's columns are one "
                "dataset's, row i against row i of the read",
            )
            publish(name, "metric", step)
        elif isinstance(step, Fit):
            body = _scope(spec, step.steps, visible, trainers, name)
            _fit(spec, where, name, step, body)
            publish(name, "fit", step, "")
            for one in step.train:
                publish(f"{name}.{one}", "trained", one, "")
            for ref, (kind, node, _) in body.items():
                # the body's values, as its held-out pass left them
                publish(f"{name}.{ref}", kind, node, name)
            fits.add(name)
    return produced


def _call(
    spec: Spec,
    where: str,
    name: str,
    step: _Call,
    visible: dict[str, Ref],
    fit: str,
    trainers: dict[str, str],
    fits: set[str],
) -> tuple[dict[str, Read], dict[str, Write]]:
    """One forward or generate: its data, and every read and write in force,
    each resolved where this step is."""
    try:
        reads, writes = spec.ops(step)
    except ValueError as refusal:
        raise ValueError(f"{where}: {refusal}") from None
    if isinstance(step.data, str):
        _refuse(
            step.data in spec.data,
            f"{where}: undeclared dataset {step.data!r}; declared: {sorted(spec.data)} — or "
            'write one in place, {"path": …}',
        )
    # how many steps a continuation-frame position may name: none on a forward
    bound = step.max_new_tokens if isinstance(step, Generate) else 0

    def site(what: str, ref: str | Site) -> Site:
        _refuse(
            not isinstance(ref, str) or ref in spec.sites,
            f"{where}: {what}: undeclared site {ref!r}; declared: {sorted(spec.sites)}",
        )
        return spec.site(ref)[1]

    def frame(what: str, pos: Where) -> None:
        if pos.frame != "generated":
            return
        _refuse(bound, f"{where}: {what}: a position in the continuation frame needs a generate step")
        _refuse(
            pos.index is None or pos.index < bound,
            f"{where}: {what}: step {pos.index} of a {bound}-token decode",
        )

    for read_name, read in reads.items():
        what = f"read {read_name!r}"
        component = site(what, read.site).component
        _featurizer(spec, f"{where}: {what}", read.featurizer, trainers, fits)
        if site(what, read.site).layers == "all":
            # one value with the layer axis first; what would make it more
            # than a read — a parameter set, a cut over the decode — is one
            # site, or one step, and every layer is many
            _refuse(
                read.featurizer == "identity",
                f"{where}: {what}: a read at every layer takes no featurizer; one featurizer "
                "is one parameter set at one site, and every layer is many sites",
            )
            _refuse(
                read.pos.frame == "prompt" or (read.pos.index is not None and read.pos.index >= 0),
                f"{where}: {what}: a read at every layer is at one decode step or in the prompt; "
                f"{read.pos.spelling()} is cut from every step of the decode",
            )
        if read.view == "logits":
            _refuse(
                component in RESIDUAL_STREAM,
                f"{where}: {what}: view 'logits' projects the residual stream through the head, "
                f"and {component!r} is not the residual stream (one of {sorted(RESIDUAL_STREAM)})",
            )
            _refuse(
                read.featurizer == "identity",
                f"{where}: {what}: a featurized read cannot also be viewed as logits",
            )
        frame(what, read.pos)

    for write_name, write in writes.items():
        what = f"write {write_name!r}"
        component = site(what, write.site).component
        _refuse(
            site(what, write.site).layers != "all",
            f"{where}: {what}: a write is at one layer — a write at every layer is as many "
            "experiments; sweep `layers` for them",
        )
        _refuse(
            not address.describe().get(component, {}).get("read_only", False),
            f"{where}: {what}: {component!r} is read-only — the model's input, not an activation",
        )
        _featurizer(spec, f"{where}: {what}", write.featurizer, trainers, fits)
        if isinstance(write.operand, str):
            _operand(where, what, name, write, visible, fit)
        frame(what, write.pos)
        if write.pos.frame == "generated":
            # A write happens *during* the decode, so it may only name a step
            # the run has already reached. Every other form of the
            # continuation frame is a cut of the finished text — the last
            # real token, the row's stop token, where it said the answer —
            # and there is nothing to write into a step that has not happened.
            _refuse(
                write.pos.all or (write.pos.index is not None and write.pos.index >= 0),
                f"{where}: {what}: a write in the continuation frame is at a step the decode has "
                f"reached — {{'index': k}} with k >= 0, or {{'all': true}}. {write.pos.spelling()} "
                "names a cut of the finished continuation, which is something to read and not "
                "something to write into",
            )
            _refuse(
                write.pos.scope is None,
                f"{where}: {what}: a write in the continuation frame takes no scope; "
                f"{write.pos.spelling()} is only known once the decode has finished",
            )

    # renormalize restores the norm the writes before it at its site changed
    order = list(writes.items())
    for index, (write_name, write) in enumerate(order):
        if write.mechanism != "renormalize":
            continue
        here = [one for one, other in order if spec.site(other.site)[1] == spec.site(write.site)[1]]
        _refuse(
            len(here) > 1 and here[-1] == write_name,
            f"{where}: renormalize {write_name!r} restores the norm the other writes at its site "
            "changed, so it comes after at least one of them and last among them; first, or "
            "alone, it is the identity",
        )

    if isinstance(step, Generate) and (reads or writes):
        # The engine walks a tapped decode step by step, and an early EOS
        # makes that walk outrun the run and drop what follows it — the
        # step's own ids among it (FINDINGS §12.4).
        _refuse(
            step.generation.get("min_new_tokens") == step.max_new_tokens,
            f"{where}: a generate step with reads or writes decodes exactly `max_new_tokens`: say "
            f"`min_new_tokens: {step.max_new_tokens}`. An early stop would end the decode before "
            "the taps have run",
        )
    return reads, writes


def _operand(where: str, what: str, name: str, write: Write, visible: dict[str, Ref], fit: str) -> None:
    """A write's operand, when it is a reference: a read of an earlier step,
    or a mean — and, either way, of a window this write can take."""
    ref = str(write.operand)
    _refuse(
        not ref.startswith(f"{name}."),
        f"{where}: {what}: operand {ref!r} is read by this same step; a write takes a value "
        "that exists before the call it is in",
    )
    found = visible.get(ref)
    _refuse(
        found is not None,
        f"{where}: {what}: operand {ref!r} is nothing before this step; an operand is "
        "`<step>.<read>` of an earlier step, the name of a mean, or a number",
    )
    assert found is not None
    kind, source, scope = found
    _refuse(
        kind in ("read", "mean"),
        f"{where}: {what}: operand {ref!r} is {SAID[kind]}; an operand is a read of an earlier step "
        "or a mean" + ("; a pca basis is loaded as a featurizer, not written at a site" if kind == "pca" else ""),
    )
    _refuse(
        scope in (fit, ""),
        f"{where}: {what}: operand {ref!r} is a value of {scope!r}'s held-out pass, which is saved, "
        "not written",
    )
    want, have = write.pos.width, source.pos.width  # None: ragged, known per row
    if kind == "read":
        _refuse(
            scope == fit,
            f"{where}: {what}: operand {ref!r} is a read from outside the fit's body, whose rows "
            "the fit shuffles; reduce it to a mean, or read it in the body",
        )
        _refuse(
            source.view == "raw",
            f"{where}: {what}: operand {ref!r} is a logits view, which is vocabulary-wide and "
            "cannot be written back at a site",
        )
        # a ragged side has no width until the rows are known; the run
        # checks those row by row
        _refuse(
            have is None or want is None or have == want,
            f"{where}: {what} covers {want} position(s) but its operand {ref!r} was read over "
            f"{have}; the windows must match",
        )
        return
    # A mean over a ragged read is one vector and lands anywhere; a mean
    # over a rectangle keeps its window, and the write must match it.
    _refuse(
        have is None or want == have or (want is None and have == 1),
        f"{where}: {what} covers {want if want is not None else 'a varying number of'} "
        f"position(s) but the mean {ref!r} keeps a window of {have}; the windows must match",
    )


def _featurizer(spec: Spec, where: str, ref: str, trainers: dict[str, str], fits: set[str]) -> None:
    """A featurizer, as a read or a write names it: `identity`, a declared
    one no fit trains, or `<fit>.<name>` of a fit that has run — or of the
    fit whose body this is, where it is the parameter being trained.

    One spelling for a trained parameter everywhere is what lets a fit's
    body and the scoring after it list one declared intervention. And a
    parameter named before its fit has run would score untrained without
    complaint, so that is refused, with the fix.
    """
    if ref == "identity":
        return
    head, dot, tail = ref.partition(".")
    if not dot:
        _refuse(
            ref in spec.featurizers,
            f"{where}: undeclared featurizer {ref!r}; declared: {sorted(spec.featurizers)}",
        )
        _refuse(
            ref not in trainers,
            f"{where}: featurizer {ref!r} is trained by step {trainers.get(ref)!r}; name it "
            f"{trainers.get(ref)}.{ref} — or, for a deliberate untrained baseline, declare a "
            "second featurizer that no fit names",
        )
        return
    _refuse(
        trainers.get(tail) == head,
        f"{where}: featurizer {ref!r}: step {head!r} trains no {tail!r} "
        f"(trained: {sorted(f'{fit}.{one}' for one, fit in trainers.items())})",
    )
    _refuse(
        head in fits,
        f"{where}: featurizer {ref!r} is used before step {head!r} trains it, so it would run "
        "on the untrained parameter. Move it after the fit — or, for a deliberate untrained "
        "baseline, declare a second featurizer that no fit names",
    )


def _fit(spec: Spec, where: str, name: str, fit: Fit, body: dict[str, Ref]) -> None:
    """A fit's own fields, against its body."""
    clash = sorted(set(fit.train) & {step for step, _ in fit.steps.items()})
    _refuse(
        not clash,
        f"{where}: {clash} is both a step of its body and a featurizer it trains, and "
        f"`{name}.{clash[0] if clash else ''}` can name only one of them",
    )
    metrics = {ref for ref, (kind, _, _) in body.items() if kind == "metric"}
    gates = {one for one in fit.train if spec.featurizers[one].kind == "gate"}
    _refuse(
        set(fit.anneal) <= gates,
        f"{where}: anneals {sorted(set(fit.anneal) - gates)}; only a gate this fit trains has a "
        f"temperature (those are {sorted(gates)})",
    )
    terms = metrics | {f"{gate}.mask" for gate in gates}
    for _, term in fit.objective:
        _refuse(
            term in terms,
            f"{where}: objective names {term!r}, which is neither a metric of its body nor the "
            f"mask of a gate it trains (one of {sorted(terms)})",
        )
    _refuse(
        fit.early_stop.metric in metrics,
        f"{where}: early_stop watches {fit.early_stop.metric!r}, which is not a metric of its "
        f"body ({sorted(metrics)})",
    )
    # The held-out run is the body with every dataset it names replaced, so
    # each must have a name to be replaced by, and each must be: one left out
    # would be scored on the rows the fit trained on, beside ones that were not.
    for inner, one in fit.steps.items():
        _refuse(
            not isinstance(one, _Call) or isinstance(one.data, str),
            f"{where}: step {inner!r} writes its dataset in place; a fit's body declares each one "
            "under `data`, so `eval.data` can say what its held-out run uses instead",
        )
    named = {one.data for _, one in fit.steps.items() if isinstance(one, _Call) and isinstance(one.data, str)}
    named |= {one.dataset for _, one in fit.steps.items() if isinstance(one, _Metric)}
    for trained, held_out in fit.eval.data.items():
        _refuse(
            trained in named,
            f"{where}: eval replaces {trained!r}, which its body does not use ({sorted(named)})",
        )
        _refuse(held_out in spec.data, f"{where}: eval: undeclared dataset {held_out!r}")
    missing = sorted(named - set(fit.eval.data))
    _refuse(
        not missing,
        f"{where}: eval.data says nothing of {missing}, which its body uses; the held-out run "
        "would score them on the rows the fit trained on. Map each to its held-out dataset — "
        "or to itself, which is the train-equals-test ablation, and says so",
    )
