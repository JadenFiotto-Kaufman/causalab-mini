"""The frozen plan: what will run, in forward order.

A plan holds strings, integers and nothing else — no envoys, no tensors, no
tokenizer, no document. It pickles with plain `pickle`, which is the test that
keeps it honest. Everything that needed a decision (which rows, which tokens,
which position, which module path, which order) was decided by `build`, on the
client, before the session opened.

The shape, top down:

    Plan
      forwards: one per (model, input), already in execution order
        Forward(name, input, input_ids, attention_mask, taps)
          taps: one per address, in forward order
            Tap(address, writes, reads)        writes run before reads at the
              WriteOp(name, positions, operand, mechanism, featurizer)  same
              ReadOp(name, positions, featurizer)                       address
      metrics:     MetricOp(name, kind, of, ids)
      featurizers: FeaturizerOp(name, kind, k, d, parametrization, seed, trained)
      train:       TrainPlan(epochs, evaluation, objective, …) or None
      saves:       SaveFile(file_path, value, example_ids, unit, …)

A fit is a request, so a fit is one plan: `TrainPlan` holds the rows of every
update it will make, already batched, already tokenized, already in the order
the seed puts them in — as **plans**, because a training step is this same plan
over different rows, and a twin type beside `Plan` would be a lie about that.

This file is the shape only. `build.py` is the compiler that fills it in.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.address import Address
from ..shapes import ExampleIds, Positions, TokenIds, TokenRows


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class ReadOp:
    name: str
    positions: Positions
    featurizer: str = "identity"


@dataclass(frozen=True)
class WriteOp:
    name: str
    positions: Positions
    operand: str  # the name of a read, produced by an earlier forward
    mechanism: str
    featurizer: str


@dataclass(frozen=True)
class Tap:
    address: Address
    writes: tuple[WriteOp, ...]
    reads: tuple[ReadOp, ...]


@dataclass(frozen=True)
class Forward:
    name: str  # "original" or an intervened model's name
    input: str  # the data role its rows come from
    input_ids: TokenRows
    attention_mask: TokenRows
    taps: tuple[Tap, ...]


@dataclass(frozen=True)
class MetricOp:
    name: str
    kind: str
    of: str  # the read it binds to
    ids: tuple[TokenIds, ...]  # one vocabulary id per row, per operand


@dataclass(frozen=True)
class FeaturizerOp:
    """One parameter set. `d` is derived from (model, site) here on the client,
    because the block may not decide anything from a tensor — including how wide
    the tensor it is about to rotate is."""

    name: str
    kind: str
    k: int
    d: int
    parametrization: str
    seed: int
    trained: bool


@dataclass(frozen=True)
class SaveFile:
    file_path: str
    value: str
    example_ids: ExampleIds = ()
    unit: str = ""
    estimand_version: str = ""
    produced_by: str = ""
    #: For a `.safetensors` bundle: the ArtifactIdentity stamped into its header,
    #: as pairs because a plan holds no dicts. Empty for a metric table.
    identity: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TrainPlan:
    """Every update the fit will make, as plans over the rows of that update.

    `epochs` is already shuffled: the seed covers data order, and data order
    decides which rows share a padded batch, which is a tokenizer question and
    therefore a client-side one. The *parameter* seed travels as data and is
    drawn inside the session.
    """

    epochs: tuple[tuple["Plan", ...], ...]
    evaluation: "Plan"
    objective: tuple[tuple[float, str], ...]
    params: tuple[str, ...]
    lr: float
    weight_decay: float
    eval_metrics: tuple[str, ...]
    early_stop: str
    patience: int
    mode: str


@dataclass(frozen=True)
class Plan:
    forwards: tuple[Forward, ...]
    metrics: tuple[MetricOp, ...]
    saves: tuple[SaveFile, ...] = ()
    featurizers: tuple[FeaturizerOp, ...] = ()
    train: TrainPlan | None = None
