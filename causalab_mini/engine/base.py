"""What an engine is: the contract, and nothing else.

An engine is a runtime — a way of loading a model and reaching the tensors
inside it. Everything a plan *means* is the same on every runtime and lives in
`steps.py` as plain functions taking an engine; nothing here has a body, so an
engine inherits no behaviour it did not ask for.

An engine is an object, not a namespace, because it holds the model it loaded.
`load` is a classmethod and everything else is an instance method.

The contract in two halves. What the **compiler** asks, because a plan is
compiled against a particular model and none of it may be decided later:

    load(spec, weights)  the model — or, with weights=False, its shape only
    tokenizer            resolves prompts and answer columns
    num_layers           bounds the layer band a site may name
    locate(component, layer) -> Address      resolve a site, now, on the client
    width(address) -> int                    how wide the tensor there is

and what the **run** asks:

    execute(plan, remote) -> Plan            the whole request
    forward(forward, values, featurizers)    one forward, tapped

That is all. The walk over steps, the fit loop, the metrics and the write
algebra are shared, so a second engine is these seven members and no more.

One rule for an engine that traces: a block may load the engine and the plan,
never the document, the tokenizer or the dataset — nnsight ships every name a
block loads as a whole pickled object. `tests/test_structure.py` checks it.
"""

from __future__ import annotations

from typing import Any

from ..address import Address
from ..plan import Forward, Plan


class EngineError(ValueError):
    pass


class Engine:
    #: The loaded model. What kind of object it is, is the engine's business.
    model: Any
    #: Whether the model has weights. An engine loaded with `weights=False`
    #: answers everything the compiler asks — the tokenizer, the layer count,
    #: the widths, an interior's operation — from the config and a meta-device
    #: module tree, in well under a second for any size of model. It cannot
    #: run, and `execute` on it refuses by name.
    weights: bool

    # ----------------------------------------------------------------- #
    # loading
    # ----------------------------------------------------------------- #

    @classmethod
    def load(cls, spec: Any, weights: bool = True, **options: Any) -> "Engine":
        """The engine, holding the model the document named.

        `spec` is a model block from either authoring format — both carry
        `key`, `revision` and `dtype`, and an engine needs nothing else.
        `weights=False` is how a document is compiled, validated and explained
        on a machine that will never run it.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------- #
    # what the compiler asks
    # ----------------------------------------------------------------- #

    @property
    def tokenizer(self) -> Any:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    def locate(self, component: str, layer: int | None = None) -> Address:
        """The address of `(component, layer)` on this model.

        A module boundary is just the pair. An interior also needs the
        operation inside the forward resolved, and how that is done is the
        runtime's business — which is why this is asked of the engine and not
        of the address.
        """
        raise NotImplementedError

    def width(self, address: Address) -> int:
        """The size of the tap's last axis — the `d` a featurizer's `k` is a
        subspace of. Derived from (model, site) and never authored."""
        raise NotImplementedError

    # ----------------------------------------------------------------- #
    # what the run asks
    # ----------------------------------------------------------------- #

    def execute(self, plan: Plan, remote: bool | str = False) -> Plan:
        """Run `plan` and return it, filled in.

        The plan that comes back is not always the one passed in: an engine
        whose run happens elsewhere fills in a copy and hands that back.
        """
        raise NotImplementedError

    def forward(
        self,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        """Run one forward with its taps applied, leaving what it read in
        `values`. This is the only place an engine touches a model's insides."""
        raise NotImplementedError
