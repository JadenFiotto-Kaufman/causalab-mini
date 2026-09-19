"""What an engine is: the contract, and nothing else.

An engine is how a plan reaches tensors. Everything else — the walk over the
steps, the fit loop, the metrics, the write algebra — is the same whatever
runtime you are on, and lives in `steps.py` as plain functions that take an
engine as their first argument. Nothing is inherited, so a second engine
inherits no loop it did not ask for.

The contract is two classmethods, because an engine is a way of doing things
and not a thing with state. (That also keeps it cheap inside a traced block: a
class ships by reference out of a registered package, where an instance would
ship whole.)

    execute(model, plan, remote)  ->  the plan, filled in
    forward(model, forward, values, featurizers)      run one forward, tapped

Neither has a default. `execute` in particular is three lines that differ
completely between runtimes — one opens a session and saves the plan into it,
another opens nothing at all — and an engine that inherited a default would be
inheriting someone else's idea of what a request is.
"""

from __future__ import annotations

from typing import Any

from ..plan import Forward, Plan


class Engine:
    @classmethod
    def execute(cls, model: Any, plan: Plan, remote: bool | str = False) -> Plan:
        """Run `plan` against `model` and return the plan, filled in.

        The plan that comes back is not always the one passed in: an engine
        whose run happens elsewhere fills in a copy and hands that back.
        """
        raise NotImplementedError

    @classmethod
    def forward(
        cls,
        model: Any,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        """Run one forward with its taps applied, leaving what it read in
        `values`. This is the only place an engine touches a model's insides."""
        raise NotImplementedError
