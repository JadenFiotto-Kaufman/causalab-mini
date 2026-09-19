"""The nnterp engine: nnsight traces, one session, local or on NDIF.

The whole request is one `model.session(...)`: not a session per forward, not
a session per training step, not a loop outside that opens sessions. `remote`
is passed to that one session and is the only difference between running here
and running on NDIF — there is no second code path.

Results come home because the session saves the **plan**. `nnsight.save`
pushes the server's copy back and hands you the object to fill in; locally
that is the plan you passed in, remotely it is the copy that will be shipped
home. Either way the steps write into the object the save handed back, so
nothing has to be reattached afterwards and nothing is lost by filling in the
wrong copy.

Everything inside a traced block is a module-level function taking plan data
and tensors. No `self`, no document, no tokenizer: nnsight ships every name a
block loads as a whole pickled object, so a block that reads one of those
ships it. `cls` is allowed — an engine is a stateless class and pickles by
reference out of a registered package.
"""

from __future__ import annotations

from typing import Any

import nnsight
import torch

from ..model.address import Address, AddressError
from ..ops import intervene
from ..plan import Forward, Plan
from . import steps
from .base import Engine


class NNterpEngine(Engine):
    @classmethod
    def execute(cls, model: Any, plan: Plan, remote: bool | str = False) -> Plan:
        if remote:
            # Our own package is not installed on an NDIF server, so the
            # functions the block calls have to ship by value.
            nnsight.register("causalab_mini")
        with model.session(remote=remote):
            executed = nnsight.save(plan)
            steps.run(cls, model, executed)
        return executed

    @classmethod
    def forward(
        cls,
        model: Any,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        with model.trace(batch(forward)):
            apply_taps(model, forward, values, featurizers)


def batch(forward: Forward) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes."""
    return {
        "input_ids": torch.tensor(forward.input_ids),
        "attention_mask": torch.tensor(forward.attention_mask),
    }


def apply_taps(
    model: Any, forward: Forward, values: dict[str, Any], featurizers: dict[str, Any]
) -> None:
    """One pass over the addresses of one forward, in forward order.

    `values` carries reads between the forwards of one pass: an operand is a
    read name, and its tensor was produced by an earlier forward. A read's
    featurizer is applied here too — `v_cf` is `Qᵀx`, not `x` — and it is the
    same object the write's `inverse` will use, which is what makes one
    featurizer name one parameter set.
    """
    for tap in forward.taps:
        for write_op in tap.writes:
            patched = intervene.apply_write(
                read(model, tap.address),
                write_op.positions,
                values[write_op.operand],
                write_op.mechanism,
                featurizers[write_op.featurizer],
                tap.address.seq_axis,
            )
            write(model, tap.address, patched)
        for read_op in tap.reads:
            gathered = intervene.gather(
                read(model, tap.address), read_op.positions, tap.address.seq_axis
            )
            values[read_op.name] = featurizers[read_op.featurizer].featurize(gathered)[0].clone()


def read(model: Any, address: Address) -> Any:
    """The tensor at `address`, during a trace.

    At a module boundary the output may be a bare tensor or a tuple whose
    first element is the hidden state, and which one it is depends on the
    transformers version, not on anything we can see in the document — so it
    is decided from the value. Inside a forward the value is one argument of
    one call, and nothing is ambiguous.

    This is the engine's half of an address: `address` says *where*, in terms
    that are true of the architecture, and this says how to reach there with
    nnsight. A different engine says it differently.
    """
    if not address.interior:
        value = getattr(address.envoy(model), address.side)
        return value[0] if isinstance(value, tuple) else value
    args, _ = operation(model, address).inputs
    return args[address.arg]


def write(model: Any, address: Address, tensor: Any) -> None:
    """Put a tensor back: rebuilding the tuple if there was one, or rebuilding
    the call's arguments around the new one."""
    if not address.interior:
        envoy = address.envoy(model)
        current = getattr(envoy, address.side)
        setattr(
            envoy,
            address.side,
            (tensor, *current[1:]) if isinstance(current, tuple) else tensor,
        )
        return
    call = operation(model, address)
    args, kwargs = call.inputs
    index = address.arg
    call.inputs = ((*args[:index], tensor, *args[index + 1 :]), kwargs)


def operation(model: Any, address: Address) -> Any:
    """The `.source` operation an interior address names."""
    if address.op is None:
        raise AddressError(
            f"component {address.component!r} is an interior; build its address "
            "with Address.locate(model, ...) so the operation is resolved"
        )
    return getattr(address.envoy(model).source, address.op)
