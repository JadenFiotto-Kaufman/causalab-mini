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
and tensors. No document, no tokenizer, no `self`: nnsight ships every name a
block loads as a whole pickled object, so a block that reads one of those
ships it. The engine is bound to a local first and passed explicitly — it is
the one object the block genuinely needs, and all it carries is the model,
which ships as a reference to the loaded module either way.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import nnsight
import torch

from .... import address as address_module
from ....address import Address, AddressError
from ....ops import intervene
from ....plan import Forward, Plan
from ... import provenance, steps
from ...base import Engine
from .loading import load


class NNterpEngine(Engine):
    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def load(cls, spec: Any, **options: Any) -> "NNterpEngine":
        return cls(load(spec, **options))

    # ----------------------------------------------------------------- #
    # what the compiler asks
    # ----------------------------------------------------------------- #

    @property
    def tokenizer(self) -> Any:
        return self.model.tokenizer

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    def locate(self, component: str, layer: int | None = None) -> Address:
        """The address, with an interior's operation resolved here on the
        client: it is named by the loaded checkpoint's forward, so a document
        that cannot be addressed should fail at compile time and not inside
        someone else's process."""
        address = Address(component, layer, family=getattr(self.model.config, "model_type", None))
        address_module.check(self.model.config, address)
        if address.call_site is None:
            return address
        source = address.resolve(self.model).source
        return replace(address, op=find_op(source, address.call_site))

    def heads(self, address: Address) -> int:
        return address_module.head_count(self.model.config, address)

    def width(self, address: Address) -> int:
        return address_module.width(self.model.config, address)

    # ----------------------------------------------------------------- #
    # what the run asks
    # ----------------------------------------------------------------- #

    def execute(self, plan: Plan, remote: bool | str = False, batch_size: int | None = None) -> Plan:
        if remote:
            # Our own package is not installed on an NDIF server, so the
            # functions the block calls have to ship by value.
            nnsight.register("causalab_mini")
        plan.provenance.update(provenance.record(self, remote, batch_size))
        engine, model = self, self.model
        with model.session(remote=remote):
            executed = nnsight.save(plan)
            steps.run(engine, executed, batch_size=batch_size)
        return executed

    def forward(
        self,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        model = self.model
        if not forward.decode:
            with model.trace(batch(forward)):
                apply_taps(model, forward, values, featurizers)
            return
        # The continuation frame: one generate trace, `tracer.iter` walking
        # the steps. Prompt-frame taps apply at step 0, the prefill; a step's
        # taps at that step; an `"all"` write at every step. Greedy, and EOS
        # held off so the bound holds and the loop never outruns the run.
        decode, prompt = forward.decode, len(forward.input_ids[0])
        with model.generate(
            batch(forward), max_new_tokens=decode, min_new_tokens=decode, do_sample=False
        ) as tracer:
            for step in tracer.iter[:decode]:
                apply_taps(model, forward, values, featurizers, step)
            values[f"{forward.name}.generated"] = tracer.result[:, prompt:].clone()


def batch(forward: Forward) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes."""
    return {
        "input_ids": torch.tensor(forward.input_ids),
        "attention_mask": torch.tensor(forward.attention_mask),
    }


def apply_taps(
    model: Any,
    forward: Forward,
    values: dict[str, Any],
    featurizers: dict[str, Any],
    step: int | None = None,
) -> None:
    """One pass over the addresses of one forward, in forward order.

    `values` carries reads between the forwards of one pass: an operand is a
    read name, and its tensor was produced by an earlier forward. A read's
    featurizer is applied here too — `v_cf` is `Qᵀx`, not `x` — and it is the
    same object the write's `inverse` will use, which is what makes one
    featurizer name one parameter set.

    `step` is None for a plain forward, and the decode step inside a
    generate trace. A tap applies when its frame is this step: the prompt
    frame at step 0 (or a plain forward), a step's own taps at that step,
    an `"all"` write at every step.
    """
    for tap in forward.taps:
        if not intervene.applies(tap.step, step):
            continue
        for write_op in tap.writes:
            patched = intervene.apply_write(
                read(model, tap.address),
                intervene.at_step(write_op.at, read(model, tap.address), tap.address.seq_axis, tap.step, step),
                intervene.resolve_operand(values, write_op.operand),
                write_op.mechanism,
                featurizers[write_op.featurizer],
                tap.address.seq_axis,
                write_op.params,
            )
            write(model, tap.address, patched)
        for read_op in tap.reads:
            tensor = read(model, tap.address)
            gathered = intervene.gather(
                tensor,
                intervene.at_step(read_op.at, tensor, tap.address.seq_axis, tap.step, step),
                tap.address.seq_axis,
            )
            if read_op.view == "logits":
                # The logit lens: the residual pushed through the final norm and
                # head, here, inside the trace — an envoy called on a value runs
                # its module on it. Note the GEMM has M = rows·w rather than
                # rows·seq, which rounds differently from the model's own
                # logits by an ulp or so (FINDINGS §10).
                gathered = model.lm_head(model.ln_final(gathered))
            values[read_op.name] = featurizers[read_op.featurizer].featurize(gathered)[0].clone()


def find_op(source: Any, call_site: str) -> str:
    """The single operation of a module's `.source` whose call site contains
    `call_site`, or a refusal naming everything the forward does have.

    Matching the *source line* rather than the operation's name is the whole
    point. nnsight names an operation `{callable}_{occurrence}` and gives
    assignments the same namespace as calls, so on transformers 5.17 the
    attention forward has both `attention_interface_0` (the assignment
    `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)`) and
    `attention_interface_1` (the call). A name match on "attention_interface"
    hits both; the needle `"attention_interface("` is call-shaped and hits one.
    """
    hits = [op.name for op in source if call_site in op.text.split("\n")[op.line - 1]]
    if len(hits) != 1:
        raise AddressError(
            f"{call_site!r} matches {len(hits)} operations {hits} of this forward; "
            f"an address serves exactly one. The forward's operations are: "
            f"{list(source.names)}"
        )
    return hits[0]


def read(model: Any, address: Address) -> Any:
    """The tensor at `address`, during a trace.

    At a module boundary the output may be a bare tensor or a tuple whose
    first element is the hidden state, and which one it is depends on the
    transformers version, not on anything we can see in the document — so it
    is decided from the value. Inside a forward the tensor is one argument of
    one call, or one element of its return, and the address says which.

    This is the engine's half of an address: `address` says *where*, in terms
    that are true of the architecture, and this says how to reach there with
    nnsight. A different engine says it differently.
    """
    if not address.interior:
        return address.get(getattr(address.resolve(model), address.side))
    call = operation(model, address)
    if address.handle == "output":
        return call.output if address.arg is None else call.output[address.arg]
    args, _ = call.inputs
    return args[address.arg]


def write(model: Any, address: Address, tensor: Any) -> None:
    """Put a tensor back: rebuilding the tuple if there was one, or rebuilding
    the call's arguments around the new one."""
    if not address.interior:
        envoy = address.resolve(model)
        setattr(envoy, address.side, address.put(getattr(envoy, address.side), tensor))
        return
    call = operation(model, address)
    index = address.arg
    if address.handle == "output" and index is None:
        call.output = tensor
        return
    assert index is not None
    if address.handle == "output":
        current = call.output
        call.output = (*current[:index], tensor, *current[index + 1 :])
        return
    args, kwargs = call.inputs
    call.inputs = ((*args[:index], tensor, *args[index + 1 :]), kwargs)


def operation(model: Any, address: Address) -> Any:
    """The `.source` operation an interior address names."""
    if address.op is None:
        raise AddressError(
            f"component {address.component!r} is an interior; build its address "
            "with engine.locate(...) so the operation is resolved"
        )
    outer = getattr(address.resolve(model).source, address.op)
    if address.inner is None:
        return outer
    # The callee's own source: only openable here, inside the trace, because
    # which function the call dispatches to is a run-time fact.
    inner = outer.source
    if address.inner not in inner.names:
        raise AddressError(
            f"component {address.component!r}: the function {address.op!r} dispatches to "
            f"has no operation {address.inner!r}; it has {list(inner.names)}"
        )
    return getattr(inner, address.inner)
