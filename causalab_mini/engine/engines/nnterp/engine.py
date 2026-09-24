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

Nothing of ours ships by value. The block's module references — `steps`,
`intervene`, `plan_module` — and the plan's own classes resolve by import on
the far side, so an NDIF server has to have `causalab_mini` and `nnterp`
installed at the client's versions. A server that does not is a
`ModuleNotFoundError`, not a silent divergence.
"""

from __future__ import annotations

from typing import Any

import nnsight
import torch

from .... import address as address_module
from ....address import Address
from ....ops import intervene
from ....plan import Forward, Generate, Plan
from ....plan import plan as plan_module
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
        return address_module.locate(self.model, component, layer)

    def heads(self, address: Address) -> int:
        return address_module.head_count(self.model, address)

    def width(self, address: Address) -> int:
        return address_module.width(self.model, address)

    # ----------------------------------------------------------------- #
    # what the run asks
    # ----------------------------------------------------------------- #

    def execute(self, plan: Plan, remote: bool | str = False, batch_size: int | None = None) -> Plan:
        plan.provenance.update(provenance.record(self, remote, batch_size))
        engine, model = self, self.model
        with model.session(remote=remote):
            # What comes home is plain — strings and tensors. The client
            # already has the plan, so only what fills it in needs the trip,
            # and nothing of ours has to survive the way back. (`remote="local"`
            # never noticed: it does not serialize the way back. FINDINGS §19.)
            home = nnsight.save({})
            steps.run(engine, plan, steps.start(plan, batch_size))
            home.update(plan_module.results_of(plan))
            # A server serves the dtype *it* chose; the document's is only a
            # request. Say what ran, where the run record is.
            home["served_dtype"] = str(model.lm_head.weight.dtype)
        plan.provenance["served_dtype"] = home.pop("served_dtype")
        plan_module.fill(plan, home)
        return plan

    def forward(self, forward: Forward, values: dict[str, Any], featurizers: dict[str, Any]) -> Any:
        model, made = self.model, {}
        with model.trace(batch(forward)) as tracer:
            apply_interventions(model, forward, values, featurizers)
            made["logits"] = tracer.result.logits
        return made["logits"]

    def generate(self, step: Generate, values: dict[str, Any], featurizers: dict[str, Any]) -> Any:
        # The continuation frame: one generate trace, `tracer.iter` walking
        # the decode steps. Prompt-frame taps apply at step 0, the prefill; a
        # step's taps at that step; an `"all"` write at every step. A decode
        # with no taps has no loop: an early EOS makes a loop outrun the run
        # and drop what follows it (FINDINGS §12.4), which would be the ids.
        model, prompt, made = self.model, len(step.input_ids[0]), {}
        with model.generate(batch(step), max_new_tokens=step.max_new_tokens, **step.generation) as tracer:
            if step.taps:
                for index in tracer.iter[: step.max_new_tokens]:
                    apply_interventions(model, step, values, featurizers, index)
            made["ids"] = tracer.result[:, prompt:].clone()
        return made["ids"]


def batch(forward: Forward) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes."""
    return {
        "input_ids": torch.tensor(forward.input_ids),
        "attention_mask": torch.tensor(forward.attention_mask),
    }


def apply_interventions(
    model: Any,
    forward: Forward,
    values: dict[str, Any],
    featurizers: dict[str, Any],
    step: int | None = None,
) -> None:
    """One walk over the addresses of one forward, in forward order.

    `values` holds the operands this call's writes take, by name — each
    produced by an earlier step — and receives what it reads. A read that
    names a featurizer is featurized here — `v_cf` is `Qᵀx`, not `x` — by the
    same object the write's `inverse` will use, which is what makes one
    featurizer name one parameter set; a read that names none is the tensor.

    `step` is None for a plain forward, and the decode step inside a
    generate trace. A tap applies when its frame is this step: the prompt
    frame at step 0 (or a plain forward), a step's own taps at that step,
    an `"all"` write at every step.
    """
    for tap in forward.taps:
        if not intervene.applies(tap.step, step):
            continue
        # what a renormalize measures against: this address before any write
        original = read(model, tap.address) if tap.writes else None
        for write_op in tap.writes:
            patched = intervene.apply_write(
                read(model, tap.address),
                intervene.at_step(write_op.at, read(model, tap.address), tap.address.seq_axis, tap.step, step),
                intervene.resolve_operand(values, write_op.operand),
                write_op.mechanism,
                None if write_op.featurizer is None else featurizers[write_op.featurizer],
                tap.address.seq_axis,
                write_op.params,
                original,
                write_op.features,
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
                gathered = intervene.softcap(
                    model.lm_head(model.ln_final(gathered)), softcapping(model)
                )
            if read_op.featurizer is not None:
                with intervene.exact(gathered):
                    gathered = featurizers[read_op.featurizer].featurize(gathered)[0]
            values[read_op.name] = gathered.clone()


def softcapping(model: Any) -> float | None:
    """The bound this family puts on its logits, or None. Gemma-2 is the one
    that has it; `model.logits` is capped and `lm_head_output` is not."""
    return getattr(model.config, "final_logit_softcapping", None)


def read(model: Any, address: Address) -> Any:
    """The tensor at `address`, during a trace: nnterp's accessor, at the
    layer — a whole-model one at `None`. The accessor knows the module, the
    operation inside its forward where there is one, the side, and where in
    the value the tensor sits; the address is only its name and a layer."""
    return model.internals[address.accessor][address.layer]


def write(model: Any, address: Address, tensor: Any) -> None:
    """Put a tensor back where `read` found it; the accessor rebuilds whatever
    the tensor was reached through — a tuple, a call's arguments."""
    model.internals[address.accessor][address.layer] = tensor
