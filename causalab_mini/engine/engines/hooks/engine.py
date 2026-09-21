"""The hooks engine: a plain HuggingFace model and `register_forward_hook`.

The simplest runtime that can execute a plan, and a control for the nnterp
engine beside it. No session, no envoys, no tracing, no remote: one
`model(**batch)` per forward with a hook on each module a tap names, and the
hook does the reading and the writing on the way out.

What it costs to not be standardized is `loading.standardized`, and what it
costs to not be nnsight is the interior: a forward hook sees a module's
boundary and nothing between, so `attention_query` is refused by name rather
than approximated. Everything else — the walk over steps, the fit loop, the
metrics, the write algebra — is the shared code in `engine/steps.py` and
`ops/`, unchanged, which is the claim this engine was written to test.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .... import address as address_module
from ....address import Address, AddressError
from ....ops import intervene
from ....plan import Forward, Plan, Tap
from ... import provenance, steps
from ...base import Engine, EngineError
from .loading import HooksEngineError, load, standardized


class HooksEngine(Engine):
    def __init__(self, model: Any, tokenizer: Any, dispatched: bool = True) -> None:
        self.model = model
        self._dispatched = dispatched
        # Two things nnterp hands over with the model and nobody else does: the
        # tokenizer, and the standardized names an address is written in.
        self._tokenizer = tokenizer
        self._names = standardized(model)

    @classmethod
    def load(cls, spec: Any, **options: Any) -> "HooksEngine":
        return cls(*load(spec, **options), dispatched=options.get("dispatch", True))

    # ----------------------------------------------------------------- #
    # what the compiler asks
    # ----------------------------------------------------------------- #

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def num_layers(self) -> int:
        """Counted off the stack rather than read off the config, because the
        stack is already translated and `num_hidden_layers` is a config key
        some families spell `n_layer`."""
        return len(self._names.layers)

    def locate(self, component: str, layer: int | None = None) -> Address:
        """The address of `(component, layer)`, or a refusal.

        A hook fires at a module boundary, so an interior is not reachable at
        all: nnsight recompiles the forward (`.source`) and hooks have no
        equivalent. The two ways to build one would both be a different
        engine — patch the module's `forward` to expose the tensor, or hook
        the `q_proj` submodule and reimplement here the reshape and the rotary
        embedding that the address says have already happened by the time the
        query is the query.
        """
        address = Address(component, layer, family=getattr(self.model.config, "model_type", None))
        if address.interior:
            raise AddressError(
                f"component {component!r} is an interior — one argument of one call "
                "inside a module's forward — and a forward hook only sees the "
                "module's boundary. It is addressed through nnsight's `.source`, "
                "which has no hooks equivalent; run this document on the nnterp "
                "engine."
            )
        return address

    def heads(self, address: Address) -> int:
        return address_module.head_count(self.model.config, address)

    def width(self, address: Address) -> int:
        return address_module.width(self.model.config, address)

    # ----------------------------------------------------------------- #
    # what the run asks
    # ----------------------------------------------------------------- #

    def execute(self, plan: Plan, remote: bool | str = False, batch_size: int | None = None) -> Plan:
        if not self._dispatched:
            raise EngineError(
                "this engine was loaded with dispatch=False: a meta-device shell, "
                "enough to compile and explain a document. Hooks have no server to "
                "run it on; load with weights to execute."
            )
        if remote:
            raise HooksEngineError(
                f"remote={remote!r}: there is no remote for hooks. A hook is a Python "
                "callable registered on a module object in this process, and NDIF runs "
                "the model in another one — there is nothing here to ship. Remote is "
                "the nnterp engine's, where the whole request is one nnsight session."
            )
        plan.provenance.update(provenance.record(self, remote, batch_size))
        steps.run(self, plan, batch_size=batch_size)
        return plan

    def forward(
        self,
        forward: Forward,
        values: dict[str, Any],
        featurizers: dict[str, Any],
    ) -> None:
        """One forward with its taps applied.

        The taps arrive in forward order and nothing here depends on it: a
        hook fires when its module runs. What does matter is that every handle
        is removed — a leaked hook would intervene on the *next* forward,
        which is a silently wrong number rather than an error.
        """
        # The decode step, shared by every hook of this forward: a hook on
        # the root fires once per pass and counts. None for a plain forward.
        clock = {"step": 0 if forward.decode else None}
        handles = [
            _install(self._names, tap, values, featurizers, clock) for tap in forward.taps
        ]
        if forward.decode:
            handles.append(self.model.register_forward_hook(lambda *_: _tick(clock)))
        try:
            if not forward.decode:
                self.model(**batch(forward, self.model.get_input_embeddings().weight.device))
            else:
                prompt = len(forward.input_ids[0])
                ids = self.model.generate(
                    **batch(forward, self.model.get_input_embeddings().weight.device),
                    max_new_tokens=forward.decode,
                    min_new_tokens=forward.decode,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                values[f"{forward.name}.generated"] = ids[:, prompt:].clone()
        finally:
            for handle in handles:
                handle.remove()


def batch(forward: Forward, device: Any = None) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes — where the model's
    first layer is. nnsight does this move for its engine; here nothing does,
    which a CPU-only suite never noticed."""
    return {
        "input_ids": torch.tensor(forward.input_ids, device=device),
        "attention_mask": torch.tensor(forward.attention_mask, device=device),
    }


def _tick(clock: dict[str, Any]) -> None:
    """One more decode step has run."""
    clock["step"] = int(clock["step"] or 0) + 1


def _install(
    names: Any, tap: Tap, values: dict[str, Any], featurizers: dict[str, Any], clock: dict[str, Any]
) -> Any:
    """Hook one address, on the side it is addressed on."""
    module = tap.address.resolve(names)
    if tap.address.side == "output":
        return module.register_forward_hook(intervene_at(tap, values, featurizers, names, clock))
    return module.register_forward_pre_hook(
        intervene_before(tap, values, featurizers, names, clock), with_kwargs=True
    )


def intervene_before(
    tap: Tap, values: dict[str, Any], featurizers: dict[str, Any], names: Any, clock: dict[str, Any]
) -> Callable[[Any, Any, Any], Any]:
    """The same body, on the way *in*.

    A pre-hook is handed the call's arguments and may return replacements.
    Which argument is the activation is the thing nnsight never has to ask:
    an envoy's `.input` knows, and here the block may be called positionally
    or by keyword depending on the family's own loop, so this finds the one
    tensor argument and refuses if there is not exactly one. FINDINGS §6.
    """

    def hook(module: Any, args: Any, kwargs: Any) -> Any:
        if not intervene.applies(tap.step, clock["step"]):
            return None
        if args:
            activation, where = args[0], None
        else:
            tensors = [key for key, value in kwargs.items() if torch.is_tensor(value)]
            if len(tensors) != 1:
                raise HooksEngineError(
                    f"{type(module).__name__} was called with no positional argument "
                    f"and {len(tensors)} tensor keyword arguments {tensors}, so which "
                    "one the activation is cannot be decided here"
                )
            where = tensors[0]
            activation = kwargs[where]
        activation = _apply(tap, activation, values, featurizers, names, clock["step"])
        if where is None:
            return (activation, *args[1:]), kwargs
        return args, {**kwargs, where: activation}

    return hook


def _apply(
    tap: Tap, activation: Any, values: dict[str, Any], featurizers: dict[str, Any], names: Any, step: int | None
) -> Any:
    """This address's writes, then its reads. A read in a model sees that
    model's writes, so at one address the writes go first."""
    original = activation  # what a renormalize measures against
    for write in tap.writes:
        activation = intervene.apply_write(
            activation,
            intervene.at_step(write.at, activation, tap.address.seq_axis, tap.step, step),
            intervene.resolve_operand(values, write.operand),
            write.mechanism,
            featurizers[write.featurizer],
            tap.address.seq_axis,
            write.params,
            original,
        )
    for read in tap.reads:
        gathered = intervene.gather(
            activation,
            intervene.at_step(read.at, activation, tap.address.seq_axis, tap.step, step),
            tap.address.seq_axis,
        )
        if read.view == "logits":
            gathered = names.lm_head(names.ln_final(gathered))
        with intervene.exact(gathered):
            values[read.name] = featurizers[read.featurizer].featurize(gathered)[0].clone()
    return activation


def intervene_at(
    tap: Tap, values: dict[str, Any], featurizers: dict[str, Any], names: Any, clock: dict[str, Any]
) -> Callable[[Any, Any, Any], Any]:
    """The hook for one address: its writes, then its reads, on the way out.

    A forward hook is handed the module's output and may return a replacement,
    and that return value is the whole of this engine's write path — there is
    no envoy to assign to. Writes run before reads at one address because a
    read in a model sees that model's writes; the nnterp engine implements the
    same rule by ordering two statements.

    Like there, the output may be a bare tensor or a tuple whose first element
    is the hidden state, and which one it is is decided from the value — a
    bare tensor at both taps on both families under transformers 5.17, but
    which one it is is a property of the version and not of the document.
    """

    def hook(module: Any, args: Any, output: Any) -> Any:
        if not intervene.applies(tap.step, clock["step"]):
            return None
        address = tap.address
        activation = _apply(tap, address.get(output), values, featurizers, names, clock["step"])
        return address.put(output, activation)

    return hook
