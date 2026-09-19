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

from ....address import Address, AddressError
from ....ops import intervene
from ....plan import Forward, Plan, Tap
from ....plan.document import ModelSpec
from ... import steps
from ...base import Engine
from .loading import HooksEngineError, load, standardized


class HooksEngine(Engine):
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        # Two things nnterp hands over with the model and nobody else does: the
        # tokenizer, and the standardized names an address is written in.
        self._tokenizer = tokenizer
        self._names = standardized(model)

    @classmethod
    def load(cls, spec: ModelSpec, **options: Any) -> "HooksEngine":
        return cls(*load(spec, **options))

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
        address = Address(component, layer)
        if address.interior:
            raise AddressError(
                f"component {component!r} is an interior — one argument of one call "
                "inside a module's forward — and a forward hook only sees the "
                "module's boundary. It is addressed through nnsight's `.source`, "
                "which has no hooks equivalent; run this document on the nnterp "
                "engine."
            )
        if address.side != "output":
            raise AddressError(
                f"component {component!r} is addressed on its {address.side}, and this "
                "engine only hooks outputs. An input-side module boundary is "
                "`register_forward_pre_hook` and the same body; no component in "
                "`address.py` needs one yet."
            )
        return address

    def width(self, address: Address) -> int:
        """The tap's width, off the config.

        The attribute names happen to be the config's own — nnterp publishes
        `hidden_size` and `vocab_size` on the handle because it read them off
        the config — so the translation here is `model.config` and nothing
        more.
        """
        attribute = address.width_attribute
        if attribute is None:
            raise AddressError(
                f"the width of {address.component!r} is not derivable here; a config "
                "has hidden_size and vocab_size and nothing for an attention interior"
            )
        return int(getattr(self.model.config, attribute))

    # ----------------------------------------------------------------- #
    # what the run asks
    # ----------------------------------------------------------------- #

    def execute(self, plan: Plan, remote: bool | str = False) -> Plan:
        if remote:
            raise HooksEngineError(
                f"remote={remote!r}: there is no remote for hooks. A hook is a Python "
                "callable registered on a module object in this process, and NDIF runs "
                "the model in another one — there is nothing here to ship. Remote is "
                "the nnterp engine's, where the whole request is one nnsight session."
            )
        steps.run(self, plan)
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
        handles = [
            tap.address.resolve(self._names).register_forward_hook(
                intervene_at(tap, values, featurizers)
            )
            for tap in forward.taps
        ]
        try:
            self.model(**batch(forward))
        finally:
            for handle in handles:
                handle.remove()


def batch(forward: Forward) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes."""
    return {
        "input_ids": torch.tensor(forward.input_ids),
        "attention_mask": torch.tensor(forward.attention_mask),
    }


def intervene_at(
    tap: Tap, values: dict[str, Any], featurizers: dict[str, Any]
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
        activation = output[0] if isinstance(output, tuple) else output
        for write in tap.writes:
            activation = intervene.apply_write(
                activation,
                write.positions,
                values[write.operand],
                write.mechanism,
                featurizers[write.featurizer],
                tap.address.seq_axis,
            )
        for read in tap.reads:
            gathered = intervene.gather(activation, read.positions, tap.address.seq_axis)
            values[read.name] = featurizers[read.featurizer].featurize(gathered)[0].clone()
        return (activation, *output[1:]) if isinstance(output, tuple) else activation

    return hook
