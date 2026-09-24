"""The hooks engine: a plain HuggingFace model and `register_forward_hook`.

The simplest runtime that can execute a plan, and a control for the nnterp
engine beside it. No session, no envoys, no tracing, no remote: one
`model(**batch)` per forward with a hook on each module a tap names, and the
hook does the reading and the writing on the way out.

What it costs to not be standardized is `loading.standardized`, and what it
costs to not be nnsight is the inside of a forward: a hook sees a module's
boundary and nothing between, so `attention_query` — an operation inside the
attention's forward — is refused by name at compile time rather than
approximated. Everything else — the walk over steps, the fit loop, the
metrics, the write algebra — is the shared code in `engine/steps.py` and
`ops/`, unchanged, which is the claim this engine was written to test.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

import torch

from .... import address as address_module
from ....address import Address, AddressError
from ....ops import intervene
from ....plan import Forward, Generate, Plan, Tap
from ... import provenance, steps
from ...base import Engine, EngineError
from .loading import HooksEngineError, load, standardized


class HooksEngine(Engine):
    def __init__(self, model: Any, tokenizer: Any, shell: Any, dispatched: bool = True) -> None:
        self.model = model
        self._dispatched = dispatched
        # What nnterp hands over with the model and nobody else does: the
        # tokenizer, the standardized names an address is written in, and —
        # through a weightless shell of the same checkpoint — which child
        # module each component is on this family, and every width.
        self._tokenizer = tokenizer
        self._names = standardized(model, shell)
        self._shell = shell

    @classmethod
    def load(cls, spec: Any, **options: Any) -> "HooksEngine":
        from ..nnterp.loading import load as load_shell

        model, tokenizer = load(spec, **options)
        return cls(model, tokenizer, load_shell(spec, dispatch=False), dispatched=options.get("dispatch", True))

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

        The same nnterp row the other engine reads — module path, side, and
        where in the value the tensor is — so what a hook reaches is what an
        accessor reaches. A row that names an operation inside a module's
        forward is the exception, and it is refused here, at compile time: a
        hook fires at a module boundary, and nnsight reaches inside by
        recompiling the forward (`.source`), which hooks have no equivalent
        of. Building one would be a different engine — patch the module's
        `forward` to expose the tensor, or hook `q_proj` and reimplement the
        reshape and the rotary embedding the query has had by the time it is
        the query.
        """
        # The row is a fact about the checkpoint that nnterp knows; this
        # engine's model is not an nnterp one, so it asks the shell.
        address = address_module.locate(self._shell, component, layer)
        if address.inside:
            raise AddressError(
                f"component {component!r} is an operation inside a module's forward — "
                "one argument or result of one call — and a forward hook only sees the "
                "module's boundary. nnsight reaches it through `.source`, which has no "
                "hooks equivalent; the nnterp engine reaches it."
            )
        return address

    def heads(self, address: Address) -> int:
        return address_module.head_count(self._shell, address)

    def width(self, address: Address) -> int:
        return address_module.width(self._shell, address)

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
        steps.run(self, plan, steps.start(plan, batch_size))
        return plan

    def forward(self, forward: Forward, values: dict[str, Any], featurizers: dict[str, Any]) -> Any:
        """One forward with its taps applied.

        The taps arrive in forward order and nothing here depends on it: a
        hook fires when its module runs. What does matter is that every handle
        is removed — a leaked hook would intervene on the *next* forward,
        which is a silently wrong number rather than an error.
        """
        with _hooked(self._names, forward, values, featurizers, {"step": None}):
            output = self.model(**batch(forward, self.model.get_input_embeddings().weight.device))
        return output.logits

    def generate(self, step: Generate, values: dict[str, Any], featurizers: dict[str, Any]) -> Any:
        """One decode with its taps applied at their steps. A hook on the root
        fires once per model pass and counts the step every tap shares."""
        clock = {"step": 0}
        with _hooked(self._names, step, values, featurizers, clock):
            ticking = self.model.register_forward_hook(lambda *_: _tick(clock))
            try:
                ids = self.model.generate(
                    **batch(step, self.model.get_input_embeddings().weight.device),
                    max_new_tokens=step.max_new_tokens,
                    # nnsight passes the tokenizer's pad id for its engine;
                    # here nothing does, unless the document said one
                    **{"pad_token_id": self._tokenizer.pad_token_id, **step.generation},
                )
            finally:
                ticking.remove()
        return ids[:, len(step.input_ids[0]) :].clone()


def batch(forward: Forward, device: Any = None) -> dict[str, Any]:
    """The plan's integers, as the tensors a forward takes — where the model's
    first layer is. nnsight does this move for its engine; here nothing does,
    which a CPU-only suite never noticed."""
    return {
        "input_ids": torch.tensor(forward.input_ids, device=device),
        "attention_mask": torch.tensor(forward.attention_mask, device=device),
    }


@contextlib.contextmanager
def _hooked(names: Any, forward: Forward, values: dict[str, Any], featurizers: dict[str, Any], clock: dict[str, Any]) -> Iterator[None]:
    """Every tap of `forward` hooked for as long as the block runs, and every
    handle removed however it ends."""
    handles = [_install(names, tap, values, featurizers, clock) for tap in forward.taps]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _tick(clock: dict[str, Any]) -> None:
    """One more decode step has run."""
    clock["step"] = int(clock["step"] or 0) + 1


def _install(
    names: Any, tap: Tap, values: dict[str, Any], featurizers: dict[str, Any], clock: dict[str, Any]
) -> Any:
    """Hook one address, on the side it is addressed on."""
    module = resolve(tap.address, names)
    if tap.address.io == "output":
        return module.register_forward_hook(intervene_at(tap, values, featurizers, names, clock))
    return module.register_forward_pre_hook(
        intervene_before(tap, values, featurizers, names, clock), with_kwargs=True
    )


def resolve(address: Address, names: Any) -> Any:
    """The raw module an address names, against the standardized tree. The
    path is in nnterp's spellings, and inside a block two of them are
    renames (`self_attn`, `mlp`) that the raw tree does not have: those go
    through the per-layer lists `standardized()` built.

    The empty path is the model itself — nnterp writes a whole-model place
    that way (`logits` is the model's own output), and a layer's own
    boundary the same way relative to the layer.
    """
    if not address.path:
        return names.model
    segments = address.path.split(".")
    if segments[0] != "layers":
        return address.resolve(names)
    layer = int(segments[1])
    module = names.layers[layer]
    for index, segment in enumerate(segments[2:]):
        if index == 0 and segment == "self_attn":
            module = names.attentions[layer]
        elif index == 0 and segment == "mlp":
            module = names.mlps[layer]
        else:
            module = getattr(module, segment, None)
        if module is None:
            raise AddressError(f"component {address.component!r}: {address.path!r} does not exist on this model")
    return module


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
            None if write.featurizer is None else featurizers[write.featurizer],
            tap.address.seq_axis,
            write.params,
            original,
            write.features,
        )
    for read in tap.reads:
        gathered = intervene.gather(
            activation,
            intervene.at_step(read.at, activation, tap.address.seq_axis, tap.step, step),
            tap.address.seq_axis,
        )
        if read.view == "logits":
            gathered = intervene.softcap(names.lm_head(names.ln_final(gathered)), names.softcap)
        if read.featurizer is not None:
            with intervene.exact(gathered):
                gathered = featurizers[read.featurizer].featurize(gathered)[0]
        values[read.name] = gathered.clone()
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

    Where the tensor is inside the module's output is nnterp's row to say —
    a bare tensor, the first element of a tuple, or a field of the output
    object — and it says it as a `Selection` with `get`/`put`. This engine
    asks for the row's, which is how it reaches `logits` (the model's own
    output, whose `logits` field is the tensor) with no rule of its own:
    unwrapping a tuple's first element is `FirstIfTuple`, one of them.
    """

    def hook(module: Any, args: Any, output: Any) -> Any:
        if not intervene.applies(tap.step, clock["step"]):
            return None
        select = selection(tap.address, names)
        activation = _apply(
            tap, output if select is None else select.get(output),
            values, featurizers, names, clock["step"],
        )
        return activation if select is None else select.put(output, activation)

    return hook


def selection(address: Address, names: Any) -> Any:
    """Where the tensor is inside the value at this place, as nnterp's row
    says it. `None` is the value untouched.

    The row is asked for by the accessor name the plan already carries, off
    the same weightless shell this engine asks for a module's spelling and
    for every width — so there is no second table here and no third spelling
    of "the first element of a tuple"."""
    return names.shell.internals[address.accessor].address.select
