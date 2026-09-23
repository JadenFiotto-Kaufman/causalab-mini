"""A plan, as text a person or an agent can read.

`explain` is what a document *means*: every forward, every tap, every
position spec and width, every save — the compiled tree, printed. It needs
no weights, because a plan is compiled against a meta model, so it is the
loop an author lives in: write, explain, fix.
"""

from __future__ import annotations

from typing import Any

from .plan import Featurizers, Fit, Observe, Plan, Step, Weights


def _features(at: Any) -> str:
    """Which part of the feature axis, when the selection names one."""
    return "" if at.take is None else f" features={list(at.take)}/{at.groups}"


def _pos(at: Any) -> str:
    """A position, as the document wrote it: the spec, not the integers.

    There are no integers to print. A plan is compiled without rows in hand
    and the run resolves the spec against its own tokenizer, per row — so
    what `explain` can honestly say is what was asked for, and what the
    *run* reports is what was found (`results["positions"]`).
    """
    return "-" if at.where is None else at.where.spelling()


def explain(step: Step, name: str = "<root>") -> str:
    return "\n".join(_lines(step, name, 0))


def _lines(step: Step, name: str, depth: int) -> list[str]:
    pad = "  " * depth
    saves = [one.file_path for one in step.saves]
    tail = f"  saves={saves}" if saves else ""
    out: list[str] = []
    if isinstance(step, Plan):
        out.append(f"{pad}{name}: Plan{tail}")
        for child, one in step.steps.items():
            out += _lines(one, child, depth + 1)
    elif isinstance(step, Featurizers):
        out.append(f"{pad}{name}: Featurizers{tail}")
        for one in step.specs:
            if one.kind == "gate":
                origin = f"loaded from {one.source}" if one.source else "θ=0"
                out.append(f"{pad}    {one.name}: gate d={one.d} {origin} trained={one.trained}")
                continue
            origin = f"loaded from {one.source}" if one.source else f"seed={one.seed}"
            out.append(
                f"{pad}    {one.name}: {one.kind} k={one.k} d={one.d} "
                f"{one.parametrization + ' ' if one.kind == 'subspace' else ''}{origin} trained={one.trained}"
            )
    elif isinstance(step, Fit):
        out.append(
            f"{pad}{name}: Fit  {len(step.epochs)} epochs x {len(step.epochs[0])} update  "
            f"lr={step.lr} objective={step.objective} params={step.params}  "
            f"early_stop={step.early_stop!r} patience={step.patience}"
            f"{f' anneal={step.anneal}' if step.anneal else ''}{tail}"
        )
        out += _lines(step.epochs[0][0], "epochs[0][0]", depth + 2)
        out += _lines(step.evaluation, "evaluation", depth + 2)
    elif isinstance(step, Observe):
        def _out(o):
            if o.reduce == "none":
                return f"{o.name}={o.read}"
            return f"{o.name}={o.reduce}{'' if o.k is None else o.k}({o.read})"
        outputs = f"  outputs={[_out(o) for o in step.outputs]}" if step.outputs else ""
        out.append(f"{pad}{name}: Observe  metrics={[m.name + '/' + m.kind + ('' if m.rows is None else f' rows={list(m.rows)}') for m in step.metrics]}{outputs}{tail}")
        for forward in step.forwards:
            rows, width = len(forward.input_ids), len(forward.input_ids[0]) if forward.input_ids else 0
            decode = f"  decode={forward.decode}" if forward.decode else ""
            out.append(f"{pad}    forward {forward.name!r} on {forward.input!r}  ({rows} rows x {width} tokens){decode}")
            for tap in forward.taps:
                address = tap.address
                where = address.component + (f"[{address.layer}]" if address.layer is not None else "")
                where += "" if tap.step is None else f" @step {tap.step}"
                for write in tap.writes:
                    args = [] if write.operand is None else [str(write.operand)]
                    args += [f"{k}={v}" for k, v in write.params.items()]
                    out.append(
                        f"{pad}      write {write.name!r} at {where} pos={_pos(write.at)}{_features(write.at)} "
                        f"{write.mechanism}({', '.join(args)}) via {write.featurizer!r}"
                        f"{'' if write.features is None else f' on its features {list(write.features)}'}"
                    )
                for read in tap.reads:
                    # a read the run cuts out of the continuation is one op
                    # per decode step in the plan and one read in the
                    # document; print the document's
                    if read.stack and tap.step != 0:
                        continue
                    name = read.stack or read.name
                    at = where.split(" @step")[0] if read.stack else where
                    steps = f" over {forward.decode} steps" if read.stack else ""
                    view = "" if read.view == "raw" else f" as {read.view}"
                    out.append(
                        f"{pad}      read  {name!r} at {at}{steps} pos={_pos(read.at)}{_features(read.at)} via {read.featurizer!r}{view}"
                    )
    elif isinstance(step, Weights):
        out.append(f"{pad}{name}: Weights  names={list(step.names)}{tail}")
    return out
