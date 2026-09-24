"""A plan, as text a person or an agent can read.

`explain` is what a document *means*: every forward, every tap, every
position spec and width, every save — the compiled tree, printed. It needs
no weights, because a plan is compiled against a meta model, so it is the
loop an author lives in: write, explain, fix.
"""

from __future__ import annotations

from typing import Any

from .plan import Featurizers, Fit, Forward, Generate, Metric, Plan, Reduce, Step


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
    elif isinstance(step, Forward):
        rows, width = len(step.input_ids), len(step.input_ids[0]) if step.input_ids else 0
        call = "forward"
        if isinstance(step, Generate):
            call = "generate" + "".join(
                f" {key}={value}" for key, value in {"max_new_tokens": step.max_new_tokens, **step.generation}.items()
            )
        kept = f"  keeps={list(step.keep)}" if step.keep else ""
        out.append(f"{pad}{name}: {call} on {step.input!r}  ({rows} rows x {width} tokens){kept}{tail}")
        for tap in step.taps:
            address = tap.address
            where = address.component + (f"[{address.layer}]" if address.layer is not None else "")
            where += "" if tap.step is None else f" @step {tap.step}"
            for write in tap.writes:
                args = [] if write.operand is None else [str(write.operand)]
                args += [f"{k}={v}" for k, v in write.params.items()]
                out.append(
                    f"{pad}    write {write.name!r} at {where} pos={_pos(write.at)}{_features(write.at)} "
                    f"{write.mechanism}({', '.join(args)}) via {write.featurizer!r}"
                    f"{'' if write.features is None else f' on its features {list(write.features)}'}"
                )
            for read in tap.reads:
                # a read the run cuts out of the continuation is one op per
                # decode step in the plan and one read in the document;
                # print the document's
                if (read.stack and tap.step != 0) or (read.layered and read.name != f"{read.layered}@0"):
                    continue
                label = read.stack or read.layered or read.name
                at = where.split(" @step")[0] if read.stack else where
                if read.layered:
                    at = f"{address.component}[every layer]" + where.partition("]")[2]
                steps = f" over {step.max_new_tokens} steps" if read.stack and isinstance(step, Generate) else ""
                view = "" if read.view == "raw" else f" as {read.view}"
                out.append(
                    f"{pad}    read  {label!r} at {at}{steps} pos={_pos(read.at)}{_features(read.at)} via {read.featurizer!r}{view}"
                )
    elif isinstance(step, Metric):
        rows = "" if step.rows is None else f" rows={list(step.rows)}"
        out.append(f"{pad}{name}: metric {step.kind}({step.of}){rows}{tail}")
    elif isinstance(step, Reduce):
        out.append(f"{pad}{name}: reduce {step.reduce}{'' if step.k is None else step.k}({step.of}){tail}")
    return out
