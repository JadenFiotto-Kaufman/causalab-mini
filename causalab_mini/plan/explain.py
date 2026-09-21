"""A plan, as text a person or an agent can read.

`explain` is what a document *means*: every forward, every tap, every
resolved position and width, every save — the compiled tree, printed. It
needs no weights, because a plan is compiled against a meta model, so it is
the loop an author lives in: write, explain, fix.
"""

from __future__ import annotations

from .plan import Featurizers, Fit, Observe, Plan, Step, Weights


def _pos(positions: tuple[tuple[int, ...], ...]) -> str:
    """`(10, 10)` for unit windows, `[8:11], [8:11]` for wider ones, and `-`
    for a row with no window — an excluded measurement."""
    if positions and all(len(window) == 1 for window in positions):
        return str(tuple(window[0] for window in positions))
    return ", ".join(f"[{w[0]}:{w[-1] + 1}]" if w else "-" for w in positions)


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
            origin = f"loaded from {one.source}" if one.source else f"seed={one.seed}"
            out.append(
                f"{pad}    {one.name}: {one.kind} k={one.k} d={one.d} "
                f"{one.parametrization} {origin} trained={one.trained}"
            )
    elif isinstance(step, Fit):
        out.append(
            f"{pad}{name}: Fit  {len(step.epochs)} epochs x {len(step.epochs[0])} update  "
            f"lr={step.lr} objective={step.objective} params={step.params}  "
            f"early_stop={step.early_stop!r} patience={step.patience}{tail}"
        )
        out += _lines(step.epochs[0][0], "epochs[0][0]", depth + 2)
        out += _lines(step.evaluation, "evaluation", depth + 2)
    elif isinstance(step, Observe):
        def _out(o):
            if o.reduce == "none":
                return f"{o.name}={o.read}"
            return f"{o.name}={o.reduce}{'' if o.k is None else o.k}({o.read})"
        outputs = f"  outputs={[_out(o) for o in step.outputs]}" if step.outputs else ""
        out.append(f"{pad}{name}: Observe  metrics={[m.name + '/' + m.kind for m in step.metrics]}{outputs}{tail}")
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
                        f"{pad}      write {write.name!r} at {where} pos={_pos(write.positions)} "
                        f"{write.mechanism}({', '.join(args)}) via {write.featurizer!r}"
                    )
                for read in tap.reads:
                    view = "" if read.view == "raw" else f" as {read.view}"
                    out.append(
                        f"{pad}      read  {read.name!r} at {where} pos={_pos(read.positions)} via {read.featurizer!r}{view}"
                    )
    elif isinstance(step, Weights):
        out.append(f"{pad}{name}: Weights  names={list(step.names)}{tail}")
    return out
