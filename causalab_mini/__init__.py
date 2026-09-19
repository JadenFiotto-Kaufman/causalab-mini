"""causalab-mini: a small, readable reimplementation of causalab's intervention engine.

The whole design rests on one rule: **a plan is pure data, an engine turns it
into tensors**. Nothing but the compiler is allowed to decide anything from a
tensor, and the plan has no `execute` — which is what makes a second engine
possible at all.

The pipeline, which is also the reading order:

    documents/*.json
        -> plan/      the request, as a tree of steps  (document -> Plan)
        -> engine/    one runtime executes it          (Plan -> tensors)
        -> Plan.write the manifest, as files

and the rest, each named for what it is allowed to know:

    address.py  WHERE a tensor lives: module path, side, sequence axis, tap
                order. The only file that knows model internals. It does not
                know how to reach there — that is the engine's.
    data/       the corpus: a dataset ref -> padded token rows. Client-side.
    ops/        agnostic: gather, scatter, the write seam, featurizers,
                metrics. Knows nothing about models at all.
    engine/     `base.py` is the contract, `steps.py` is what a plan means on
                any runtime, and `engines/<name>/` is one directory per
                runtime — today `engines/nnterp`, nnsight traces in one
                session, local or on NDIF.

A plan holds its own results, so what a run produced is navigable where it
happened: `root.steps["fit"].results["train/loss"]`. One document can be
several experiments — see `plan/sweep.py` — and then the root plan holds one
child plan per point. `shapes.py` is shared vocabulary, names for the tuples
that travel, and `cli.py` is the whole pipeline in ten lines.
"""
