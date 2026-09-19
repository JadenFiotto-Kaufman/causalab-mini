"""causalab-mini: a small, readable reimplementation of causalab's intervention engine.

The whole design rests on one rule: **a plan is pure data, a block turns it
into tensors**. Nothing but the compiler is allowed to decide anything from a
tensor, and nothing but `model/` is allowed to know what a model looks like.

The pipeline, which is also the reading order:

    documents/*.json
        -> plan/      the request, as pure data       (document -> Plan)
        -> session/   one nnsight session             (Plan -> tensors)
        -> output.py  the manifest, as files

and three supporting packages, each named for what it is allowed to know:

    data/    the corpus: a dataset ref -> padded token rows. Client-side.
    model/   the ONLY model-aware code: loading, and addressing a component.
    ops/     agnostic: gather, scatter, the write seam, featurizers, metrics.
             Knows nothing about models and may not import `model/`.

`shapes.py` is shared vocabulary — names for the tuples that travel — and
`cli.py` is the whole pipeline in ten lines.
"""
