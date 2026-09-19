# HANDOFF — read this first

Written 2026-09-19 at the end of a long session, for whoever picks this up next.
Three files carry state here and they are different things:

- **`NOTES.md`** — ground truth. What the copied documents contain, field by
  field, and the exhaustive minimum feature set. It is the spec; trust it.
- **`FINDINGS.md`** — evidence. Every fact about model internals this project
  had to encode itself, for a separate proposal to move that knowledge into
  nnterp. Append to it; do not summarise it away.
- **`HANDOFF.md`** (this file) — where the project stands, what was decided but
  **not yet built**, and the wider context this sits in.

---

## 1. What this project is

A clean-room reimplementation of causalab's intervention engine, deliberately
simple. Two goals the owner stated:

1. Test whether the abstractions in the real engine are the best ones, by
   rebuilding without them.
2. Be readable enough that someone learns how causalab works from it. The real
   engine is ~4,000 lines across 11 modules and is hard to follow.

It **imports nothing from causalab**. Only the JSON documents were copied.

## 2. State as of this handoff

`master`, clean tree, no remote. **120 tests passing**
(`CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`, ~6 s), `uvx pyright` at 0
errors. **2,300 source lines** across 22 files in `causalab_mini/`.

The package is five sub-packages and a short spine, each named for what it is
allowed to know:

    __init__.py  shapes.py  cli.py  output.py      vocabulary + spine
    plan/     document.py  plan.py  build.py       the request, as pure data
    data/     rows.py      encoding.py             the corpus -> padded tokens
    model/    loading.py   address.py              the ONLY model-aware code
    ops/      intervene.py metrics.py featurizer.py   agnostic: tensors only
    session/  run.py       observe.py  train.py    the one nnsight session

`plan/document.py` is 702 of those lines and was deliberately left whole: it is
one concept (the protocol surface) and splitting it would need a third file for
the shared refusal helpers, which is more concepts, not fewer.

Working end to end: activation patching, one `.source` interior
(`attention_query`), GPT-2 as a reach-only probe, and a DAS fit — all inside
**one** nnsight session, with `remote="local"` producing bit-identical results.

Documents in `documents/`: `minimal_cpu.json` (patching, shipped),
`das.json` (shipped, unrunnable here — Llama-3.1-8B), `das_cpu_reduction.json`
(authored, four changes from `das.json`), `attention_query_cpu.json` (authored,
the interior), `gpt2_cpu.json` (authored, the reach probe). Authored documents
say so in their own `header.description`.

## 3. Rules that must not be broken

These are load-bearing. Several tests enforce them.

1. **Imports nothing from causalab.** Ever.
2. **One session for the whole request.** `run.execute` opens exactly one
   `model.session(remote=remote)`. Not a session per forward, not a lazy
   per-read path. `remote=True` on that session is the *only* difference
   between local and remote — there is no second code path.
3. **A plan is pure data; a block turns it into tensors.** A *fresh* plan holds
   strings, ints and tuples only (see §4 for how this changes).
4. **No trace body may reference a client object** — no executor, document,
   tokenizer. `tests/test_structure.py` is an AST tripwire over every
   `with ….trace(`/`.session(` block, package-wide; it has a vacuity guard.
5. **`model/` is the only package that knows anything about models**, and
   `model/address.py` the only file that knows their internals. `ops/` knows
   nothing about them and may not import `model/`. No file sits on both sides.
6. **The client never decides anything from a tensor.** The plan carries a
   spec; the block resolves it, including any dynamic case. (Nothing here needs
   a dynamic case yet. In the real engine this is how the generated frame and
   the DeltaNet fire count work.)

## 4. DECIDED BUT NOT BUILT — the step/plan refactor

This was settled in discussion with the owner at the very end of the session and
**no code has been written for it**. It is the next task.

### The problem
`session/run.py`'s `execute` hardcodes a workflow: build featurizers → maybe fit → observe →
save weights, with an `if plan.train is not None` in the middle. That
conditional is the executor knowing about document shape. The sequence is code
when it should be data.

### The design

- **`Plan` becomes a step, and plans nest.** The composite is `Plan` itself:
  `Plan.execute(model, state)` is `for step in self.steps: step.execute(...)`.
  Three experiments in a row is a root plan whose three steps are plans. A fit's
  per-minibatch sub-plan is the same mechanism.
- **Leaves stay at roughly the four kinds we already have** (build featurizers,
  fit, observe, save weights). **No registry, no plugin protocol, no generic
  composite tower beyond `Plan`.** If eight step types appear to express two
  document shapes, the abstraction has cost more than it bought.
- **A plan carries its own results**, so they are navigable at any depth:
  `root.experiments[1].observe.results["iia"]` rather than a flat dict with
  qualified string keys. The owner argued this and is right; an earlier
  proposal of mine (flat dict + graft by path) was worse.
- **How results come home: `nnsight.save(plan)` at the root of the session.**
  This is the owner's solution and it is the correct mechanism. A save on a
  container pushes the server's copy home and *replaces* the client's binding;
  locally the block and caller share the object so it is a no-op. One line, one
  code path, nothing to reattach. `.save()` is mounted onto every object, not
  just tensors, so a dataclass works.
  - **Do not** rely on the block mutating the client's plan across the wire.
    That silently works locally and silently does nothing remotely. It is
    exactly the bug the real engine hit with its fire tally.
- **Run state is separate from the plan during execution**: a small mutable
  object holding **values** (activations moving between forwards, never leave)
  and **featurizers** (stateful, possibly trained, never leave). Results live on
  the plan nodes, not here.
- **Values scope per sub-plan.** Three experiments must not see each other's
  activations.
- **Featurizer inheritance is declared by the enclosing plan.** Both "three
  experiments each training a rotation" (isolated) and "one rotation evaluated
  three ways" (shared) are real experiments, so this cannot be decided once by
  the framework; it belongs in the data.

### Known costs, accepted
- A plan that has been run holds tensors, so it is no longer shippable as-is.
  Re-running means clearing results or copying first.
- The purity test changes from "a plan holds only strings and integers" to
  "a **fresh** plan does". That is a more honest statement of the property.
- Saving the plan round-trips the whole thing, including token ids. A few KB
  here. If plans ever get large, save a results sub-tree instead; the design
  does not have to change to allow that.

### The test that decides whether it earned itself
Write a document that is three experiments, and see whether it needs anything
beyond a root plan with three children. All 120 existing tests should still
pass, with only the purity test's wording changing. If it is a refactor rather
than a redesign, that is the proof.

## 5. Findings from this project worth carrying

Full detail in `FINDINGS.md`; these are the ones that reach past mini.

- **No family axis was needed.** `Address.locate` returns *equal* addresses on
  tiny Llama and tiny GPT-2 for all three components, including the interior,
  though the trees share no module path. nnterp absorbs the family axis for
  module boundaries, and the interior's op is resolved per model at load time
  rather than tabulated. The real engine carries a family-keyed table; this
  suggests it may not need one.
- **Binding-suffix addressing is a demonstrated bug, twice.** On GPT-2,
  `query_states_0` is the cross-attention query on a branch that never runs and
  `query_states_1` is the real one; Llama has only `query_states_0`. An address
  written as a variable-binding suffix reads a dead branch on one family and the
  right tensor on the other, silently. Independently, the nnterp design work
  measured the same class of failure on `attn_weights_1/2`. **The real
  causalab's `sources.py` addresses scores and probabilities this way.** Address
  the *call*, not the binding. This should be fixed on causalab PR #4.
- **The clone in `ops.scatter` is load-bearing under gradients**, not hygiene.
  The in-place spelling raises the moment you differentiate, and activation
  patching never tells you because it never differentiates.
- **Model weights arrive with gradients enabled** and a backward accumulates
  onto them. "The model is frozen" is a property of the optimizer's parameter
  list and nothing else. The real causalab and NDIF both freeze explicitly;
  neither is correct by default.
- **A subspace swap leaves the complement untouched only as arithmetic** — in
  fp32 it moves by up to ~4e-7, because the complement is reconstructed by a
  projection rather than copied. Bit-identity needs an axis-aligned write.
- **`document.py` is a third of the project** (702 of 2,300 lines), almost all
  refusals. The weight of causalab is in its document surface, not its
  execution.
- **A layer-0 query interchange is a no-op** when the two prompts share a length
  and a last token. It looks exactly like a broken write.
- **There is no machine-readable protocol schema.** `docs/intervention_protocol.md`
  is 372 KB of authoritative prose and the Python implements it.

## 6. Environment

- `uv` (0.12.1). `CONTRIBUTING.md` has the commands.
- Tests: `CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`. Type check:
  `uvx pyright` (needs `extraPaths` for nnterp — already in `pyproject.toml`).
- **`nnsight` and `nnterp` are editable local paths** (`[tool.uv.sources]`)
  pointing at `/home/localjadenfk/wd/nnsight` and `/home/localjadenfk/wd/nnterp`.
  **These are shared working checkouts and can move underneath this project.**
  At handoff: nnsight is a detached HEAD at `524c33fc` (the commit of nnsight
  PR #729); nnterp is on branch `standardize-internals`.
- Tiny CPU models: `hf-internal-testing/tiny-random-LlamaForCausalLM` pinned to
  a commit SHA (see the documents), and a tiny GPT-2. Tiny GPT-2 cannot run the
  weekdays documents — `" Friday"` is four tokens there — hence
  `documents/data/counting` and `gpt2_cpu.json`.

## 7. The wider context this sits in

Three other pieces of work are in flight. None of them blocks mini, but mini
produces evidence for the second.

- **causalab**, the real engine: one PR,
  <https://github.com/JadenFiotto-Kaufman/causalab/pull/4>, in the owner's fork
  (no push access to `goodfire-ai`). 26 commits: an executor-free training loop,
  training on NDIF as one session per fit, a pre-queue version guard, and the
  read-side lift. Local checkout `/home/localjadenfk/wd/causalab` on
  `train-loop-executor-free`.
  - Still to do before it merges upstream: pin nnsight/nnterp to git revs (CI
    cannot sync while they are editable paths); the A3B cross-engine parity
    golden has never run (needs ~67 GB; the shared GPU box has ~55 GB free); a
    first run against a real NDIF deployment; and the eager-default divergence
    (the two engines' loaders default to different attention implementations, so
    `--engine` changes numerics on a document that does not pin it).
- **nnterp**: PR <https://github.com/ndif-team/nnterp/pull/60> (hybrids, the
  `attn_implementation` kwarg fix, remote construction staying on meta), plus
  branch `standardize-internals` carrying a **design-only commit `334e4ef`**:
  a proposal for nnterp to own *naming and reaching* while causalab keeps
  *meaning and policy*. `eproperty` was investigated and rejected (it cannot be
  cloudpickled by value, and nnsight auto-registers working-tree modules, so one
  descriptor breaks remote for every checkout user). Two blockers are the
  owner's: causalab ships no LICENSE so moving code upstream needs a
  relicensing grant, and nnterp has no CI that runs its suite.
- **nnsight**: PR <https://github.com/ndif-team/nnsight/pull/729>, a per-host
  cache for the remote environment lookup.

## 8. How the owner works

- They want to be **grilled before a project starts** so the two sides are
  aligned, and they answer numbered questions directly.
- Standing preference, saved to memory: **solve inside existing abstractions
  rather than adding parallel ones**. Fewer concepts, easier to understand, more
  general. Do not mint a twin type beside an existing one.
- They want **honest reporting**: say what was measured versus quoted, name what
  was not run, and correct an earlier overstatement explicitly rather than
  quietly. Two corrections were owed during this session and both mattered.
- Prefer a plain function to a class, a string to an enum, a dict to a registry,
  and no indirection that cannot be justified in one sentence.
