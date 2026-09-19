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

`master`, clean tree, no remote. **124 tests passing**
(`CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`, ~6 s), `uvx pyright` at 0
errors. **2,649 source lines** across 24 files in `causalab_mini/`.

The package is five sub-packages and a short spine, each named for what it is
allowed to know:

    __init__.py shapes.py address.py cli.py       vocabulary, the architecture
                                                  map, the entry point
    plan/   document.py plan.py build.py write.py  the request, as pure data
    data/   rows.py     encoding.py                the corpus -> padded tokens
    ops/    intervene.py metrics.py featurizer.py  agnostic: tensors only
    engine/ base.py     steps.py                   the contract, and what a
                                                   plan means on any runtime
      engines/nnterp/   engine.py  loading.py      one directory per runtime

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
3. **A plan is pure data; an engine turns it into tensors.** A *fresh* plan
   holds strings, ints, tuples and dicts only — its `results` dicts are empty
   until it runs, and they are the only mutable thing in the tree. A plan has
   no `execute`: a plan that knew how to run itself would only run on one
   engine.
4. **No trace body may reference a client object** — no executor, document,
   tokenizer. `tests/test_structure.py` is an AST tripwire over every
   `with ….trace(`/`.session(` block, package-wide; it has a vacuity guard.
5. **`address.py` is the only file that knows anything about model
   internals**, and it says *where* only — which module path (in nnterp's
   standardized names), which side, which axis the sequence runs along, what
   order taps go in. Reaching there is the engine's
   (`engine/engines/nnterp/engine.py`'s `read`/`write`). `ops/` knows nothing
   about models at all.
6. **An engine is six members and no more**: `load`, `tokenizer`,
   `num_layers`, `locate`, `width`, `execute`, `forward` — the first four are
   what the compiler asks of a runtime, the last two what the run asks.
   Everything else lives in `engine/steps.py` and is shared.
   `tests/test_engine.py` pins this: it asserts the override set is exactly
   the contract, and runs a real compiled plan on an engine that has no model
   and no session.
7. **An engine holds its model.** `Engine.load(spec)` is how a model enters
   the project, and `plan.build(document, data_root, engine)` compiles
   against the engine, not against a handle — because what a tokenizer is,
   how a site is located and how wide it is are all runtime questions.
6. **The client never decides anything from a tensor.** The plan carries a
   spec; the block resolves it, including any dynamic case. (Nothing here needs
   a dynamic case yet. In the real engine this is how the generated frame and
   the DeltaNet fire count work.)

## 4. The step/plan refactor — BUILT

What §4 used to describe as decided-but-unbuilt is in. How it landed, and
where it differs from the plan written here before it was built:

- **`Plan` is a step and plans nest.** `Plan.steps` is an ordered
  `{name: Step}` dict; the leaves are `Featurizers`, `Observe`, `Fit`,
  `Weights`. A document compiles to a root plan with two to four steps.
- **A plan carries its own results**, at the node that produced them.
  `root.step("fit", Fit).epochs[0][0].results["ce"]` is the metric of one
  training update. `Plan.result(name)` is the flat lookup; it descends through
  steps but **stops before a fit's internal passes**, or every fitted document
  would have six ambiguous `iia`s.
- **`nnsight.save(plan)` at the top of the session is how results come home**,
  exactly as predicted. Verified with a probe before anything was built: a
  frozen dataclass with nested children and mutable `results` dicts round-trips
  on both the local and the serialized path, tensors landing at every depth.
- **`Plan.write(out_dir)` writes the manifest**, recursing: a nested plan
  writes into a directory named by its step name, so a plan's path in the tree
  is its path on disk. A one-plan document writes into `out` exactly as before.
- **Steps have no `execute` method.** This is the owner's correction and it is
  the load-bearing one: an `Engine` classmethod executes a step, so a second
  engine (torch hooks, vLLM) is possible and `plan/` stays free of torch and
  nnsight. `Engine` itself has **no implementation** — an engine that opens
  nothing should not inherit a session.
- **There is no run state object.** `values` — the activations a write's
  operand names — are born and die inside one `Observe`, so the only thing
  crossing steps is the live featurizer dict.
- **No featurizer-isolation flag.** Each point rebuilds its own parameters
  before using them and execution is sequential, so nothing was needed yet.

Still to do, and the test that decides whether nesting earned itself: **a
sweep**. The protocol's own spelling for "three experiments in one document"
is `{"sweep": [...]}` at a field — `featurizers.rot.seed: {"sweep": [0,1,2]}`
with no `train` block is its random-subspace control. The agreed shape is to
lower a sweep on the **document**, before compiling: N documents, N `build`
calls, N child plans under one root. Nothing in the engine changes. Sweeps are
on NOTES §8's out-of-scope list, so the slice is narrow: one wrapper at one
field, every other sweep form refused by name.

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
- **`document.py` is a third of the project** (702 of 2,555 lines), almost all
  refusals. The weight of causalab is in its document surface, not its
  execution.
- **A layer-0 query interchange is a no-op** when the two prompts share a length
  and a last token. It looks exactly like a broken write.
- **A `yield` cannot appear inside an nnsight block.** nnsight recompiles a
  `with model.session(...)` body as a standalone function, so a
  `@contextmanager` whose body opens a session is a `SyntaxError: 'yield'
  outside function` at call time. This killed an `Engine.open` hook and forced
  the better design: `execute` owns the whole session literally.
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
