# HANDOFF — read this first

Written 2026-09-19 at the end of a long session, for whoever picks this up next.
Three files carry state here and they are different things:

- **`REVIEW.md`** — the design, audited against the three goals (express
  many workflows, agent-configurable, human-usable). What holds, the five
  load-bearing changes in order, the agent-facing CLI, and the order of work.
  Read this before starting anything new.
- **`SURVEY.md`** — the gap. Every feature the real causalab has that mini
  does not, rated by what it would cost mini and by how many experiments
  want it, with the corpus of 24 shipped documents triaged one by one. Read
  §2 (five architectural verdicts) and §3 (what to do first) even if you read
  nothing else.
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

`master`, clean tree, pushed to GitHub (private). **527 tests passing**
(`CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`, ~30 s), `uvx pyright` at 0
errors. **6,948 lines** across 30 files in `causalab_mini/`.

The package is five sub-packages and a short spine, each named for what it is
allowed to know:

    __init__.py shapes.py address.py cli.py       vocabulary, the architecture
                                                  map, the entry point
    plan/   spec.py     plan.py                   the request, as pure data
            build.py    write.py  sweep.py
    data/   rows.py     tokens.py                  the corpus -> padded tokens
    ops/    intervene.py metrics.py featurizer.py  agnostic: tensors only
            locate.py                              a position spec -> indices
    engine/ base.py     steps.py                   the contract, and what a
                                                   plan means on any runtime
      engines/nnterp/   engine.py  loading.py      one directory per runtime
      engines/hooks/    engine.py  loading.py      plain HF + forward hooks

Working end to end: activation patching, one `.source` interior
(`attention_query`), GPT-2 as a reach-only probe, and a DAS fit — all inside
**one** nnsight session, with `remote="local"` producing bit-identical results.

There is now a **second engine**, `engines/hooks`: a plain
`AutoModelForCausalLM` driven by `register_forward_hook`, no nnsight anywhere.
It runs `patching.json` and the DAS fit to numbers bit-identical to the
nnterp engine's, refuses the interior and `remote` by name, and needed no
change to `steps.py`, `ops/`, `plan/` or `address.py`. What it had to supply by
hand — and what turned out to be free — is FINDINGS §6. It is not wired into
the CLI: `--engine` is a flag nobody has needed yet.

Documents: **29** in `documents/v2/` and **five** in `documents/real/`, which
pin real checkpoints and are compiled but not run by the suite. Among the v2
ones, ported from causalab's own corpus: `multi_position_patch.json` (three
disjoint writes at one site in one call, the bit-for-bit twin of
`window_patch.json`'s one window), `hydra_effect.json` (a read taken inside
one intervened call is the operand of a write in another — the only
cross-call operand chain in causalab's corpus) and `random_subspace.json`
(the matched-k control: three untrained seeded rotations, no fit at all).
Authored for mini: `attention_query.json` (an interior),
`gpt2_reach.json` (the reach probe on a second family) and
`pos_sweep.json` (one document, three experiments). Authored documents say
so in their own `header.description`. causalab's protocol format has no
reader here; `documents/intervention_protocol.md` is its specification,
which the vocabulary still follows.

## 3. Rules that must not be broken

These are load-bearing. Several tests enforce them.

1. **Imports nothing from causalab.** Ever.
2. **One session for the whole request.** `NNterpEngine.execute` opens exactly one
   `model.session(remote=remote)`. Not a session per forward, not a lazy
   per-read path. `remote=True` on that session is the *only* difference
   between local and remote — there is no second code path.
3. **One document format, one compiler.** `spec.py` reads the document,
   whose `steps` are what runs — forwards, generates, metrics, reduces,
   fits — and whose step names are how later steps and `saves` reach what
   each produced. `build.py` compiles it, handing every model call to
   `build._forward`.
4. **A plan is pure data; an engine turns it into tensors.** A *fresh* plan
   holds strings, ints, tuples and dicts only — its `results` dicts are empty
   until it runs, and they are the only mutable thing in the tree. A plan has
   no `execute`: a plan that knew how to run itself would only run on one
   engine.
5. **No trace body may reference a client object** — no executor, document,
   dataset, or bare `tokenizer`. `tests/test_structure.py` is an AST tripwire
   over every `with ….trace(`/`.session(` block, package-wide; it has a
   vacuity guard. The run does use a tokenizer — resolving a position is
   its job — and reaches it through the *model*: `model.tokenizer` is one of
   nnsight's persistent objects, so it is written as an id and a server
   resolves it to the served checkpoint's own tokenizer. A bare name closed
   over would be pickled by value instead, and would be the wrong object.
6. **`address.py` is the only file that knows anything about model
   internals**, and since FINDINGS §23 most of what it knows it asks nnterp
   for. A boundary is the *name of a nnterp accessor*, and which child
   module that is on a checkpoint, whether the block has the place at all,
   whether it is one per layer, **where it sits in the forward pass**, and
   every width and head count are nnterp's — `locate` stamps the resolved
   child, side and rank into the `Address` and checks `per_layer` against
   nnterp's answer, so the two cannot drift. The table is a floor and not a
   fence: a name only `model.internals` has is addressable, because
   `RenameConfig(addresses={...})` is how a user adds a place, or moves one
   nnterp has wrong for their model. The four places inside the attention
   (`attention_query/key/scores/z`) are nnterp rows too —
   `attention_queries`, `attention_keys`, `attention_scores`,
   `attention_head_outputs`, hard-coded `.source` ops checked on 26 families
   by nnterp's `test_source_ops.py` — so an address is an accessor and a
   layer and nothing else, and the engine's `read`/`write` are
   `model.internals[name][layer]`. What is left that is purely mini's: the
   per-head kind, the key axis, the seq axis, the width *attribute name*,
   and read-only. `ops/` knows nothing about models at all.
7. **An engine is nine members and no more**: `load`, then `tokenizer`,
   `num_layers`, `locate`, `width` and `heads` — what the compiler asks of a
   runtime — then `execute`, `forward` and `generate`, what the run asks.
   (`heads` joined when a site could name them: like `width`, it is a
   question about the checkpoint that only its holder can answer.
   `generate` is beside `forward` because a decode is a different call with
   a different result: it takes the step's generate arguments and returns
   the ids.)
   Everything else lives in `engine/steps.py` and is shared.
   `tests/test_engine.py` pins this: it asserts the override set is exactly
   the contract, and runs a real compiled plan on an engine that has no model
   and no session.
8. **An engine holds its model.** `Engine.load(spec)` is how a model enters
   the project, and `plan.build_request(raw, data_root, engine)` compiles
   against the engine, not against a handle — because what a tokenizer is,
   how a site is located and how wide it is are all runtime questions.
9. **The client never decides anything from a tensor.** The plan carries a
   spec; the block resolves it, including any dynamic case. **Positions are
   the standing example**: a plan carries the `Where` a document wrote and
   the per-row text a text anchor names, and `engine/steps.py` turns those
   into integers against the model's own tokenizer, per row, inside the
   session. See §5.

## 4. Positions are a spec, resolved where the model is — BUILT

A position used to be an integer the compiler worked out against a tokenizer
the client happened to have. It is now one `shapes.Where`, three independent
fields and nothing else, and the *run* resolves it:

    frame                 which sequence: "prompt", or "generated"
    scope                 which run of it: an `Anchor` — a variable's text,
                          a segment the frame located, or both
    index/span/last/all   how much of that run

A position's run is the prompt's own tokens: it starts after whatever the
tokenizer puts in front of every prompt, so `{"index": 0}` is the first word
on Llama (which prepends a BOS) and on GPT-2 (which does not). Nothing
addresses the prefix itself yet.

`{"index": -1}` is the last token; `{"index": -1, "scope": {"variable":
"entity"}}` is the last token of *this row's* entity; `{"frame": "generated",
"index": -1}` is the last token of this row's continuation, which is its
stop token on a row that stopped (the frame is cut *after* it — a stop
token is a token the model produced). A bare `-1` is sugar for
`{"index": -1}` and is what every shipped document still writes.

- **One resolver, `ops/locate.py`.** A `Frame` is one padded batch as text —
  each row's content span, what it decodes to, and the character each token
  starts at, built by decoding growing prefixes, so no fast tokenizer is
  needed. `locate(frame, where, anchors)` is three steps: find the run, cut
  the run, bounds-check. An integer position is the scope-free case of those
  same three steps, which is why there is one resolver and not two. It
  imports the standard library, `torch` and `shapes.py` and nothing else;
  on a server it resolves by reference from the installed `causalab_mini`,
  like the rest of the package (FINDINGS §24).
- **`engine/steps.py` calls it**, once per forward per window of rows, with
  `engine.tokenizer` — which is `model.tokenizer`, one of nnsight's
  persistent objects, so on a server it is the *served checkpoint's own*.
  `Selection.positions` is filled in on a copy; the plan stays fresh. **The
  engine contract did not change**: an engine is still handed integers.
- **Three refusals moved to run time**, and this is the design's real cost: a
  write with nothing to write on a row, a ragged write whose operand is a
  different width, and a metric none of whose rows could be placed. Which
  rows a text anchor is in is a question about the model's own tokenization,
  and the compiler no longer pretends to know it. A client-side pre-check
  (`explain --precheck`) is designed and not built. Each of those names the
  row's own reason — `alignment_missing`, `alignment_ambiguous`,
  `out_of_range` — because the three want different fixes.
- **A cut of a fixed width that fits no row is refused, not reported.**
  `{"last": 12}` on a nine-token row names the same number of tokens on
  every row, so a row it does not fit is a document that is wrong about its
  own prompts. Only an *anchored* cut reports per row, because which rows
  carry a word is data.
- **What a run reports, it reports in the frame it resolved in.** A tap in
  the continuation frame is given the one position its decode step
  processes, and where in the *continuation* that was is said by the code
  that has that frame. The prompt frame says nothing about it.
- **The character map is built only for a step that has a position the
  document does not already fix.** It is O(L) decode calls of O(L) work per
  row, once per forward; a fit whose every position is a bare `-1` would
  build 42 of them and read none.
- **Eligibility is two halves meeting in the run.** A metric step's `rows`
  is the column half, decided where the data is; the position half is what
  the run could place. A forward with a position the document leaves open
  reports `results["positions"]` — the window, the reason and the decoded
  tokens per op — and a metric of one of its reads carries its
  `results["eligible"]`, the intersection, and that record. A step with no
  such position reports neither.
- **The continuation frame is cut per row at its first stop token.** A read
  whose cut only the finished text can settle (`{"index": -1}`, a scope)
  compiles to one ordinary read per decode step carrying a `stack` name, and
  the run puts them back together and cuts them — so neither engine needed a
  line. A write may only name a step the decode has reached.
- **Chat turns are segments, and the data says so.** A forward whose field
  holds a list of `{"role", "content"}` messages is rendered through the
  checkpoint's own chat template at compile time, and the character span of
  each turn's content travels in the plan beside the anchors. A position
  then names a turn by its own role — `{"segment": "user"}`, or
  `{"segment": "user[1]"}` when that role speaks twice, with the bare name
  `alignment_ambiguous` exactly as a variable occurring twice is. The
  template's control tokens are between the turns and in none of them.
  `{"segment": …, "variable": …}` composes. `documents/v2/chat_turn.json`.
- **Deferred, deliberately**: the client-side pre-check and the CLI work
  around it (`--precheck`, `validate` warnings).
- **Not run**: a real NDIF deployment. `remote="local"` pins the whole
  mechanism (it serializes, hides the local modules and resolves the
  persistent objects exactly as a server does) and an anchored document comes
  back identical, but no request has gone to ndif.us — and `remote="local"`
  does **not** serialize the way home (FINDINGS §19). `results["positions"]`
  and `results["eligible"]` are the first non-tensor, three-deep payloads to
  go through `nnsight.save({})`: nested dicts of tuples of ints and strings,
  with no class of ours in them, so they satisfy §19.6's rule and should be
  fine — but "should" is the word until a real run says otherwise. A run
  against the local stack is queued.

## 5. The step/plan refactor — BUILT

What §4 used to describe as decided-but-unbuilt is in. How it landed, and
where it differs from the plan written here before it was built:

- **`Plan` is a step and plans nest.** `Plan.steps` is an ordered
  `{name: Step}` dict; the leaves are the kinds of thing a document runs —
  `Forward`, `Generate`, `Metric`, `Reduce`, `Fit` — plus `Featurizers`
  (declaring a parameter set is what builds it) and `Weights` (what a fit
  trained, when saved). A steps-first document compiles one for one: its
  steps are the plan's, under their own names. A `Fit` holds a plan of its
  body's steps per minibatch, and one over the held-out rows.
- **A plan carries its own results**, at the node that produced them.
  `root.step("fit", Fit).epochs[0][0].result("ce")` is the metric of one
  training update. `Plan.result(name)` is the flat lookup; it descends through
  steps but **stops before a fit's updates**, or every fitted document would
  have six ambiguous `iia`s.
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
- **One state per `steps` list, and a step is the unit.** The walk is "for
  each step, run it": what a step produces — a read by its op's name, a
  decode's ids, a metric, a reduction — goes into `State.values`, so a
  write's operand is simply the name of an earlier value, and each step's
  windows of rows slice it by its own layout. A nested plan gets a copy; a
  fit's update gets a copy that keeps each value's graph until the
  optimizer step, and everything else is published detached. A step reports
  its own provenance, when it has a position the document leaves open.
- **No featurizer-isolation flag.** Each point rebuilds its own parameters
  before using them and execution is sequential, so nothing was needed yet.

### The sweep, and what it proved

Built, in `plan/sweep.py` and `build_request`. `documents/v2/pos_sweep.json`
is one document that is three experiments — the same interchange patched at
the last token, the one before it, and the one before that.

A sweep is lowered on the **raw JSON, before anything is compiled**: the
wrapper is replaced by each value in turn and each resulting document is
compiled on its own, so a point is an ordinary document with its own
addresses, its own tokenization and its own digest. A sweep of `model` or
`header` is refused by name.

**The whole cost was 22 lines in `build.py`, one refusal in the document and
a new file that is mostly refusals.** Nothing in `engine/`, `steps.py`,
`plan/plan.py`, `plan/write.py`, `ops/` or `address.py` changed — the engine
walks the same tree it always walked, and the writer already recursed. That
is the evidence that the nesting earned itself. Two behaviours fell out
rather than being built: `root.result("iia")` refuses with "3 in this plan"
because `iia` means three things now, and the three points write into
`pos=-1/`, `pos=-2/` and `pos=-3/` because a plan's path in the tree is its
path on disk.

## 6. Findings from this project worth carrying

Full detail in `FINDINGS.md`; these are the ones that reach past mini.

- **A second runtime needed no family table either, and agreed to the bit.**
  Translating nnterp's standardized names onto a raw HuggingFace tree is one
  function: the decoder is `base_model`, the head is `get_output_embeddings()`,
  and only the layer stack has to be guessed (the decoder's one `ModuleList`).
  nnsight's forced left padding is the one silent default that had to be
  copied for the two engines to be comparable at all — invisible on rotary
  models, 0.38 on a padded GPT-2 row. FINDINGS §6.
- **No family axis was needed.** `Address.locate` returns *equal* addresses on
  tiny Llama and tiny GPT-2 for all three components, including the interior,
  though the trees share no module path. nnterp absorbs the family axis, the
  interiors included: one hard-coded op serves every family that calls
  transformers' attention interface, and a family that does not says so.
- **Binding-suffix addressing is a real trap, and mini measured it — but the
  claim about causalab was overstated and is corrected here.** On GPT-2,
  `query_states_0` is the cross-attention query on a branch that never runs
  and `query_states_1` is the real one; Llama has only `query_states_0`. An
  address written as a variable-binding suffix reads a dead branch on one
  family and the right tensor on the other, silently. That measurement
  stands, and it is why mini addresses the *call*.
  **What was wrong:** an earlier version of this note said the real
  causalab's `sources.py` addresses scores and probabilities this way and
  that it works by luck. It does not. Its documented rule is a substring
  match with refusal on ambiguity and a preference for the hit whose own
  source line *calls* the symbol — the fix mini's `find_op` used before the
  interiors became nnterp rows — and it says
  "NEVER a hardcoded `_n` suffix for a symbol that appears once". A suffix is
  spelled only where two *live* ops share a symbol. Checked against
  transformers 5.17: llama and gpt2 bind `attn_weights` in the same order, so
  `_1` is the post-mask softmax input and `_2` the softmax output on both;
  GPT-2's extra `.type(value.dtype)` line lands at `_3`, downstream of both.
  The residual risk is narrower than claimed: a release that inserts another
  `attn_weights` assignment *upstream* of the softmax moves both silently,
  because the substring still matches exactly one name and the
  refuse-on-ambiguity guard never fires.
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
- **The protocol reader was a quarter of the project** (702 of 2,818 lines when
  it was written), almost all refusals. The weight of causalab is in its
  document surface, not its execution.
- **A layer-0 query interchange is a no-op** when the two prompts share a length
  and a last token. It looks exactly like a broken write.
- **A `yield` cannot appear inside an nnsight block.** nnsight recompiles a
  `with model.session(...)` body as a standalone function, so a
  `@contextmanager` whose body opens a session is a `SyntaxError: 'yield'
  outside function` at call time. This killed an `Engine.open` hook and forced
  the better design: `execute` owns the whole session literally.
- **There is no machine-readable protocol schema.** `docs/intervention_protocol.md`
  is 372 KB of authoritative prose and the Python implements it.

## 7. Environment

- `uv` (0.12.1). `CONTRIBUTING.md` has the commands.
- Tests: `CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`. Type check:
  `uvx pyright` (needs `extraPaths` for nnterp — already in `pyproject.toml`).
- **`nnsight` and `nnterp` are editable local paths** (`[tool.uv.sources]`)
  pointing at `/home/localjadenfk/wd/nnsight` and `/home/localjadenfk/wd/nnterp`.
  **These are shared working checkouts and can move underneath this project.**
  At handoff: nnsight is a detached HEAD at `524c33fc` (the commit of nnsight
  PR #729); nnterp is on branch `standardize-internals`.
- **`--engine ndif` needs the server to have our code.** `causalab_mini` and
  `nnterp` must be installed there at the client's versions; nothing of either
  ships by value any more. That makes the two sides one codebase rather than
  two, and it is why a stock ndif.us cannot run a mini document: it has neither
  package. Self-hosting is the path — build the NDIF image with the three
  checkouts (`nnsight`, `nnterp`, `causalab_mini`) installed into it, and a
  request that names a module the server lacks fails loudly.
- Tiny CPU models: `hf-internal-testing/tiny-random-LlamaForCausalLM` pinned to
  a commit SHA (see the documents), and a tiny GPT-2. Tiny GPT-2 cannot run the
  weekdays documents — `" Friday"` is four tokens there — hence
  `documents/data/counting` and `documents/v2/gpt2_reach.json`.

## 8. The wider context this sits in

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

## 9. How the owner works

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
