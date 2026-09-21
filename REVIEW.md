# REVIEW — the format and the architecture, against three goals

Written 2026-09-21, after the owner set the goals and answered seven
questions. The goals:

1. **Express a wide variety of intervention and interpretability workflows.**
2. **Be easily configured by an agent** — agents author and run workflows as
   JSON through a CLI instead of writing nnsight code.
3. **Be easily understood and usable by a human** — interp researchers and
   workflow authors generally, with no obligation to match causalab's names.

The premises this review takes from the answers: **v2 (`spec.py`) is the
product** and the protocol reader is a compatibility shim; compilation **must
not require model weights**; every workflow class is in scope, plus logit
lens; a step's outputs should be **referenceable by later steps**, ephemeral
unless saved; ragged positions are needed; the hooks engine is a
**measurement fixture** that refuses what it cannot do.

---

## 0. Two answers owed

### Ragged spans, plainly

Today every row contributes exactly one position. A read at `pos: -1` over
four rows is four vectors — a rectangle, `(4, width)`. Everything downstream
assumes that rectangle: `gather` returns it, a featurizer rotates it, a
metric reduces each row of it to one number, a save writes one number per
row.

A *span* is several positions per row: "the tokens of the entity name". Now
row 0's entity is ` Paris` (one token) and row 1's is ` New York` (two). The
read is no longer a rectangle. It is a list of rows of *different lengths* —
that is all "ragged" means — and every consumer that assumed the rectangle
has to be told what to do with a row that has two positions where another has
one. A write is the sharpest case: swapping a two-token span into a one-token
span has no obvious meaning, so the protocol has *policies* for it.

There is a cheap half. If every row's window is the same length — the last
three tokens, say — the read is a rectangle again, `(rows, window, width)`,
and nothing downstream changes shape. That is why §2.D says windows first.

### Why compile ahead, and how to do it without weights

The compiler consults the model for four things: the tokenizer (to resolve
`pos: -1` against the padding it produces, and answer columns to token ids),
`num_layers`, each site's width, and — for an interior — the module's
`.source`. **None of these needs a weight.** A meta-device load
(`StandardizedTransformer(key, dispatch=False)`) supplies all four in 0.38 s
for the tiny model and would take seconds for a 70B, since it reads config
and tokenizer and instantiates modules on `meta`.

So the answer to "why pre-compute at all" is that the reasons still hold and
the cost was never real:

- **A document that cannot run fails in milliseconds, on the client,** with a
  path — not after a 70B model has loaded on a server.
- **The plan is fully decided before the session opens.** That is what makes
  it a few KB of strings and integers that ship to NDIF, and what keeps the
  block trivial: it reads integers, it never decides from a tensor, it never
  needs the tokenizer (which would ship whole).
- **What ran is reproducible from the plan alone.**

Resolving "as they happen" would move all three decisions into the session,
which is exactly what HANDOFF rules 5 and 9 exist to prevent. The change
needed is nothing new on the engine: `load(spec, dispatch=False)`. §2.A.

---

## 1. Verdict

**What holds, and should be protected.**

- **The plan is a pure-data tree with results and saves on its nodes.** This
  serves all three goals at once: an agent gets a schema and a shape it can
  generate; a human reads the tree top to bottom; the engine walks it. Every
  later change below is expressed as *more kinds of node*, not a different
  shape.
- **One session per request, `remote` as the only switch.** Unchanged since
  day one and still the right rule.
- **The engine contract** (seven members) and the fixture role of the hooks
  engine. Two runtimes agree to the bit on single writes; that is enough
  generality, and the contract should be shaped around nnsight from here.
- **The write seam**, `inverse(do(featurize(x)), err, x)`. One line covers
  patching and DAS and, per the survey, five of eight mechanisms and all six
  featurizer kinds. (Its `x` is redundant — §6.)
- **`address.py` as the only file that knows model internals**, and the
  refuse-by-name culture, now backed by pydantic's `extra="forbid"`.

**What does not hold, in order of how much it blocks.**

| | blocks | cost |
|---|---|---|
| **A** compilation needs weights | goal 2 outright: no `validate`/`explain` without a GPU; no authoring against a remote model | 2 |
| **B** nothing crosses steps but featurizers, implicitly, unordered | goal 1: harvest→ablate, choose→patch, logit lens; goal 2: a *silent* wrong answer from a valid document | 3 |
| **C** one intervention per document | goal 1: clean-vs-treatment as steps, any two-experiment document | 2 |
| **D** one position per row | goal 1: spans, `all`, per-row variables, anything from a continuation | 3 |
| **E** no generation | goal 1: decode probes, steering, behavioral | 3–4 |
| **F** no agent-facing CLI or discovery | goal 2: an agent cannot ask what components, kinds or data exist | 1–2 each |
| **G** thin vocabulary: 1 mechanism, 1 featurizer kind, 4 metrics, 11 of 56 components | goal 1, by breadth | 1–2 each |

---

## 2. The load-bearing changes

### A. Weightless compilation

`Engine.load(spec, **options)` passes the runtime's options through, and
`dispatch=False` — nnsight's own spelling — gives a meta shell: for nnterp
literally that keyword; for hooks, `from_config` under
`torch.device("meta")`. `build_spec` works unchanged against it. Whether the
shell can *run* is not the engine's question: nnsight's runs on NDIF, hooks'
has nowhere to, and only the latter refuses.

This unlocks every verb in §3 that is not `run`, and it is the precondition
for an agent that authors on one machine and runs on another.

### B. Cross-step outputs — one ephemeral dict per `steps` list

The owner's design, which replaces an earlier path-based proposal here:

    state = {}
    for step in steps:
        run(step, state)

A step declares `outputs` — names for values it makes available to the
steps after it — and the engine puts them in `state` as it goes. A later
step references one with `{"ref": "mean"}`. The dict is a local of the walk:
it is never saved, never shipped, and dies with the plan, so an output is
ephemeral unless a save on the producing step also names it. Sweep points
are sibling *plans*, each with its own `steps` list and its own `state`, so
they cannot see each other's outputs.

This is not a new mechanism. `featurizers` is already this dict, threaded
through `steps.run` for the one channel that exists today; the change is to
generalize it. Three rules keep it honest:

- **An output is a tensor, never a coordinate.** It may be a write's
  operand, a featurizer's initial basis, a metric's target. It may *not* be a
  layer number or a position. "Pick the best layer, then patch there" would
  make the plan undecidable before the session opens, and that rule is
  load-bearing. That case is two documents.
- **The compiler validates existence and order.** A reference to a name no
  earlier sibling outputs is refused with its path. This is the same check
  that fixes today's silent failure: `score` before `fit` scores an untrained
  rotation and nothing complains. Featurizer use gets the same check.
- **The compiler marks what to keep.** Today a pass's reads die inside it.
  An `Observe` that declares `outputs: {"acts": "v_cf"}` has that read kept
  and published; nothing is kept that nothing names — the client decides.

What it costs: an `outputs` field on the step models, a `Reference` model,
the order check in `Spec._cross_check`, and a resolution step in `steps.run`
before an operand reaches `apply_write`. The engine's `forward` does not
change: by the time it runs, operands are tensors.

### C. Named interventions

`intervention` becomes `interventions: {name: …}`; a step names one with
`"intervention": "clean"`; when there is exactly one, the name is optional.
This is the smallest change that lets a document hold a treatment and a
control as *steps* rather than as intervened models inside one experiment,
and it is what makes "score with the rotation, then score without" writable.

### D. Positions: windows, then ragged

Two stages, and the first is most of the value at a fraction of the cost.

**Uniform windows.** `pos` accepts `{"last": 3}` or `{"span": [a, b]}` with
the same width on every row. `Positions` becomes a window per row; `gather`
returns `(rows, window, width)`; featurizers are pointwise and do not care;
metrics reduce over the window or emit one row per position. Everything stays
rectangular. A multi-position patch that today needs three writes becomes one.

**Ragged.** `{"variable": "entity"}`, `{"all": true}`, `{"column": c}`.
Windows differ per row; reads become flat-plus-offsets; writes need the
protocol's landing policies (`refuse`, `exact_length_buckets`,
`padded_masked`); a row that cannot be aligned becomes an *excluded
measurement* still in the denominator — copy that asymmetry from causalab
wholesale, it is correct. This is where per-row eligibility enters the
output tables, and it should enter with it, not before.

### E. The continuation frame

Generation is a second frame that positions resolve in, and it exists only
after a forward has run. Design it now so D does not have to be redone:
every position carries a `frame` (`prompt` by default); a `Forward` may carry
a decode budget; a read at a continuation position is per step. Build it
after A–D. Whether it is an eighth engine member or a widened `forward` can
be decided then; nnsight's answer is one `generate` trace walked with
`tracer.iter`, which fits inside `forward`.

---

## 3. The agent surface

The old CLI's verbs, minus what causalab needed for its own history
(`migrate`, `pin`, `--resume`), plus what an agent authoring from scratch
needs and causalab never offered — ways to *ask* before writing. Every verb
takes `--json` so its output is parseable.

| verb | needs | what it answers |
|---|---|---|
| `schema` | nothing | the JSON Schema of a document (`Spec.model_json_schema()`) |
| `vocab` | nothing | components, mechanisms, featurizer kinds, metric kinds, position forms — each with its constraints (needs eager attention; hooks refuses; read-only) |
| `model <key>` | config + tokenizer | layers, widths, which components resolve on this family, padding side |
| `tokens <key> <text>…` | tokenizer | whether each string is one token — the `" Friday"`/`" Ottawa"` trap, which is the commonest way a metric column is wrong |
| `data <ref>` | disk | columns, row count, splits, a sample row |
| `validate <doc>` | nothing | refusals with paths |
| `explain <doc>` | meta model | the compiled tree, as `examples/das_walkthrough.ipynb` prints it — forwards, taps, resolved positions, widths, saves |
| `run <doc> --engine --out [--remote]` | weights | the run |

`explain` is the one agents will live in: it is the plan, and the plan is
what the document *means*. A generate–explain–fix loop needs nothing else.

---

## 4. The human surface — notes on the v2 shape

- **`do: {"swap": "v_cf"}`** uses the mechanism name as a key. Compact for a
  human; poor for a schema and for an agent, because the set of valid keys
  is not enumerable in JSON Schema. `{"mechanism": "swap", "operand": "v_cf"}`
  is one line longer and fully typed. Recommend the change, now, while there
  are two documents.
- **Implicit `original`, implicit `identity`, the inserted `featurizers`
  step.** Keep all three. Each is the universal default and spelling it would
  be noise; the cost is that `explain` shows a step the document did not
  write, which the notebook already explains in one sentence.
- **`steps` is a dict, so order is significant.** Every mainstream JSON parser
  preserves object order and pydantic does too, but the JSON standard does
  not promise it. Keep the dict — it reads far better than a list of named
  objects — and say in the schema description that order is execution order.
- **`roles` / `rows`.** The split (field at root, dataset per step) is right;
  the names are adequate. `roles` could be `inputs`; not worth a rename.
- **Sweeps are missing from v2.** `sweep.points` operates on raw JSON and
  works on a v2 document unchanged; wire `build_spec` through it and add the
  cross product. Cost 1.
- **Provenance.** An output directory still does not contain the document
  that produced it, or the engine and versions that ran. For an agent that
  runs fifty variants this is not hygiene; it is the only way to tell them
  apart. Cost 2, and it should land before the CLI does.

---

## 5. Cheap fills, by goal-1 class

Ordered by what each class needs, all cost 1–2 unless marked.

- **Residual-stream patching / DAS** — done. Add: literal operands (zero
  ablation), the additive mechanisms (`add_scaled`, `gaussian`; `renormalize`
  needs the ordering rule, cost 3), `early_stop.mode: min`, `pca` kind.
- **Attention-head work** — per-head slice on an address (cost 2), the
  remaining attention components, `attention_probs` (needs eager). Path
  patching then falls out of C + D.
- **Logit lens** — a `view: "logits"` on a read: project a residual read
  through the model's final norm and head inside the block (causalab's
  `head.py` does exactly this). With a layer sweep and a `top_k` or
  `token_prob` metric, that is the whole method. Cost 2.
- **Harvest → reduce → ablate** — B, plus `save.reduce` (cost 1) and a
  literal-or-reference operand. Mean ablation becomes one document.
- **Feature-level** — the `gate` featurizer family (cost 3: it has a
  train/eval mode and needs the objective to reach inside a featurizer),
  SAE as a loaded featurizer (cost 2), chains (cost 3).
- **Multi-model comparison** — out of scope for one document by design (one
  model per session). Two documents and a comparison step is the right shape;
  do not bend the rule.
- **Sweeps and campaigns** — sweeps in v2 (cost 1), cross products (cost 2).
  Interning and cohorts stay out.

---

## 6. Debts inside the engine

Recorded so they are not rediscovered.

- `inverse(f, err, x)` — `x` is redundant; causalab's is `inverse(f, err)` and
  in a chain the intermediate `x`s do not exist. Drop it when chains land.
- The fit pre-materializes every epoch as its own `Observe`; the payload grows
  with epochs for no information. Carry index lists, select rows in the block.
- Early stopping keeps the *last* weights, not the best. If `eval/iia` is
  reported beside `rot`, they should agree on which epoch they came from.
- Every training update computes every metric, including ones only the
  early-stop reads. Harmless here; not at scale.
- Cayley solves a full `d×d` system per access. Woodbury, when `d` is real.

---

## 7. Order of work

> Status: step 1 landed 2026-09-21. `Engine.load(spec, **options)` passes
> the runtime's own options through, so `dispatch=False` is a meta shell on
> both engines — and, for nnterp, the same shell that runs on NDIF, which
> is why the CLI's third engine is `ndif`: nnterp, undispatched, executed
> remotely. `document.json` and `run.json` in every output directory; the
> verbs `schema`, `vocab`, `model`, `tokens`, `data`, `validate`, `explain`,
> `run`, each with `--json`.

1. **A + provenance + the read-only CLI verbs** (`schema`, `vocab`, `model`,
   `tokens`, `data`, `validate`, `explain`). This is goal 2 delivered: an
   agent can author, check and understand a document with no GPU.
2. ~~**The ordering validator, sweeps in v2, the `do` rename.**~~ Landed
   2026-09-21. `Spec` refuses a step that uses a featurizer before the step
   that trains it, with the fix in the message; `build_request` is the one
   entry point for both formats and lowers sweeps — including cross
   products, labelled `k=8,seed=0` — before either compiler sees a point; a
   write is `mechanism` and `operand`, two fields a schema can enumerate.
3. ~~**B + C.** References and named interventions.~~ Landed 2026-09-21.
   One `State` per `steps` list — the owner's `state = {}; for step in
   steps: run(step, state)` — carrying the live featurizers and what earlier
   siblings published; a nested plan gets its own outputs. A step declares
   `outputs` (a read, kept or averaged over rows); a later write names one
   with `{"ref": …}`; the compiler checks existence, order, name collisions
   and row counts. `interventions` is plural and a step names its own.
   `documents/v2/mean_ablation.json` is the proof: three steps, three
   experiments, a mean that crosses them without touching disk. Logit lens
   still needs the `view: "logits"` read (§5).
4. **D, windows first.** Then ragged, with eligibility.
5. **Vocabulary throughout**, in whatever order the documents being written
   demand.
6. **E.**

The test of each item is the same one this project has used from the start:
write the document first, see whether it needs anything beyond a new kind of
node, and if it does not, the shape is still right.
