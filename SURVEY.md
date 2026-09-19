# SURVEY — what causalab has that causalab-mini does not

Written 2026-09-19 from five parallel surveys (protocol surface, execution,
learning and measurement, the run surface, and the shipped document corpus),
each reading both codebases and reporting against one shared scale. Every
claim here is keyed to a real path; where a survey's claim was about mini's
own correctness or about the real causalab, it was verified by running code
before being written down.

**causalab**: 102,807 lines of Python (`/home/localjadenfk/wd/causalab`,
`train-loop-executor-free`). **causalab-mini**: 3,086. The ratio is 33:1, but
the ratio of *concepts* is nothing like that, and that is the useful result.

## How to read this

**Cost** — what it would take mini to have the feature:

| | |
|---|---|
| **1 extension** | a new row in a table mini already has. No new concept. |
| **2 new leaf** | one implementation behind an existing contract; a file plus tests. |
| **3 new seam** | a new abstraction, or a change to an existing contract. |
| **4 structural** | bends a load-bearing rule, or needs a whole subsystem. |
| **5 don't / differently** | conflicts with the architecture, or is scale-and-ops machinery. |

**Reach** — `broad` (most real experiments), `narrow` (a recognisable class),
`edge` (one paper, one family, one deployment). The two cells that matter are
cheap+broad and expensive+edge.

---

## 1. The shape of the gap

Most of the distance is **table entries, not architecture**: 56 components
against 3, 8 `do` mechanisms against 1, 11 metric kinds against 3, 6
featurizer kinds against 1, 24 shipped documents against 5. Nearly all of
that is cost 1 or 2.

The expense concentrates in **four ideas**, one paragraph each:

- **Ragged windows.** Every position form whose width varies by row
  (`{"all": true}`, `{"variable": v}`, `{"column": c}`, scoped spans), and
  everything downstream of it: ragged write policies, alignment
  cardinalities, per-row eligibility. A read's value stops being one tensor.
- **The continuation frame.** Generation as a *second* frame that positions
  resolve in, existing only after a forward has run.
- **The routed-MoE face.** A ragged per-expert view with width-0 rows, on a
  flattened `(batch·position, …)` geometry that is not mini's contract shape,
  with a routing table known only after the router runs.
- **`code` blocks.** A locator resolved without importing, a transitive-import
  digest, an AST checker, and then a real `importlib.import_module` inside the
  forward.

And in **three scale subsystems** that mini should not have: cross-point
interning, cohorts, and CUDA-graph capture.

## 2. Five architectural verdicts

This is what the exercise was for.

**The seven-member engine contract survives.** The large majority of
causalab's execution machinery is a new body for `forward`/`locate`/`load` or
a new row in `address.py` — no eighth member. **Generation is the one genuine
eighth-member candidate**: `forward(forward, values, featurizers)` cannot
express "run the prefill, then walk N decode steps and accumulate". Three
features break the rules outright — cross-point interning, prefix resume and
cohorts — all for the same reason: they need mutable state that outlives a
step and spans sibling plans, against "nothing crosses steps except the
featurizers".

**The write seam holds, and a gate is a featurizer.** Settled by causalab's
own code (`class Gate(Stage)`, `featurizers.py:833`; `MECHANISMS` has no mask
entry): `σ(θ)·x_cf + (1−σ(θ))·x_base` falls out of featurize→swap→inverse
unchanged, and `dbm.json` is `das.json`'s document shape exactly. All six
featurizer kinds fit the line. What does break it is **the mechanism
classes**: five of eight are `f→f`, but `add_scaled` and `gaussian` are a
second *additive* class summed after the absolute write, and `renormalize`
needs the pre-write value and must run last. Mini's one-`do`-per-write cannot
express "one absolute plus N additive at one address".

**Fit-inside-one-session holds — and causalab independently agrees.**
`nnterp_engine/fit.py:249` opens exactly one `model.session(remote=remote)`
for a whole fit and builds the stages and the optimizer inside it from a
picklable spec. Mini reached the same design from the other end. Nothing
about the session is what makes any training feature awkward; causalab's own
two refusals are about a client-side redraw and about gradients on a remote
job, not about the loop.

**Nested plans subsume about 9% of the workflow layer** (~1,100 of 12,464
lines): the layout and the addressing. `nested.py` is 695 lines of
`mount`/`qualified`/`rebase` that mini got for free by making `Plan` a
`Step`. What nesting does *not* buy is everything that makes a workflow a
second *run* rather than a deeper one — a step is another document, compiled
separately, talking to its successors through files on disk. Scheduling,
cross-step references, caching and plotting are genuinely different concerns,
not missing step types.

**"A plan is pure data" held everywhere it was tested**, including under a
sweep, a second engine and a fit. The only features that break it are the
three scale subsystems above.

## 3. Do first — cheap and broad

Merged across the five surveys, in order of what each buys. **Status as of
2026-09-19**: rows 8 and 11 are done and row 6 was a survey error; the rest
stand.

| # | do this | cost | why now |
|---|---|---|---|
| 1 | **Read a featurizer bundle back** (`file_path` + identity check) | 2 | Mini already *writes* a 14-field stamp and can never consume one, so the header is inert and the fit→apply chain — half of causalab's shipped templates — is unreachable. |
| 2 | **Write the canonical document into the output directory** | 2 | An output dir currently holds a 64-hex `produced_by` that cannot be inverted: **a mini result does not contain the experiment that produced it.** |
| 3 | **Real eligibility columns** | 2 | `"eligible": True` is a literal in `write.py`. An excluded measurement and a genuine zero are indistinguishable in mini's output today. |
| 4 | **Freeze the model at load** (`eval()`, `requires_grad_(False)`) | 1 | Two lines per loader. `FINDINGS.md` §1.15 already diagnoses this and mini declined to fix it; every backward accumulates `.grad` on model weights for no reason. |
| 5 | **Record engine, dependency versions, and a code digest** | 2 | ~20 lines. nnsight and nnterp are editable checkouts that move underneath the project, and two engines exist whose attention defaults are known to differ. |
| ~~6~~ | ~~**Endpoint-disjoint fit splits**~~ | — | **SURVEY ERROR — already implemented.** `plan/build.py:239` refuses two refs that share a row, by name ("the two must be endpoint-disjoint"), and deliberately permits one ref named twice as the visible train-equals-test ablation. The learning survey reported this absent; it is not. |
| 7 | **`fit_diagnostics.json`** | 2 | Two numbers for a subspace, computed where `Weights` already stands — and what stops a meaningless fit reporting a perfect score. |
| 8 | ~~**The nine module-boundary components**~~ | 1 | **DONE.** Eight added (`embeddings`, `block_input`, `attention_output`, `mlp_input`, `mlp_output`, `ln_final`, plus the interiors `attention_key` and `attention_z`); eleven now. `input_ids` and `attention_probs` were left — see FINDINGS §7. |
| 9 | **Literal scalar operands** (`{"swap": 0.0}`) | 1–2 | Zero ablation, the cheapest baseline there is, is a type widening on `WriteOp.operand`. |
| 10 | **`add_scaled`, `lerp`, `clamp`, `gaussian`** | 1–2 | One function each behind the existing `Mechanism` protocol. (`renormalize` is cost 3 — it needs the ordering rule.) |
| 11 | ~~Authorable `subspace.seed`~~ ✓, `pca` kind, `early_stop.mode: "min"` | 1–2 | The seed is **DONE** and bought `random_subspace_cpu.json`. `pca` and the minimizing objective remain. |
| 12 | **Per-head feature slice on an address** | 2 | One field plus a slice in `gather`/`scatter`. Head-level work is a large share of real interpretability. |
| 13 | **Sweep `{"range": …}` and multi-field cross products** | 1–2 | The two commonest sweep spellings; the plan tree already carries the results. |
| 14 | **Refuse a non-differentiable metric in an objective** | 1 | Mini will happily put `match` in a loss and train on a zero gradient. |
| 15 | **Microbatching by row window inside `forward`** | 2 | Entirely behind the existing seam, and the precondition for running anything bigger than a tiny model. |
| 16 | **The faithful-server harness** (test-only) | 2 | Mini's whole remote claim rests on `remote="local"`, which causalab documents as checking "pickling and imports, and nothing past them". |
| 17 | **`validate` / `explain` / `digest` verbs, `--set`, `--engine`, CI** | 1–2 | Mini already computes everything these print. `--engine` is unwired rather than unwanted. |

### What is done, and what it cost

| item | landed | what it actually took |
|---|---|---|
| The component vocabulary (row 8) | `1d0835b` | Eleven components, not nine: `attention_key` and `attention_z` came along because they are the same call as `attention_query`. Three things the table did not anticipate — the sort key needed a third band for `embeddings`, an interior needed to say whether it means a call's argument or its return, and the hooks engine needed pre-hooks plus a rule for which argument is the activation. FINDINGS §7. |
| Authorable featurizer seed (row 11) | `0c5cb1e` | One field; the plumbing was already there. |
| Three documents (§7) | `0c5cb1e` | `multi_position_patch` needed no code, `hydra_effect` needed `token_logit`, `random_subspace_control` needed the seed. |
| Refusals for unimplemented surface | `4bf0f96` | Four silent-acceptance bugs, §9. |

**And one finding that came only from porting a document**: the two engines
are *not* bit-identical when several writes are installed at one address —
1.49e-08 on one row. Cross-engine bit-parity was a property of one write, not
of writing (FINDINGS §8). It is not on any to-do list here because it is
inside nnsight, but it changes what "the engines agree" means.

## 4. Expensive but broad — plan for these

Four cost-3 items with `broad` reach. Each is a real seam change; none should
be done casually, and three of them are one decision.

- **Uniform-width span positions** (`{"span": [a,b)}`, static `{"indices": […]}`).
  The cheap half of the position story: width is the same on every row, so
  nothing becomes ragged. **This is the fork in the road** — take this one
  before deciding about raggedness.
- **Ragged positions and per-row eligibility.** The expensive half. A read's
  value becomes ragged, metric rows gain a position, and an unalignable read
  row becomes an excluded measurement still in the denominator while an
  unalignable *write* row is refused before any forward. Copy that asymmetry
  wholesale.
- **`params` — free tensors owned by no featurizer.** How mean ablation gets
  its mean in: a harvest saved with `reduce: mean`, reloaded as a constant,
  swapped in. Needs a second operand namespace beside reads.
- **The canonical form.** Mini's digest is over raw JSON, so **two spellings
  of one experiment digest differently** — an authored `"featurizer":
  "identity"` versus an omitted one, or two orderings of an IM's `writes`.

## 5. Don't bring back

- **Everything CUDA-graph** (`cuda_graphs.py`, `graph_cohort.py`,
  `graph_reuse.py`, the `_gather` override, `GraphPool`) — ~3,300 lines whose
  only product is host-time removal on one validated model path, gated by an
  eligibility table longer than several of mini's modules. The cohort variant
  is explicitly *not* bit-identical.
- **Kernel and loader work** — `kernels/moe_glue*.py`, `lean_experts_path`,
  `prompt_masks`, `compile_cache.py`, `fastersafetensors`. All measured, all
  real, all throughput-only, none of it changes a number.
- **Lazy group execution and `reset_reads`.** Mini's client-computed schedule
  and per-`Observe` values make both unnecessary. This is mini's architecture
  being *easier*, and it should be recorded as such rather than ported.
- **The two-level engine seam.** causalab's engines reach past their own
  contract (`cohort.py` and `graph_cohort.py` both disable private-usage
  checking with a written rationale). Mini's one contract with no escape
  hatch is the thing causalab wants, not the other way round.
- **`token_positions.py`** (1,506 lines) — its own docstring calls it
  superseded and partly inert.
- **Binding-suffix `.source` addressing** — see §8; the trap is real even
  though the accusation against causalab was not.
- **Two reference grammars** (`{"artifact": …}` vs `{"step": …}`) — causalab's
  own spec calls it "the one wart" and lists aligning them as future work.
- **Coordinate-suffixed bundle keys and `entry` selectors** — ~400 lines
  solving a filename collision mini's per-point directories do not create.
- **`migrate`** — mini has one protocol version and no history.
- **`clip_grad_norm`, `train.checkpoint`, non-fp32 `train.precision`,
  gradient accumulation** — all four are parsed by causalab and implemented by
  nobody (see §8). Copying them would mean silently ignoring an author.
- **Schema-to-Markdown renderers** — they stop a 372 KB prose spec drifting
  from the code. Mini's docs *are* the code.
- **Governance vocabulary** (controls, `waive`, certification, supersession
  retention, the remote event sink) — a multi-person preregistered-pipeline
  layer; the event sink ships with nothing behind it.

## 6. Bring, but differently — including where mini is worse

Four places the survey found mini's version worse than causalab's, and they
are the most useful rows in the whole exercise:

| what | mini today | what to do |
|---|---|---|
| **The write seam's third argument** | `inverse(f, err, x)` | Drop `x`. causalab's is `inverse(f, err)`: it computes `err` from the pre-write value at featurize time, every kind recovers what it needs from it, and **in a chain the intermediate `x`s do not exist**. `err` is the general answer; `x` is a second, partly-redundant channel. |
| **Cayley** | a full `(d, d)` `linalg.solve` per access (`ops/featurizer.py:37`) | causalab uses the Woodbury low-rank form: one `k×k` solve, `O(dk²)`. Correct and cheap at d=16; a 4096³ solve per access on a real residual stream. |
| **Featurizer caching** | recomputes `basis` on every access (its own comment admits it) | Cache per forward scope — but note the subtlety: causalab shares the *forward* and replays the *backward* per access so gradient accumulation stays bit-identical, and keys a gate's entry on the mode. |
| **Pre-materialized epochs** | `Fit.epochs` is every update, built client-side, so the payload grows with epochs | Keep one `Observe` template per forward group and carry the batch partition as index lists, selecting rows in the block. This is the one change the one-session design actually asks for. |

Also bring-differently: the **row budget** (raise an error naming the knob,
rather than probing and halving), **`explain --engine`** (print which engine
will run; do not preflight a router), the **static model registry** (mini's
engine-derived widths cannot drift — add a registry only for a weightless
`dry-run` verb, and never let a digest read it), and **per-family address
tables** (nnterp absorbs the family axis; a table should be the fallback for
what it does not standardize, not the default).

## 7. The shipped corpus — 24 documents by distance

Distances use the same scale. Every dataset these need ships with causalab in
exactly the shape `data/rows.py` reads, so data is never the blocker; 19 of 24
name Llama-3.1-8B and 4 name Qwen3.6-35B-A3B, which is a practical blocker on
this CPU box, not a design one.

> **Three of these are now ported** — `multi_position_patch_cpu.json`,
> `hydra_effect_cpu.json` and `random_subspace_cpu.json` — with the
> `token_logit` metric and an authorable featurizer seed that the latter two
> needed. `interchange` and `weekdays_8b_interchange` were deliberately not:
> they are `minimal_cpu` at another layer, so they would add a file and no
> coverage.

**Distance 1 — runs today, or after a mechanical retarget (5).**
`minimal_cpu`, `das`, `weekdays_8b_interchange`, `interchange`,
`multi_position_patch`. **`multi_position_patch.json` was verified running on
mini** after retargeting to the tiny Llama: three disjoint absolute writes in
one intervened model, which none of mini's five documents exercises.

**Distance 2 — one new leaf (2).** `random_subspace_control` (needs
`featurizers.<name>.seed`, one key); `hydra_effect` (needs `attention_output`
and the `token_logit` metric — and its five-intervened-model graph, with a
read inside one IM feeding a write in another, **was verified working on mini
already**).

**Distance 3 — a new seam (13).** `weekdays_das_apply`, `mean_ablation`,
`mean_harvest`, `harvest`, `path_patching`, `dbm`, `dbm_apply`, `dbm_head`,
`dbm_head_apply`, `das_pca_init`, `weekdays_das_sweep`,
`weekdays_locate_scan`, `attention_band_patch`. The recurring blockers are
`method.positions` as a name table (8 of 24), the `gate` featurizer family
(4), loading a fitted featurizer (4), and multi-axis sweeps (3).

**Distance 4 — structural (4).** `dbm_expert_neuron`,
`dbm_expert_neuron_apply` (the ragged per-expert face — note these are *not*
blocked by the checkpoint: `tiny-random/qwen3.5-moe` runs on CPU),
`probe_generate`, `probe_variable` (the continuation frame).

**Distance 5 (1).** `workflows/weekdays_8b.json` — two of its six steps are
`{"type": "script"}` naming causalab module paths, so porting it means
reimplementing a knee-finder and a heatmap plotter, neither of which tests
anything about the intervention engine.

### The next five tests, in order

1. **`multi_position_patch`** (zero code) — three absolute writes coexisting
   at disjoint positions in one intervened model.
2. **`hydra_effect`** ported *with* `attention_output` — the only cross-IM
   operand chain in the corpus, plus the cheapest possible proof that the
   address table generalizes past the rows it was born with.
3. **`random_subspace_control`** — one authorable key; proves a subspace is a
   complete object *without* a fit, and sweeps a featurizer parameter rather
   than a position.
4. **The zero-ablation half of `mean_ablation`** (`do: {"swap": 0.0}`) — the
   operand seam is not a read-name seam.
5. **`weekdays_das_apply`** against mini's own `rot.safetensors` — the round
   trip. The first test where one mini run's output is another's input, and
   what makes `dbm_apply`, `dbm_head_apply` and `das_pca_init` cheap after.

Tests 4 and 5 are the last two that can be written before `method.positions`
is needed.

## 8. The 56-component vocabulary

Mini implements 3. Of the other 53:

- **10 are a one-line table row** (cost 1–2): `embeddings`, `block_input`,
  `attention_output`, `mlp_input`, `mlp_output`, `ln_final`, `input_ids`,
  `attention_probs`, `attention_key`, `attention_z`. These sit on accessors
  nnterp already standardizes, and their widths are on the handle.
- **24 need a family axis, a width off the config, or a new contract**
  (cost 3): the norm taps, the pre-RoPE projections, `attention_premix`
  (needs `head_dim`), `attention_result` (**derived**, not tapped — the value
  is not the tensor at the address), `mlp_activation` (names *different
  tensors* on Llama and GPT-2), the router and shared-expert boundaries.
- **19 are a whole model family away** (cost 4): the ten Gated-DeltaNet
  kernel-boundary slots, the three nnterp-only `.source` bindings, and the
  six routed-MoE dispatch slots.

**32 are module boundaries, 23 are interiors, 1 is derived.** The count is
the evidence for the nnterp proposal in `FINDINGS.md` §2: the table has 56
rows, and mini can currently justify writing 13 of them.

## 9. Bugs this survey found

**In mini — four, all one shape, all now fixed** (commit `4bf0f96`). A
document could say something mini does not implement, be accepted, have it
silently dropped, and still produce a **different digest** — stamping an
artifact identity that names an experiment which never ran. `data.shuffle`
(the shuffled-source *control*) was the worst: `RoleSpec.from_json` was the
one parser with no strict-key check. Unknown top-level groups (`axes`) and
unknown `method` sections (`path_patching`) also loaded. Separately, a
non-finite metric was written as a bare `NaN`, which is not JSON.

**In causalab — three, worth reporting upstream.** All verified here:

- **`clip_grad_norm` is parsed, validated and applied by nothing.**
  `grep -rn clip_grad causalab/` returns exactly two lines, both in
  `protocol/schema.py`. A document can author it and be silently ignored.
  (`train.checkpoint` and non-fp32 `train.precision` are the same shape:
  parsed, consumed by nobody, and no engine declares the capability.)
- **`order_rng` and `mask_rng` are seeded identically**
  (`training/state.py:286,289`, both `manual_seed(spec.seed)`), while
  `draw.py:302` exists precisely to decorrelate streams via
  `sha256(f"{seed}:{purpose}")`. One stream was decorrelated and two were not.
  The comment is explicit about being *local* and silent about being
  *correlated*.
- **The remote version guard omits `transformers`.** causalab's `GUARDED`
  (`nnterp_engine/versions.py:38`) is `("causalab", "nnterp", "nnsight",
  "torch")`. nnsight's own `CRITICAL_PACKAGES` (`ndif.py:26`) is `{"nnsight",
  "transformers", "torch"}` — so the one package nnsight flags that causalab
  does not guard is the one the `.source` address table is most exposed to.
  `sources.py:29-33` says it outright: "the suffix moves when a transformers
  release adds or removes a line — that is how transformers 5 broke nnterp's
  GPT-2 dropout address", and the table is stamped to transformers 5.16.1.
  Partly mitigated: a skew that *moves* an op name surfaces as `match_op`'s
  inventory-bearing error, wrapped with the server's `transformers.__version__`.
  Not covered: a release that keeps the names and changes the semantics — the
  pre-mask/post-mask `attn_weights_1` distinction is exactly that kind of fact
  — and a module-boundary-only run never calls `match_op` at all. One token.
- **`{"span": [a,b], "relative_to": …}` parses and silently means something
  else** (`schema.py:2915-2954`): the resolver offsets an `index` only, so a
  span spelled that way falls through to the content frame. The document says
  one address and the run uses another, with no refusal.

### Why a remote payload carries ~85 KB of nnterp

nnsight's `_hide_local_modules` (`intervention/backends/local.py:83`) pops
every loaded module whose root is not in `_SERVER_MODULES` — `{torch, numpy,
transformers, accelerate, diffusers, einops, peft, nnsight}` — and strips
non-`site-packages` entries from `sys.path`. Neither `causalab` nor `nnterp`
is in that set, so a remote `StandardizedTransformer` registers itself for
by-value pickling and its classes ride in every payload. `nnsight.register`
is a thin wrapper over `cloudpickle.register_pickle_by_value`, which covers a
module's plain functions but still pickles an `lru_cache` wrapper by
reference to its defining module — so registration is **not** a substitute
for installing the package server-side, and that is why `versions.py` exists
at all. Scale only; nothing about it changes a value.

## 10. What the survey did not settle

- `train.anneal`'s dotted target (`gate.theta.temperature`) — whether it is a
  closed vocabulary or an open attribute path decides whether `dbm.json` is
  cost 3 or 4.
- The emit site of the `overlapping_write_unproven` reason code was not found.
- `validate.py:1820-1831` gates rule 9 on *both* writes carrying `dims`,
  which the code's own comment flags as an open spec question.
