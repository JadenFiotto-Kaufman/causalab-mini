# causalab-mini — ground truth for the design

This file is the reading I did of causalab before any design exists. It walks
the three documents copied into `documents/` field by field, states exactly
what a runtime must do about each field, and lists — as explicitly as it can —
everything the real protocol supports that these three documents do **not**
touch.

Nothing in `documents/` is generated. Everything except the three documents
marked "written by me" below is a byte-for-byte copy of a file in
`/home/localjadenfk/wd/causalab`. (Two of those three were authored after this
reading, for cases the copied corpus cannot reach: see §7 and §10.8 for the
`.source` interior, and §9.1 for why a second model family needs its own rows.
Each says so in its own `header.description`.)

---

## 0. What is in `documents/`

| file | copied from | bytes | what it is |
|---|---|---|---|
| `intervention_protocol.md` | `docs/intervention_protocol.md` | 372 346 | the protocol specification (prose + tables) |
| `minimal_cpu.json` | `causalab/configs/protocols/minimal_cpu.json` | 1 903 | the activation-patching document |
| `das.json` | `causalab/configs/protocols/das.json` | 2 827 | the DAS document |
| `das_cpu_reduction.json` | **written by me**, derived from `das.json` | — | a CPU-runnable reduction of `das.json`; see §4 |
| `attention_query_cpu.json` | **written by me**, derived from `minimal_cpu.json` | — | the same interchange with its site at the `attention_query` interior; no copied document taps one (§7) |
| `gpt2_cpu.json` | **written by me**, derived from `minimal_cpu.json` | — | the same interchange on `tiny-random-gpt2`, over `data/counting` — the weekdays answers are multi-token there (§9.1) |
| `data/counting/train.json` | **written by me** | — | 4 rows with single-token answers under the GPT-2 tokenizer |
| `data/weekdays/train.json` | `tests/protocol/fixtures/data/weekdays/train.json` | 1 383 | 4 rows — what `minimal_cpu.json` reads |
| `data/weekdays/data.json` | `tests/protocol/fixtures/data/weekdays/data.json` | 1 389 | 4 rows, 2 `train` + 2 `test` — what `das_cpu_reduction.json` reads |
| `data/natural_domains_arithmetic/data/weekdays.json` | `causalab/tasks/natural_domains_arithmetic/data/weekdays.json` | 35 199 | 49 rows, 30 `train` + 19 `test` — what `das.json` reads |

**No paths were edited.** causalab resolves a `dataset` ref as a relative path
under a *data root*: the ref `weekdays/train` becomes `<root>/weekdays/train.json`
and `natural_domains_arithmetic/data/weekdays` becomes
`<root>/natural_domains_arithmetic/data/weekdays.json`. I laid `documents/data/`
out so that it *is* that root, so all three documents resolve unchanged.

I verified this against causalab's own loader (read-only; nothing in the source
trees was touched):

```
causalab validate documents/das.json              --data --data-root documents/data   -> OK, digest f72244bae35b9716…
causalab validate documents/minimal_cpu.json      --data --data-root documents/data   -> OK, digest b4a4a8a720f8fc51…
causalab validate documents/das_cpu_reduction.json --data --data-root documents/data  -> OK, digest 04a97a661d8f253b…
```

(The two tiny-model documents need the tiny Llama registered from its HF config
first — `registry.register_model(registry.model_info_from_hf_config(key, cfg))`.
That is true of the *shipped* `minimal_cpu.json` too: it is deliberately outside
causalab's static model registry, and `run` registers it from the checkpoint
config. My reduction fails and passes identically to the shipped one.)

---

## 1. The protocol specification: what "the protocol json" actually is

**There is no JSON Schema, and no machine-readable definition of any kind.** I
looked for one:

- `find … -name '*.schema.json'` over the repo returns nothing outside
  `.venv/` (only setuptools' own schemas).
- There is no `causalab … schema` CLI verb; the CLI's subcommands are
  `migrate` plus the verb list built in `causalab/cli.py` around line 116.
- `grep -rn 'json_schema|jsonschema'` over `causalab/` returns nothing.

What exists is:

1. **`docs/intervention_protocol.md`** — 372 KB of prose and tables, headed
   "Intervention Protocol — specification, `protocol_version` 3". This is the
   authority. It is normative: it uses "refused", "load error", "closed
   vocabulary", numbered validation rules (`[V4]`, `[P2]`, rule 19, …) that the
   code emits by number.
2. **`causalab/protocol/schema.py`** — 232 KB of Python: frozen dataclasses plus
   a strict parser. Its own docstring says it "is the authoring surface of
   `docs/intervention_protocol.md`", i.e. it implements the Markdown, not the
   other way round. It is executable, but it is causalab code and this project
   imports nothing from causalab, so it is not copied.
3. **`causalab/protocol/canonical.py`** — the canonical form (§7 of the spec):
   a deterministic re-emission used for digests. Also Python, also not copied.

So: **the spec is prose only, and I copied the prose file.** A "canonical form"
definition exists but as code, not as a document.

The parts of the spec that matter for the three documents below are §1 (document
layout), §2.1–2.12 (the sections), §4 (execution semantics), §2.10 (metrics) and
§2.11 (train). §3 (sweeps), §3.1 (`at_once`), §3.2 (path blocks and `axes`), §7
(canonical form and digests), §8 (engine contract) and §9 (CLI) are not exercised
by any of the three.

### Document layout, in one table

Four top-level groups, all required, recommended (not enforced) in this order:

| key | content |
|---|---|
| `header` | `protocol_version` (required, the string `"3"`), optional `title`, optional `description`. `title`/`description` are authoring metadata and do not enter the digest. |
| `model` | which network, and how it is realized numerically. |
| `data` | which rows, per role. |
| `method` | the experiment: twelve optional/required subsections, of which `sites`, `reads` and `save` are required. |

`method`'s subsections, in canonical order: `segments`, `positions`, `sites`✓,
`featurizers`, `params`, `code`, `reads`✓, `writes`, `intervened_models`,
`metrics`, `train`, `save`✓.

---

## 2. `documents/minimal_cpu.json` — the activation-patching document

### 2.1 Why this one

I wanted the smallest *real* file with an actual write in it, on a small model.
Every shipped/tested JSON with a `writes` section, sorted by size:

| document | why not |
|---|---|
| `probe_generate.json` (1 168) | one write, but it is a *steering* write (`add_scaled`) and its read is in the **generated** frame — it needs greedy decoding and the continuation position frame. Not activation patching, and strictly more machinery. |
| `interchange.json` / `02_interchange_im.json` (1 305 / 1 258) | **exactly `minimal_cpu.json`'s method**, on `meta-llama/Llama-3.1-8B` at layer 18. Same one write, same two reads, same two metrics. Rejected only because the model does not fit CPU testing. |
| `mean_ablation.json` (1 463) | one write and one metric — smaller on the metric axis — but its operand is a `params` entry loaded from *a prior run's artifact* (`harvest/acts.safetensors`). It cannot run standalone; causalab's own test suite exempts it for that reason. |
| **`minimal_cpu.json` (1 600)** | **chosen.** One write, two reads, two metrics, one layer, a tiny random Llama pinned to a commit SHA, fp32, CPU. It is causalab's standalone-install smoke assertion — the one document the project itself certifies as "runs end to end on a laptop". |
| `weekdays_8b_interchange.json`, `drift_locate_scan_im.json`, all the `mcqa_*` scans | 8B models, and/or a `sweep` wrapper. |

The brief asked for "one metric". `minimal_cpu.json` has two. I took the small
model over the metric count deliberately: the two metrics are `match` and
`logit_diff`, which are the two most common scorings in the whole corpus, and
the alternative with one metric costs an 8B checkpoint. If the design wants one
metric, delete the `iia` entry and its `save` line — nothing else depends on it.

### 2.2 Field by field

```json
"header": {"protocol_version": "3", "description": "…"}
```

- `protocol_version`: the string `"3"`. Compared for equality, never ordered.
  A runtime refuses anything else.
- `description`: free text. **Not** part of the experiment's identity. The
  engine does nothing with it. (This document's description is worth reading:
  it explains that `revision` is a commit SHA rather than a branch precisely so
  an upstream re-upload cannot move what CI asserts.)

```json
"model": {
  "key": "hf-internal-testing/tiny-random-LlamaForCausalLM",
  "revision": "9fb191250dd56d0ba7ec9785a025ed29c03d5998",
  "dtype": "fp32"
}
```

- `key`: an HF model id (or a name in causalab's own model registry). The
  network *as a name*.
- `revision`: the checkpoint revision. Here a 40-char commit SHA.
- `dtype`: the compute dtype the weights are realized in. Closed set:
  `fp32` (the default) | `bf16` | `fp16`. **Precision is part of the
  experiment**, not of the run: the spec is explicit that the same document at
  `bf16` and at `fp32` is two different experiments.

What a runtime must do: load that checkpoint at that revision, in that dtype.
Nothing else. (`quantization` and `attn_implementation` also live here; neither
document authors them.)

```json
"data": {
  "base":           {"dataset": "weekdays/train", "field": "input"},
  "counterfactual": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
}
```

- Roles are the keys. `base` is required; `counterfactual` is optional and
  singular (if its value were an array, references would index it as
  `counterfactual[j]`).
- `dataset`: a ref — a relative path under the data root, optionally followed
  by `#<split>`. **No `#` here**, so the whole table is used. (`das.json` uses
  the `#train` form.)
- `field`: which column of the row supplies the prompt text. `[0]` indexes a
  list-valued column, so `counterfactual_inputs[0]` is the first — and here
  only — counterfactual prompt of each row.
- **Rows are paired by index.** Base row *i* meets counterfactual row *i*. The
  base role is never permuted. Both roles must have the same row count.
- **`base` is the schema of the pair**: every column a metric names is read off
  the *base* row, never off the counterfactual.

The resolved table (`documents/data/weekdays/train.json`) is 4 rows. Row 0:

```json
{"input": "If today is Thursday, tomorrow is",
 "counterfactual_inputs": ["If today is Saturday, tomorrow is"],
 "counterfactual_inputs_variables": [{"entity": "Saturday"}],
 "entity": "Thursday",
 "answer": " Friday", "base_answer": " Friday",
 "cf_answer": " Sunday", "label": " Sunday",
 "split": "all"}
```

Column meanings, for a reader who has never seen causalab:

| column | meaning | used by |
|---|---|---|
| `input` | the base prompt | `data.base.field` |
| `counterfactual_inputs` | list of alternative prompts | `data.counterfactual.field` |
| `base_answer` | what the model should say on `input` with no intervention (` Friday`) | `logit_diff.b` |
| `cf_answer` | what the model should say if the swapped-in information won (` Sunday`) | `match.expected`, `logit_diff.a` |
| `label` | the training target — here identical to `cf_answer` | not used by this document; used by `das.json`'s `cross_entropy` |
| `answer` | an alias of `base_answer` in this table | unused here |
| `entity`, `counterfactual_inputs_variables` | prompt variables, for `variable` positions | unused here |
| `split` | which split the row belongs to; `"all"` means one undivided pool | the `#split` fragment |

A leading space in an answer is **normalized away** by the resolver — `" X"` and
`"X"` name the same answer, and `token_form` alone decides the surface form.
This is a real trap: the `["X", " X"]` "cover both forms" idiom is inert.

```json
"method": {
  "sites": {
    "target":  {"component": "block_output", "layers": [0]},
    "lm_head": {"component": "lm_head"}
  },
```

- `sites` is **the complete tap inventory**. Every read and every write must
  name a site declared here, including the head. There are no implicit names.
- `component`: one of a closed vocabulary of 56 names. Here two of them:
  - `block_output` — the residual stream leaving transformer block *L*. The
    spec pins the block algebra: `block_mid = block_input + attention_output`
    and `block_output = block_mid + mlp_output`.
  - `lm_head` — the vocabulary projection. Layer-less, so no `layers` key.
- `layers`: a **band** — a non-empty, strictly increasing list of depth
  indices. `[0]` is the one-layer band, which behaves exactly as the scalar it
  replaced. A band longer than one is one address spanning N layers; **no
  document here uses one**.
- `head` / `expert` / `stream` are optional sub-axes on a site. Not used here.

```json
  "reads": {
    "v_cf":   {"site": "target",  "pos": -1, "model": "original", "input": "counterfactual"},
    "logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
  },
```

- A read is `featurize(activation at (site, pos) in model)[dims]`. With no
  `featurizer` and no `dims` (this document), it is just the activation.
- `pos`: `-1` is *sugar* for `{"index": -1}` — one token per row, counting from
  the end of the sequence. Negative counts from the end; non-negative counts
  from the content start, rebased past any chat prefix.
- `model`: `original` is the reserved name for the un-intervened forward (it is
  never declared); anything else must be a declared `intervened_models` entry.
- `input`: `base` | `counterfactual` | `counterfactual[j]`. For a read on an
  intervened model this restates the IM's own `input` and is **cross-checked** —
  a mismatch is a load error, not a silent override.
- **A read in model M sees the activation with all of M's writes applied**,
  upstream *and at the same address*. To read the un-written value, read in
  `original`.
- Reads never carry `do`.

So: `v_cf` is the layer-0 residual at the last prompt token of the
*counterfactual* prompt, from a clean forward. `logits` is the vocabulary
projection at the last prompt token of the *base* prompt, in the patched model.

```json
  "writes": {
    "patch": {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}
  },
```

- A write is an **inert definition**. It carries no `model` and no `input`. It
  executes inside every intervened model that lists it, and nowhere else.
- `do` has exactly one key, from a closed set of nine mechanisms. `swap` means
  `f ← op`, and is in the **absolute** class.
- The operand `"v_cf"` is a read name. An operand may only be a read name, a
  param name, or a literal scalar — never a tensor, never a closure.
- Per (site, overlapping pos, model): **at most one absolute write**, any number
  of additive ones. Absolutes apply first, then additive deltas are summed. This
  is what makes write sets order-free.
- **Rule 21**: an operand must be read at or above the address it lands on.
  Here they are the same address on two different inputs, which is legal and is
  the ordinary interchange idiom.
- The full write semantics are `write(inverse(scatter(do(f[dims]) into f), err))`.
  With no featurizer, `featurize` is the identity and `err` is 0, so this
  collapses to "replace the tensor at (block_output layer 0, last token) with
  `v_cf`".

```json
  "intervened_models": {
    "patched": {"input": "base", "writes": ["patch"]}
  },
```

- `input` is **mandatory** — which role's rows this model runs on.
- `writes` is unordered (the canonical form sorts it).
- Every declared write must appear in at least one intervened model.
- Cross-model data flow has exactly one channel: a read in model A may be the
  operand of a write in force in model B. The graph IM → writes → operand reads
  → IMs must be acyclic; it is the execution schedule.

Here the graph is: `patched` needs `patch`, `patch` needs `v_cf`, `v_cf` is read
in `original` on the counterfactual input. So two forwards: `original` over the
counterfactual rows, then `patched` over the base rows.

```json
  "metrics": {
    "iia":        {"kind": "match",      "of": "logits", "expected": "cf_answer",
                   "token_form": "space_prefixed"},
    "logit_diff": {"kind": "logit_diff", "of": "logits", "a": "cf_answer",
                   "b": "base_answer",  "token_form": "space_prefixed"}
  },
```

- `of` names a **read**, and a metric binds to exactly one read, hence to
  exactly one (model, input) pair.
- `a`, `b`, `expected`, `target`, `token` name **dataset columns** — the answer
  is per row, so the document names the column and the table carries the string.
  (Contrast `class_probs.groups` and `token_logits.tokens`, which hold literal
  token strings; neither is used here.)
- `match`: the result per example is a 0/1 indicator — does the argmax of the
  read equal the token id of the row's `expected` string? Unit `fraction`,
  estimand `match/v1`. Its optional `mode` is `exact` by default and *is*
  materialized into the canonical form.
- `logit_diff`: the result per example is `logits[id(a)] − logits[id(b)]`.
  Unit `logit`, estimand `logit_diff/v1`.
- `token_form` is **required** on every kind that resolves a string. Four
  values: `auto` (try `" " + s`, fall back to `s`; refuses if the two forms
  disagree), `bare`, `space_prefixed`, `id` (exact integer vocabulary ids from
  the column). Both metrics here pin `space_prefixed`.
- A column value must resolve to **one** token. A multi-token value is refused,
  not silently scored on its first piece. (`mode: "first_token"` opts into
  first-piece credit; not used here.)
- Both metrics are **token-space** kinds, so their read must be a *plain*
  `lm_head` read — no `featurizer`, no `dims`. `logits` is exactly that.
- What a metric consumes: `logits` at (lm_head, pos −1) has shape
  `(n_rows, vocab)`; a metric gathers per-row scalars out of it.

```json
  "save": [
    {"value": "iia",        "model": "patched", "input": "base", "file_path": "iia.json"},
    {"value": "logit_diff", "model": "patched", "input": "base", "file_path": "logit_diff.json"}
  ]
```

- `save` is mandatory, non-empty, and **the complete manifest of everything
  that leaves the run**. Nothing is written that is not listed.
- `model`/`input` **restate** the binding already resolved from the
  declarations, and are cross-checked. They are drift protection, never a
  second source of truth.
- `file_path` is relative to the run's output directory. Only two formats
  exist: `.json` for per-example metric tables, `.safetensors` for dense
  numerics.
- A metric table is a JSON **array of row objects**, one row per example, each
  `{example_id, metric, value, eligible, unit, estimand_version, produced_by, …}`.
  `example_id` is the base row's label — the `example_id` column if the table
  has one, otherwise the zero-based row index as a string. `produced_by` is the
  point digest. Labels repeat on every row, deliberately, so `jq` and a human
  can both read the file. **One file per metric.**
- Rule: every metric must be saved. There is no eval-only exemption.

---

## 3. `documents/das.json` — the DAS document

### 3.1 Why this one

Every shipped or tested document with a `train` section:

| document | why not |
|---|---|
| `04_das_im.json` (1 869) | the *same method* as `das.json`, differing only in dataset ref (`weekdays/data#train` instead of `natural_domains_arithmetic/data/weekdays#train`). It is the test-suite twin, not the shipped template. Worth knowing it exists — it is the smallest DAS JSON in the repo. |
| `mcqa_das_fit.json` (2 171) | sweeps `k` over 8 values (1…128), plus a named `positions` entry. |
| `weekdays_das_sweep.json` / `08_weekdays_das_sweep_im.json` | sweep, and take their layer from a **prior run's artifact** (`{"artifact": …, "key": "best_layer"}`). |
| `mcqa_gate_fit.json`, `dbm*.json` | `gate` featurizers (DBM), not `subspace` — a different method. |
| `das_pca_init.json` (3 035) | rank sweep plus `init` from a saved PCA basis — two more features. |
| **`das.json` (2 408 chars, 2 827 bytes)** | **chosen.** One `subspace` featurizer, one `k`, one layer, no sweep, no `init`, no anneal, no phases. `docs/methods/das.md` §7 lists it first, described as "one rotation on `block_output`, fitted under `ce`". |

**No trim of protocol features was necessary**: `das.json` already has no
sweeps, no `at_once`, one layer and one `k`. The only thing wrong with it for
this project is the model — see §4.

### 3.2 What is different from `minimal_cpu.json`

`das.json` is `interchange.json` plus a featurizer plus a `train` block. The
`sites`, `writes` shape, `intervened_models` and the `reads` addresses are
identical. Three things are new.

**(a) `featurizers`.**

```json
"featurizers": {"rot": {"kind": "subspace", "k": 8, "parametrization": "cayley"}}
```

- A featurizer is a pair of maps: `featurize(x) → (f, err)` and
  `inverse(f, err) → x̂`. For `subspace` they are `featurize(x) = (Qᵀx, 0)` and
  `inverse(f, 0) = Qf`, with `Q` an orthonormal `(d, k)` basis.
- `k`: the width of the feature space — the first `k` columns of the rotation
  are the subspace the interchange acts in. The site's other `d − k` directions
  pass through untouched. **`k` is the only width authored**; `d` is *derived*
  from (model, site) and may never be written in the document.
- `parametrization`: how the stored parameter becomes `Q`. Closed set:
  `cayley` (the Cayley transform from the start basis, rank `k`, `O(d k²)` per
  access — the default the shipped templates author), `matrix_exp`, `stiefel`.
- One parameter slot, auto-declared and named `rot.weight`. Slots are never
  authored.
- **One name, several sites**: a featurizer name is a *parameter set*. The same
  name used at a read and at a write is **one** rotation, and gradients from
  both accumulate on it. There is no tying field; naming is the tying.
- **Error-term contract**: `err` and the unselected dims always come from the
  pre-write value at the address. That is what makes a `dims` write a subspace
  swap and a zero write an ablation of only the feature contribution.

**(b) the featurizer on the read and the write.**

```json
"reads":  {"v_cf": {…, "featurizer": "rot"}},
"writes": {"patch": {…, "featurizer": "rot", "do": {"swap": "v_cf"}}}
```

The read gives `Qᵀ x_cf` (a `k`-vector). The write computes, at the base
activation `x_b`: `f = Qᵀ x_b`, `f ← Qᵀ x_cf`, then `x̂ = Q f + (x_b − Q Qᵀ x_b)`.
That is: **replace the base activation's component in the subspace with the
counterfactual's, leave the orthogonal complement alone.** That single sentence
is what DAS *is*.

**(c) different metrics.**

```json
"iia": {"kind": "logit_diff", "of": "logits", "a": "cf_answer", "b": "base_answer",
        "token_form": "space_prefixed"},
"ce":  {"kind": "cross_entropy", "of": "logits", "target": "label",
        "token_form": "space_prefixed"}
```

- **Trap worth flagging**: the metric *named* `iia` is a `match` in
  `minimal_cpu.json` and a `logit_diff` in `das.json`. The name is the author's;
  the arithmetic is `kind`'s. Nothing in causalab ties the two.
- `cross_entropy`: cross-entropy of the read's distribution against the single
  token id of the row's `target` column. Unit `nat`, estimand
  `cross_entropy/v1`. In the weekdays tables `label == cf_answer`, so the
  training target is "say the counterfactual answer".

**(d) `train`.** See §5.

**(e) a third `save` entry.**

```json
{"value": "rot", "site": "target", "file_path": "rot.safetensors"}
```

- A trained featurizer's save entry names `site` instead of `model`/`input`,
  and it too is cross-checked.
- Rules, all load errors: every trained featurizer must be saved; an untrained
  or `file_path`-loaded featurizer may **not** be saved; writes and intervened
  models are never saveable.

---

## 4. `documents/das_cpu_reduction.json` — the derived variant

`das.json` is already minimal in protocol features, so this is **not** a
reduction of the *method*. It is a reduction of the *scale*, because
`meta-llama/Llama-3.1-8B` at `bf16` is not something this project can run.

Exactly four changes against `das.json`, and nothing else:

1. `model.key` / `model.revision` → the tiny random Llama at the commit SHA
   `minimal_cpu.json` pins (`9fb191250dd56d0ba7ec9785a025ed29c03d5998`).
2. `model.dtype` `bf16` → `fp32`.
3. `sites.target.layers` `[18]` → `[0]` (the tiny Llama has 2 layers).
4. The three dataset refs `natural_domains_arithmetic/data/weekdays#train` /
   `#test` → `weekdays/data#train` / `#test` — the 4-row fixture table (2 train
   rows, 2 test rows, one table with two splits, which is the shape the spec
   prefers).

Everything else is `das.json` verbatim: `k: 8`, `parametrization: cayley`, the
two metrics, the whole `train` block, the three `save` entries.

Two consequences a reader should know, both stated in the file's own header:

- The tiny Llama's `hidden_size` is **16**, so `k: 8` is half the residual
  stream. This is a smoke test, not a result.
- `batch.pairs` is 16 but the train split is 2 rows, so a batch is the whole
  split and an "epoch" is one update.

---

## 5. The DAS document's training requirements

```json
"train": {
  "objective":  [[1.0, "ce"]],
  "params":     ["rot"],
  "optimizer":  {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
  "steps":      {"epochs": 10},
  "batch":      {"pairs": 16},
  "precision":  {"feature": "fp32", "loss": "fp32"},
  "eval":       {"every": {"epochs": 1}, "metrics": ["iia"],
                 "split": "natural_domains_arithmetic/data/weekdays#test"},
  "early_stop": {"metric": "iia", "patience": 3, "mode": "max"},
  "seed": 0
}
```

**What is trained.** A rotation — one orthonormal `(d, k)` basis `Q`, stored as
the single parameter slot `rot.weight` and mapped to `Q` by the Cayley
transform. Not a mask, not a probe, not any model weight. **The model is
frozen**; the only trainable tensor in the whole run is `rot.weight`.
`train.params` is the *only* trainability declaration in the protocol — there is
no `requires_grad` anywhere else.

**The objective.** The positional spelling `[[weight, term], …]`; here one term,
weight `1.0`, over the metric `ce`. The loop **minimizes** `Σ wᵢ · termᵢ`. The
sign of the weight *is* the direction: a positive weight on a cross-entropy
minimizes it; a `-1` on a `logit_diff` would maximize the margin. There is no
`maximize` flag. (The alternative named spelling
`{name: {"weight": w, "metric": name}}` and the regularizer terms `l1`/`l2`/`l0`
exist and are not used here.)

**Step budget.** `{"epochs": 10}` — ten passes over the training rows. The
alternative unit is `{"updates": n}`.

**Batch.** `{"pairs": 16}` — sixteen base+counterfactual **pairs** per update,
not sixteen rows. On the 30-row `#train` split that is two updates per epoch
(16 + 14), so at most 20 updates.

**Optimizer.** `adamw`, lr `1e-3`, weight decay `0.0`. `lr` and `weight_decay`
may also be a mapping keyed by the entries of `params` (one optimizer group per
entry); here they are plain numbers. `schedule` defaults to `constant`.

**Precision.** `{"feature": "fp32", "loss": "fp32"}` — the featurizer's
arithmetic and the loss are computed in fp32 even though the model runs in
`bf16`. The model's own dtype lives in `model.dtype` and nowhere else. An
engine that cannot honour the declared loop precision must refuse the document
at load rather than digest one precision and run another.

**What an eval pass needs.** `eval` is `{every, split, metrics}`:

- `every: {"epochs": 1}` — one eval per epoch. (`{"updates": n}` is refused
  unless the engine declares a `train_eval_updates` capability.)
- `split` is a dataset ref exactly like a `data` entry's, and its content digest
  enters the canonical form. When it names a *different* ref from the training
  rows the two must be **endpoint-disjoint**; the same ref for both is the
  visible train-equals-test ablation and is allowed.
- `metrics: ["iia"]` — the metrics computed on that split.
- The pass runs **in eval mode**: no gradients, and any gate is hard rather than
  relaxed (irrelevant for a subspace, which has no soft/hard split).
- The split's rows are read and encoded once per point; only the trained
  model's forward group is re-run per pass.

**Early stopping.** `{"metric": "iia", "patience": 3, "mode": "max"}` — stop when
the eval `logit_diff` has not improved (increased) for 3 consecutive eval passes.

**Seed.** `0`, covering both parameter initialization and data order. A
`subspace` may also carry its own `seed`; absent, it uses this one (or `0` when
there is no `train` block at all, which is what makes an *untrained* subspace a
reproducible random rank-`k` basis).

**What comes out.** `rot.safetensors` — a safetensors bundle holding the fitted
`weight` slot, with an `ArtifactIdentity` stamped into the **header**. The
identity carries: the producing point's digest, model key + revision, model
dtype and quantization, tokenizer, the site record, `k`, the parametrization,
the featurizer dtype, the trained-on data ref *and its content digest*, the
engine, the applied implementations, and the code revision. It is checked on any
later `file_path` load; a mismatch refuses. "A rotation fitted against bf16
weights is not the same artifact as one fitted against fp32 weights, and the
stamp is what says so."

Plus, beside the bundle, `fit_diagnostics.json`, which for a `subspace` records
the saved rotation's `orthonormality_deviation` (`max|QᵀQ − I|`) and whether it
is `within_tolerance` — so a rotation a later document could not load is flagged
in the run that produced it.

And the two metric tables `iia.json` and `ce.json`, exactly as in §2.

---

## 6. The minimum feature set — what causalab-mini must implement

This is the exhaustive in-scope list, derived only from the three documents.

### 6.1 Document structure

- Four groups: `header`, `model`, `data`, `method`. Order carries no meaning.
- `header.protocol_version == "3"`; `header.description` ignored.
- `method` subsections actually used: `sites`, `featurizers`, `reads`, `writes`,
  `intervened_models`, `metrics`, `train`, `save`. **Never** used: `segments`,
  `positions` (as a named table — `pos` is always inline sugar), `params`,
  `code`.

### 6.2 `model`

- `key` (str), `revision` (str), `dtype` ∈ {`fp32`, `bf16`}.
- Nothing else. No `quantization`, no `attn_implementation`.

### 6.3 `data`

- Exactly two roles: `base`, `counterfactual`.
- `dataset` ref forms needed: **both** — a bare ref (`weekdays/train`) and a
  `#split` fragment (`natural_domains_arithmetic/data/weekdays#train`,
  `weekdays/data#train`). A split is a *column of one table*, selected by value.
- `field` forms needed: a bare column name (`input`) and a list index
  (`counterfactual_inputs[0]`). No deeper indexing.
- Rows paired by index; base never permuted; equal row counts required.
- Dataset columns that must be readable off a **base** row:
  `input`, `counterfactual_inputs` (list of str), `base_answer`, `cf_answer`,
  `label`, `split`. (`answer`, `entity`, `*_variables`, `*_forms`, `number`,
  `result`, `scoring_digest`, `string_mode` appear in the tables but no copied
  document reads them.)
- Row label = `example_id` column if present, else the zero-based index as a
  string. **Neither table carries `example_id`**, so index labels are what these
  three documents need.
- A leading space in an authored answer string is stripped before `token_form`
  decides the form.

### 6.4 `sites`

- Fields needed: `component`, `layers`.
- Components needed: exactly **two** — `block_output` and `lm_head`.
- `layers`: a one-element list only. Both the sugar `18` and the list `[18]`
  must canonicalize the same way if digests matter; `das.json` and
  `minimal_cpu.json` both author the list form.
- `lm_head` is layer-less and takes no `layers`.
- No `head`, no `expert`, no `stream`.
- Both components are **module boundaries** in nnterp's standardized tree —
  `layers[i]` output and the head. Neither needs `.source`.

### 6.5 Position specs

- **Exactly one form: `pos: -1`**, the bare-int sugar for `{"index": -1}`, at
  every read and every write in all three documents. One token per row, counted
  from the end of the sequence.
- Cardinality is `one_to_one` by construction (an `index` is one token on every
  input), so no alignment machinery is needed.
- Positions are never resolved to integers in the document; resolution is a
  runtime service against the encoded batch (pad side matters: with left
  padding, `-1` is the last real token; with right padding it is not).

### 6.6 `featurizers`

- One kind: `subspace`. (`identity` is the implicit default when a read or
  write names no featurizer.)
- Fields: `k` (int), `parametrization` (only `cayley` needed).
- One auto-declared slot: `<name>.weight`.
- `d` is derived from (model, site) and never authored.
- One featurizer name used at both a read and a write = one parameter set,
  gradients accumulating from both.
- `err` is always 0 for `subspace`; the untouched complement comes from the
  pre-write value.
- No `dims`, no chains/lists, no `file_path`, no `entry`, no `init`, no `seed`,
  no `dtype`.

### 6.7 `reads`

- Fields: `site`, `pos`, `model`, `input`, and optionally `featurizer`.
- `model` values needed: the reserved `original`, and one declared IM name.
- `input` values needed: `base`, `counterfactual`. (Never `counterfactual[j]`.)
- A read in an IM sees that IM's writes applied.
- No `dims`.

### 6.8 `writes` and `do`

- Fields: `site`, `pos`, optionally `featurizer`, and `do`.
- **One mechanism: `{"swap": <read name>}`.** Absolute class.
- Operand kinds needed: **a read name only**. Never a param, never a literal.
- Ordering rule still has to hold even with one write: at most one absolute per
  (site, overlapping pos, model).
- Rule 21 (operand read at or above the landing address) is satisfied trivially:
  same site, same position, different input.
- No `ragged` policy — every `pos: -1` write is one token per row, so widths are
  uniform by construction.

### 6.9 `intervened_models`

- One entry, named `patched`, in all three documents.
- Fields: `input` (mandatory), `writes` (list of write names).
- `original` reserved and never declared.
- The IM/read graph must be acyclic. In all three documents it is a two-node
  chain: `original`(counterfactual) → `patched`(base).
- **Forward count: 2 per point** — one `original` pass over the counterfactual
  rows to produce `v_cf`, one `patched` pass over the base rows.

### 6.10 `metrics`

Three kinds, and no others:

| kind | fields | consumes | produces per row | unit |
|---|---|---|---|---|
| `match` | `of`, `expected`, (`mode` defaulting to `exact`) | `logits` of shape `(n_rows, vocab)` at one position; the row's `expected` column as one string | `1.0` if `argmax(logits) == id(expected)` else `0.0` | `fraction` |
| `logit_diff` | `of`, `a`, `b` | the same `(n_rows, vocab)`; two string columns | `logits[id(a)] − logits[id(b)]` | `logit` |
| `cross_entropy` | `of`, `target` | the same `(n_rows, vocab)`; one string column | `−log_softmax(logits)[id(target)]` | `nat` |

- `token_form` is **required** on all three, and the only value used is
  `space_prefixed`.
- All three bind to a **plain** `lm_head` read (no featurizer, no dims).
- A column value must resolve to exactly one token id.
- `unit` and `estimand_version` are derived, never authored, and are written on
  every output row.
- Eligibility: a row whose answer column is `null`/absent/empty is an *excluded
  measurement*, not a zero. Neither table has such rows, so the machinery is
  needed only if the design wants it.

### 6.11 `save`

Two of the three entry shapes:

- read/metric: `{"value", "model", "input", "file_path"}` → a `.json` array of
  row objects, one file per metric.
- trained featurizer: `{"value", "site", "file_path"}` → a `.safetensors`
  bundle with an identity header.

Not needed: the `{"kind": …}` non-value entries, and `reduce`.

Rules that must hold: `save` non-empty; every metric saved; every trained
featurizer saved; `model`/`input`/`site` cross-checked against the declarations.

### 6.12 `train`

Needed: `objective` (positional spelling, one metric term, weight `1.0`),
`params` (one featurizer name), `optimizer` (`adamw`, scalar `lr`, scalar
`weight_decay`), `steps` (`epochs` only), `batch` (`pairs`), `precision`
(`feature`, `loss`), `eval` (`every: {epochs}`, `split`, `metrics`),
`early_stop` (`metric`, `patience`, `mode: "max"`), `seed`.

Minimizes `Σ wᵢ · termᵢ`. Gradients flow through the patched forward into
`rot.weight` and nowhere else.

### 6.13 Dataset shapes, concretely

- A table is a JSON array of flat row objects. Every row carries `split`.
- `documents/data/weekdays/train.json` — 4 rows, all `split: "all"`.
- `documents/data/weekdays/data.json` — 4 rows, 2 `train` + 2 `test`.
- `documents/data/natural_domains_arithmetic/data/weekdays.json` — 49 rows,
  30 `train` + 19 `test`; carries the extra columns `*_forms`, `number`,
  `result`, `scoring_digest`, `string_mode` that no copied document reads.

---

## 7. What each document needs from a model

Identical across all three, which is convenient:

**Modules touched.**

| site | nnterp accessor | what happens there |
|---|---|---|
| `block_output` layer L | `model.layers_output[L]` | **read** (counterfactual forward) and **written** (patched forward), at one token position |
| `lm_head` | `model.logits` (equivalently the head module's output) | **read only**, at one token position |

Both are module boundaries. **Neither requires `.source`.** Worth stating
plainly: *none of the three copied documents exercises a `.source` interior at
all.* The brief's "exactly one `.source` interior" therefore needed a fourth
document, which is `documents/attention_query_cpu.json` (`attention_query`,
shipped and worked; see FINDINGS §1.8–1.11) —
in causalab the `.source`-only components are the attention interior
(`attention_probs`, `attention_premix`, `attention_z`, the pre-RoPE projections),
the MLP interior (`mlp_activation`, `mlp_neuron_output`) and the whole routed-MoE
interior.

**Tensors read.** At `block_output`: `(batch, seq, d_model)`, sliced to
`(batch, d_model)` at one position — or `(batch, k)` after `Qᵀ`. At `lm_head`:
`(batch, seq, vocab)`, sliced to `(batch, vocab)`.

**Tensors written.** Only `block_output` at one position: a `(batch, d_model)`
slice replaced in place (after `inverse`, for `das.json`). Nothing else is ever
written.

**Tokenizer, and what for.** Yes, required, for three distinct jobs:

1. **Encoding prompts.** Both roles' texts, batched. Padding side decides what
   `pos: -1` means, so the padding side is part of correctness, not of
   performance.
2. **Resolving answer strings to token ids.** Every metric needs
   `id(" Sunday")` under `token_form: space_prefixed`. A multi-token answer must
   refuse, not silently take its first piece.
3. **The answer-form pre-flight.** Before any forward, a `bare` token_form over a
   table that carries space-prefixed answers must be refused with both surface
   forms decoded. Not triggered by these documents (all three use
   `space_prefixed`), but it is what stops a metric crediting a token the model
   never emits.

**Gradients.** Only `das.json` / `das_cpu_reduction.json`. The backward must
reach `rot.weight` through the patched forward — which means the head has to be
run the way the model runs it (causalab explicitly notes that its `lm_head`
gather optimization is disabled for a read the fit differentiates through, "so
the training gradient is the model's to the bit").

**No decoding.** No document generates. No `max_new_tokens` anywhere.

---

## 8. Not in scope — protocol features these three documents do not use

This list is as important as §6. Everything below is real causalab protocol
surface that **nothing** in `documents/` needs.

**Whole sections never authored**

- `segments` (§2.2.1) — named sequence segments, the `chat` frame,
  `apply_chat_template`, `system` turns, `assistant_prefix`, `continuation`.
  All three documents are plain text, `prefix_lengths` 0.
- `positions` as a named table — `pos` is always inline.
- `params` (§2.6) — free/constant tensors owned by no featurizer. (This is how
  mean ablation gets its mean in.)
- `code` (§2.8.1) — `locator`, `args`, `data_inputs`, `env_inputs`,
  `row_roles`, `source_sha256`, the closure walk, the undeclared-read AST
  checker.
- `axes` (§3.2) — correlated row tuples and dependent axes.
- `path_patching` (§3.2) — the declarative sender/receiver sugar.

**Sweeps and multi-point machinery**

> Partly implemented since this was written: `{"sweep": [literal, …]}` at
> **one** field is lowered by `plan/sweep.py` into one point per value, and a
> swept document compiles to a root plan with one child plan per point. The
> rest of this list still stands.

- `{"sweep": {"range": [...]}}` on any field, and more than one swept field
  (their cross product).
- `{"at_once": [...]}` (§3.1) — N sites live in one forward, member naming
  (`a[layers=10]`), `names` templates, fan-out by reference.
- Multi-point documents among the three copied documents: all three are
  **one point**. `documents/pos_sweep_cpu.json` is authored, not shipped.
- **Cohorts** (§4) — points of one campaign that declare `train` on the same
  model and rows fitting together in one batched forward, each keeping its own
  seed, schedule, objective and early-stop decision.
- **Cross-point interning** (§4, §8) — running each distinct forward group once
  and letting every point gather its own taps from one capture; group identity
  by digest; `RunResult.forwards`.
- Swept bundles: per-entry `ArtifactIdentity`, the `entries` table in a
  safetensors header, `entry` selectors like `{"k": 8, "seed": 0}`.

**Positions**

- `{"variable": "x"}` — per-row windows located from prompt variables.
- `{"column": "c"}` — per-row windows from a column's string.
- `{"span": [a, b]}`, `{"all": true}`, `{"segment": "s"}`.
- `scope` and `relative_to` anchors.
- **Generated positions**: `{"generated": {"max_new_tokens": n}}`, the
  continuation frame, greedy decode, EOS handling, ragged continuations,
  per-step metric rows with a `step` column, `matched: false`.
- **Span algebra**: `indices`, `union`, `intersection`, `before`, `after`,
  `between`, `atomic`.
- **Ragged positions** and the `ragged` write policies `refuse` /
  `exact_length_buckets` / `padded_masked`, and the `ragged_write_unsupported`
  refusal.
- The `alignment` key and its five cardinalities (`one_to_one`, `one_to_many`,
  `many_to_one`, `absent`, `ambiguous`), declared-vs-observed checking, and the
  reason codes `alignment_missing` / `alignment_ambiguous`.
- `edit_groups` — per-row character spans, `atomic` coordinated edits, and the
  refusal to address one constituent without its siblings.

**Sites**

- 53 of the 56 components. In particular: `input_ids`, `embeddings`,
  `block_input`, `block_mid`, the three norm taps, every attention-interior tap
  (`attention_probs`, `attention_scores`, `attention_z`, `attention_premix`,
  `attention_key/value*`, `attention_gate`, `attention_result`,
  `attention_output`), every MLP tap (`mlp_input`, `mlp_activation`,
  `mlp_neuron_output`, `mlp_output`), `ln_final`, the whole Gated-DeltaNet
  `delta_*` family, and the whole MoE family.
- **Layer bands** longer than one (`"layers": [10, 11, 12]` as one address),
  and band lowering to per-layer members.
- `head` sub-axis (and its GQA-narrower KV-space head count).
- **`expert` faces** — the ragged per-expert view of the routed interior, flat
  rows plus per-example widths, width-0 rows for unchosen experts, the
  `featurizer`/`dims` refusal on that face, `expert_permutation`, the dispatch
  pin on `experts_implementation`.
- `stream` (`full_attention` / `linear_attention`).

**Featurizers**

- `pca`, `sae`, `standardize`, and the whole `gate` family (DBM): `sigmoid`,
  `clamp`, `hard_concrete`, `budget` parametrizations; `group` ∈
  `head`/`expert_neuron`/`site`; `axis: position`; `dead` rules; `top_k`
  readout; `k_schedule`; `pool`.
- `subspace` fields not used: `init` (from a PCA basis, with its
  `init_produced_by` / `init_trained_on` / `init_components` / `init_digest`
  stamps), `seed` (and therefore the untrained random rank-`k` control),
  `matrix_exp` and `stiefel` parametrizations, `file_path`, `entry`, `dtype`.
- **Featurizer chains** (`"featurizer": ["rot", "gate"]`), per-stage `err`
  lists, width derivation along a chain, and the empty-list / repeated-stage
  refusals.
- `dims` on a read or a write.

**`do` mechanisms** — eight of nine unused: `add_scaled`, `lerp`, `affine`,
`gaussian` (with `seed` and `axis: tp_duplicated|tp_split`), `renormalize`,
`clamp`, `pytorch_fn`. Also unused: additive writes at all, multiple writes at
one address, and therefore the absolute-then-additive ordering rule in anger.

**Operands** — param-name operands and literal-scalar operands (e.g.
`{"swap": 0.0}` for zero ablation, `{"swap": "mu"}` for mean ablation).

**Metrics** — eight of eleven kinds: `soft_accuracy`, `token_logit`, `kl`,
`js` (and `restrict`), `class_probs`, `token_logits`, `top_k` (and `by`),
`decode`. Also: `token_form` values `auto`, `bare`, `id`; `match` `mode:
first_token` and multi-form `expected` columns; authored `unit` /
`estimand_version`; `minimum_count`; the `ids` domain; the `scoring_digest` /
`string_mode` translation table; `n_eligible` / `n_considered` / `excluded`
aggregate reporting.

**Training** — everything beyond §5: the named objective spelling; regularizer
terms `l1` / `l2` / `l0` with `reduce` and `costs` and `parameter_count`;
`constraint` and its Lagrangian dual pair; `anneal` (linear and geometric);
`control` (PID on a term weight); `phases` and `freeze_masks`; per-entry `lr` /
`weight_decay` mappings; `optimizer.schedule: linear_warmup_decay` and
`warmup_frac`; `steps: {"updates": n}`; `checkpoint`; `early_stop` with
`mode: "min"`.

**Data verbs** — `shuffle: {seed}` (the resample control), `draw: {kind:
uniform, eval: j}` (counterfactual sets drawn per row per epoch), multiple
counterfactual roles, prepared token sequences (`<column>_encoding` with
`tokenizer_digest`, `text_sha256`, `input_ids`, `offset_mapping`).

**Artifacts and references** — `{"artifact": <ref>, "key": <field>}` value
wrappers resolved at load; loading a fitted featurizer by `file_path`;
`location_ledger`, `trajectory` and `rank` save kinds; `save.reduce`
(`mean`/`sum`/`std`/`median`/`count`).

**Model realization** — `quantization` (`int8`/`nf4`/`fp4`, `compute_dtype`,
`double_quant`, `int8_threshold`), `attn_implementation`, `fp16`.

**Execution machinery** — elision (stopping a forward after the deepest tap),
prefix resume from a cached un-intervened pass (`base_digest`, `prefix_reuse`),
CUDA-graph capture, tensor parallelism and `PositionFrame` shard maps,
fire-count checking per write member, the whole capability/refusal table.

**Identity machinery** — the canonical form and point digests (§7), `--set`
addressing, `causalab migrate` from protocol_version 2, run receipts, workflows
and their `pins` / `control` / `waive` / `reduction` blocks, `example_id`
uniqueness checking, `scoring_digest`.

---

## 9. Environment facts

### 9.1 Tiny models used for CPU testing

From `/home/localjadenfk/wd/causalab/tests/neural/engines/nnterp_engine/conftest.py`
(lines 51–58) — the four fixture model ids, verbatim:

```python
TINY_LLAMA       = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_QWEN35_MOE  = "tiny-random/qwen3.5-moe"
TINY_GPT2        = "hf-internal-testing/tiny-random-gpt2"
TINY_GPT_NEOX    = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
```

That file also states the loading convention both engines use: **fp32, eager
attention, CPU**, so any disagreement is an executor bug rather than a kernel or
dtype story.

The pinned revision `9fb191250dd56d0ba7ec9785a025ed29c03d5998` for
`tiny-random-LlamaForCausalLM` comes from `minimal_cpu.json` and is reused by
`tests/neural/engines/pytorch_hooks/test_fan_out_run.py` (line 33). **It is
already in the local HF cache.**

Measured configs (I loaded them):

| model | layers | d_model | heads | kv heads | ffn | vocab | tokenizer |
|---|---|---|---|---|---|---|---|
| `tiny-random-LlamaForCausalLM` @ that SHA | 2 | 16 | 4 | 4 | 64 | 32 000 | `LlamaTokenizer` (sentencepiece) |
| `tiny-random-gpt2` | 5 | 32 | 4 | — | — | 1 000 | byte-level BPE, near-character |

**Tokenization facts that decide whether these documents can run:**

- Tiny Llama: `" Friday"` → `[28728]`, `"Friday"` → `[28728]` — the *same id*.
  This is a sentencepiece family, so the space-prefixed and bare forms coincide
  and the bare-form pre-flight never fires. `" Sunday"` → `[16340]`. All the
  weekday answers are single tokens, so `token_form: space_prefixed` with
  `mode: exact` works. `"If today is Thursday, tomorrow is"` → 10 tokens
  (without specials).
- Tiny GPT-2: `" Friday"` → `[304, 82, 271, 288]` and `"Friday"` →
  `[38, 82, 271, 288]` — **four tokens each, and different**. The weekdays
  answers are *not* single tokens under this tokenizer, so a `match` or
  `logit_diff` over them would be refused (multi-token answer under
  `mode: exact`). **GPT-2 in this project needs its own dataset**, or
  `mode: "first_token"`, or single-character answers. This is the single
  biggest environment gotcha I found.

Note also the tiny Llama's config emits a transformers warning
(`pad_token_id … got -1`) on load. Harmless, but it will appear in every run.

### 9.2 `uv` or plain venv

**`uv`.** It is installed (`/home/localjadenfk/.local/bin/uv`, version 0.12.1),
causalab uses it (`uv.lock`, `[tool.uv.sources]`, `uv sync --frozen` in CI), and
the decisive reason is the dependency shape: nnsight and nnterp are **local
editable checkouts**, not releases. causalab expresses that in `pyproject.toml`:

```toml
[tool.uv.sources]
nnsight = { path = "/home/localjadenfk/wd/nnsight", editable = true }
nnterp  = { path = "/home/localjadenfk/wd/nnterp",  editable = true }
```

A standalone project here wants the same two lines. With plain venv that is two
manual `pip install -e` calls with nothing recording them, and no lock.

Versions currently resolvable (from causalab's own `.venv`, Python 3.10.20):

- nnsight `0.8.1.dev125+ga8ee93782`, importing from
  `/home/localjadenfk/wd/nnsight/src/nnsight/`
- nnterp from `/home/localjadenfk/wd/nnterp/nnterp/` (no `__version__`
  attribute; branch `standardize-internals`, also pushed as
  `origin/nnsight-0.8-causalab`)
- torch `2.9.0+cu128`

Both packages declare `requires-python = ">=3.10"`; nnterp's runtime
dependencies are just `nnsight>=0.8` and `transformers`.

⚠️ The nnsight checkout at `/home/localjadenfk/wd/nnsight` is currently on
branch `remote-env-per-host` (HEAD `524c33fc`, "ndif: cache the remote
environment per host"), **not** detached at a PR commit. An editable install
follows the working tree, so whatever that checkout is on is what
causalab-mini will import. If the project needs a specific commit, pin it
before the first `uv sync` — or better, don't share the checkout.

⚠️ Do **not** use the base conda interpreter (`/home/localjadenfk/miniconda3`,
Python 3.13): it has an old nnsight without `TransformersModel`, so
`import nnterp` fails there.

---

## 10. Observations, not decisions

Opinions I formed while reading. They are not design proposals and the design
is yours.

1. **The three documents share one method skeleton.** `interchange.json`,
   `minimal_cpu.json` and `das.json` are literally the same `sites` / `reads` /
   `writes` / `intervened_models`, with `das.json` adding a featurizer and a
   `train` block. Whatever else is true of the real engine's abstractions, the
   *documents* are pushing hard toward "one interchange, parameterized".
2. **`featurizer` is the seam where activation patching becomes DAS.** The only
   difference between the two methods, at the intervention level, is whether
   `featurize` is the identity or `Qᵀ`. The error-term contract (`err` and
   unselected dims come from the pre-write value) is what makes that one seam
   cover both. It is worth checking whether it needs to be a named table at all
   for a corpus this size, versus a single optional map on the write.
3. **`reads` is doing two different jobs.** `v_cf` is an *operand* — an
   intermediate the write consumes, never scored, never saved. `logits` is an
   *observable* — scored by metrics and the only thing `save` ever points at.
   The real engine gives them one noun. Whether that is the right call is
   exactly the sort of thing this project could answer.
4. **`intervened_models` may be more machinery than these documents need.** All
   three declare exactly one IM, over `base`, with exactly one write. The whole
   acyclic-graph apparatus resolves, here, to "run the counterfactual first,
   then the base with a patch".
5. **The `save` manifest is doing real work and is cheap.** "Nothing leaves the
   run that is not listed" is one rule, and it is what makes an output directory
   auditable. I would keep it even in a mini.
6. **The restated bindings (`save[].model`/`input`/`site`, `reads[].input` on an
   IM) are pure redundancy, cross-checked.** They cost a check and buy drift
   protection in a hand-edited JSON. In a small corpus they may buy nothing.
7. **`token_form` being required is the single best small decision in the
   spec.** The spec's justification (four production bugs, with the gpt2 `"?"` =
   30 vs `" ?"` = 5633 case measured) is concrete, and the cost is one key.
8. **No copied document touches a `.source` interior.** If the project wants
   exactly one, it needs a fourth case — `attention_probs` or `mlp_activation`
   at one layer would be the natural pick, and both have shipped causalab
   documents to copy the shape from (`attention_band_patch.json`,
   `dbm_head.json`).
9. **`k: 8` on a 16-wide residual stream is not a meaningful DAS fit.** If the
   DAS path is meant to *show something*, tiny-random weights will not. A real
   small model (gpt2 proper, or `Llama-3.2-1B`) with a single-token-answer task
   would. The tiny models prove plumbing, not method.
10. **Splits as a column of one table, rather than two files, is a genuinely
    good idea** and costs nothing: `weekdays/data#train` and `#test` cannot
    share a row, because a row declares exactly one split, and anyone can check
    it from the bytes.
