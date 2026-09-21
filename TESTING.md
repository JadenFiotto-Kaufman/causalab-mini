# Exploratory testing, 2026-09-21: the consolidated fix list

Six agents tested commit `13d05cb` read-only, one frontier each. Their repro
scripts are under `scratchpad/agents/{arch_a,arch_b,combos,authoring,robust}/`
(bippu, session scratch) and `/disk/u/localjadenfk/mini/agent_real/` (hakone).
Status: `[ ]` open, `[x]` fixed with a pytest. Severity order within a batch.

## Batch 1 — silent wrong answers in the core
- [x] 1.1 A save resolves its metric from the FIRST intervention declaring that name, not the step's (`build._spec_saves`): wrong unit, wrong eligibility mask, shifted rows in the table; reverse case is a bare StopIteration in `write._file`.
- [x] 1.2 Hooks engine passes no `position_ids`: on a left-padded batch it runs a different forward from nnterp (GPT-2: 0.29 off on padded rows; RoPE hides it). Fix: mask-derived position_ids in `hooks.batch()` and generate.
- [x] 1.3 `renormalize` is a silent no-op when it and the write it follows are in different decode frames: `original` is per tap, and prompt-frame taps sort before step taps unconditionally.
- [x] 1.4 Under decode, a prompt-frame read does not see a `{"step":"all"}`/step-0 write at the same address (same root as 1.3).
- [x] 1.5 v2 accepts two absolute writes at one (site, overlapping pos, model); last wins. The protocol format refuses. Lift the check.
- [x] 1.6 Sweep labels are not injective: duplicate values / colliding axes silently drop points (4096 -> 16). Refuse duplicate labels. Also sanitise `/` in labels.
- [x] 1.7 `<model>.generated` is keyed by model only: two forwards of one model (base, counterfactual) collide, alphabetically-last input wins. Key by (model, input) or refuse.
- [x] 1.8 Saving `original.generated` when no read names `original`: validate+explain pass, KeyError at write time after all the GPU work.
- [x] 1.9 A metric/output/read named `*.generated` is overwritten by the generated ids. Reserve the suffix.
- [x] 1.10 pydantic lax mode: `"pos": true` means position 1; `"pos": "-1"`, `"layers": [true]`, `"decode": "3"` coerced. `strict=True` on `Node` (keep int->float for params).
- [x] 1.11 An output (`reduce: pca` etc.) bundle carries no identity stamp, and an empty stamp means NO checks: a basis loads at any layer/component/model. Stamp outputs; refuse unstamped bundles (the repo's own pca/sae artifacts are unstamped) unless explicitly allowed.
- [x] 1.12 `_identity` omits the site's `heads`/`units`: a rotation fitted at head 1 loads at head 2.
- [x] 1.13 `{"column": c}`: first substring match when the text occurs twice; matches inside words; empty value resolves to the first token instead of an excluded row.
- [x] 1.14 A metric may bind to a read that is not vocabulary-shaped (mlp_activation, units/heads on lm_head, a featurized read): IndexError on tiny models, silently a neuron's activation labelled a logit on a real one.
- [x] 1.15 A fit's per-update results never come home from a real server (`children()` stops at eval); `remote="local"` cannot show it and a test pins the lossy set. Decide: ship them or stop promising them.

## Batch 2 — crashes and data loss in the core
- [x] 2.1 `gaussian` has never worked from a document: `params: dict[str, float]` coerces seed to 7.0; `manual_seed` refuses a float.
- [x] 2.2 A write's `features` bound uses the component's full width, ignoring the site's heads/units -> IndexError in the trace.
- [x] 2.3 Operands are never moved to the write's device: `add_scaled`/`lerp`/`{"ref"}` die under `device_map="auto"`; `swap` survives by accident.
- [x] 2.4 A `weights` save to a path not ending `.safetensors` silently writes `[]` (exit 0). A metric saved to `.safetensors` loses its table columns. Make the extension contract two-way and refuse.
- [x] 2.5 A fit cannot save its own `train/loss` / `train/eval`: validate advertises them, `_spec_saves` raises StopIteration.
- [x] 2.6 `save.file_path` escapes `--out` (`../`, absolute); dataset refs escape `--data-root`. Contain both.
- [x] 2.7 A save silently overwrites document.json / run.json / another save. Refuse collisions.
- [x] 2.8 Sweeps have no size guard: `{"range":[0,1e9]}` took a process past 110 GB. MAX_POINTS before any deepcopy.
- [x] 2.9 Ragged read empty on every row -> float index tensor IndexError; refuse at compile (misspelt column).
- [x] 2.10 Under decode a multi-position lm_head read silently collapses to one position. Refuse.
- [x] 2.11 (all but `save path is a directory` now refused too) Raw tracebacks: intervened model named `original`; `counterfactual_inputs[3]` out of range; missing/empty dataset; `--batch-size -1` (KeyError) / `0` (ignored); `{"span":"ab"}` TypeError; `{"all": false}` means all; `{"last":0}`, `{"span":[3,1]}` pass validate; save path is a directory.
- [~] 2.12 (done: `steps: {}`, Infinity, duplicate example_ids, duplicate JSON keys. Kept by design: a model with `writes: []` is the clean control. OPEN: stale files in a reused `--out`) Nonsense accepted: `steps: {}`; a model with `writes: []`; `scale: Infinity`; duplicate example_ids; duplicate JSON keys; stale files in a reused `--out`.
- [x] 2.13 A missing/misspelt `steps` reroutes a v2 document to the protocol parser ("missing top-level group 'data'"). Route on `protocol_version`.
- [ ] 2.14 Protocol format: 24% of mutations are raw KeyError/AttributeError (`document.py` indexes `raw[...]` at ~30 sites). `_need(raw, key, where)`.

## Batch 3 — architectures
Fixtures (pinned tiny checkpoints) are listed in the two architecture reports; smallest covering set:
llama, gpt2, qwen3 (trl, head_dim 128), gemma2 (trl), gemma3_text, phi, phi3, gpt_neox, bloom, olmo2, opt, qwen3_moe, deepseek_v3, gpt_oss, mistral (hf-internal, tokenizer).
- [ ] 3.1 Family-parametrized invariant suite: every component locates+reads or is refused AT COMPILE TIME; `width()` == the tensor's width; residual identities hold (`embeddings == block_input@0`, `block_output@L == block_input@L+1`, `block_input + attention_output == block_mid`, `mlp_input_norm == mlp_input`, `block_mid + mlp_output == block_output`) or an override says why not.
- [ ] 3.2 `locate()` must resolve boundary paths too (meta shell has the tree): today Phi/NeoX/OPT documents validate, explain, and die in the trace / on the server; `causalab-mini model` prints `ok` for components that do not exist. `resolve` must also drop `None` children (StableLM).
- [ ] 3.3 Sandwich/post-norm families: Gemma2, Gemma3(+text), OLMo2 — `block_mid` is bit-identical to attention_output, `mlp_input_norm` is not what the MLP consumes, `attention_output`/`mlp_output` are not the contributions. `_OVERRIDES` (gemma: `pre_feedforward_layernorm`; olmo2: see report) + comments.
- [ ] 3.4 Parallel-residual families (GPT-NeoX w/ use_parallel_residual, GPT-J, Falcon, StableLM, Phi): `block_mid`/`mlp_input_norm` are the block INPUT on NeoX (module exists). Refuse by name via a config-aware check (`refuse` hook beside `address.check`).
- [ ] 3.5 BLOOM (and MPT/DBRX): the residual is added INSIDE attention and MLP, so `attention_output` is the stream and `mlp_output` is the block output. Override to `dense` / `dense_4h_to_h`.
- [ ] 3.6 `attention_premix` width is `hidden_size`; must be heads*head_dim (`width="head_dim"`): Qwen3, Gemma2/3, Qwen3-MoE, DeepSeek. One unit read 64 numbers.
- [ ] 3.7 MLA (DeepSeek V2/V3): query/key dim is qk_nope+qk_rope, z/premix is v_head_dim; `config.head_dim` is neither.
- [ ] 3.8 Gemma2 `final_logit_softcapping`: `lm_head` is not the model's logits (token_prob 0.274 vs 0.226). Lens or refuse. Also `attn_logit_softcapping` vs the scores docstring.
- [ ] 3.9 Path alternatives: premix `dense|out_proj`; neuron_output `dense_4h_to_h|fc_out|fc2`; activation `activation_fn|gelu_impl`; input norm `self_attn_layer_norm|ln_attn`. NOT Falcon-40B `ln_mlp` for block_mid.
- [ ] 3.10 `width()` for intermediate size: try n_inner, intermediate_size, ffn_hidden_size, ffn_dim, 4*hidden; raise AddressError not AttributeError. MoE: refuse mlp_activation/mlp_neuron_output on a sparse block by the resolved module's class (per layer: DeepSeek-V3 layer 0 is dense).
- [ ] 3.11 Hooks `standardized()` cannot be built for OPT, BLOOM, Mixtral, Qwen2/3-MoE, DeepSeek V2/V3. Use nnterp's name lists; resolve lazily so an unresolvable `mlps` costs only the MLP components.
- [ ] 3.12 Fast SentencePiece tokenizers make `" one"` two tokens (`['▁','▁one']`): no document runs on Phi-3 / hf Mistral / Mixtral. Fix `encoding.token_id`.
- [ ] 3.13 Interiors on pre-`attention_interface` families (GPT-J `self._attn(`, BLOOM `F.softmax(`, Falcon three branches) and GPT-OSS sinks (`scores_0` is the pattern, `F_softmax_0` has the sink column): per-family overrides or named refusals. OPT's FFN is 2-D (flattened rows) — refuse with the reason.
- [ ] 3.14 MoE vocabulary proposal: router_logits / router_weights / router_choice; needs a "flat rows" marker. Defer; record.

## Batch 4 — the agent's CLI (goal 2)
- [ ] 4.1 `--json` emits nothing on error; every failure is a traceback. One try/except in `main`: JSON error object, exit 2 = refused document, 1 = crash; strip the pydantic wall to the sentence.
- [ ] 4.2 `run` prints no numbers: a metric is discarded unless saved, silently. Always print/return a per-metric summary; write `summary.json`.
- [ ] 4.3 `{"sweep"}`, `"as"`, `range` are invisible to `schema` and `vocab`. `vocab` hard-codes 2 of 7 position forms; unknown-form error names none. Derive both; add dataset ref grammar, role field subscripts, `.generated`, the extension contract, optional `intervention`.
- [ ] 4.4 `validate` misses 10 classes `explain`/`run` catch; 4 need no model, 3 need only `--data-root`. Add a data pass; document what it does not check.
- [ ] 4.5 `model`: report resolved revision sha, num_heads/head_dim per head component, and only truly-resolving components (3.2). New verbs: `guide` (draft in the authoring report), `template <kind>`, `positions <key> <text>`, `tokens --data --column`, `run --dry-run`, `--set path=value`.
- [ ] 4.6 `--json explain` returns a string; return the plan tree as data.
- [~] 4.7 (done: `produced_by`. OPEN: the run.json fields, the safetensors slot name) `produced_by` is "" on every v2 table; `provenance.document_digest` is never called. run.json lacks device_map, requested dtype, attn_implementation, seed, data digests. Safetensors slot is always named `weight`.
- [x] 4.8 Every step must give rows for every role, even ones its intervention never reads.

## Batch 5 — scale
- [ ] 5.1 A loaded featurizer's bytes are duplicated per sweep point and re-read per point (real SAE: 6.4 GB plan). Share one blob. `plan.source` ships per point though the server never reads it.
- [ ] 5.2 `sae`: `activation: "topk"` with k (Qwen-Scope is TopK-50; through ReLU the code is dense, L0=688). `.pt` dict + transposed orientation: convert or refuse with the reason.
- [ ] 5.3 Gate L1 is a mean, so pressure is w/d: toy weights do nothing at d=2048. Offer `<gate>.mask_sum` or document the scaling.
- [ ] 5.4 `--batch-size` does not bound the concatenated lm_head reads (rows x vocab). Reduce metrics inside the window.
- [ ] 5.5 Tolerances in tests should be width-relative (5e-5 at d=2048, not 1e-6).
- [ ] 5.6 Cosmetic: `aten.allclose.default` / nnterp scan warning printed on every run.

## Verified correct (keep as regression anchors)
All mechanisms and featurizers vs hand formulas (0.0); 18 v2 documents x batch sizes; 26/27 documents x local/serialized/batched; nothing of ours in the return payload; positions never land on a pad; two engines bit-identical on GPU across 4 real families (25,578 numbers) incl. generation; bf16 tracks fp32 incl. the DAS rank curve; DAS across two GPUs; no memory leak across 32 traces; `select=None` (element 0) right on all 24 families; GQA head counts right everywhere.
