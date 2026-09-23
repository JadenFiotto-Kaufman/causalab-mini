# FINDINGS

A running list of **every fact about model internals this project had to encode
itself** — every module path, side, tuple index, operation name and shape
assumption that nnterp did not hand us as data — plus what the document format
made us implement twice.

All of it is in one file, `causalab_mini/address.py`, with the
padding-dependent part in `causalab_mini/data/encoding.py` and one autograd fact in
`causalab_mini/ops/intervene.py`. That concentration is the result this slice was built to
produce: if the facts below moved into nnterp, `address.py` would be a lookup
and nothing else in the project would change. Adding DAS — a rotation, a fit,
and a training loop inside the session — added **no** new module path, side,
tuple index or operation name; §1.13–§1.15 are the three things it did add, and
none of them is an address.

Measured against nnterp at `334e4ef` (dist `1.3.1.dev64+g0e4b401af`), nnsight
`0.8.1.dev125+ga8ee93782` and transformers 5.17.0, on
`hf-internal-testing/tiny-random-LlamaForCausalLM` @ `9fb19125` and
`hf-internal-testing/tiny-random-gpt2`.

---

## 1. Model facts we had to encode

### 1.1 `block_output` is `layers.{L}`, side `output`

*Where it came from:* read off `nnterp/rename_utils.py` `LayerAccessor.get_module`,
which builds `model.layers[layer]` and then takes `.output`.

nnterp does standardize the tree, so the *path* is stable across architectures —
that is the whole value of nnterp and it is real. What it does not provide is the
path **as data**. `model.layers_output[L]` is a live accessor: it must be called
inside the trace, against a model object, and it cannot be put in a plan,
pickled, sorted, compared, or shown to a user. So a project whose plan is pure
data has to write down the string `"layers.{layer}"` and the side `"output"`
itself, and that string is now a copy of nnterp's internal convention with no
link back to it.

### 1.2 `lm_head` is `lm_head`, side `output` — and that is not `model.logits`

*Where it came from:* `nnterp/rename_utils.py` `LM_HEAD_NAMES` (the rename
target), plus `StandardizedTransformer.logits`, which is `self.output.logits` —
the **model's** output, not the head module's.

They are two different taps. We chose the module boundary, because the protocol's
site vocabulary is about modules. They happen to be bit-identical here:
`tests/test_patching.py::test_the_engines_metrics_equal_hand_computed_ones`
computes the metrics from `model.logits` and compares with `torch.equal` against
the engine reading `lm_head.output`, and it passes. That equality is a fact about
this architecture (no logit softcapping, no logit scaling, no final transform
between the head and the returned logits), not something nnterp promises. On a
Gemma-2 it would be false. **We encoded an architecture assumption by choosing
one of two taps that nnterp exposes under names that do not say they differ.**

### 1.3 A module output may be a tuple, and which it is cannot be known up front

*Where it came from:* `LayerAccessor.__getitem__`, which does
`target[0] if isinstance(target, tuple) else target`, and `__setitem__`, which
rebuilds `(value, *current[1:])`.

`address.read`/`address.write` reimplement exactly that. It is decided **from the
value at run time**, because it is a property of the transformers version, not of
anything in the document: on transformers 5.17 a Llama decoder layer returns a
bare tensor, on earlier versions a `(hidden, …)` tuple. nnterp has the answer
(`layers_output.returns_tuple(layer)`), but only after an access has populated
it, and only for its own accessors — not for a path.

This is the single most copied-by-hand fact in the file, and it is the one most
likely to be silently wrong in a project that guesses: `output[0]` on a bare
tensor selects **batch row 0** and still has a plausible shape.

### 1.4 Forward order between named components

*Where it came from:* nothing in nnterp. It came from nnsight's
`OutOfOrderError`, i.e. from being told after the fact.

`address.order` encodes: `block_output` at layer L comes before `block_output` at
layer L+1, and every `block_output` comes before `lm_head`. nnsight enforces the
ordering strictly (our first attempt at a test read layer 0 again after reading
the head and failed with
`'model.model.layers.0.output.i0' was requested but the model already ran past it`),
but no library will *tell* you the order of two named taps, so the engine has to
carry a private total order over the component vocabulary. With 3 components that
is a dict of 3 entries; causalab's vocabulary has 56, and every one of them needs
a rank — including the interior ones, where the rank is inside a single module's
forward.

### 1.5 Writing back needs the same tuple knowledge, in reverse

*Where it came from:* `LayerAccessor.__setitem__`.

`envoy.output = tensor` is not enough in general: if the output was a tuple, the
write must rebuild it around the new hidden state. Same fact as §1.3, needed
again on the write path, and a project that only ever *reads* will not discover
it.

### 1.6 Shapes

- `block_output` is `(batch, seq, hidden)`; we slice one position per row.
- `lm_head` output is `(batch, seq, vocab)` — **and this holds only because a
  plain forward keeps every position.** transformers' `logits_to_keep` slices the
  head to the last position under generation. Nothing in nnterp or nnsight
  announces which regime you are in; our position indices would silently address
  the wrong axis entry if it changed. We assume it, we do not check it.

nnterp *does* give us `num_layers`, `hidden_size` and `vocab_size` as plain
integers on the handle, and they are the counterexample that shows what good
looks like: `plan.build` uses `model.num_layers` to refuse a site naming a layer
the model does not have, on the client, before anything runs. Every fact in
§1.1–§1.6 wants to be available in exactly that form.

### 1.6b `attention_query` is a call, not a module, and not a binding either

*Where it came from:* `print(model.attentions[0].source)` on both tiny models,
read line by line. Nothing derived it; there is no table anywhere that says it.

The component is "the query tensor as the attention implementation receives
it". It never crosses a module boundary — `q_proj` is too early (pre-RoPE, and
not head-shaped the way the kernel wants), and the attention module's output is
far too late. It exists only as an argument of one call inside
`LlamaAttention.forward` / `GPT2Attention.forward`:

```
 ALL_ATTENTION_FUNCTIONS_get_interface_0 -> 20     attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
 attention_interface_0                   ->  +     ...
                                            21         self.config._attn_implementation, eager_attention_forward
                                            22     )
 attention_interface_1                   -> 24     attn_output, attn_weights = attention_interface(
                                            25         self,
                                            26         query_states,
```

Four facts had to be written down, and `causalab_mini/address.py`'s
`_COMPONENTS["attention_query"]` is exactly those four:

1. **The module is `attentions.{layer}`** — nnterp's accessor, not a real module
   path (see §1.11).
2. **The operation is the call whose call site contains `"attention_interface("`.**
   Not a name: nnsight's `{callable}_{occurrence}` namespace holds assignments
   and calls *together*, so this forward has `attention_interface_0` — the
   assignment that looks the implementation up — and `attention_interface_1` —
   the call that runs it. A substring match on the name `attention_interface`
   matches **three** operations (the `get_interface` call, the name it binds,
   and the call), and `causalab_mini` refuses on that with the whole inventory
   printed; `"attention_interface("` is call-shaped and matches one. An
   occurrence suffix is a property of the transformers version and of which
   branches exist in that particular forward, so nothing hard-codes
   `attention_interface_1`: `Address.locate(model, …)` resolves it against the
   loaded checkpoint, on the client, and the plan then carries the resolved
   name as a string.
3. **The query is positional argument 1.** The interface is called
   `attention_interface(self, query, key, value, attention_mask, **kwargs)` —
   `args[0]` is the attention *module*, so nnsight's `.input` (its first
   argument) is the module, not the query. `.inputs` and an index is the only
   honest reach.
4. **The sequence is axis 2.** At this point the query is
   `(batch, head, seq, head_dim)` — measured `(2, 4, 5, 4)` on tiny Llama and
   `(2, 4, 10, 8)` on tiny GPT-2 — not `(batch, seq, width)`. Every position
   resolution in this project produces one index per row into *the sequence*,
   so an address has to say which axis that is. `ops.gather`/`ops.scatter` take
   it; every module boundary passes 1 and this one passes 2.

Two things we checked rather than assumed:

- **It needs no eager attention.** Both checkpoints load with
  `config._attn_implementation == "sdpa"`, and the read, the write and the
  change in the logits all happen under it. That is because q/k/v are arguments
  *to* the dispatch, one level above the kernel: `attention_probs` and
  `attention_scores` live inside `eager_attention_forward`, which sdpa never
  calls, but the interface call itself is in the module's own forward whatever
  implementation is selected.
- **It is past RoPE.** `test_the_query_is_head_shaped_and_already_rotated`
  compares the tensor at the address with `q_proj.output` reshaped the way the
  forward reshapes it; they differ. So `attention_query` is the post-rotation
  query, which is what "as the implementation receives it" has to mean.

### 1.6c Forward order acquired a rank *inside* a block

*Where it came from:* nnsight's `OutOfOrderError`, again, and by construction.

§1.4 said a private total order over the component vocabulary was needed.
Adding one interior showed the order is not one-dimensional: `attention_query`
at layer L runs **before** `block_output` at layer L, and after `block_output`
at layer L−1. The sort key is now `(inside the stack?, depth, rank within the
block)`. Reading `q_proj.output` after reading the interface call in the same
trace raises, so the intra-module order is enforced just as strictly as the
inter-module one — which means an engine with several interior components
(causalab has about twenty) must rank them against each other *within one
module's forward*, and nothing in nnsight or nnterp publishes that ranking.
`print(module.source)` shows it to a human, in execution order; there is no
programmatic "op A precedes op B".

### 1.6d An op name is available as data, and it is the only one that is

`Source.names` and each op's `.line`/`.text` are readable **outside** a trace,
so the matcher runs on the client and a bad address is a load error. This is the
one place where the thing we needed was available as data rather than as a live
accessor, and it is worth saying so: it is what makes an interior address
carryable in a pure-data plan at all.

The cost is that reaching `.source` instruments the module's forward for the
rest of the process (nnsight documents the overhead as a few percent when not
tracing). Our session-scoped model fixture is shared by every test and the
numbers did not move, but a project that resolves an address against a model it
then benchmarks should know.

### 1.7 Padding side decides what `pos: -1` means

*Where it came from:* the tokenizer, by inspection —
`tokenizer.padding_side == "left"` for this checkpoint, and a printed
`attention_mask`.

Position resolution is not a model-internals fact in the module-path sense, but
it is the same kind of unowned knowledge: `-1` means "the last real token", and
whether that is the last index of the padded row depends on the padding side.
`encoding.positions` therefore derives each row's content span from the attention
mask rather than trusting `padding_side`, and refuses a non-contiguous mask. No
library we depend on offers "the index of the k-th content token of row i".

### 1.8 Answer strings to single token ids

*Where it came from:* NOTES.md §9.1, re-measured here.

`tokenizer.encode(s, add_special_tokens=False)` with the space-prefixed surface
form, refusing anything that is not exactly one token. On this sentencepiece
tokenizer `" Friday"` and `"Friday"` are the *same* id (28728), so the
`token_form` distinction is inert here and would not be on a byte-level BPE —
which means a test that passes on tiny-Llama proves nothing about the
`token_form` machinery. nnterp has `add_prefix_false_tokenizer`, which is
adjacent to this problem but does not solve it.

### 1.8b The second family needed its own rows, and nothing else

*Where it came from:* running the engine on `hf-internal-testing/tiny-random-gpt2`.

`documents/gpt2_cpu.json` is minimal_cpu.json's method on a GPT-2, and the one
thing it could not reuse is the **data**: tiny GPT-2's byte-level vocabulary
spells `" Friday"` as `[304, 82, 271, 288]`, so a single-token answer metric
refuses (`test_the_shipped_weekdays_answers_are_not_single_tokens_here` asserts
the refusal, by the message, on the shipped document). `documents/data/counting`
is four rows whose answers (`" one"` … `" four"`) are one token each here.

That is the scope line this slice draws, and it held: the **engine** generalized
with no change at all, and the **metrics** needed a new table. Also worth
recording, because it is the counterexample to §1.8: on this tokenizer `" two"`
and `"two"` are different ids, so `token_form: space_prefixed` is load-bearing
here and inert on the tiny Llama. A `token_form` test that passes only on the
sentencepiece model proves nothing; this is the model that can prove it.

### 1.11 Family axes: where they were needed, and why not in `address.py`

**This is the headline of the GPT-2 slice: no entry in `_COMPONENTS` needed a
family axis.** One table, three components, two families, and
`test_one_address_serves_both_families` asserts that `Address.locate` returns
*equal* addresses on both models — same component, same layer, same resolved
operation.

That is not because the two models are alike. Their real module paths share
nothing:

| | tiny Llama | tiny GPT-2 |
|---|---|---|
| a block | `model.model.layers.0` | `model.transformer.h.0` |
| its attention | `model.model.layers.0.self_attn` | `model.transformer.h.0.attn` |
| the head | `model.lm_head` | `model.lm_head` |

It is because of two decisions, each of which is a fact worth writing down:

1. **A path in `_COMPONENTS` is written in nnterp's accessors, not in module
   names.** `layers.{layer}` resolves `model.layers[0]`, `attentions.{layer}`
   resolves `model.attentions[0]`, and those accessors are nnterp's standardized
   tree, which *is* the family axis — absorbed by a dependency instead of
   written here. (Both spellings work: nnterp also aliases the attribute, so
   `layers.0.self_attn` reaches GPT-2's `attn`. We use the accessor, because it
   is the one nnterp documents.)
2. **The interior's operation is resolved per model rather than tabulated.**
   Even if the two forwards had spelled the call differently, the family
   difference would have landed in the matcher's needle, not in a family column
   — and the needle would then be the thing to think about.

They *do* spell it identically (`attention_interface_1` on both), and the
reason matters more than the fact. GPT-2's attention forward is much bigger: a
cross-attention branch, an `is_updated` cache branch, a `using_eager` branch
with a second attention implementation (`self__upcast_and_reordered_attn_0`,
which the Llama forward has no counterpart for). None of that moves the suffix,
because **a call-op suffix counts calls of one symbol** and the forward calls
`attention_interface` once.

A *binding* suffix does not survive the same trip, and here it is, measured:

| operation | tiny Llama | tiny GPT-2 |
|---|---|---|
| `query_states_0` | the query, projected and reshaped (line 10) | the **cross-attention** query (line 27), on a branch this model never runs |
| `query_states_1` | — | the query this model actually uses (line 46) |

So an address spelled as the binding `query_states_0` reads the right tensor on
one family and a never-executed branch on the other, with no error. This is the
same shape as the `attn_weights_1`/`attn_weights_2` divergence that motivated
addressing the call, reproduced on a different pair of names, and it is the
single best argument for "address the call".

### 1.12 A layer-0 query is a no-op for prompts that share their last token

*Where it came from:* a GPT-2 interchange at `attention_query` layer 0 that
changed nothing, and looked exactly like a broken write.

The query at layer 0 is a function of the block input at that position, which is
the token embedding plus the position embedding and nothing else. The four
`counting` prompts have the same length and the same final token, so their
layer-0 queries at `pos: -1` are *the same tensor*, and swapping one for another
is arithmetically a no-op —
`test_the_query_at_layer_0_carries_nothing_a_prompt_pair_differs_in` asserts
both halves (identical results, and identical query rows). The same document at
layer 2 moves the logits.

It did not show up on the tiny Llama because the weekdays prompts have
*different lengths*, so RoPE alone makes the layer-0 queries differ. A test that
happens to pick prompts of unequal length proves the write works; one that picks
equal lengths proves nothing, and the engine cannot tell you which you did.

### 1.13 A tap's width is the one model fact nnterp hands over as data

*Where it came from:* `StandardizedTransformer.hidden_size` / `.vocab_size`,
already on the handle.

DAS needs `d`: `k` is the only width a document authors, and `d` is derived from
(model, site). For the two components that can carry a featurizer here it is a
plain integer attribute on the nnterp handle, readable on the client with no
trace — so `Address.width(model)` is a two-line lookup, and a `k` wider than its
site is a load error before anything runs.

This is §1.6's point made twice over: every fact in §1.1–§1.6 wants to be
available in exactly this form, and this one is. What is *not* available is the
width of an attention interior — `head_dim` is on the config, not on the
standardized handle — so `_COMPONENTS["attention_query"].width` is `None` and a
featurizer there is refused rather than guessed.

### 1.14 `scatter`'s clone is load-bearing for a fit, not hygiene

*Where it came from:* the first training loop, which raised
`one of the variables needed for gradient computation has been modified by an
inplace operation: [torch.FloatTensor [2, 16]], which is output 0 of AsStrided,
is at version 1`.

`ops.scatter` copies the activation and writes one position into the copy. For
activation patching that reads as defensiveness — nothing downstream would have
noticed the in-place edit. For a fit it is the difference between a backward
pass and an exception: the obvious spelling,
`model.layers_output[0][:, -1, :] = edited`, mutates the tensor autograd
recorded as an input to the operations that produced it, and torch refuses.

The finding is the shape of the failure, not the fix: **the in-place write is
correct until the moment you differentiate through it**, and nothing about
activation patching tells you that. An engine that only ever ran forward
interventions would have shipped it.

### 1.15 A model's parameters are frozen, and nobody said so

*Where it came from:* `train.params`, and the absence of anything else.

`train.params` is the protocol's only trainability declaration, so the
optimizer's parameter list is literally it. But the tiny Llama's weights arrive
with `requires_grad=True` — nothing in nnterp or nnsight freezes them, and the
backward pass through the patched forward *does* reach them and accumulates
`.grad` on every model weight the graph touches. It changes nothing here,
because the optimizer only holds `rot.weight`, but the memory is real and a
longer fit on a real model would notice. "The model is frozen" is a property of
which tensors the optimizer was given, not a property of the model.

*Closed 2026-09-21:* both loaders now `eval()` and `requires_grad_(False)`
the model. One trap on the way: nnsight loads weights lazily, on the first
trace, by **replacing** the module — so a freeze applied after construction
lands on the meta shell and is thrown away with it (measured: 0 of 21
parameters trainable after load, 21 of 21 after the first trace). The nnterp
loader therefore dispatches at load unless asked for a shell. On NDIF the
served model is the server's to freeze.

### 1.9 dtype names

`{"fp32": torch.float32, "bf16": torch.bfloat16}` in `engine/engines/nnterp/loading.py`. Small, but it is
the document's vocabulary mapped onto torch's by hand, and `model.dtype` is part
of the experiment's identity, so getting it wrong silently produces a different
experiment.

### 1.10 Loading

`nnterp.StandardizedTransformer` defaults `device_map="auto"`; on a machine whose
CUDA driver torch refuses, every run needs `device_map="cpu"` passed explicitly
(the CLI takes `--device-map`). Also: the model must be loaded **before** the
plan is compiled, because the plan needs the tokenizer and the layer count. That
ordering is forced by the design, not by nnterp, but it is worth stating: there
is no "compile the experiment without the checkpoint" path.

---

## 2. What we would ask nnterp for

In priority order, each of which would delete code here:

1. **A name → (path, side) table as data**, for the component vocabulary — the
   thing `address.locate` is. Not an accessor: a mapping we can read without a
   trace, put in a plan, and print.
2. **Tuple-ness as data**, per (module, side), known before the access — the
   thing `address.read`/`write` rediscover from every value.
3. **A total forward order over named taps**, so an engine does not have to
   maintain a private rank per component.
4. **Content-span position resolution** from an encoded batch: "row i, token k
   from the end", padding-side-independent.
5. **Which axis of the tensor at a tap is the sequence**, and — for an interior —
   **which argument of a call carries which tensor**. Both are per-address
   constants (2 and 1 for `attention_query`); both are currently a comment
   beside a number in `_COMPONENTS`.
6. **An operation's kind, from nnsight rather than from nnterp: is this name a
   call or an assignment?** This is the one thing the matcher wanted and could
   not have. `Source.names` is a flat tuple of strings in which
   `attention_interface_0` (an assignment) and `attention_interface_1` (a call)
   are indistinguishable, so the only way to say "the call" is to look at the
   source line — `op.text.split("\n")[op.line - 1]`, reconstructed from the two
   attributes `Source.__repr__` uses to render its gutter. It works, and it is
   eight lines, but it means an address is matched against *source text*, so
   reformatting a transformers forward could move an address that renaming
   nothing did. A `kind` on an operation, or a `calls` view over a `Source`,
   would let the needle be a symbol again.

What nnterp *does* give, and it is the reason (1) needs so little: the
standardized accessors (`layers`, `attentions`, `lm_head`) already absorb the
family axis, so an address written in accessor spellings needs no family column
at all (§1.11). The gap is that they are live accessors and not a table.

(1)–(3) are `address.py` in its entirety. (4) is half of `data/encoding.py`.

---

## 3. Redundancy in the document format

**Restated bindings, cross-checked.** `save[].model` / `save[].input` restate the
binding already implied by `save[].value` → metric → read; a read on an
intervened model restates that model's `input`. Both are implemented here as
equality checks (`document._saves`, `document._reads`), about six lines total.
They caught nothing in a three-document corpus, and they cost almost nothing; in
a hand-edited JSON they are the only thing standing between a typo and a silently
different experiment. Keep, but know that they are decoration until a document is
edited by a person.

**Two spellings of one position.** `pos: -1` and `{"index": -1}` are the same
thing; `document._pos` accepts both in three lines. Likewise a site's `layers`
has a scalar sugar and a list form (we accept the list only, because that is what
every shipped document authors). Two spellings mean two things to canonicalize
before anything can be digested.

**The same position resolved twice.** In all three documents the read `v_cf` and
the write `patch` name the same site and the same `pos`, and the plan therefore
carries two identical `positions` tuples for one address. Nothing is wrong with
it, but it shows that "a read at an address" and "a write at an address" are
mostly the same object in this corpus, differing by `do`.

**The dataset is loaded and encoded twice.** `data.base` and
`data.counterfactual` point at the same table in all three documents, with
different `field`s. Two forwards over different columns is real work; re-reading
the same JSON file is not. A plan that interned tables by ref would fix it, and
we deliberately did not, because the corpus is three documents.

**Columns that duplicate each other.** In the weekdays tables `answer` ==
`base_answer` and `label` == `cf_answer` on every row. Only the protocol's naming
makes them distinct; a reader of the data cannot tell which column a metric
*should* use.

**One `seed`, two random number generators, on two sides of the session
boundary.** `train.seed` covers "both parameter initialization and data order".
Parameter initialization has to happen *inside* the session — a parameter drawn
on the client is a tensor the block only receives a copy of, and an optimizer
over it trains nothing. Data order has to happen *outside* it — which rows share
a minibatch decides how they are padded, and tokenization is client-side by this
project's first rule. So one authored integer becomes a `torch.Generator` in the
block and a `random.Random` in the compiler, and a reader of the document cannot
tell that the field is doing two jobs in two processes.

**Trainability is stated three times.** A featurizer is trained if it is in
`train.params`; it must then appear in `save`; and it must appear at a read or a
write or it is unused. Each of the three is checkable against the other two, and
`document._cross_check` checks all three pairs (about ten lines). The redundancy
is real, and unlike the restated bindings above it is not obviously decoration:
`params` is the one that *means* something, and the other two are the rules that
stop a fit producing nothing or a document saving a random basis.

**`precision.feature` and `precision.loss` are two fields with one legal
combination here.** Both must be `fp32`. They are separable in causalab (an
fp32 loop over a bf16 model), and the spec is right that an engine which cannot
honour the declared precision must refuse the document rather than digest one
precision and run another — which is what this one does. But in this corpus they
are a single bit spelled as two.

**The `save` manifest and `fit_diagnostics.json` contradict each other.** `save`
is "the complete manifest of everything that leaves the run: nothing is written
that is not listed", and causalab then writes `fit_diagnostics.json` beside the
bundle without listing it. We kept the manifest rule, so the rotation's
`orthonormality_deviation` is not a file: it is an assertion in
`tests/test_das.py`, and the facts that would have gone in the header
(`k`, `d`, parametrization, model, dtype, the fitted rows and their content
digest) are in the safetensors metadata, which is inside a listed file.

**A metric's name says nothing about its arithmetic.** `iia` is a `match` in
`minimal_cpu.json` and a `logit_diff` in `das.json`. This is not redundancy, it
is a trap, and it is worth one line in any document linter: the name is the
author's, the arithmetic is `kind`'s.

**`intervened_models` for one intervened model.** Every document in the corpus
declares exactly one, over `base`, with exactly one write, so the whole
acyclic-graph apparatus resolves to "run the counterfactual forward first". We
implemented the general scheduler anyway (`plan._schedule`, 25 lines) because it
is the only part of the protocol that says what order anything happens in, and
because writing "counterfactual then base" by hand would have hidden the one rule
the corpus is actually exercising.

---

## 5. What the DAS slice can and cannot promise

### 5.1 "The complement is untouched" is a statement in exact arithmetic

The defining property of a subspace swap is that the orthogonal complement of
the rotation survives it. In exact arithmetic it does. In fp32 it does not, and
cannot: `inverse` computes `Qf + (x − QQᵀx)`, so the complement that comes out
is *reconstructed by a projection*, not copied. Measured on a 16-wide activation
with values of order 1 and `k=4`, the complement moves by at most **3.6e-07**
and the swapped-in subspace lands to within **3.0e-07** — rounding, and about
what one fp32 projection costs.

Bit-identity is available only to an axis-aligned featurizer, where the
complement really is copied — `dims` on a write, which this corpus does not use.
So the test asserts the property at `atol=1e-5` and says why, and the one
`torch.equal` in it is about the *other positions* of the tensor, which are
copied and therefore exact.

### 5.2 What the fit demonstrates

Plumbing, not method, and the document says so in its own header. `k: 8` on a
16-wide residual stream of a randomly-initialized 2-layer Llama, over a 2-row
train split, is not a scientific DAS fit and no `iia` it produces means
anything. What the run does show is that every mechanical claim holds: the loss
falls monotonically on the term the document named, the rotation is still
orthonormal afterwards, the same seed gives the same weights to the bit, the
fitted artifact reloads, and `remote="local"` reproduces all of it.

One reproducibility caveat, measured: the same fit run locally and run through
`--remote local` produces **the same weights and the same metadata** but not the
same `rot.safetensors` bytes. safetensors serializes its header map in
non-deterministic order (the first differing byte is 27, inside the JSON
header), so an artifact is content-reproducible and not byte-reproducible, and
anything that hashes the file rather than its contents will see spurious drift.

It also shows one thing worth knowing that a good fit would have hidden: the
objective (`ce`) and the watched metric (`iia`, a `logit_diff`, mode `max`)
*disagree* here — `ce` falls while `iia` also falls — so early stopping fires
after four of the ten epochs. That is the machinery working, on a model where
there is nothing to find.

---

## 4. Things this slice does not know, and should

- **Whether a site's component even exists on the loaded model.** We check the
  layer index against `num_layers`, and — for an interior only — that its
  operation is in the forward. `block_output` on a model with no `layers` envoy
  would still fail inside the trace.
- **Whether the head tap is the last position only.** See §1.6.
- **Whether a read and a write at the same address in the same model are
  ordered correctly beyond "writes first".** The rule is implemented (a tap's
  writes run before its reads); no document in the corpus exercises it.
- **Whether the rotation it fitted is the rotation causalab would have fitted.**
  `cayley` here is `(I − A)⁻¹(I + A)E` with `A = WEᵀ − EWᵀ` built densely and
  solved for `k` columns. That is the spec's "Cayley transform from the start
  basis, rank `k`" as arithmetic, but the spec also says `O(d k²)` per access,
  which implies a Woodbury solve this does not do. The *basis* is the same; the
  cost is not, and the numerics of a dense `d`-by-`d` solve and a `2k`-by-`2k`
  one differ in the last bits. Nothing here can tell you which one causalab's
  digest was computed against.
- **How much a fit costs.** `Subspace.basis` recomputes the Cayley transform on
  every access — twice per write, once per read — because the optimizer steps
  the parameter between accesses and a cached `Q` would be stale. At `d=16` that
  is free. At `d=4096` it is the first thing to fix, and the fix (cache per
  forward, invalidate on `optimizer.step`) needs a place to hang the cache that
  the current `Featurizer` protocol does not have.

---

## 6. What a second engine had to supply that nnterp was providing

A second engine was built to measure exactly this:
`causalab_mini/engine/engines/hooks/`, a `HooksEngine` over a plain
`AutoModelForCausalLM` + `AutoTokenizer` with
`torch.nn.Module.register_forward_hook` doing the reading and the writing. No
nnterp, no nnsight, no session, no `.source`. It is 262 lines in three files
against the nnterp engine's 242 — 119 of them statements, the rest docstrings
and comments — and, the result that matters, it runs the same documents to
**bit-identical** numbers.

Nothing outside the new directory changed. `engine/steps.py`, `ops/`, `plan/`
and `address.py` are untouched; `engine/__init__.py` gained an export. So the
seven-member contract held for a runtime that shares no execution code with the
first, which is the claim HANDOFF §3.6 wanted tested.

### 6.1 The translation is one function with three statements

`Address.path` is written in nnterp's standardized names, and a raw Llama has
`model.layers` while a raw GPT-2 has `transformer.h`. All of the translating is
`loading.standardized`, which returns an object carrying the standardized
attributes so `Address.resolve` walks it unchanged:

| standardized name | raw Llama | raw GPT-2 | how it is found |
| --- | --- | --- | --- |
| `layers` | `model.layers` | `transformer.h` | the decoder's only `nn.ModuleList` |
| `lm_head` | `lm_head` | `lm_head` | `model.get_output_embeddings()` |
| `attentions` | `model.layers[L].self_attn` | `transformer.h[L].attn` | not supplied — refused, see §6.6 |

**It needed no family table, and that is the honest headline: most of what
nnterp's rename table does, transformers already publishes.** The decoder is
`model.base_model` (`base_model_prefix` is `"model"` on Llama, `"transformer"`
on GPT-2) and the head is `get_output_embeddings()`, which *is* `lm_head` on
both. Only the layer stack has no published accessor, so it is "the decoder's
only `ModuleList`" — true of both families here, and precisely the guess
nnterp replaces with a maintained list of names. A model with two `ModuleList`
children under its decoder (an encoder-decoder, a hybrid with a separate
adapter stack) breaks the guess, and the code raises rather than picking the
first.

This measurement cuts both ways. For these two families the standardization
was worth about ten lines — but those ten lines encode an assumption nnterp has
evidence for across dozens of architectures and we have evidence for across
two.

### 6.2 What nnterp was silently providing, item by item

- **The tokenizer.** `StandardizedTransformer.tokenizer` comes with the model;
  a plain load is two `from_pretrained` calls that nothing pairs, and the
  engine has to hold both objects. `Engine.model` is one attribute, so the
  tokenizer lives in a private one behind the `tokenizer` property — the
  contract already had the right shape for this and nothing had to change.
- **Padding side.** The real one. See §6.3.
- **`num_layers`.** `len(self._names.layers)`, once the stack is translated.
  `config.num_hidden_layers` would also have worked on both families —
  transformers' `attribute_map` maps GPT-2's `n_layer` onto it — so this was
  free, and counting the translated stack merely avoids depending on that map.
- **`hidden_size` / `vocab_size`.** Free. `address.py` names them as *nnterp
  handle attributes*, and they are spelled identically on `model.config`
  (again via `attribute_map`, for GPT-2's `n_embd`), so `width` is
  `getattr(self.model.config, attribute)`. Measured equal to nnterp's on the
  tiny Llama: 16 and 32000. Worth recording as a latent trap rather than a
  problem: the `width` column in `_COMPONENTS` is documented as naming one
  runtime's API, and happens to name the config's too.
- **Tuple-versus-tensor at a module boundary.** §1.3's question survives into
  the hook, which is handed the module's output and may return a replacement,
  so the same `output[0] if isinstance(output, tuple) else output` pair
  appears. Measured on transformers 5.17: at both `layers.{L}` and `lm_head`,
  on both families, a forward hook sees a **bare `Tensor`** — the tuple branch
  never fired. It is kept because §1.3's point is that this is a property of
  the transformers version, which is not something either engine can see.
- **Eval mode.** Free, and load-bearing: `from_pretrained` returns a model in
  eval mode, so GPT-2's dropout is off. A `model.train()` anywhere would have
  broken parity non-deterministically, and nothing in the contract or in
  `steps.py` says who owns that.
- **`device_map`.** A plain load defaults to CPU, where nnterp defaults to
  `"auto"` and needs `--device-map cpu` on this machine (§1's last entry). The
  hooks loader defaults to `"cpu"`, so that footgun is nnterp's rather than
  transformers'.
- **The attention implementation.** Both loaders got `sdpa` here, so it did not
  divide the two engines. That is luck, not agreement: nnterp forces `eager`
  when `enable_attention_probs` is on, and causalab's two engines already
  diverge on this exact default (HANDOFF §7). A cross-engine golden that does
  not pin `attn_implementation` is one nnterp default away from being a
  comparison of two different experiments.
- **The dtype table.** Copied verbatim. `{"fp32": float32, "bf16": bfloat16}`
  is the document's vocabulary rather than nnterp's, and both loaders spell it
  out.

### 6.3 Padding side is nnsight's decision, and it is invisible on Llama

`nnsight/modeling/transformers.py` does `self.tokenizer.padding_side = "left"`
unconditionally on every model it loads. A plain `AutoTokenizer` does whatever
the checkpoint's `tokenizer_config.json` says: left for
`tiny-random-LlamaForCausalLM`, **right** for `tiny-random-gpt2`. The hooks
loader therefore sets `padding_side = "left"` itself, with a comment pointing
at the nnsight line, and that one assignment is a precondition of the parity
result below.

Measured on the `weekdays/train` batch (row 0 is 11 tokens, rows 1–3 are 9 and
so carry two pads):

- **Llama, left versus right padding: max divergence 1.5e-08 on `logit_diff`,
  which is rounding.** Rotary attention is a function of position
  *differences*, so shifting every real token in a row by the same offset
  cancels. `iia` is identical.
- **Tiny GPT-2, same comparison: 0.38 on the last-real-token logits of a padded
  row**, and 0.0 on the unpadded one. Learned absolute position embeddings do
  not cancel; the padded row's tokens are simply at different positions.

So padding side is not a formality, it is only invisible on the family this
project's golden happens to use. `encoding.positions` is already
padding-side-independent (§1, `starts`/`ends`), which is exactly what let the
same document compile either way and produce numbers that quietly differ.

### 6.4 The parity result

`tests/test_hooks_engine.py`, CPU, fp32, `tiny-random-LlamaForCausalLM`
@ `9fb19125`. Each engine compiles its own plan — the tokenizer and the widths
are engine questions — and then:

- **`documents/minimal_cpu.json`: bit-identical**, `torch.equal` on both
  results. `logit_diff = [-0.027417995, 0.180859923, 0.035145037, -0.193306163]`
  and `iia = [0, 0, 0, 0]`, the same values `test_patching.py` derives by hand
  from nnterp's own accessors.
- **`documents/das_cpu_reduction.json`: bit-identical through a fit**, which is
  the stronger claim — ten AdamW updates, a backward through each engine's own
  intervention path, early stopping after four epochs — and `torch.equal` holds
  on the fitted `(16, 8)` rotation itself, not only on the metrics over it.

No tolerance was loosened anywhere. Agreement to the bit says the intervention
is the *same arithmetic on the same tensors* under both runtimes: nnsight's
envoy assignment and a hook's return value are two spellings of one write, and
`ops/` does the rest identically.

### 6.5 What the contract did not give the engine, and did not need to

Nothing. The seven members were enough, and the two halves split the way HANDOFF
§3.6 claims: `load`/`tokenizer`/`num_layers`/`locate`/`width` are answered from
the loaded objects on the client, `execute`/`forward` are the run. Two smaller
observations:

- **`execute` gets the whole plan rather than a callback**, which is what let a
  session-less engine exist at all: the hooks engine's `execute` is
  `steps.run(self, plan); return plan`, and the nnterp engine's single session
  turns out to be the special case rather than the baseline.
- **`values` and `featurizers` cross into `forward` as plain dicts**, so a hook
  closes over them and writes reads straight in. No run-state object was missed
  — HANDOFF §4's "there is no run state object" holds on a second runtime.

### 6.6 What the hooks engine refuses, by name

- **`attention_query`, and every interior.** A hook fires at a module boundary;
  the query tensor never crosses one. nnsight reaches it by recompiling the
  forward (`.source`), and there is no hooks equivalent. To support it, a hooks
  engine would need one of: a patched `forward` on the attention module that
  exposes the tensor (a per-family fork of transformers code), or a hook on
  `q_proj` plus a reimplementation *inside the engine* of the head reshape and
  the rotary embedding — because the address says `attention_query` is
  projected, reshaped to `(batch, head, seq, head_dim)` and rotated, with the
  sequence on axis 2. Either one moves per-family model code back into the
  engine, which is the cost this project exists to measure. `locate` raises
  `AddressError` naming `.source`, so `attention_query_cpu.json` fails at
  compile time on the client rather than mid-forward.
- **Input-side module boundaries.** None exist in `_COMPONENTS` today (only the
  interior is input-side), and the fix is `register_forward_pre_hook` with the
  same body. Refused rather than written blind.
- **`remote`, in any form.** A hook is a Python callable registered on a module
  object in this process; there is nothing to ship. `execute` raises on any
  truthy `remote`, `"local"` included, because running here silently would
  produce correct numbers under a false claim about where they came from.

### 6.7 One thing that was harder than expected

Hook lifetime. The nnterp engine's taps live and die with the
`with model.trace(...)` block; hooks outlive the forward unless removed, and a
leaked handle intervenes on the *next* forward — a wrong number, not an error,
and the next forward here is usually the one being scored. The registration
loop is therefore wrapped in `try/finally`, and a test asserts the module's
`_forward_hooks` count is unchanged afterwards. It is four lines, and it is the
only place where hooks are more *dangerous* than tracing rather than merely
more limited.


## 7. Eleven components, and what the second engine had to guess

`address.py` grew from three components to eleven: `embeddings`,
`block_input`, `attention_query`, `attention_key`, `attention_z`,
`attention_output`, `mlp_input`, `mlp_output`, `block_output`, `ln_final`,
`lm_head`. Three facts came out of doing it.

### 7.1 The sort key needed a third band

The old key was "everything inside the layer stack, by depth and by stage,
before everything after it" — because every layerless tap it knew
(`lm_head`) was downstream of every layer. `embeddings` is upstream of all of
them, so `_Component` grew a `band` (0 before the stack, 1 inside, 2 after)
and the key became `(band, layer, stage)`. nnsight enforces the order, so
this is checkable rather than notional: `test_components.py` reads all eleven
in one trace in sorted order, and a wrong key raises rather than returning a
wrong number.

### 7.2 An interior is not one shape

`attention_query` and `attention_key` are arguments 1 and 2 of the same call.
`attention_z` is that call's **return**. So an interior address needs to say
*which handle* — `inputs` or `output` — and `_Component` grew a `handle`
field and the engine a two-line branch. A table of `(path, side, seq_axis)`
could not have expressed the third one; this is a concrete instance of what
FINDINGS §2 asks nnterp to carry as data.

### 7.3 Of six standardized names, transformers publishes three

The hooks engine has to translate nnterp's names against a raw tree. The
split is exactly half:

| standardized | published by transformers | how the hooks engine finds it |
| --- | --- | --- |
| `embed_tokens` | ✅ `get_input_embeddings()` | published |
| `lm_head` | ✅ `get_output_embeddings()` | published |
| the decoder | ✅ `base_model` / `base_model_prefix` | published |
| `layers` | ❌ | the decoder's only `nn.ModuleList` |
| `attentions` / `mlps` | ❌ | the block's child whose class name ends in `Attention` / `MLP` |
| `ln_final` | ❌ | the decoder's only normalization child outside the stack |

The three guesses are conventions, not contracts. `LlamaAttention`/`GPT2Attention`
and `LlamaMLP`/`GPT2MLP` happen to agree on a suffix; nothing requires that,
and a family that names its mixer `Mixer` breaks it. Each guess raises rather
than picking a first match, because the failure mode is reading a real tensor
from the wrong module, which nothing downstream could detect.

The resulting translation, which is what nnterp is worth for these two
families, is pinned in `tests/test_components.py::RAW_PATHS`:

| standardized | raw Llama | raw GPT-2 |
| --- | --- | --- |
| `embed_tokens` | `model.embed_tokens` | `transformer.wte` |
| `layers.0` | `model.layers.0` | `transformer.h.0` |
| `attentions.0` | `model.layers.0.self_attn` | `transformer.h.0.attn` |
| `mlps.0` | `model.layers.0.mlp` | `transformer.h.0.mlp` |
| `ln_final` | `model.norm` | `transformer.ln_f` |
| `lm_head` | `lm_head` | `lm_head` |

### 7.4 An input-side tap needs to know an argument's *name*

`block_input` and `mlp_input` are the first input-side components. In nnsight
that is `envoy.input` and the envoy knows which argument that is. With hooks
it is `register_forward_pre_hook`, which is handed `(args, kwargs)` — and a
decoder layer may be called positionally or by keyword depending on the
family's own loop. The engine takes `args[0]` when there is one and otherwise
the single tensor-valued keyword, refusing if there is not exactly one. That
"which argument is the activation" question is a third thing nnsight answers
for free.


## 8. Several writes at one address cost one ulp across engines

`documents/multi_position_patch_cpu.json` installs three absolute writes at
one address, at positions −4, −3 and −2. It is the only document in the suite
where the nnterp and hooks engines do not agree to the bit: `logit_diff`
differs by **1.49e-08 on one of four rows**, one ulp at fp32. Every
single-write document still asserts `torch.equal`.

Ruled out, each by its own probe:

- the operands the three writes consume — bit-identical;
- the activation the three writes compute — bit-identical;
- the tensor actually installed at the address, **including its strides and
  contiguity** — bit-identical;
- the weights — identical tensors;
- the attention implementation — `sdpa` on both sides;
- the KV cache — forcing `use_cache=False` on the hooks side changes nothing
  (this was the leading hypothesis, from the 1-ulp `DynamicCache` re-layout
  the real causalab hit);
- tracing itself — an un-intervened forward through both engines on the same
  batch is bit-identical.

So it enters *after* the replacement is installed, in how nnsight continues a
forward whose intermediate value has been assigned. That is nnsight's
internals rather than mini's, and the test records the number rather than
loosening quietly: `atol=1e-7` with the measurement in its docstring.

The reason this matters beyond one document: **cross-engine bit-parity is a
property of one write, not of writing.** A suite that only ever wrote once
would have reported exact agreement and been believed.

**Sharpened 2026-09-21, once windows existed.** `documents/v2/window_patch.json`
replaces the same three positions in *one* write, and diverges across
engines by exactly the same 1.49e-08 on the same one row; the single-position
document is exact. Measured side by side:

| document | writes | positions replaced | max diff |
|---|---|---|---|
| `v2/patching.json` | 1 | 1 | 0 |
| `v2/window_patch.json` | 1 | 3 | 1.49e-08 |
| `multi_position_patch_cpu.json` | 3 | 3 | 1.49e-08 |

So it looked as though it was about **how many positions are replaced at
an address**. The two multi-position documents agree with *each other* to
the bit on one engine.

**Corrected the same day, once ragged reads existed.** `documents/v2/
entity_mean_ablation.json` installs ONE write at ONE position — a `(16,)`
mean the two engines compute bit-identically — and the ablated logits differ
by 7.45e-09 on one row. Zero ablation at that position is exact; the
unit-window mean at that position is exact. The only thing that changed is
the values written. So the count inference was wrong too: the divergence is
**input-dependent**, enters after a replacement is installed, and is on the
order of one ulp when it appears. What survives every revision of this
section is the method: a parity claim is a property of the documents that
were run, and each new document is a new measurement, not a confirmation.


## 9. nnsight mounts `.save()` on `object`, and pydantic notices

Naming a field `save` on a pydantic model warns:

    UserWarning: Field name "save" in "Observe" shadows an attribute in
    parent "Node"

`pydantic.BaseModel` has no `save`. nnsight's C extension mounts one onto
**`object`** — that is how `x.save()` works on anything inside a trace — so
every class in a process that has imported nnsight inherits it, and a field
called `save` shadows it.

Harmless here (nothing calls `.save()` on a document) and the field is
`saves` now, which matches `Step.saves` anyway. Worth knowing because the
warning names pydantic and the cause is nnsight, and because any library
that defines a `save` attribute on a class will silently replace nnsight's
for instances of it.


## 10. The logit lens rounds differently from the head, by one ulp, and why

`view: "logits"` gathers a residual window `(rows, w, width)` and pushes it
through `ln_final` and `lm_head` inside the block. Compared with the model's
own logits at the same positions, measured on the tiny Llama:

| projected | max |diff| vs the model's logits |
|---|---|
| one position per row, then the head | 5.96e-08 |
| the whole sequence through the head, *then* sliced | 0 — bit-exact |

Same weights, same norm, same head; the only difference is the GEMM's `M`
(rows·1 against rows·seq), and a GEMM at a different `M` may accumulate in
a different order. The argmax agrees. Both engines project identically,
because they gather the same window and call the same two modules — so the
lens agrees across engines to the bit while disagreeing with the head by an
ulp.

causalab's `head.py` serves `lm_head` reads at named positions this way for
feasibility (an eval pass over 900 rows otherwise "materializes 5.8 GB of
logits to read 900 rows") and carries the measured warning that a *training*
read must keep the head un-elided — the backward GEMM at reduced `M` flipped
an early stop. Mini's `view` is for reading, and the compiler refuses a
featurized or written-back logits view, so the training case cannot arise
by accident.


## 11. Ragged positions: what a row's own text costs, and what the tokenizer decides

`{"column": "entity"}` locates a row's column text inside its prompt and
returns the tokens covering it. Three facts came out of making that work.

### 11.1 Character offsets by decoding prefixes, not `offset_mapping`

The mapping from a character span to a token window is built by decoding
growing prefixes of the content ids — `offsets[k] = len(decode(ids[:k]))` —
which works on any tokenizer, fast or slow, and needs no `offset_mapping`.
The substring is searched in the *decoded* text, not the original prompt,
because decoding may normalize a leading space; the search tries the text,
then `" " + text`, then the stripped text. Quadratic in the row's tokens,
which at prompt lengths is nothing.

### 11.2 The same word is one token or three, and that decides what can land

On the tiny Llama's tokenizer the weekdays split: ` Monday`, ` Friday`,
` Saturday`, ` Sunday` are one token; ` Tuesday`, ` Wednesday`, ` Thursday`
are three. So "swap the counterfactual's entity into the base's entity" —
the most natural ragged interchange there is — has rows where the source
window is three tokens and the target is one. There is no way to land that
without a policy, and the protocol has two (`exact_length_buckets`,
`padded_masked`); mini implements `refuse`, naming the rows, before any
forward. This is not an edge case: it is the *default* outcome of an entity
patch on real text.

### 11.3 A read may skip a row; a write may not

A row whose column text is not in its prompt gets an empty window. For a
read that is an excluded measurement — the row contributes no positions to
a harvest, `explain` prints `-` for it, and it stays a row. For a write it
is refused: writing nothing somewhere is not an intervention, and the row
would score as if one had happened. That asymmetry is causalab's, and it is
correct.

What did **not** need to change: `apply_write`. A ragged read gathers flat,
`(total, width)`; featurizers are pointwise; a mean over it is one vector
that broadcasts into any window. The seam held.


## 12. The continuation frame: what a decode loop looks like from a tap

`decode: N` on an intervention turns every forward into a prefill plus N
greedy steps. Four facts shaped the design, all measured on the tiny Llama
under nnsight 0.8's `generate` trace.

### 12.1 Step 0 is the whole prompt; every later step is one position

At step 0 a layer's output is `(rows, prompt, width)`; at step k ≥ 1 it is
`(rows, 1, width)` — the cache carries the rest. So a prompt-frame tap
applies at step 0 with the plan's positions as compiled, and a step-k tap
means the one position that step processes. The engine takes the last index
of whatever the sequence axis has, which is the same thing on both sides.

### 12.2 The head keeps one row under generation, even at the prefill

`lm_head.output` is `(rows, 1, vocab)` at *every* step of a generate trace,
including step 0 — transformers' `logits_to_keep` computes logits for the
last position only. In a plain forward it is `(rows, prompt, vocab)`. A
prompt-frame read at the head with position 10 therefore cannot index it
under generation; the engine treats a one-position tensor as one position.
This is the runtime's layout, not the experiment's, which is why the rule
lives in `ops.at_step` beside the frame rule and not in the compiler.

### 12.3 A prefill write reaches the continuation only through the cache

A write at layer 0 of the prompt's last token changed the generated ids on
both rows, in a run where no tap touched a decode step. The protocol's
"prefill-only" semantics are exactly this, and steering is the other case:
`{"step": "all"}`, a write at every step.

### 12.4 EOS must be held off for a bounded loop to be a bound

`max_new_tokens` is an upper bound — an EOS ends the run early, and then a
`tracer.iter[:N]` loop outruns it, warns, keeps what it saved and drops the
statements after it. `min_new_tokens=N` is what makes N steps N steps; the
engine passes both. A hooks engine counts steps with a forward hook on the
root module, which fires once per pass.

### 12.5 nnsight cannot see a `with` it cannot read

Run from stdin or `python -c`, `with model.generate(...) as tracer:` returned
a tensor and the `with` failed — nnsight decides "traced or direct" by
reading the calling frame's source, and stdin has none. From a file it
traces. Worth knowing before concluding that generation is broken.


## 13. Eighteen components, and the family axis that still was not needed

Seven more: `input_ids`, `attention_input_norm`, `attention_premix`,
`block_mid`, `mlp_input_norm`, `mlp_activation`, `mlp_neuron_output`.

### 13.1 A child's name is a question the checkpoint can answer

The first plan was `.source`: each of these is one call inside a forward, and
a call-site needle per family would find it. The probe showed something
simpler — **every one is a module boundary of a named child**, and the only
per-family fact is the child's name: `input_layernorm` / `ln_1`,
`post_attention_layernorm` / `ln_2`, `o_proj` / `c_proj`, `act_fn` / `act`,
`down_proj` / `c_proj`. nnterp standardizes the block, the mixer and the MLP;
it does not name their children. So a path may offer alternatives —
`layers.{layer}.input_layernorm|layers.{layer}.ln_1` — and `Address.resolve`
takes the one that exists, refusing if none or several do. Still one row per
component, still no family column, and because nothing here needs `.source`,
**the hooks engine reaches all of them** where it would have refused every
interior.

Had these gone through `.source`, the match would also have needed a rule
mini's `find_op` lacks: `hidden_states = self.input_layernorm(hidden_states)`
is *two* operations on one line, the call and the assignment, and a needle on
the line hits both. (causalab's `match_op` prefers the hit whose own name is
the called symbol. Mini did not need to learn that today.)

### 13.2 The same place is not always the same tensor

`mlp_neuron_output` is the down-projection's input on both families. On
Llama that is `act(gate)·up`; on GPT-2, which has no gate, it is the
activation itself — the same tensor `mlp_activation` names. The address is
right on both; what it *means* differs, and the table's comment says so
rather than the code hiding it.

### 13.3 Read-only is a property of a component

`input_ids` is integers, `(batch, seq)`, no width axis; `gather` returns
`(rows, w)` for it without being told. A write there is refused where the
document is read. And at layer 0 a third component joins `embeddings` and
`block_input` in being a no-op under interchange on these prompts:
`attention_input_norm`, the norm of a last token both prompts share.


## 14. A gate is a featurizer, and its mode is one attribute

`documents/v2/dbm.json` is `das.json` with three differences: the
featurizer's `kind`, one objective term, and an `anneal`. Nothing in the
write seam, the engines or the plan walk learned the word "mask".

### 14.1 What it took

| piece | where | size |
|---|---|---|
| `Gate`: `featurize(x) = x`, `inverse(f, _, x) = m⊙f + (1−m)⊙x` | `ops/featurizer.py` | one class |
| soft `σ(θ/T)` under update, hard `θ > 0` when scored | `Gate.mask`, reading `self.training` | four lines |
| who sets `training` | the fit loop, around its updates and its eval pass | one helper |
| temperature schedule | `fit.anneal: {gate: {start, end}}`, geometric over the updates | three lines in the loop |
| the L1 term | `<gate>.mask` joins the scored dict; `objective` already takes `.mean()` | two lines |

The survey priced this at 3, for two reasons that both dissolved. *"It has a
train/eval mode"* — but the fit loop already is the only place that knows
which of the two is happening, so the mode is an attribute it sets, not a
protocol every featurizer implements (a rotation is handed the flag and
ignores it). *"The objective must reach inside a featurizer"* — but the
objective is `Σ wᵢ · mean(termᵢ)` over a dict of tensors, and a mask is a
tensor; putting it in the dict under `<name>.mask` is the whole reach. The
same key in the eval pass is the *hard* mask, so `train/eval` grows a column
— the fraction of units kept — for free.

### 14.2 What is pinned

- All units on is `patching.json`'s interchange **bit for bit**; all off is
  the un-intervened model bit for bit. (`1·f + 0·x` is exact in IEEE.)
- θ starts at 0, and hard is `θ > 0` — so an unfitted gate writes *nothing*,
  where an unfitted rotation writes a random subspace. A seed on a gate is
  refused.
- A fitted gate written to a bundle and loaded by a second document scores
  bit-equal to the fit run's own score step. The stamp gained `kind`, so a
  rotation's bundle offered to a gate is refused by key, before its shape.
- `remote="local"` learns the same θ bit for bit; the two engines agree on θ
  to 1e-6 and on the hard mask exactly.

### 14.3 What was measured, and what it is not

L1 weight 1e-4, 2e-3, 1e-2, 5e-2 keeps **4, 2, 1, 0 of 16** units at layer 0
of the tiny Llama (20 updates, lr 0.05, T 1 → 0.05). The model is random:
cross-entropy moves in the fourth decimal (10.3940 vs 10.3946 held out), so
*which* units survive means nothing. The monotone curve reaching empty is
evidence the mechanism works, not a result about a model.

### 14.4 Not brought over

causalab's `Gate` is ~700 lines: `hard_concrete` and `clamp` and `budget`
parametrizations, `group: head` and the per-expert table, the dead-unit
rules (`freeze`, `leak`), a `fill` start value, `top_k` hard masks, and the
forward-sharing cache keyed on the mode. None is here. The one most likely
to be wanted first is a gate over **heads** at `attention_z` — which needs
that component to publish its `(heads, head_dim)` shape, which it does not
yet (`width` is refused for interiors).


## 15. Eligibility is a row list, decided where the data is

`"eligible": True` was a literal in `plan/write.py`, and `rows.column`
refused a null. Now a row whose answer column is null or empty is an
**excluded measurement** for every metric naming that column, and only for
those.

The design question was where the mask lives, and the answer is that there
is none. Which rows have an answer is a fact about the dataset, so the
compiler resolves it: a `MetricOp` carries `rows`, the indices it scores
(`None` when that is all of them, so every earlier plan is unchanged), and
`ids` only for those. At run time `observe` indexes the read by that list
and the metric functions are untouched. Consequences that fall out rather
than being written:

- a fit's loss, its held-out score and its early stop are means over
  measured rows, because they are means over what the metric returned;
- a genuine NaN is still loud — nothing was taught to skip NaNs, which is
  what a `nanmean` design would have cost;
- the other rows score **bit-equal** to the un-holed run, on both engines.

The table keeps the row: `{"value": null, "eligible": false}`, so an
excluded measurement cannot be read as a zero or quietly shorten a
denominator. The result tensor holds one value per eligible row, and the
`SaveFile` carries the booleans that re-align it.

Two refusals, both before any forward: a metric with **no** eligible row in
a pass (including one minibatch of a fit — its loss would be the mean of
nothing), and a column **no** row has, which is a misspelling and not an
exclusion. `causalab-mini data <ref>` now reports which columns have holes.

Not done: a *position's* ineligibility (a `{"column": …}` window the row
does not contain) reaching a metric. A metric reads a unit window, so that
case cannot be authored yet; when a metric over a located position exists,
its `rows` is this same field.


## 16. Where the activation is inside the boundary's value

Until now both engines carried the same two lines — a tuple's first element,
or the tensor itself — and nothing could say otherwise. That rule is a
convention: a block returning `(router_logits, hidden)` would have handed
over the wrong tensor without complaint.

Now it is the *default* of a field, `select`, on the component row, and the
two lines live once, in `Address.get` / `Address.put`:

| `select` | meaning |
|---|---|
| `None` | decided from the value, as before. Stays the default because tensor-vs-tuple is a property of the transformers version, and pinning it would be wrong on half the installs. |
| a path, `(1,)`, `("hidden_states", 0)` | walked in to read; containers rebuilt on the way out to write (tuples, named tuples, lists, mappings). Pure data. |
| `Lens(get, put)` | two module-level functions, for what a path cannot say — e.g. an activation packed `(tokens, d)`. |

**It has to be a pair.** "A function that indexes the tuple" is a read; a
write must hand the module back its whole value with one part replaced. A
path gives both directions for free, which is why it is the first choice
and the function form the escape hatch.

**The family axis finally arrived, as a list of exceptions.** `a|b` paths
cover a child that is *named* differently, because existence distinguishes
them. The same name handing over a different *structure* cannot be probed
that way (and guessing the hidden state by shape is the silent failure this
removes), so `_OVERRIDES[(config.model_type, component)]` holds the fields
of a row that differ on that family. The engine stamps `family` on the
address when it locates it — a string, so the plan is still data, and the
functions never leave the table. `Address.where` is the place without the
family, which is what "one address serves both families" now compares.

`_OVERRIDES` is **empty**: no model in this repository needs an exception.
What is pinned instead is the wiring, on the real tiny Llama through both
engines — a lens that reverses the width axis in and out is bit-invisible
to a full-width swap and is observed being called for the read and the
write; an explicit `(0,)` on the raw attention module's real `(output,
weights)` tuple equals the default bit for bit; `(1,)` reaches sdpa's `None`
weights and is refused by name. A select that does not reach a tensor is an
`AddressError`, not a swap.

Not covered: interiors keep their own `handle`/`arg`, and the hooks
engine's *input* side still picks the first positional or the single tensor
keyword before `select` could apply.


## 17. Heads are a slice of a place; the pattern is an operation inside the operation

### 17.1 `heads` belongs to the site, not the address

A site may say `"heads": [1, 3]` at any per-head tensor — `attention_query`,
`attention_key`, `attention_scores`, `attention_probs`, `attention_z`,
`attention_premix`. It compiles onto the **read and write ops**, beside their
positions, and not into the `Address`: heads are *where in the tensor*, as
positions are, and two sites naming disjoint heads of one component are one
address, hence one tap, hence ordered correctly for free. (Pinned: heads
{0,1} and {2,3} as two sites equal the unsliced patch bit for bit.)

A per-head tensor is handed on **flat**, `(rows, w, n·per_head)`, however the
model holds it — head-major in one axis (`o_proj`'s input) or as
`(heads, head_dim)` (the query, `z`). `gather` splits, picks and flattens;
`scatter` rebuilds the model's own shape with the named heads replaced.
Nothing downstream knows heads exist, which is why **DAS inside a pair of
heads needed no code**: the site is 8 wide and a `k=4` rotation fits it.
This changed what an *unsliced* interior read returns, from `(rows, w, heads,
head_dim)` to flat; no test or document depended on the old shape.

`attention_z` and `attention_premix` are the same tensor held two ways, and
the per-head patch agrees **bit for bit** between them — which is also what
lets the hooks engine do head patching at all, since `premix` is a module
boundary.

The engine contract grew to **eight** members: `heads(address)`. Like
`width`, it is a question about the checkpoint only its holder can answer,
and it is per address because key-head space is narrower under GQA. The
per-head width is the config's own `head_dim` where it has one —
`hidden/heads` is not always true.

### 17.2 The pattern

`attention_scores` and `attention_probs` are the argument and the return of
the softmax inside `eager_attention_forward`, the function the attention
module's `attention_interface(...)` call dispatches to. Three facts, each
found by probing:

- **nnsight opens a nested `.source` only inside a trace**, because the
  callee is a run-time fact. So unlike every other interior, this one cannot
  be resolved on the client against a meta shell. The table therefore holds
  the inner operation's *name* (`nn_functional_softmax_0`, identical on
  Llama and GPT-2), and the engine resolves it where the run runs, refusing
  with the callee's operation list if it is absent.
- **A needle would not have worked**: `softmax(` matches three operations on
  its one line (the call, the `.to`, the assignment) — the rule §13.1 said
  mini would eventually have to learn. A name sidesteps it.
- **It exists only under eager attention.** What *can* be checked on the
  client is the config's `_attn_implementation`, so `locate` refuses at
  compile time and says what to write: `"attn_implementation": "eager"` in
  the document's model block — the document's, because it decides which
  tensors exist and changes the last bits.

Pinned on the tiny Llama: all twenty components read in one forward in
sorted order (so the two new stages are in the right place); `softmax(scores)
== probs` bit for bit; each head's pattern sums to 1; zeroing head 2's
pattern at a position equals zeroing head 2's `z` there (1e-6) — the write
lands where it says; `remote="local"` reads the same pattern bit for bit,
so the nested source opens inside a serialized session too.

Not done: swapping a *counterfactual's* pattern in. The key axis is the
padded batch's length, so base and counterfactual patterns only line up when
their batches pad to the same width; nothing checks that on the client yet,
and torch is what would complain. The hooks engine refuses both components,
as it does every interior.


### 17.3 Corrected the same day: `heads` was the first case of `Selection`

§17.1 shipped as a `heads` field on `ReadOp` and `WriteOp`, threaded as an
extra argument through `gather`, `scatter`, `apply_write` and both engines.
That was too specific, and the threading was the symptom: positions say
*where along the sequence*, heads say *where along the features*, and the
second is one rule — view the features as `(groups, -1)`, keep some groups —
whatever the groups are called.

So an op now carries one `at: Selection(positions, groups, take)`
(`shapes.py`, integers all the way down), `gather`/`scatter` take it whole,
and the engines pass `op.at` and know nothing about heads. A bare `Positions`
is still accepted by the tensor functions as the selection of every feature,
so the fifty-odd direct callers did not change. **No subclass was added**:
an op that knew how to resolve heads would have behaviour, and ops are data
the engine executes; and heads are orthogonal to ragged/rectangular, raw/
logits and prompt/step, so a `HeadRead` would fork the type rather than
compose with it.

What generality bought, for one validator and four compiler lines: a site
may name **`units`** — single features, anywhere the width is known — so
neuron patching (`documents/v2/neuron_patching.json`) and a **gate over
chosen neurons** (DBM at a site of units: `d` = 4, nothing else changed) both
work. `heads` and `units` are exclusive on a site: they slice the same axis.

Found on the way: `mlp_activation`'s width. Llama says `intermediate_size`;
GPT-2 says `n_inner`, None meaning 4·hidden — and the tiny GPT-2 *also*
carries a stray `intermediate_size: 37` its 128-wide MLPs never read. The
test compares against the module the activation feeds, not the config,
which is the only reason that was caught. Width now lives once, in
`address.width(config, address)`, and both engines call it.


## 18. Batching is a window of rows, and it lives in the walk

`execute(plan, batch_size=N)` bounds how many rows one model call holds. It
is absent from the plan — the same experiment however it is run — and
recorded in `run.json`, because it moves the last bit of every number.

**Where it lives is the design.** Not in the compiler (chunked `Observe`
steps would need their results, their `mean`/`pca` reductions and their
per-row references re-joined by something) and not in each engine (twice the
code, and the engines would have to agree). It is one function in the shared
walk, `steps.passes`: a window of rows is a whole small pass — every forward
of the pass, in order, over a slice of the rows — handed to an **unchanged**
`engine.forward`. What the windows read is concatenated in row order, and
everything after it — metrics, eligibility, outputs, saves, a fit's loss —
sees one pass. Neither engine changed; the hooks engine got batching for
free; with no `batch_size` there is one window, which is the pass exactly as
compiled (bit-equal, pinned).

Two slices make it work, both pure data:

- `plan.window(forward, a, b)` — every per-row thing a forward holds is a
  tuple with one entry per row (ids, mask, each op's positions), so a window
  is a slice of each. Positions are absolute indices into the padded width
  the client fixed once for all rows, so they survive unchanged.
- `intervene.rows(tensor, positions, a, b)` — the rows of a *published*
  value. A window's write must meet the operand of **its** rows, so `State`
  now remembers the positions a per-row output was read over (`layout`). A
  reduced value (a mean, a basis) has no layout and every window shares it
  whole. This is the case a careless windowing gets wrong silently — right
  shape, wrong examples — and it has its own test.

**The bug the tests caught:** raggedness was re-derived from the positions
in hand, and a *window* of a ragged pass's rows can happen to be
rectangular — so one window came back `(rows, w, d)` and the next `(total,
d)`, and the concatenation failed. Whether a selection gathers flat is now
decided once by the compiler, over every row, and carried on the
`Selection` (`flat`).

Pinned across patching, windows, literal operands, the logit lens,
generation, heads, neurons, a published mean, a ragged harvest, a per-row
reference, a DAS fit and `remote="local"`, at batch sizes 1 and 3 (3 does
not divide the 4 rows): agreement to 1e-6, not bit-equality — a GEMM over
fewer rows rounds differently (§8).

**What it does not do:** free memory inside a fit's update. The loss is the
mean over the concatenated rows, so the graph of every window is alive until
the backward; a fit's memory knob is `pairs`. It does bound the held-out
pass, which runs under `no_grad`. Gradient accumulation across windows would
be the next step and is not built.


## 19. The first real runs: what a toy model and `remote="local"` could not show

2026-09-21, hakone: Llama-3.2-1B on an A100, and through a self-hosted NDIF.
The results are in `documents/real/README.md` and are the kind the library
exists to produce (a layer-12 crossover, one mover head, a 16-dimensional
subspace). This section is the **seven defects** the runs found. Every one
was invisible to 409 passing tests, because every test ran a 16-wide random
model on a CPU in one process.

### Found by a real width

1. **The Cayley start did not train.** `start_weight` drew unit-variance
   entries, so the skew matrix's singular values grow like √d; the Cayley
   transform saturates (→ −I, derivative ~1/σ²) and at d = 2048 the basis
   stops responding to its parameter. Measured: loss 3.23 → 3.29 over 60
   updates, held-out IIA 0.00 at every k. Scaled by 1/√d: 3.22 → 0.11, IIA
   1.00. At d = 16 the two are indistinguishable, which is the only reason
   DAS "worked" before.

### Found by a GPU

2. A featurizer's parameter was built on the CPU and met a CUDA activation.
   Fixed generally rather than by threading a device: a featurizer computes
   **where its activation is** (`_on`), the parameter staying one leaf for
   the optimizer — which also covers `device_map="auto"` over several GPUs
   and a server whose placement the client never learns.
3. The hooks engine left its input ids on the CPU (nnsight moves them for
   its own engine; nothing did here).
4. A gate's mask (CPU) was stacked with metrics (CUDA) in a fit's record.

### Found by a real server

5. **Python minor versions must match.** The plan's classes ship by value,
   and a 3.13 dataclass carries `__replace__ = dataclasses._replace`, which
   3.12 does not have: the payload could not be read. A property of shipping
   dataclasses by value, not fixable here; the client env was rebuilt on
   3.12. (nnsight versions must match too — the module layout moved between
   0.8.0rc1 and the dev checkout.)
6. **A filled-in plan cannot come home.** The server can *run* by-value
   classes but cannot pickle an instance of one back. `execute` now brings
   home `{step path: {name: tensor}}` — no class of ours in it, asserted on
   the pickle bytes — and fills the plan the client never gave up. It is
   also the better design: `execute` returns the plan it was passed.
7. **NDIF runs a request under autocast**, which downcast half of the
   Cayley solve (`linalg.solve: A Float, B BFloat16`). Featurizer math now
   runs outside autocast (`intervene.exact`): a featurizer declares fp32 and
   this is where that is kept. It also sidesteps the autocast weight cache
   that once froze remote training outright.

And one thing that is not a defect but was unrecorded: **the server serves
the dtype it chose** (bf16) whatever the document says. `run.json` now has
`served_dtype`.

### What held

Every design claim that could have broken did not: one document is one
NDIF job (16 experiments, or 96, or four fits with their optimizer loops);
the walk, the fit, the windowing, generation-free and nested-`.source`-free
paths all ran server-side unmodified; the hooks and nnterp engines agree
**exactly** on the real model on the GPU (672 numbers); and the same suite
passes on both machines once nine cross-engine assertions stopped claiming
bit-equality — on hakone's CPU the engines differ by one ulp (1.49e-08)
where on bippu they do not, so that was a fact about a laptop.

### A format gap the documents exposed

The layer sweep wants `pos` swept on a read **and** its write *together*.
Two `{"sweep": …}` wrappers are a cross product — 64 points, half of them
reading one position and writing another — so it is two documents instead.
A linked sweep (one axis, several fields) was the missing spelling — added
the same day as `{"sweep": […], "as": "pos"}`: wrappers sharing a name are
one coordinate. `{"sweep": {"range": [0, 16]}}` landed with it, and the two
layer-sweep documents are one again.

Not yet run for real: generation, the attention pattern (the deployment is
sdpa), ragged reads, and a gate — all pass on the GPU with the tiny model,
none has met a real one.


## 20. `clamp` and `renormalize`, and the one mechanism that needs the past

`clamp` is `f.clamp(lo, hi)`, no operand, either bound optional; `lo = hi =
0` equals the literal-zero swap bit for bit, which is its test.

`renormalize` is `f · ‖f₀‖/‖f‖` where `f₀` is the feature value **before any
write of this forward touched this address**. It is the first mechanism
whose input is not `(f, operand, params)`: its operand is not authored, the
seam supplies it (`intervene.PRE_WRITE`), and each engine keeps the
address's pre-write tensor for the length of one tap to make that possible
— one line in each. The ordering rule causalab needed an executor phase for
is a validator here: among a model's writes at one site, a renormalize comes
after at least one other and last, because first or alone it is the
identity, and a document that does nothing should not validate.
`documents/v2/steer_renormalize.json` measures its own claim with reads at
the site: the steered activation is longer, the renormalized one is exactly
as long as the original, and the two models answer differently.


## 21. `sae` and `linear`: the featurizer the error term was waiting for

The write seam has been `inverse(do(featurize(x)), err, x)` since the first
rotation, and `err` has been `None` every time. An encoder/decoder pair is
what it was for:

    featurize(x) = (f, x − decode(f))        inverse(f′, err, x) = decode(f′) + err

An intervention changes what the dictionary explains and hands back the
rest exactly. Without it every SAE experiment is also "replace the
activation by its reconstruction", and the two effects cannot be told
apart; with it, a write that changes no latent is the un-intervened model
however bad the SAE is (pinned with a *random* dictionary, whose
reconstruction is terrible). One class, `Encoder`, in SAELens's names and
orientations; `linear` is the same object without the ReLU, and a missing
`W_dec` means tied weights. Loaded, never trained, never published by a
`weights` step; `k` comes from the bundle and may exceed `d`.

**`features` on a write** is what makes it usable: the mechanism acts on
named coordinates of the *featurizer's* space and the others pass through.
It is not an SAE feature — two of a rotation's eight directions is the same
field. (A site's `heads`/`units` slice the activation; a write's `features`
slice the feature space. Two axes, two fields.) Pinned on the model:
ablating a latent moves **exactly** the rows where that latent was on.

**Loaded tensors now travel as bytes.** They were nested tuples of Python
floats, which was honest for a 16x8 rotation and is not a format for a real
SAE (tens of millions of numbers). A `FeaturizerOp` now carries the
safetensors blob itself — still plain data: it pickles, compares, ships by
value to a server, and is exactly the bytes that were checked on the client
— at four bytes a number. Every kind's constructor takes its tensors by
name, so `build` is `KINDS[kind](**tensors)` for all five.

Not done: top-k and JumpReLU activations, a published SAE run for real
(hakone has `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` cached — the obvious
next real run), and featurizer chains (standardize then rotate).


## 22. Swapping a counterfactual's attention pattern, and the check that makes it safe

§17.2 left this undone for one reason: the last axis of `attention_probs` is
the **keys of the padded batch**. Key `j` of the operand lands on key `j`
of the target, and that is the same *token* only if the two prompts are laid
out identically. If they are not, attention mass meant for a word lands on a
pad or on a different word — with every shape correct and no error from
anything.

Both masks are in the plan before any forward, so the compiler checks them
(`_check_keys`): a write at a key-axis component whose operand was read in
another forward requires the two forwards' attention masks to be equal, and
says which rows differ and by how many tokens. The table marks the two
components (`keys=True`); nothing else changed — the swap itself was already
expressible. A pattern published by an earlier step is refused, because its
layout is not in hand to check.

Pinned on the tiny Llama (`documents/data/weekdays_aligned`, the pairs that
tokenize to equal length): the patched head's pattern, read back in the
patched model, **equals** the counterfactual's bit for bit and is still a
distribution; a bystander head's is untouched; a pattern taken from the same
prompt changes nothing; the misaligned `weekdays/train` is refused by name.

On Llama-3.2-1B it answered a real question in 15 s. Head patching had found
one mover head, L12 H28. Swapping only its *pattern* at the last token does
nothing (logit-diff −2.396 against a −2.403 median over 32 heads); swapping
its whole result had given +0.05. The head looks at the same place whatever
the day is — what differs between prompts is what it reads there.


## 23. The block is nnterp's to address

The exploratory-testing pass (six agents, 24 families) found that mini's own
component table was wrong or missing on most families outside Llama and
GPT-2: Gemma-2/3 and OLMo-2 put a norm on a sublayer's output before adding
it, so `block_mid` read the attention module's raw output (bit-identical to
`attention_output`, a 17x smaller tensor than the stream); GPT-NeoX's second
norm exists and normalizes the block *input*; BLOOM adds the residual inside
its sublayers; Phi, NeoX and OPT spell their children differently; the
o_proj input is `heads * head_dim` wide, not `hidden_size`, on Qwen3 and
Gemma. Each was one more row, one more `a|b` alternative, one more override
in mini — a second family table beside nnterp's.

The decision was to have one. nnterp (PR ndif-team/nnterp#61) now addresses
the block itself: `layers_mid`, `attentions_norm_output`, `mlps_norm_output`,
`attentions_premix`, `mlps_activation`, `mlps_neurons` beside the accessors
it had, each defined by a residual identity and asserted on 26 families;
`attentions_output` / `mlps_output` are the contributions on every family;
`block_structure` says which places a block has; `head_dim`, `qk_head_dim`,
`num_kv_heads`, `intermediate_size` are published.

Mini's table shrank accordingly. A boundary row is now the *name of the
nnterp accessor* and its stage in the forward; the path alternatives, the
`select`/`Lens` pair, the `_OVERRIDES` table, and the width/head-count
config logic are gone (417 lines from 534, and every family-specific line of
those). `locate` asks the checkpoint through nnterp — `internals[name]` is
disabled with a reason where the family lacks the place, and `get_module`
refuses a layer that lacks it — and writes the resolved child (`self_attn.
o_proj`, `post_attention_layernorm`) and side into the `Address`, so the plan
still says where in plain strings. The nnterp engine reads and writes
boundaries through the accessor; widths and heads come off the model's
attributes. The hooks engine, which has no nnterp handle, loads a weightless
nnterp shell of the same checkpoint for exactly those facts and walks the
resolved path against its own standardized tree (`self_attn` / `mlp` are
nnterp renames, mapped to its per-layer lists). Ignored otherwise, by
decision.

Kept in mini: the five interiors (`attention_query/key/scores/probs/z`),
which nnterp does not address yet. The four whole-model boundaries
(`embeddings`, `input_ids`, `ln_final`, `lm_head`) were briefly mini's own
paths with an inline "first of a tuple" rule — the one convention of nnterp's
`select` default, copied — until nnterp's registry stopped being about
layers: `Address(per_layer=False)` rows, read at no layer, so *every*
boundary goes through an accessor and a family's `select`/`Lens` reaches all
of them. `locate` also asks `unavailable_on(layer)` now, so a place one layer
lacks (DeepSeek's mixture-of-experts blocks) is refused at compile time with
nnterp's reason rather than inside the trace.

Checked through mini after the port: Gemma-2 `block_input + attention_output
== block_mid` at 0.0; NeoX and Phi refuse `block_mid` / `mlp_input_norm` at
compile time, citing the parallel block; Qwen3's premix width equals the
tensor's; BLOOM's boundaries are right and only its interiors refuse. Mini's
own suite: 435 (the `select` tests went with the mechanism, to nnterp).


## 24. By reference: what registration was actually buying, and what it cost

Measured on bippu against a self-hosted NDIF built from `~/wd/ndif` with
`nnsight` (524c33fc), `nnterp` and `causalab_mini` installed into the image, so
client and server ran the same three checkouts. The question was whether
`nnsight.register("causalab_mini")` could go. It can, and two of the things it
was believed to be doing turn out not to be true.

**nnterp's own registration never fired under mini.** `StandardizedTransformer`
calls `nnsight.ndif.register("nnterp")` in the `remote=True` branch of its
constructor. `--engine ndif` never takes that branch: it builds the shell with
`dispatch=False` and passes `remote=True` to `model.session(...)` instead, which
is a different argument in a different place. Read straight out of cloudpickle's
registry:

```
before:                                                []
after StandardizedTransformer(dispatch=False):         []          <- what --engine ndif does
after StandardizedTransformer(remote=True):            ['nnterp']
```

So every remote run mini has ever done already resolved nnterp by import on the
server. Only `causalab_mini` was ever shipped.

**Shipping nnterp by value does not work anyway.** Register it and let a traced
block name the module, and the payload cannot be built at all:

```
TypeError: cannot pickle '_thread.RLock' object
```

— nnterp's module-level `logger`. A by-value module is rebuilt from its
contents, and a logger's lock is in them. This is not reachable from mini (the
block names no nnterp module), but it means the registration nnterp performs on
its own `remote=True` path is one referenced global away from failing, and that
"ship it by value" was never a usable fallback for a server without nnterp.

**What deletion is worth, measured on `weekdays_layer_sweep`.** The serialized
request payload, before compression:

| | bytes |
|---|---|
| by reference | **12 392** |
| `register("causalab_mini")` | 59 993 |

4.8x, on a document whose block calls into `steps`, `intervene` and `ops`. Both
paths returned the same numbers, so this was pure weight.

**The server is now provably the source of the code.** A block that reads its
own globals' `__file__` reports, from inside the model actor (pid matching the
actor that ran the document):

```
causalab_mini.__file__  /usr/local/lib/python3.12/site-packages/causalab_mini/__init__.py
nnterp.__file__         /usr/local/lib/python3.12/site-packages/nnterp/__init__.py
```

against the client's `/home/.../causalab-mini/causalab_mini/__init__.py`. With
registration on, the server reports the *client's* path — the module was rebuilt
from the shipped source. That is the discriminator, and it is the only one: a
by-value run and a by-reference run are otherwise indistinguishable from the
client.

**Two entries of §19 are consequences of shipping, not of remote execution.**
§19.5 (client and server Python minors must match, because a 3.13 dataclass
carries a `__replace__` 3.12 does not have) is a property of pickling *our*
classes by value; by reference the classes are the server's and the minors need
not match — 3.12.13 client against a 3.12.14 server ran clean, and the traced
block itself has always shipped as source rather than bytecode
(`nnsight.schema.request.RequestModel.serialize`). §19.6 (a filled-in plan
cannot come home, because the server cannot pickle back a class it only has by
value) also stops being true: the server has the classes. `execute` still brings
home `{step path: {name: tensor}}` and should keep doing so — the client already
holds the plan and nothing of ours needs the return trip — but the reason is now
design, not a limit.

**The price.** A stock ndif.us can no longer run a mini document at all. That
was the one thing registration bought, and it is the trade the owner took: one
codebase across both sides, loudly, instead of two that look alike.

**Verified end to end.** `documents/real/weekdays_layer_sweep.json` — 32
experiments, 42 rows, a read, a swap write and two metrics — run against a
self-hosted NDIF by reference and against the same checkpoint locally on one
A6000. At the deployment's default dtype the server serves bf16 and the numbers
differ as §19 already recorded (mean 0.051 on `logit_diff`, three of 1344 `iia`
rows flipping at the layer-12/13 crossover). Deploy it `--dtype float32` — dtype
is not part of the model key, so the deployment decides and the client changes
nothing — and all **2688 values are bit-identical** to the local run. Remote
execution, the serialization round-trip and the by-reference switch perturb the
arithmetic not at all; the whole of the difference §19 saw was served dtype.
