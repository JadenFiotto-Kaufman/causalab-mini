# FINDINGS

A running list of **every fact about model internals this project had to encode
itself** — every module path, side, tuple index, operation name and shape
assumption that nnterp did not hand us as data — plus what the document format
made us implement twice.

All of it is in one file, `causalab_mini/address.py`, with the
padding-dependent part in `causalab_mini/encoding.py`. That concentration is the
result this slice was built to produce: if the facts below moved into nnterp,
`address.py` would be a lookup and nothing else in the project would change.

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
carry a private total order over the component vocabulary. With 2 components that
is a dict of 2 entries; causalab's vocabulary has 56, and every one of them needs
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

### 1.9 dtype names

`{"fp32": torch.float32, "bf16": torch.bfloat16}` in `model.py`. Small, but it is
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

What nnterp *does* give, and it is the reason (1) needs so little: the
standardized accessors (`layers`, `attentions`, `lm_head`) already absorb the
family axis, so an address written in accessor spellings needs no family column
at all (§1.11). The gap is that they are live accessors and not a table.

(1)–(3) are `address.py` in its entirety. (4) is half of `encoding.py`.

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

## 4. Things this slice does not know, and should

- **Whether a site's component even exists on the loaded model.** We check the
  layer index against `num_layers`, and — for an interior only — that its
  operation is in the forward. `block_output` on a model with no `layers` envoy
  would still fail inside the trace.
- **Whether the head tap is the last position only.** See §1.6.
- **Whether a read and a write at the same address in the same model are
  ordered correctly beyond "writes first".** The rule is implemented (a tap's
  writes run before its reads); no document in the corpus exercises it.
