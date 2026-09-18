# FINDINGS

A running list of **every fact about model internals this project had to encode
itself** — every module path, side, tuple index, operation name and shape
assumption that nnterp did not hand us as data — plus what the document format
made us implement twice.

All of it is in one file, `causalab_mini/address.py` (67 lines), with the
padding-dependent part in `causalab_mini/encoding.py`. That concentration is the
result this slice was built to produce: if the facts below moved into nnterp,
`address.py` would be a lookup and nothing else in the project would change.

Measured against nnterp at `334e4ef` and transformers 5.17.0, on
`hf-internal-testing/tiny-random-LlamaForCausalLM` @ `9fb19125`.

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
  layer index against `num_layers`, and nothing else. `block_output` on a model
  with no `layers` envoy would fail inside the trace.
- **Whether the head tap is the last position only.** See §1.6.
- **Whether a read and a write at the same address in the same model are
  ordered correctly beyond "writes first".** The rule is implemented (a tap's
  writes run before its reads); no document in the corpus exercises it.
