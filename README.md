# causalab-mini

A deliberately small, readable reimplementation of causalab's intervention
engine. One rule underneath everything: **a plan is pure data; an engine turns
it into tensors.**

A document is its steps, in the order they run: a `forward` or a `generate`
is one model call over one dataset, with the interventions it lists in force;
a `metric` scores one of its reads; a `reduce` takes a read's mean or its
principal basis; a `fit` trains through a body of such steps. A step's name
is how everything after it reaches what it produced — `patched.logits`,
`iia`, `fit.rot` — and `saves` names what goes to disk. The compiler resolves
everything that needs a model but not a row — the widths, an interior's
operation, the prompts' token ids — on the client, without weights, and hands
the engine a plan of the same steps in strings and integers. The engine walks
it in one session; the results land on the steps that produced them.

**Where** along a sequence a read or a write acts is the one thing the plan
does not carry as an integer. It carries the spec — `-1`, `{"last": 3}`,
`{"index": -1, "scope": {"variable": "entity"}}`, `{"frame": "generated",
"index": -1}` — and the run resolves it per row against the model's own
tokenizer, so "the last token of this row's entity" is a different index on
every row and the same document on every model. What it resolved to, and why
a row had nowhere, comes home beside the numbers.

```bash
uv sync
uv run causalab-mini explain documents/v2/das.json        # the compiled plan, no weights
uv run causalab-mini run documents/v2/das.json --device-map cpu --out out
```

## Where to read

| file | what it is |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | the state of the project and the rules that hold it together — start here |
| [`REVIEW.md`](REVIEW.md) | the design audited against its three goals, and the order of work that followed |
| [`SURVEY.md`](SURVEY.md) | everything the real causalab has that this does not, rated by cost and reach |
| [`FINDINGS.md`](FINDINGS.md) | every fact about model internals and runtimes this project had to learn the hard way |
| [`NOTES.md`](NOTES.md) | a close reading of the protocol the copied documents are written in |
| [`examples/`](examples/) | a notebook that runs one document end to end and shows the plan filling in |
| [`documents/real/`](documents/real/) | Llama-3.2-1B on the weekday task, with measured results: behaviour, layer sweeps, head patching, DAS — run locally and on NDIF |
| [`documents/v2/`](documents/v2/) | the steps-first format: patching, DAS, mean ablation, zero ablation, a window, an entity harvest, a logit lens, generation, a PCA control, DBM, head-by-head and neuron patching, attention knockout, steering with renormalize, SAE feature ablation, attention-pattern patching |

The CLI's read-only verbs — `schema`, `vocab`, `model`, `tokens`, `data`,
`validate`, `explain` — need no GPU and are the loop an author lives in.
Every verb takes `--json`.

## Two engines

`--engine nnterp` runs on nnsight through nnterp's standardized names, locally
or — as `--engine ndif` — on NDIF with no local weights. `--engine hooks` is
plain torch and forward hooks: a measurement fixture that agrees with the
first to the bit on single writes and refuses what a hook cannot reach.

**`--engine ndif` runs by reference.** Nothing of `causalab_mini` or `nnterp`
travels with a request: the block's module references and the plan's classes
resolve by import on the server, which therefore has to have both packages
installed at the client's versions. A stock ndif.us cannot run a mini document
until they are installed there — the failure is a `ModuleNotFoundError` on the
server, not a silent divergence. Set `NDIF_HOST` (and `NDIF_API_KEY` where the
server has auth on) and the run is one job.

Tests: `CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`. Types: `uvx pyright`.
