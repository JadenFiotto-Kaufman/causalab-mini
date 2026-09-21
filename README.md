# causalab-mini

A deliberately small, readable reimplementation of causalab's intervention
engine. One rule underneath everything: **a plan is pure data; an engine turns
it into tensors.**

A document is a tree of steps. Each step says which rows it runs over and what
it writes out; the experiment it runs — reads, writes, intervened models,
metrics — is declared once and shared. The compiler resolves everything that
needs a model (positions against the tokenizer's padding, widths, an
interior's operation) on the client, without weights, and hands the engine a
plan of strings and integers. The engine walks it in one session; the results
land on the nodes that produced them.

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
| [`documents/v2/`](documents/v2/) | the plan-shaped format: patching, DAS, mean ablation, zero ablation, a window, an entity harvest, a logit lens, generation, a PCA control, DBM, head-by-head and neuron patching, attention knockout |

The CLI's read-only verbs — `schema`, `vocab`, `model`, `tokens`, `data`,
`validate`, `explain` — need no GPU and are the loop an author lives in.
Every verb takes `--json`.

## Two engines

`--engine nnterp` runs on nnsight through nnterp's standardized names, locally
or — as `--engine ndif` — on NDIF with no local weights. `--engine hooks` is
plain torch and forward hooks: a measurement fixture that agrees with the
first to the bit on single writes and refuses what a hook cannot reach.

Tests: `CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q`. Types: `uvx pyright`.
