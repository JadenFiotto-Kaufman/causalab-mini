# examples

- **`das_walkthrough.ipynb`** — one DAS document, start to finish: the JSON, the
  validated spec, the compiled plan tree, one nnsight session, the results filling in on
  the nodes that produced them, and the files that leave. Runs on CPU in about ten
  seconds. Outputs are committed, so it reads without being run.

To run it yourself:

```bash
CUDA_VISIBLE_DEVICES= uv run --with jupyter --with ipykernel jupyter lab examples/
```

`CUDA_VISIBLE_DEVICES=` is not optional here — the models are tiny CPU fixtures and this
machine's driver is older than the torch build.
