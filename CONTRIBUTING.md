# Working on causalab-mini

Three commands. All of them assume `uv` and this directory.

```bash
uv sync                                        # the environment, incl. editable nnsight/nnterp
CUDA_VISIBLE_DEVICES= uv run pytest tests/ -q  # the suite: CPU only, a few seconds
uvx pyright                                    # the type checker: must stay at 0 errors
```

`CUDA_VISIBLE_DEVICES=` is not optional here: the tiny models are CPU fixtures,
and this machine's driver is older than the torch build, so letting torch see a
GPU costs a warning at best.

`uvx pyright` reads its settings from `[tool.pyright]` in `pyproject.toml`,
which points it at `.venv` and at the nnterp checkout (installed editable
through an import hook, which a static checker cannot follow on its own). The
package ships a `py.typed`, so anything importing it gets the annotations.

To run a document:

```bash
uv run causalab-mini documents/minimal_cpu.json --data-root documents/data \
    --out out --device-map cpu
```

`documents/das_cpu_reduction.json` is the same command and takes a few seconds
longer: it declares a `train` block, so the run fits a rotation before it scores
anything, and writes `rot.safetensors` beside the two metric tables. The fit
happens inside the same single session as the run — see `causalab_mini/engine/` —
so `--remote local` exercises the loop, the optimizer and the backward pass over
the serialization path, exactly as `--remote true` would.

Read `NOTES.md` before changing what a document means — it is the ground-truth
reading of the protocol — and add to `FINDINGS.md` every model fact you had to
encode by hand.
