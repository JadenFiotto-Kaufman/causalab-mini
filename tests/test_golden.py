"""Every runnable document, pinned to the bytes of what it writes.

The corpus is the project's evidence that the library does what it says, so a
change that moves a number in it has to be one somebody meant. This runs each
document that the tiny CPU models can run — every plan-shaped one and every
protocol one but the 8B `das.json` — and compares each file it writes with a
hash committed in `golden/outputs.json`.

What a hash covers is what a result *is*: every value, eligibility, position,
reason and decoded token of a metric table; every tensor's dtype, shape and
bytes, and a bundle's identity stamp. What it leaves out is what names the
document rather than the result: `produced_by` (the document's digest),
a table's `metric` column (the name the document gave the metric),
`document.json` and `run.json`. So a document may be rewritten — renamed,
reshaped, moved to another format — and still be held to the same numbers.

The file is written by `python tests/test_golden.py`, and only ever from a
commit whose numbers are the ones meant: regenerating it from the code under
test would pin whatever that code does. A torch upgrade that moves the last
bit is the one legitimate reason, and then it is regenerated at the commit
the pin was taken from.
"""

import hashlib
import json
import pathlib
import sys

from typing import Any

import pytest
import torch
from safetensors import safe_open

from causalab_mini import plan
from causalab_mini.engine import NNterpEngine
from causalab_mini.plan import sweep
from causalab_mini.plan.spec import Model

REPO = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "outputs.json"
DATA_ROOT = REPO / "documents" / "data"
#: The documents the suite can run: `das.json` names an 8B checkpoint, and
#: `real/` pins real ones (compiled, never run, by `test_real_runs.py`).
DOCUMENTS = sorted(
    [f"documents/{path.name}" for path in (REPO / "documents").glob("*.json") if path.name != "das.json"]
    + [f"documents/v2/{path.name}" for path in (REPO / "documents" / "v2").glob("*.json")]
)
#: A table's columns that name the document, not the result.
NAMING = {"produced_by", "metric"}


def outputs(document: str, out: pathlib.Path, engines: dict[str, Any]) -> dict[str, str]:
    """Run one document into `out` and hash what it wrote, by relative path.
    `engines` holds one loaded model per model block, across documents."""
    raw = json.loads((REPO / document).read_text())
    block = sweep.points(raw)[0][1]["model"]
    key = json.dumps(block, sort_keys=True)
    if key not in engines:
        engines[key] = NNterpEngine.load(Model.model_validate(block), device_map="cpu")
    engine = engines[key]
    engine.execute(plan.build_request(raw, DATA_ROOT, engine)).write(out)
    return {
        str(path.relative_to(out)): _hash(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name not in ("document.json", "run.json")
    }


def _hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    if path.suffix == ".safetensors":
        with safe_open(str(path), "pt") as bundle:
            stamp = {k: v for k, v in (bundle.metadata() or {}).items() if k != "produced_by"}
            digest.update(json.dumps(stamp, sort_keys=True).encode())
            for key in sorted(bundle.keys()):
                tensor = bundle.get_tensor(key).contiguous()
                digest.update(f"{key} {tensor.dtype} {list(tensor.shape)}".encode())
                digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
    else:
        rows = [{k: v for k, v in row.items() if k not in NAMING} for row in json.loads(path.read_text())]
        digest.update(json.dumps(rows, sort_keys=True).encode())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def engines():
    return {}


@pytest.mark.parametrize("document", DOCUMENTS)
def test_a_document_writes_what_it_wrote_when_it_was_pinned(document, engines, tmp_path, monkeypatch):
    # loaded bundles are named relative to the repository, as a user runs them
    monkeypatch.chdir(REPO)
    pinned = json.loads(GOLDEN.read_text())
    assert document in pinned, f"{document} is not pinned; pin it from a commit whose numbers are meant"
    assert outputs(document, tmp_path, engines) == pinned[document]


if __name__ == "__main__":
    import os
    import tempfile

    os.chdir(REPO)
    found, loaded = {}, {}
    for one in DOCUMENTS:
        with tempfile.TemporaryDirectory() as out:
            found[one] = outputs(one, pathlib.Path(out), loaded)
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n")
    print(f"pinned {sum(len(one) for one in found.values())} files of {len(found)} documents", file=sys.stderr)
