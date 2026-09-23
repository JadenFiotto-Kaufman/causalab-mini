"""What ran: the record every output directory carries.

An output directory used to hold metric tables and a 64-hex `produced_by`
that could not be inverted — a result with no way back to the experiment
that produced it. Now a run carries two things home:

* **the document itself**, verbatim, on the root plan as `source` — set by
  the compiler, so it is there whether or not the run happened;
* **this record**, on the root plan as `provenance` — set by the engine at
  the top of `execute`: which engine, remote or not, the versions of the
  four packages whose code decides what a block does, and a digest of this
  package's own source. nnsight and nnterp are editable checkouts that move
  underneath this project, and two engines exist that are known to differ by
  an ulp on some documents; without this the files cannot say which code
  produced them.

Deliberately absent, following causalab: no hostname, no timestamp, no
absolute path. A record has to be byte-identical across machines to be worth
comparing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import pathlib
from typing import Any

#: The packages whose code decides what a run does.
WATCHED = ("torch", "transformers", "nnsight", "nnterp", "pydantic")


def record(engine: Any, remote: bool | str, batch_size: int | None = None) -> dict[str, Any]:
    return {
        "engine": type(engine).__name__,
        "remote": remote,
        # not part of the experiment, but it moves the last bit of every number
        "batch_size": batch_size,
        "versions": {name: _version(name) for name in WATCHED},
        "causalab_mini": code_digest(),
    }


def code_digest() -> str:
    """sha256 over this package's own `.py` files, in path order. Two runs
    with the same digest ran the same code, whatever git says."""
    root = pathlib.Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"
