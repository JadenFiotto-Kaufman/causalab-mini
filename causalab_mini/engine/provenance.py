"""What ran: the record every output directory carries.

An output directory used to hold metric tables and a 64-hex `produced_by`
that could not be inverted — a result with no way back to the experiment
that produced it. Now a run carries two things home:

* **the document itself**, verbatim, on the root plan as `source` — set by
  the compiler, so it is there whether or not the run happened;
* **this record**, on the root plan as `provenance` — set by the engine at
  the top of `execute`: which engine, remote or not, the versions of the
  packages whose code decides what a block does, and a digest of this
  package's own source. nnsight and nnterp are editable checkouts that move
  underneath this project, and two engines exist that are known to differ by
  an ulp on some documents; without this the files cannot say which code
  produced them.

On a remote run the versions that decide what the block does are the
*server's*, and `versions` is this process's — an assumption about a machine
it has never asked. So a remote run asks: nnsight's `/env` lookup, cached per
host, answers with the server's Python and its installed packages, and they
are recorded beside the client's under a key that says whose they are. A
server that will not say is recorded as not having said, because that is
evidence too and is not a reason to refuse a run.

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
    found = {
        "engine": type(engine).__name__,
        "remote": remote,
        # not part of the experiment, but it moves the last bit of every number
        "batch_size": batch_size,
        "versions": {name: _version(name) for name in WATCHED},
        "causalab_mini": code_digest(),
    }
    served = _served(remote)
    if served is not None:
        found["server"] = served
    return found


def _served(remote: bool | str) -> dict[str, Any] | None:
    """What the server says it is running, or why it did not say.

    `None` for a run that has no server — every local one, and
    `remote="local"`, which is this process pretending. A `remote` that is a
    URL is the host; `True` is the configured one. nnsight caches the answer
    per host, so asking once per run costs one request per process.
    """
    if not remote or remote == "local":
        return None
    from nnsight import ndif

    host = remote if isinstance(remote, str) else None
    try:
        env = ndif.get_remote_env(host)
    except Exception as unreachable:  # a server that will not say is evidence
        return {"asked": ndif.resolve_host(host), "said": f"{type(unreachable).__name__}: {unreachable}"}
    packages = env.get("packages", {})
    return {
        "asked": ndif.resolve_host(host),
        "python": env.get("python_version", "").split()[0],
        "versions": {name: packages.get(name, "not installed") for name in WATCHED},
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
