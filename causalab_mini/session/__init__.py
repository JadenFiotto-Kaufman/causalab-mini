"""The one nnsight session, and everything that happens inside it.

`run.py` opens it — exactly once per request, with `remote` as its only
local/remote difference. `observe.py` is one execution of a plan: the forwards,
the taps, the metrics. `train.py` is that same execution N times with an
optimizer in between.

Every function in here is module-level and takes plain data. nnsight ships
every name a traced block loads as a whole pickled object, so a block that
reaches for `self`, a document or a tokenizer ships it; `tests/test_structure.py`
is the tripwire that says so.
"""

from .run import execute

__all__ = ["execute"]
