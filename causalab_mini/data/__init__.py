"""The corpus: a dataset reference becomes padded token rows.

`rows.py` reads tables off disk and picks columns out of them; `encoding.py`
turns the text in those columns into the integers a forward takes, and resolves
a document's `pos` against the padding it just produced.

Both are client-side. The block never sees a row, a field name or a tokenizer —
by the time the session opens, all of this has become `tuple[int, ...]`.
"""
