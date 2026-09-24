"""The corpus: a dataset reference becomes padded token rows.

`rows.py` reads tables off disk and picks columns out of them; `tokens.py`
turns the text in those columns into the integers a forward takes, answers
what one of those integers is as an answer, and does the padding arithmetic
over a batch of them.

Both are client-side, and *where along* those integers a read or a write acts
is not here: that is a spec (`shapes.Where`) the run resolves against the
model's own tokenizer (`ops/locate.py`). The rule this buys is that the
format — `plan/spec.py`, which validates a document — needs no tokenizer to
answer how wide a cut is or which frame it names.
"""
