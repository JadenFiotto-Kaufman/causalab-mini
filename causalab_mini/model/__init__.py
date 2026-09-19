"""The only corner of the project that knows anything about models.

`loading.py` builds the handle (nnterp's StandardizedTransformer, which is what
gives `layers` and `lm_head` the same names on every architecture), and
`address.py` turns a protocol component name into a module path, a side, a
sequence axis and — for an interior — a `.source` operation.

Everything downstream of an `Address` is architecture-blind. `ops/` may not
import this package, and does not.
"""
