"""causalab-mini: a small, readable reimplementation of causalab's intervention engine.

The whole design rests on one rule: **a plan is pure data, a block turns it into
tensors**. `plan.py` compiles a document into strings and integers; `run.py`
opens one nnsight session and executes it. Nothing else is allowed to decide
anything from a tensor.
"""
