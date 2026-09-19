"""One directory per runtime. Each holds everything that is true of that
runtime and nothing that is true of the others: how it loads a model, how it
resolves an address against that model, and how it runs one forward.

What they share — what a plan means, the fit loop, the metrics, the write
algebra — is one level up, in `engine/steps.py`.
"""
