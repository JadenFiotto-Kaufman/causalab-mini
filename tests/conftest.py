import copy
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "documents" / "data"
MINIMAL = REPO / "documents" / "minimal_cpu.json"
DAS = REPO / "documents" / "das_cpu_reduction.json"


@pytest.fixture(scope="session")
def data_root():
    return DATA_ROOT


@pytest.fixture
def minimal_raw():
    """The activation-patching document, as a dict a test may edit."""
    return copy.deepcopy(json.loads(MINIMAL.read_text()))


@pytest.fixture
def das_raw():
    """The DAS document, as a dict a test may edit."""
    return copy.deepcopy(json.loads(DAS.read_text()))


@pytest.fixture(scope="session")
def model_engine():
    """An nnterp engine holding the tiny random Llama the document pins, on
    CPU in fp32. The engine loads the model: that is part of its contract, and
    a second engine loads it differently."""
    from causalab_mini.engine import NNterpEngine
    from causalab_mini.plan import document

    return NNterpEngine.load(document.Document.load(MINIMAL).model, device_map="cpu")


@pytest.fixture(scope="session")
def model(model_engine):
    """The nnterp handle itself, for tests that trace it by hand."""
    return model_engine.model


def same_numbers(a, b, atol: float = 1e-6) -> bool:
    """Two engines, one experiment: equal to the last few bits.

    These assertions were `torch.equal` while every run was on one machine.
    On hakone (torch 2.11, a different CPU) the same suite differs between
    the engines by exactly one ulp, 1.49e-08 — the intervention paths order
    their kernels differently and which way a GEMM rounds is the platform's
    business (FINDINGS §8, §19). Bit-equality across engines was a fact about
    one laptop, not a property of the library; this is the property.
    """
    import torch

    return a.shape == b.shape and torch.allclose(a.float(), b.float(), rtol=0, atol=atol)


def of_kind(plan, kind) -> list:
    """A plan's steps of one kind, in the order they run — its forwards, say.
    A plan is keyed by step name, and a test that is about what the model
    calls do has no need to spell each one's."""
    return [step for step in plan.steps.values() if isinstance(step, kind)]


def provenance(executed):
    """Where each step of a run acted and which rows it scored, by step —
    what two runs of one plan must agree on beside their numbers."""
    return {name: (one.results.get("positions"), one.results.get("eligible")) for name, one in executed.steps.items()}


def tensors(results):
    """A run's results as tensors by name, a record's parts as `<name>/<part>`
    — so two runs are compared tensor by tensor."""
    found = {}
    for name, value in results.items():
        found.update({f"{name}/{part}": one for part, one in value.items()} if isinstance(value, dict) else {name: value})
    return found
