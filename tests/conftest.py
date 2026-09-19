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
