import copy
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "documents" / "data"
MINIMAL = REPO / "documents" / "minimal_cpu.json"


@pytest.fixture(scope="session")
def data_root():
    return DATA_ROOT


@pytest.fixture
def minimal_raw():
    """The activation-patching document, as a dict a test may edit."""
    return copy.deepcopy(json.loads(MINIMAL.read_text()))


@pytest.fixture(scope="session")
def model():
    """The tiny random Llama the document pins, on CPU in fp32."""
    from causalab_mini import document, model as model_module

    doc = document.Document.load(MINIMAL)
    return model_module.load(doc.model, device_map="cpu")
