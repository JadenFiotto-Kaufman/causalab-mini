"""The steps-first corpus is the old one: same plans, same numbers.

`documents/v2/` and `documents/real/` are written in the steps-first
format, and `tests/fixtures/*_old/` keeps them as they were. Every
rewritten document must compile to the plan its old self did — the same
model calls in the same order, with the same ids, masks, taps, positions
and operands; the same metrics, reductions, fits and files. (`test_golden.py`
holds the 22 the suite can run to the bytes they wrote.) And the old one,
put through `convert_v2.py`, must still write those bytes. Names are the
one thing allowed to differ, so a plan is compared with every name taken
out: an operand is the read that produced it, an op is where it acts, a
file is where it lands.

Test-only, like the converter, and deleted with the old format;
`test_golden.py` stays as the lasting pin.
"""

import json
import pathlib

import pytest
from convert_v2 import convert
from test_golden import GOLDEN, outputs

from causalab_mini import plan
from causalab_mini.engine import NNterpEngine
from causalab_mini.plan import sweep
from causalab_mini.plan.plan import Featurizers, Fit, Forward, Metric, Plan, Reduce, Weights
from causalab_mini.plan.spec import Model

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "documents" / "data"
FIXTURES = REPO / "tests" / "fixtures"
RUNNABLE = sorted(f"documents/v2/{path.name}" for path in (REPO / "documents" / "v2").glob("*.json"))
REAL = sorted(f"documents/real/{path.name}" for path in (REPO / "documents" / "real").glob("*.json"))


@pytest.fixture(scope="module")
def engines():
    return {}


def _engine(raw, engines, **options):
    block = sweep.points(raw)[0][1]["model"]
    key = json.dumps([block, options], sort_keys=True)
    if key not in engines:
        engines[key] = NNterpEngine.load(Model.model_validate(block), **options)
    return engines[key]


def _flat(step, where="", into=None, scope=None):
    """A plan with its names taken out, as streams: forwards, metrics,
    reductions, fits, featurizers, weights and files, each in run order.
    An operand is compared as the read that produced it — or the kind of
    reduction it is — and a file as where it lands."""
    into = into if into is not None else {key: [] for key in ("forwards", "metrics", "reduced", "fits", "featurizers", "weights", "files")}
    into["files"] += [f"{where}{one.file_path}" for one in step.saves]
    if isinstance(step, Plan):
        inner = {
            op.stack or op.name: (tap.address, op.at)
            for one in step.steps.values() if isinstance(one, Forward)
            for tap in one.taps for op in tap.reads
        }
        inner |= {name: ("reduced", one.reduce) for name, one in step.steps.items() if isinstance(one, Reduce)}
        for name, child in step.steps.items():
            _flat(child, f"{where}{name}/" if isinstance(child, Plan) else where, into, {**(scope or {}), **inner})
    elif isinstance(step, Forward):
        scope = scope or {}

        def operand(value):
            return scope.get(value, ("published",)) if isinstance(value, str) else value

        into["forwards"].append((
            type(step).__name__, step.input_ids, step.attention_mask, getattr(step, "max_new_tokens", 0),
            getattr(step, "generation", {}), step.sample, step.segments, len(step.keep),
            tuple(
                (tap.address, tap.step,
                 tuple((op.mechanism, op.featurizer, op.params, op.features, op.at, operand(op.operand)) for op in tap.writes),
                 tuple((op.featurizer, op.view, op.at, bool(op.stack)) for op in tap.reads))
                for tap in step.taps
            ),
        ))
    elif isinstance(step, Metric):
        into["metrics"].append((step.kind, step.ids, step.rows, step.flat))
    elif isinstance(step, Reduce):
        into["reduced"].append((step.reduce, step.k))
    elif isinstance(step, Fit):
        updates = [_flat(one) for epoch in step.epochs for one in epoch]
        into["fits"].append((
            len(step.epochs), [len(epoch) for epoch in step.epochs], updates, _flat(step.evaluation),
            [weight for weight, _ in step.objective], step.params, step.lr, step.weight_decay,
            len(step.eval_metrics), step.patience, step.mode, step.anneal,
        ))
        into["files"] += _flat(step.evaluation, where)["files"]
    elif isinstance(step, Featurizers):
        into["featurizers"].append(step.specs)
    elif isinstance(step, Weights):
        into["weights"].append(step.names)
    return into


def _same_plan(old, new):
    one, other = _flat(old), _flat(new)
    one["files"], other["files"] = sorted(one["files"]), sorted(other["files"])
    for key in one:
        assert one[key] == other[key], key


def _old(document):
    """The document as it was, before the rewrite."""
    folder, name = document.split("/")[1:]
    return json.loads((FIXTURES / f"{folder}_old" / name).read_text())


@pytest.mark.parametrize("document", RUNNABLE + REAL)
def test_a_rewritten_document_compiles_to_the_plan_it_was(document, engines, monkeypatch):
    monkeypatch.chdir(REPO)
    old, new = _old(document), json.loads((REPO / document).read_text())
    assert [label for label, _ in sweep.points(new)] == [label for label, _ in sweep.points(old)]
    engine = _engine(old, engines, **({"dispatch": False} if document in REAL else {"device_map": "cpu"}))
    _same_plan(plan.build_request(old, DATA_ROOT, engine), plan.build_request(new, DATA_ROOT, engine))


@pytest.mark.parametrize("document", RUNNABLE)
def test_a_converted_document_writes_the_pinned_bytes(document, engines, tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    converted = tmp_path / "converted.json"
    converted.write_text(json.dumps(convert(_old(document))))
    assert outputs(str(converted), tmp_path / "out", engines) == json.loads(GOLDEN.read_text())[document]
