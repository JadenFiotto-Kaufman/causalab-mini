"""The `gate` featurizer: DBM, a learned binary mask over a site's units.

A gate is the one featurizer with a mode. While a fit updates it the mask is
soft, σ(θ/T), so a gradient reaches θ; whenever it is scored the mask is hard,
θ > 0, so the score is of a mask that could actually be shipped. Everything
else — the write seam, the fit loop, the bundle round trip — is what a
rotation already uses.
"""

import copy
import json
import pathlib

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import save_file

from causalab_mini import plan
from causalab_mini.engine import steps
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import featurizer
from causalab_mini.plan.spec_v2 import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
DBM = REPO / "documents" / "v2" / "dbm.json"
PATCHING = REPO / "documents" / "v2" / "patching.json"


@pytest.fixture
def dbm_raw():
    """One point of the swept document: the L1 weight that keeps two units."""
    raw = json.loads(DBM.read_text())
    raw["steps"]["fit"]["objective"][1][0] = 0.002
    return raw


# --------------------------------------------------------------------- #
# the object
# --------------------------------------------------------------------- #


def test_the_mask_is_soft_in_training_and_hard_otherwise():
    gate = featurizer.Gate(torch.tensor([-2.0, 0.0, 0.5, 3.0]))
    assert gate.mask.tolist() == [0.0, 0.0, 1.0, 1.0], "θ = 0 is off: the hard mask is θ > 0"

    gate.training = True
    assert torch.allclose(gate.mask, torch.sigmoid(gate.weight))
    gate.temperature = 0.01  # annealed, the soft mask is the hard one away from θ = 0
    assert torch.allclose(gate.mask[[0, 2, 3]], torch.tensor([0.0, 1.0, 1.0]), atol=1e-6)


def test_a_swap_through_a_gate_takes_the_masked_units_and_keeps_the_rest():
    gate = featurizer.Gate(torch.tensor([1.0, -1.0, 1.0]))
    here, there = torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[10.0, 20.0, 30.0]])
    features, err = gate.featurize(there)
    assert torch.equal(features, there) and err is None, "a gate's features are the units"
    assert gate.inverse(features, None, here).tolist() == [[10.0, 2.0, 30.0]]


def test_an_unfitted_gate_writes_nothing():
    """θ starts at 0, and the hard mask is θ > 0 — so a gate nobody has
    fitted is the null intervention, not a random one."""
    gate = featurizer.Gate(featurizer.start("gate", 8, 8, seed=0))
    x = torch.randn(2, 8)
    assert torch.equal(gate.inverse(torch.randn(2, 8), None, x), x)


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["featurizers"]["mask"].update(k=4), "takes no k"),
        (lambda raw: raw["featurizers"]["mask"].update(seed=1), "has no seed"),
        (lambda raw: raw["featurizers"]["mask"].update(kind="subspace"), "needs k"),
        (lambda raw: raw["steps"]["fit"]["anneal"].update(rot={"start": 1, "end": 0.1}), "only a gate"),
        (lambda raw: raw["steps"]["fit"]["anneal"]["mask"].update(end=0.0), "greater than 0"),
        (lambda raw: raw["steps"]["fit"]["objective"].append([1.0, "other.mask"]), "neither a metric"),
    ],
    ids=["k", "seed", "subspace without k", "anneal a non-gate", "anneal to zero", "foreign mask"],
)
def test_what_a_gate_document_may_not_say(dbm_raw, edit, message):
    edit(dbm_raw)
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(dbm_raw)


def test_the_fit_leaves_the_gate_hard_and_annealed(dbm_raw, data_root):
    """Run the walk by hand, so the live featurizer can be looked at: after
    the fit the gate is out of training mode, and its temperature is the
    schedule's last value."""
    engine = HooksEngine.load(Spec.model_validate(dbm_raw).model, device_map="cpu")
    built = plan.build_request(dbm_raw, data_root, engine)
    from causalab_mini.ops import intervene

    state = steps.State(featurizers=dict(intervene.FEATURIZERS))
    steps.build(built.step("featurizers", plan.Featurizers), state)
    gate = state.featurizers["mask"]
    assert gate.temperature == 1.0 and not gate.training

    fit = built.step("fit", plan.Fit)
    steps.fit(engine, fit, state)
    assert not gate.training
    assert gate.temperature == pytest.approx(0.05)
    # the eval record: the watched metric, then the fraction the hard mask keeps
    record = fit.results["train/eval"]
    assert record.shape[1] == 2
    assert record[-1, 1] == pytest.approx(float((gate.weight > 0).float().mean()))


def test_the_swept_document_is_the_sparsity_curve(data_root, model_engine):
    """More L1, fewer units: measured 2026-09-21 as 4, 2, 1, 0 of 16. The
    model is random, so which units survive means nothing; that the curve is
    monotone and reaches empty is the mechanism working."""
    executed = model_engine.execute(
        plan.build_request(json.loads(DBM.read_text()), data_root, model_engine)
    )
    kept = [int((point.result("mask") > 0).sum()) for point in executed.steps.values()]
    assert kept == sorted(kept, reverse=True) and kept[0] > kept[-1] == 0, kept


def test_the_two_engines_learn_the_same_mask(dbm_raw, data_root, model_engine):
    hooks = HooksEngine.load(Spec.model_validate(dbm_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(dbm_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(dbm_raw, data_root, hooks))
    assert torch.allclose(traced.result("mask"), hooked.result("mask"), atol=1e-6)
    assert torch.equal(traced.result("mask") > 0, hooked.result("mask") > 0)


# --------------------------------------------------------------------- #
# a loaded gate is a mask and nothing else
# --------------------------------------------------------------------- #


def _apply(dbm_raw, bundle):
    raw = copy.deepcopy(dbm_raw)
    raw["featurizers"]["mask"]["file_path"] = str(bundle)
    raw["steps"] = {"score": raw["steps"]["score"]}
    return raw


def test_the_bundle_round_trip_reproduces_the_fits_own_score(dbm_raw, data_root, model_engine, tmp_path):
    executed = model_engine.execute(plan.build_request(dbm_raw, data_root, model_engine))
    executed.write(tmp_path)

    applied = model_engine.execute(
        plan.build_request(_apply(dbm_raw, tmp_path / "mask.safetensors"), data_root, model_engine)
    )
    assert torch.equal(applied.result("iia"), executed.result("iia"))


def test_an_all_on_gate_is_plain_patching_and_an_all_off_gate_is_nothing(
    dbm_raw, data_root, model_engine, tmp_path
):
    """The two ends of the mask pin the write: every unit on is the
    interchange `patching.json` does, bit for bit; every unit off is the
    un-intervened model."""
    patching = json.loads(PATCHING.read_text())
    patching["steps"]["score"]["rows"] = dbm_raw["steps"]["score"]["rows"]
    whole = model_engine.execute(plan.build_request(patching, data_root, model_engine)).result("logit_diff")
    patching["interventions"]["patching"]["models"]["patched"]["writes"] = []
    nothing = model_engine.execute(plan.build_request(patching, data_root, model_engine)).result("logit_diff")

    scored = {}
    for label, theta in (("on", torch.ones(16)), ("off", -torch.ones(16))):
        save_file({"weight": theta}, str(tmp_path / f"{label}.safetensors"))
        raw = _apply(dbm_raw, tmp_path / f"{label}.safetensors")
        scored[label] = model_engine.execute(plan.build_request(raw, data_root, model_engine)).result("iia")
    assert torch.equal(scored["on"], whole)
    assert torch.equal(scored["off"], nothing)
    assert not torch.equal(whole, nothing)


def test_a_bundle_of_another_kind_is_refused(dbm_raw, data_root, model_engine, tmp_path):
    """A rotation's bundle is `(d, k)`; a gate's is `(d,)`. The stamp says
    which, and the shape would too."""
    save_file({"weight": torch.ones(16, 4)}, str(tmp_path / "rot.safetensors"), metadata={"kind": "subspace"})
    with pytest.raises(plan.PlanError, match="kind: bundle says 'subspace', document says 'gate'"):
        plan.build_request(_apply(dbm_raw, tmp_path / "rot.safetensors"), data_root, model_engine)


def test_a_gate_fit_survives_being_shipped(dbm_raw, data_root, model_engine):
    """The mode flag, the anneal and the mask term all live in the walk, so
    they ship with it: `remote="local"` serializes the session as a remote
    run would, and learns the same θ bit for bit."""
    here = model_engine.execute(plan.build_request(dbm_raw, data_root, model_engine))
    shipped = model_engine.execute(plan.build_request(dbm_raw, data_root, model_engine), remote="local")
    assert torch.equal(here.result("mask"), shipped.result("mask"))
    assert torch.equal(
        here.step("fit", plan.Fit).results["train/eval"],
        shipped.step("fit", plan.Fit).results["train/eval"],
    )
