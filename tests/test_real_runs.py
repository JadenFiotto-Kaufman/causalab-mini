"""What only a real model, a real GPU and a real server could teach.

Every fix in this file was found on 2026-09-21 running `documents/real/` on
hakone — Llama-3.2-1B locally on an A100 and through a self-hosted NDIF —
and none of them was visible to a CPU suite on a 16-wide random model, or to
`remote="local"`. The tests pin the *mechanism* of each fix at a size a
laptop can run. FINDINGS §19.
"""

import json
import pathlib
import pickle

import pytest
import torch

from causalab_mini import plan
from causalab_mini.ops import featurizer, intervene
from causalab_mini.plan import plan as plan_module

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_the_cayley_start_is_trainable_at_a_real_width():
    """Unscaled, the start's singular values grow like √d and the Cayley
    transform saturates: the basis stops responding to its parameter. The
    gradient of a fixed readout of Q, per unit of parameter, is the measure —
    at d = 1024 the 1/√d start is orders of magnitude more responsive."""
    d, k = 1024, 8
    readout = torch.randn(d, k, generator=torch.Generator().manual_seed(1))

    def responsiveness(weight):
        weight = weight.clone().requires_grad_(True)
        (featurizer.cayley(weight) * readout).sum().backward()
        return float(weight.grad.norm())

    scaled = featurizer.start_weight(d, k, seed=0)
    assert scaled.std() == pytest.approx(d**-0.5, rel=0.1)
    assert responsiveness(scaled) > 100 * responsiveness(scaled * d**0.5)
    # and it is still an orthonormal basis, which no scale can break
    q = featurizer.cayley(scaled)
    assert torch.allclose(q.T @ q, torch.eye(k), atol=1e-4)


def test_featurizer_math_runs_outside_a_servers_autocast():
    """NDIF wraps a request in `torch.autocast`. Inside one, half of the
    Cayley solve is downcast and `linalg.solve` refuses the pair — so the
    write seam steps out of autocast for the featurizer's own arithmetic."""
    rot = featurizer.Subspace(featurizer.start_weight(16, 4, seed=0))
    tensor = torch.randn(2, 5, 16)
    operand = torch.randn(2, 1, 4)
    plain = intervene.apply_write(tensor, ((4,), (4,)), operand, "swap", rot)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        # the hazard: two fp32 tensors, and their product is not fp32. (On
        # CUDA this is what split the solve's operands; CPU autocast happens
        # not to cast the same ops, so the dtype is what is asserted here.)
        assert (tensor @ tensor.mT).dtype == torch.bfloat16
        with intervene.exact(tensor):
            assert (tensor @ tensor.mT).dtype == torch.float32
        inside = intervene.apply_write(tensor, ((4,), (4,)), operand, "swap", rot)
    assert inside.dtype == tensor.dtype and torch.equal(inside, plain)


def test_a_featurizer_computes_where_its_activation_is():
    """The parameter is one leaf wherever it was built; the arithmetic
    follows the activation. `meta` stands in for a GPU a laptop lacks: a
    CPU parameter meeting a meta activation must produce a meta result,
    rather than the device error the first GPU run died with."""
    for made in (featurizer.Subspace(featurizer.start_weight(8, 2, 0)),
                 featurizer.Basis(torch.eye(8)[:, :2]),
                 featurizer.Gate(torch.ones(8))):
        x = torch.empty(3, 1, 8, device="meta")
        f, err = made.featurize(x)
        assert made.inverse(f, err, x).device.type == "meta", type(made).__name__
        assert made.weight.device.type == "cpu"


def test_what_comes_home_from_a_run_is_plain(data_root, model_engine):
    """A server runs the plan by value and cannot pickle one of its classes
    back. So results travel as `{step path: {name: tensor}}` — strings and
    tensors — and fill the plan the client never gave up."""
    raw = json.loads((REPO / "documents" / "v2" / "das.json").read_text())
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    home = plan_module.results_of(executed)

    assert set(home) == {"fit", "fit/eval", "score", "weights"}
    blob = pickle.dumps(home)
    assert b"causalab_mini" not in blob, "nothing of ours rides home"

    fresh = plan.build_request(raw, data_root, model_engine)
    plan_module.fill(fresh, pickle.loads(blob))
    assert torch.equal(fresh.result("rot"), executed.result("rot"))
    assert torch.equal(
        fresh.step("fit", plan.Fit).evaluation.results["iia"],
        executed.step("fit", plan.Fit).evaluation.results["iia"],
    )


def test_the_run_record_says_what_dtype_was_served(data_root, model_engine):
    """The document's dtype is a request; a server serves its own (NDIF: bf16
    under an fp32 document). What ran is in `run.json`."""
    raw = json.loads((REPO / "documents" / "v2" / "patching.json").read_text())
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    assert executed.provenance["served_dtype"] == "torch.float32"


def test_the_real_documents_are_valid_and_compile_without_weights(data_root):
    from causalab_mini.plan import sweep
    from causalab_mini.plan.spec import Spec

    found = sorted((REPO / "documents" / "real").glob("*.json"))
    assert len(found) == 4
    for path in found:
        for _, point in sweep.points(json.loads(path.read_text())):
            Spec.model_validate(point)
