"""The continuation frame: a forward that decodes, and taps at its steps.

`decode: N` on an intervention makes every forward a prefill plus N greedy
steps. A position may then be `{"step": k}` — the one position step k
processes — and a write may be `{"step": "all"}`, every step, which is what
steering is. Prompt-frame taps apply at the prefill and reach the
continuation only through the cache. The generated ids come home as
`<model>.generated`.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan import sweep
from causalab_mini.plan.explain import explain
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
PROBE = REPO / "documents" / "v2" / "generate_probe.json"


@pytest.fixture
def probe_raw():
    return json.loads(PROBE.read_text())


def _unswept(raw, step=1):
    raw = json.loads(json.dumps(raw))
    raw["interventions"]["generate"]["reads"]["logits"]["pos"] = {"step": step}
    return raw


def test_a_decoding_forward_yields_ids_and_a_per_step_read(probe_raw, data_root, model_engine):
    raw = _unswept(probe_raw, step=2)
    built = plan.build_request(raw, data_root, model_engine)
    forwards = built.step("score", plan.Observe).forwards
    assert [f.decode for f in forwards] == [3, 3]
    # the patched forward: the prompt-frame write at layer 0, then the head at step 2
    taps = forwards[1].taps
    assert [(t.address.component, t.step) for t in taps] == [("block_output", None), ("lm_head", 2)]
    assert "decode=3" in explain(built) and "@step 2" in explain(built)

    executed = model_engine.execute(built)
    results = executed.step("score", plan.Observe).results
    assert tuple(results["patched.generated"].shape) == (4, 3)
    assert tuple(results["original.generated"].shape) == (4, 3)
    assert results["p_answer"].shape == (4,)


def test_a_prefill_write_reaches_the_continuation_through_the_cache(probe_raw, data_root, model_engine):
    """The patch is made once, at step 0. If what the model says at step 2
    differs from the un-patched run's, it got there — and the only road is
    the cache. The comparison is on the answer's probability, not the greedy
    token: this tiny random model generates the same three tokens for every
    row whatever the input, so an argmax flip would need a far larger push
    than one interchange (zeroing the residual does flip it; FINDINGS §12)."""
    raw = _unswept(probe_raw, step=2)
    patched = model_engine.execute(plan.build_request(raw, data_root, model_engine))

    clean = json.loads(json.dumps(raw))
    one = clean["interventions"]["generate"]
    del one["writes"], one["models"], one["reads"]["v_cf"]
    one["reads"]["logits"]["model"] = "original"
    clean["steps"]["score"]["saves"] = [{"value": "p_answer", "file_path": "p_answer.json"}]
    plain = model_engine.execute(plan.build_request(clean, data_root, model_engine))

    assert not torch.equal(
        patched.step("score", plan.Observe).results["p_answer"],
        plain.step("score", plan.Observe).results["p_answer"],
    )


def test_sweeping_the_step_is_one_point_per_decode_step(probe_raw, data_root, model_engine):
    built = plan.build_request(probe_raw, data_root, model_engine)
    assert list(built.steps) == ["step=0", "step=1", "step=2"]
    executed = model_engine.execute(built)
    curve = [executed.step(p, plan.Plan).step("score", plan.Observe).results["p_answer"] for p in built.steps]
    assert len({tuple(c.tolist()) for c in curve}) == 3, "three steps, three readings"


def test_the_two_engines_generate_the_same_tokens(probe_raw, data_root, model_engine):
    """Greedy decoding on both, the same prefill patch: the same ids. The
    per-step probabilities are compared to the ulp, as everywhere."""
    raw = _unswept(probe_raw, step=1)
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(raw, data_root, hooks))
    a, b = traced.step("score", plan.Observe).results, hooked.step("score", plan.Observe).results
    assert torch.equal(a["patched.generated"], b["patched.generated"])
    assert torch.allclose(a["p_answer"], b["p_answer"], rtol=0, atol=1e-7)


def test_steering_is_a_write_at_every_step(probe_raw, data_root, model_engine):
    """`{"step": "all"}` with add_scaled: the counterfactual's last-token
    residual, scaled, added at every decode step at the last position."""
    raw = _unswept(probe_raw, step=2)
    one = raw["interventions"]["generate"]
    one["writes"]["patch"] = {"site": "target", "pos": {"step": "all"}, "mechanism": "add_scaled",
                              "operand": "v_cf", "params": {"scale": 4.0}}
    built = plan.build_request(raw, data_root, model_engine)
    tap = built.step("score", plan.Observe).forwards[1].taps[0]
    assert tap.step == "all" and tap.writes[0].mechanism == "add_scaled"
    steered = model_engine.execute(built).step("score", plan.Observe).results["patched.generated"]
    assert tuple(steered.shape) == (4, 3)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda one: one["reads"]["logits"].update(pos={"step": 3}), "step 3 of a 3-token decode"),
        (lambda one: one.update(decode=0), "a step position needs `decode` > 0"),
        (lambda one: one["reads"]["logits"].update(pos={"step": "all"}), "'all' is for writes"),
        (lambda one: one["reads"]["logits"].update(pos={"step": -1}), "non-negative integer"),
    ],
    ids=["past the budget", "no decode", "a read at all steps", "a negative step"],
)
def test_what_a_step_may_not_be(probe_raw, edit, message):
    raw = _unswept(probe_raw, step=1)
    edit(raw["interventions"]["generate"])
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(raw)
