"""Two frames at one address are one moment; two engines run one forward.

Found by exploratory testing (TESTING.md 1.2–1.4, 2.1, 2.3). Under `decode`
a tap is an address *and a frame*, and handled tap by tap a prompt-frame op
and an every-step op at one address did not see each other. And the hooks
engine numbered pad tokens as positions, which rotary models forgive and
GPT-2 does not.
"""

import copy
import json
import pathlib

import pytest
import torch
from conftest import same_numbers

from causalab_mini import plan
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import intervene
from causalab_mini.plan import document, sweep
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2 = REPO / "documents" / "v2"


def _doc(name):
    return sweep.points(json.loads((V2 / name).read_text()))[0][1]


@pytest.fixture
def steer():
    """steer_renormalize.json, decoding three tokens."""
    raw = _doc("steer_renormalize.json")
    raw["interventions"]["steer"]["decode"] = 3
    return raw


@pytest.mark.parametrize("engine_name", ["nnterp", "hooks"])
@pytest.mark.parametrize("frames", [(-1, {"step": "all"}), ({"step": "all"}, -1)], ids=["steer@prompt", "steer@all"])
def test_a_renormalize_restores_the_norm_across_frames(engine_name, frames, steer, data_root, model_engine):
    """It used to be bit-identical to not renormalizing at all: `original`
    was taken per tap, so the later frame measured against the already
    steered value; and the prompt frame always ran first, whatever the
    document's order said."""
    one = steer["interventions"]["steer"]
    one["writes"]["steer"]["pos"], one["writes"]["restore"]["pos"] = frames
    engine = model_engine if engine_name == "nnterp" else HooksEngine.load(Spec.model_validate(steer).model, device_map="cpu")
    got = engine.execute(plan.build_request(steer, data_root, engine)).step("steer", plan.Observe).results
    assert torch.allclose(got["norm_after"].norm(dim=-1), got["norm_before"].norm(dim=-1), rtol=1e-5)
    assert (got["norm_steered"].norm(dim=-1) > got["norm_before"].norm(dim=-1)).all()


@pytest.mark.parametrize("engine_name", ["nnterp", "hooks"])
def test_a_prompt_frame_read_sees_an_every_step_write_at_its_address(engine_name, data_root, model_engine):
    """Two spellings of one place and moment — the last prompt position at
    the prefill — used to disagree."""
    raw = _doc("zero_ablation.json")
    one = raw["interventions"]["zeroed"]
    site = one["writes"]["zero"]["site"]
    one["decode"] = 2
    one["writes"]["zero"]["pos"] = {"step": "all"}
    one["reads"]["at_prompt"] = {"site": site, "pos": -1, "model": "zeroed", "input": "base"}
    one["reads"]["at_step"] = {"site": site, "pos": {"step": 0}, "model": "zeroed", "input": "base"}
    raw["steps"]["zeroed"]["outputs"] = {"prompt_view": {"read": "at_prompt"}, "step_view": {"read": "at_step"}}
    engine = model_engine if engine_name == "nnterp" else HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    got = engine.execute(plan.build_request(raw, data_root, engine)).step("zeroed", plan.Observe).results
    assert got["step_view"].abs().sum() == 0
    assert got["prompt_view"].abs().sum() == 0, "it read the un-ablated activation"


def test_the_two_engines_agree_on_a_padded_gpt2_batch(data_root):
    """`weekdays/train` tokenizes to unequal lengths under GPT-2, so the
    batch is left-padded; `counting`, which the old parity test used, is
    rectangular and could not see this. Absolute positions make it visible:
    the engines differed by 0.29 on exactly the padded rows."""
    spec = document.Document.load(REPO / "documents" / "gpt2_cpu.json").model
    v2 = _doc("patching.json")
    v2["model"] = {"key": spec.key, "revision": spec.revision, "dtype": "fp32"}
    # no metric: the weekday names are several tokens here. Compare the logits.
    v2["interventions"]["patching"]["metrics"] = {}
    v2["steps"]["score"]["saves"] = []
    v2["steps"]["score"]["outputs"] = {"p": {"read": "logits"}}
    traced, hooked = NNterpEngine.load(spec, device_map="cpu"), HooksEngine.load(spec, device_map="cpu")
    built = plan.build_request(v2, data_root, traced)
    masks = built.step("score", plan.Observe).forwards[0].attention_mask
    assert len({sum(row) for row in masks}) > 1, "the batch must actually be padded for this to test anything"
    a = traced.execute(built).result("p")
    b = hooked.execute(plan.build_request(v2, data_root, hooked)).result("p")
    assert same_numbers(a, b)


def test_gaussian_runs_from_a_document(data_root, model_engine):
    """It never had: a document's `seed: 7` arrives as `7.0`, and
    `manual_seed` refuses a float. Only the op was tested, with a Python int."""
    raw = _doc("patching.json")
    raw["interventions"]["patching"]["writes"]["patch"] = {
        "site": "target", "pos": -1, "mechanism": "gaussian", "params": {"seed": 7, "scale": 0.5}}
    raw["interventions"]["patching"]["reads"].pop("v_cf")
    once = model_engine.execute(plan.build_request(raw, data_root, model_engine)).result("logit_diff")
    again = model_engine.execute(plan.build_request(raw, data_root, model_engine)).result("logit_diff")
    assert torch.equal(once, again)


def test_an_operand_is_moved_to_where_it_is_written():
    """Under `device_map="auto"` a read on one GPU feeds a write on another.
    `swap` survived that by accident (scatter moves what it writes);
    `add_scaled` and `lerp` died inside the mechanism. No second device
    here, so the operand reports where it was asked to go."""

    class Elsewhere:
        def __init__(self, tensor):
            self.tensor, self.asked = tensor, None

        def to(self, device):
            self.asked = device
            return self.tensor

    here = torch.zeros(2, 4, 8)
    for mechanism, params in (("add_scaled", {"scale": 2.0}), ("lerp", {"t": 0.5}), ("swap", {})):
        operand = Elsewhere(torch.ones(2, 1, 8))
        intervene.apply_write(here, ((3,), (3,)), operand, mechanism, params=params)
        assert operand.asked == here.device, mechanism
