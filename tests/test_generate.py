"""The continuation frame: a forward that decodes, and taps at its steps.

`decode: N` on an intervention makes every forward a prefill plus N greedy
steps, and `{"frame": "generated", …}` is a position along what it said.

Two halves, and the difference between them is what the *run* has to have
seen. `{"index": k}` with k >= 0 is a step the decode reaches, so a tap
fires there and a write may say it — steering is `{"all": true}`, every
step. Everything else — the last real token, the row's stop token, where
the model said the row's own text — is a cut of a continuation that does
not exist until the decode has finished, so the read fires at every step
and `engine/steps.py` cuts the stack afterwards. A write may not name one
of those at all, and says so.

Prompt-frame taps apply at the prefill and reach the continuation only
through the cache. The generated ids come home as `<model>.generated`.
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
    raw["interventions"]["generate"]["reads"]["logits"]["pos"] = {
        "frame": "generated", "index": step
    }
    return raw


def _at(raw, pos):
    raw = json.loads(json.dumps(raw))
    raw["interventions"]["generate"]["reads"]["logits"]["pos"] = pos
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
    assert list(built.steps) == ["index=0", "index=1", "index=2"]
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
    """`{"frame": "generated", "all": true}` with add_scaled: the last-token
    residual, scaled, added at every decode step at the last position."""
    raw = _unswept(probe_raw, step=2)
    one = raw["interventions"]["generate"]
    one["writes"]["patch"] = {"site": "target", "pos": {"frame": "generated", "all": True},
                              "mechanism": "add_scaled", "operand": "v_cf",
                              "params": {"scale": 4.0}}
    built = plan.build_request(raw, data_root, model_engine)
    tap = built.step("score", plan.Observe).forwards[1].taps[0]
    assert tap.step == "all" and tap.writes[0].mechanism == "add_scaled"
    steered = model_engine.execute(built).step("score", plan.Observe).results["patched.generated"]
    assert tuple(steered.shape) == (4, 3)


GENERATED = {"frame": "generated"}


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda one: one["reads"]["logits"].update(pos={**GENERATED, "index": 3}),
         "step 3 of a 3-token decode"),
        (lambda one: one.update(decode=0), "a generated position needs `decode` > 0"),
        (lambda one: one["writes"]["patch"].update(pos={**GENERATED, "index": -1}),
         "a write in the continuation frame is at a step the decode has reached"),
        (lambda one: one["writes"]["patch"].update(pos={**GENERATED, "all": True,
                                                        "scope": {"segment": "eos"}}),
         "takes no scope"),
        (lambda one: one["writes"]["patch"].update(pos={**GENERATED, "index": 9}),
         "step 9 of a 3-token decode"),
    ],
    ids=["past the budget", "no decode", "a write at the last token",
         "a write at the stop token", "a write past the budget"],
)
def test_what_a_continuation_position_may_not_be(probe_raw, edit, message):
    """A read may name any cut of the continuation, because the run has the
    whole of it by the time it chooses. A write happens during the decode,
    so it may only name a step the decode has reached."""
    raw = _unswept(probe_raw, step=1)
    edit(raw["interventions"]["generate"])
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(raw)


# --------------------------------------------------------------------- #
# the cuts only the finished continuation can settle
# --------------------------------------------------------------------- #


def test_the_last_generated_token_is_read_out_of_every_step(probe_raw, data_root, model_engine):
    """`{"index": -1}` cannot name a decode step in advance, so the read fires
    at every one of them and the run cuts the stack against the continuation
    it produced. Here nothing stops, so the last token is the last step — and
    the number is the one that step's own read gets."""
    dynamic = plan.build_request(_at(probe_raw, {**GENERATED, "index": -1}), data_root, model_engine)
    taps = dynamic.step("score", plan.Observe).forwards[1].taps
    stacked = [op for tap in taps for op in tap.reads if op.stack]
    assert [(op.name, op.stack) for op in stacked] == [(f"logits@{k}", "logits") for k in range(3)]

    last = model_engine.execute(dynamic).step("score", plan.Observe)
    static = model_engine.execute(
        plan.build_request(_unswept(probe_raw, step=2), data_root, model_engine)
    ).step("score", plan.Observe)
    assert torch.equal(last.results["p_answer"], static.results["p_answer"])
    assert last.results["positions"]["logits"]["rows"] == ((2,),) * 4


def test_a_read_over_a_stack_is_refused_when_it_would_hold_too_much(
    probe_raw, data_root, model_engine, monkeypatch
):
    """The one cost of buffering every step, stated: `rows x decode x width`
    numbers. It is nothing here and a gigabyte on a long decode over a wide
    site, so there is a line — and the three numbers are the compiler's, so
    it is drawn before a model is loaded rather than after the memory has
    been held."""
    import sys

    # `plan.build` is the compiler *function* — the package re-exports it over
    # the module's own name, which is the trap test_structure.py names as
    # `write_module`. So the module is fetched by its import path.
    monkeypatch.setattr(sys.modules["causalab_mini.plan.build"], "STACK_LIMIT", 1024)
    with pytest.raises(plan.PlanError, match="keeps every decode step"):
        plan.build_request(_at(probe_raw, {**GENERATED, "index": -1}), data_root, model_engine)


# --------------------------------------------------------------------- #
# what the model said, and whether it stopped
# --------------------------------------------------------------------- #


ANSWER = REPO / "documents" / "v2" / "generated_answer.json"


def test_whether_the_model_said_it_is_a_result_and_not_an_exception(data_root, model_engine):
    """`documents/v2/generated_answer.json`. Two of these four rows' `said`
    text is in what the model generated and two are not, and the spec is the
    same for all four — so which rows can be scored is the run's answer, per
    row, with the reason beside it."""
    raw = json.loads(ANSWER.read_text())
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    score = executed.step("score", plan.Observe)

    assert score.results["eligible"]["p_answer"] == (True, True, False, False)
    said = score.results["positions"]["at_said"]
    assert said["reason"] == ("", "", "alignment_missing", "alignment_missing")
    assert said["tokens"] == ("' substr'", "' energ'", "", "")
    # and the step it landed on is the row's own, not a number counted once
    assert said["rows"] == ((1,), (2,), (), ())
    assert score.results["p_answer"].shape == (2,)


def test_a_row_that_never_stopped_says_so_rather_than_ending_the_run(data_root, model_engine):
    """`{"scope": {"segment": "eos"}}` is where the row stopped. Mini holds
    EOS off so the decode runs to its bound and the batch stays rectangular
    — a deliberate difference — so no row stops here, and every row comes
    back `alignment_missing` instead of the run failing."""
    raw = json.loads(ANSWER.read_text())
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    stop = executed.step("score", plan.Observe).results["positions"]["at_stop"]
    assert stop["rows"] == ((),) * 4
    assert set(stop["reason"]) == {"alignment_missing"}


def test_the_continuation_frame_prints_as_itself(data_root, model_engine):
    """And a read the run cuts out of the continuation prints as the one
    read the document wrote, not as the six ops the plan carries."""
    text = explain(plan.build_request(json.loads(ANSWER.read_text()), data_root, model_engine))
    assert "read  'at_said' at lm_head over 6 steps" in text
    assert "pos={generated index:-1 scope:{variable:said}}" in text
    assert "pos={generated index:-1 scope:{segment:eos}}" in text
    assert "at_said@" not in text


def test_a_tap_in_the_continuation_frame_reports_where_it_was(
    probe_raw, data_root, model_engine, tmp_path
):
    """The prompt frame has nothing true to say about a tap that acted at a
    decode step, so it says nothing and the continuation says it instead:
    the step, and the token the model produced there. The table carries it,
    which is where a reader asks "of what token" about a number."""
    raw = _at(probe_raw, {**GENERATED, "index": 2})
    one = raw["interventions"]["generate"]
    one["writes"]["patch"]["pos"] = {**GENERATED, "all": True}
    one["writes"]["patch"].update(mechanism="add_scaled", params={"scale": 4.0})
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    where = executed.step("score", plan.Observe).results["positions"]

    assert where["logits"]["rows"] == ((2,),) * 4, "the decode step, not a prompt index"
    assert where["patch"]["rows"] == ((0, 1, 2),) * 4, "a steering write is every step"
    assert set(where["logits"]["reason"]) == {""}
    # the prompt-frame read is still reported against the prompt
    assert where["v_cf"]["rows"] == ((10,),) * 4 and where["v_cf"]["tokens"] == ("' is'",) * 4
    # and what the model said at step 2 is what the provenance shows
    generated = executed.step("score", plan.Observe).results["patched.generated"]
    said = model_engine.tokenizer.decode([int(generated[0][2])])
    assert said in where["logits"]["tokens"][0]

    executed.write(tmp_path)
    row = json.loads((tmp_path / "p_answer.json").read_text())[0]
    assert row["positions"] == [2] and row["reason"] == ""
