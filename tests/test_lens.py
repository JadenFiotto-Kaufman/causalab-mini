"""The logit lens: a residual read pushed through the model's final norm and
head, so every layer can be asked what it would say if it were the last.

`view: "logits"` on a read is the projection; a layer sweep is the lens, or
`layers: "all"`, which is every layer in one forward; `token_prob` on the
answer is what to plot against depth.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import plan
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.plan.explain import explain
from causalab_mini.plan import sweep
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
LENS = REPO / "documents" / "v2" / "logit_lens.json"


@pytest.fixture
def lens_raw():
    return json.loads(LENS.read_text())


def test_the_lens_reads_a_probability_per_row_at_every_layer(lens_raw, data_root, model_engine):
    executed = model_engine.execute(plan.build_request(lens_raw, data_root, model_engine))
    for point in ("layers=0", "layers=1"):
        p = executed.step(point, plan.Plan).result("p_answer")
        assert p.shape == (4,) and ((p > 0) & (p < 1)).all(), point


def test_the_two_engines_project_identically(lens_raw, data_root, model_engine):
    """Both engines gather the same window and push it through the same two
    modules, so — unlike the model's own logits — the lens agrees to the bit."""
    # the document is swept, so the model block comes off the first lowered point
    hooks = HooksEngine.load(Spec.model_validate(sweep.points(lens_raw)[0][1]).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(lens_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(lens_raw, data_root, hooks))
    for point in ("layers=0", "layers=1"):
        a, b = (one.step(point, plan.Plan) for one in (traced, hooked))
        assert torch.equal(a.result("p_answer"), b.result("p_answer")), point
        assert torch.equal(a.result("top1"), b.result("top1")), point


def test_the_lens_is_one_document_with_one_point_per_layer(lens_raw, data_root, model_engine):
    built = plan.build_request(lens_raw, data_root, model_engine)
    assert list(built.steps) == ["layers=0", "layers=1"]  # a one-layer band sweeps as its layer
    text = explain(built)
    assert "at block_output[0] pos={index:-1} via 'identity' as logits" in text
    assert "at block_output[1]" in text

    executed = model_engine.execute(built)
    per_layer = [
        executed.step(point, plan.Plan).result("p_answer")
        for point in built.steps
    ]
    assert not torch.equal(per_layer[0], per_layer[1]), "two layers, two answers"


def test_the_lens_projection_matches_the_head_within_an_ulp(model_engine):
    """The number behind the docstring above, measured directly."""
    m = model_engine.model
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), "attention_mask": torch.ones(2, 4, dtype=torch.long)}
    with m.trace(batch):
        resid = m.layers[1].output
        resid = resid[0] if isinstance(resid, tuple) else resid
        one = m.lm_head(m.ln_final(resid[:, -1:, :])).save()
        whole = m.lm_head(m.ln_final(resid))[:, -1:, :].save()
        true = m.lm_head.output[:, -1:, :].save()
    assert torch.equal(whole, true), "the whole sequence, then sliced: exact"
    # On bippu the one-position projection differs from the head by an ulp
    # (the GEMM has fewer rows); on hakone it does not. Either way it is
    # within one — which way a GEMM rounds is the platform's business.
    assert (one - true).abs().max() < 1e-7
    assert torch.equal(one.argmax(-1), true.argmax(-1))


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["sites"].update(resid={"component": "lm_head"}), "'lm_head' is not the residual stream"),
        (lambda raw: raw["sites"].update(resid={"component": "attention_query", "layers": [0]}), "'attention_query' is not the residual stream"),
        (lambda raw: raw["steps"]["lens"]["reads"]["logits"].update(featurizer="rot"), "cannot also be viewed as logits"),
    ],
    ids=["the head itself", "an interior", "a featurized read"],
)
def test_where_a_logits_view_is_refused(lens_raw, edit, message):
    lens_raw["sites"]["resid"]["layers"] = [0]  # un-sweep so it is one document
    lens_raw["featurizers"] = {"rot": {"kind": "subspace", "k": 4, "parametrization": "cayley"}}
    lens_raw["steps"]["lens"]["reads"]["rot_user"] = {"site": "resid", "pos": -1, "featurizer": "rot"}
    edit(lens_raw)
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(lens_raw)


def test_a_logits_view_cannot_be_written_back():
    raw = json.loads((REPO / "documents" / "v2" / "patching.json").read_text())
    raw["steps"]["counterfactual"]["reads"]["v_cf"]["view"] = "logits"
    with pytest.raises(ValidationError, match="is a logits view, which is vocabulary-wide"):
        Spec.model_validate(raw)


# --------------------------------------------------------------------- #
# every layer, in one forward
# --------------------------------------------------------------------- #

EVERY = REPO / "documents" / "v2" / "logit_lens_all.json"


def test_every_layer_in_one_forward_is_the_sweep_point_by_point(lens_raw, data_root, model_engine):
    """`layers: "all"` is one forward whose read is taken at each layer and
    stacked, the layer axis first; the swept document is one forward per
    layer. They are the same lens, to the bit."""
    every = model_engine.execute(plan.build_request(json.loads(EVERY.read_text()), data_root, model_engine))
    swept = model_engine.execute(plan.build_request(lens_raw, data_root, model_engine))
    assert every.result("p_answer").shape == (2, 4)
    for layer in (0, 1):
        point = swept.step(f"layers={layer}", plan.Plan)
        for name in ("p_answer", "top1"):
            assert torch.equal(every.result(name)[layer], point.result(name)), (name, layer)


def test_a_table_of_every_layer_has_a_row_per_layer_and_example(data_root, model_engine, tmp_path):
    executed = model_engine.execute(plan.build_request(json.loads(EVERY.read_text()), data_root, model_engine))
    executed.write(tmp_path)
    rows = json.loads((tmp_path / "p_answer.json").read_text())
    assert [(row["layer"], row["example_id"]) for row in rows] == [(layer, str(row)) for layer in (0, 1) for row in range(4)]
    assert [row["value"] for row in rows[4:]] == pytest.approx(executed.result("p_answer")[1].tolist())


def test_the_two_engines_agree_on_every_layer(data_root, model_engine):
    raw = json.loads(EVERY.read_text())
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(raw, data_root, hooks))
    assert torch.equal(traced.result("p_answer"), hooked.result("p_answer"))


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda raw: raw["steps"]["lens"].update(interventions={"writes": {"zero": {"site": "resid", "pos": -1, "mechanism": "swap", "operand": 0.0}}}),
         "a write at every layer is as many experiments"),
        (lambda raw: raw["steps"]["lens"]["reads"]["logits"].update(view="raw", featurizer="rot")
         or raw.update(featurizers={"rot": {"kind": "subspace", "k": 4}}),
         "a read at every layer takes no featurizer"),
        (lambda raw: raw["steps"].update(mean={"kind": "reduce", "reduce": "mean", "of": "lens.logits"}),
         "a reduction of it is not implemented"),
        (lambda raw: raw["steps"].update(patched={
            "kind": "forward", "data": "prompts", "field": "input",
            "interventions": {"writes": {"w": {"site": {"component": "block_output", "layers": [0]}, "pos": -1,
                                               "mechanism": "swap", "operand": "lens.logits"}}}}),
         "is a read at every layer; an operand is"),
    ],
    ids=["a write", "a featurizer", "a reduction", "an operand"],
)
def test_what_every_layer_may_not_be(edit, message):
    """One value with the layer axis first is something to score and save.
    A write at every layer is as many experiments, a featurizer is one
    parameter set at one site, and a reduction over rows would first have to
    say which axis the rows are."""
    raw = json.loads(EVERY.read_text())
    edit(raw)
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(raw)
