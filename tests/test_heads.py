"""Per-head sites, and the attention pattern.

A per-head tensor is handed on flat — every head side by side — whether the
model holds it as one head-major axis (`o_proj`'s input) or as two (`the
query`, `z`). A site may name `heads`, and then it *is* those heads: its
width is theirs, a write leaves the others alone, and nothing downstream of
`gather`/`scatter` knows heads exist.

The pattern itself, `attention_probs`, is an operation inside the operation:
the softmax in the eager attention function the attention call dispatches
to. It exists only under eager attention, and the document says so.
"""

import copy
import json
import pathlib

import pytest
import torch
from pydantic import ValidationError
from conftest import model_block, of_kind

from causalab_mini import plan
from causalab_mini.address import AddressError
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import intervene
from causalab_mini.plan.spec import Spec
from causalab_mini.shapes import Selection

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCHING = REPO / "documents" / "v2" / "patching.json"
HEADS = REPO / "documents" / "v2" / "head_patching.json"
KNOCKOUT = REPO / "documents" / "v2" / "attention_knockout.json"
DAS = REPO / "documents" / "v2" / "das.json"


def _at(component, heads=None):
    raw = json.loads(PATCHING.read_text())
    raw["sites"]["target"] = {"component": component, "layers": 0}
    if heads is not None:
        raw["sites"]["target"]["heads"] = heads
    return raw


def _score(raw, data_root, engine, name="logit_diff"):
    return engine.execute(plan.build_request(raw, data_root, engine)).result(name)


@pytest.fixture(scope="module")
def eager_engine():
    from causalab_mini.plan import sweep

    _, point = sweep.points(json.loads(KNOCKOUT.read_text()))[0]
    return NNterpEngine.load(Spec.model_validate(point).model, device_map="cpu")


# --------------------------------------------------------------------- #
# the split, as tensor code
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("held", ["two axes", "flat"])
def test_a_head_slice_is_the_same_however_the_model_holds_the_tensor(held):
    batch, seq, heads, per = 2, 5, 4, 3
    two_axes = torch.arange(batch * seq * heads * per, dtype=torch.float32).reshape(batch, seq, heads, per)
    tensor = two_axes if held == "two axes" else two_axes.reshape(batch, seq, heads * per)
    positions = ((4,), (2,))

    got = intervene.gather(tensor, Selection(positions, heads, (1, 3)))
    assert got.shape == (2, 1, 2 * per)
    assert torch.equal(got[0, 0], torch.cat([two_axes[0, 4, 1], two_axes[0, 4, 3]]))

    written = intervene.scatter(tensor, Selection(positions, heads, (1, 3)), torch.zeros(2, 1, 2 * per))
    view = written.reshape(batch, seq, heads, per)
    assert view[0, 4, [1, 3]].abs().sum() == 0
    assert torch.equal(view[0, 4, [0, 2]], two_axes[0, 4, [0, 2]]), "the other heads are untouched"
    assert torch.equal(view[0, 3], two_axes[0, 3]), "and so is every other position"
    assert written.shape == tensor.shape


def test_a_ragged_window_splits_heads_too():
    tensor = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
    got = intervene.gather(tensor, Selection(((0, 1), (4,)), 4, (2,)))
    assert got.shape == (3, 2)
    assert torch.equal(got[2], tensor[1, 4, 4:6])


# --------------------------------------------------------------------- #
# on the model
# --------------------------------------------------------------------- #


def test_every_head_at_once_is_the_unsliced_patch(data_root, model_engine):
    whole = _score(_at("attention_z"), data_root, model_engine)
    assert torch.equal(_score(_at("attention_z", [0, 1, 2, 3]), data_root, model_engine), whole)
    # and z is o_proj's input: the same tensor, held two ways
    assert torch.equal(_score(_at("attention_premix"), data_root, model_engine), whole)


@pytest.mark.parametrize("head", [0, 3])
def test_one_head_of_z_is_the_same_head_of_the_projections_input(head, data_root, model_engine):
    via_z = _score(_at("attention_z", [head]), data_root, model_engine)
    via_premix = _score(_at("attention_premix", [head]), data_root, model_engine)
    assert torch.equal(via_z, via_premix)
    assert not torch.equal(via_z, _score(_at("attention_z"), data_root, model_engine))


def test_two_sites_of_disjoint_heads_are_one_tap_and_add_up(data_root, model_engine):
    """Heads are a slice of a place, like positions are — not a different
    place. Two sites at one component share its address, hence its tap."""
    raw = _at("attention_z", [0, 1])
    raw["sites"]["rest"] = {"component": "attention_z", "layers": 0, "heads": [2, 3]}
    reads = raw["steps"]["counterfactual"]["reads"]
    reads["v_rest"] = {**reads["v_cf"], "site": "rest"}
    writes = raw["steps"]["patched"]["interventions"]["writes"]
    writes["patch_rest"] = {**writes["patch"], "site": "rest", "operand": "counterfactual.v_rest"}

    built = plan.build_request(raw, data_root, model_engine)
    (tap,) = [t for t in of_kind(built, plan.Forward)[-1].taps if t.writes]
    assert [(w.at.groups, w.at.take) for w in tap.writes] == [(4, (0, 1)), (4, (2, 3))]
    assert torch.equal(
        model_engine.execute(built).result("logit_diff"),
        _score(_at("attention_z"), data_root, model_engine),
    )


def test_the_swept_document_gives_a_different_answer_per_head(data_root, model_engine):
    executed = model_engine.execute(
        plan.build_request(json.loads(HEADS.read_text()), data_root, model_engine)
    )
    assert list(executed.steps) == ["heads=0", "heads=1", "heads=2", "heads=3"]
    scores = [point.result("logit_diff") for point in executed.steps.values()]
    assert len({tuple(one.tolist()) for one in scores}) == 4


def test_the_hooks_engine_patches_a_head_at_the_boundary_it_can_reach(data_root, model_engine):
    raw = _at("attention_premix", [2])
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    assert hooks.heads(hooks.locate("attention_premix", 0)) == 4
    assert torch.allclose(
        _score(raw, data_root, hooks), _score(raw, data_root, model_engine), rtol=0, atol=1e-7
    )


def test_a_rotation_over_two_heads_is_as_wide_as_two_heads(data_root, model_engine):
    """The reason the slice is handed on flat: DAS inside a pair of heads
    needs nothing but a site that names them."""
    raw = json.loads(DAS.read_text())
    raw["sites"]["target"] = {"component": "attention_z", "layers": 0, "heads": [0, 1]}
    raw["featurizers"]["rot"]["k"] = 4
    built = plan.build_request(raw, data_root, model_engine)
    (spec,) = built.step("featurizers", plan.Featurizers).specs
    assert (spec.d, spec.k) == (8, 4)
    assert torch.isfinite(model_engine.execute(built).step("fit", plan.Fit).results["train"]["loss"]).all()

    raw["featurizers"]["rot"]["k"] = 9
    with pytest.raises(plan.PlanError, match="not a subspace of the 8-wide site"):
        plan.build_request(raw, data_root, model_engine)


def test_what_a_headed_site_may_not_say(data_root, model_engine):
    with pytest.raises(ValidationError, match="not a per-head tensor"):
        Spec.model_validate(_at("block_output", [0]))
    with pytest.raises(ValidationError, match="distinct head indices"):
        Spec.model_validate(_at("attention_z", [1, 1]))
    with pytest.raises(plan.PlanError, match="head 4 of a 4-head tensor"):
        plan.build_request(_at("attention_z", [4]), data_root, model_engine)


# --------------------------------------------------------------------- #
# the pattern
# --------------------------------------------------------------------- #


def test_the_pattern_needs_eager_attention_and_says_how_to_get_it(data_root, model_engine):
    raw = json.loads(KNOCKOUT.read_text())
    with pytest.raises(AddressError, match='say "attn_implementation": "eager"'):
        plan.build_request(raw, data_root, model_engine)  # this engine runs sdpa


def test_the_pattern_is_a_distribution_over_keys_per_head(data_root, eager_engine):
    raw = json.loads(KNOCKOUT.read_text())
    raw["sites"]["one_head"]["heads"] = [0]
    executed = eager_engine.execute(plan.build_request(raw, data_root, eager_engine))
    pattern = executed.result("clean.pattern")

    rows, window, flat = pattern.shape
    assert (rows, window) == (4, 1) and flat % 4 == 0
    per_head = pattern.reshape(rows, 4, flat // 4)
    assert torch.allclose(per_head.sum(-1), torch.ones(rows, 4), atol=1e-6)


def test_the_scores_are_what_the_softmax_turns_into_the_pattern(data_root, eager_engine):
    raw = json.loads(KNOCKOUT.read_text())
    raw["sites"]["one_head"]["heads"] = [0]
    raw["sites"]["scores"] = {"component": "attention_scores", "layers": 0}
    reads = raw["steps"]["clean"]["reads"]
    reads["scores"] = {**reads["pattern"], "site": "scores"}
    del raw["steps"]["patched"]["interventions"]
    raw["steps"]["saves"]["clean.scores"] = "scores.safetensors"
    executed = eager_engine.execute(plan.build_request(raw, data_root, eager_engine))

    rows = executed.result("clean.scores").shape[0]
    scores = executed.result("clean.scores").reshape(rows, 4, -1)
    assert torch.equal(scores.softmax(-1), executed.result("clean.pattern").reshape(rows, 4, -1))


def test_knocking_out_a_head_is_zeroing_its_z(data_root, eager_engine):
    """The check that the write landed where it says: a head whose pattern is
    all zero at a position mixes nothing there, so its z is zero — the same
    intervention, stated one operation later."""
    knock = json.loads(KNOCKOUT.read_text())
    knock["sites"]["one_head"]["heads"] = [2]
    via_pattern = _score(knock, data_root, eager_engine)

    via_z = copy.deepcopy(knock)
    via_z["sites"]["one_head"] = {"component": "attention_z", "layers": 0, "heads": [2]}
    assert torch.allclose(via_pattern, _score(via_z, data_root, eager_engine), rtol=0, atol=1e-6)

    unpatched = copy.deepcopy(knock)
    del unpatched["steps"]["patched"]["interventions"]
    assert not torch.equal(via_pattern, _score(unpatched, data_root, eager_engine))


def test_the_pattern_survives_being_shipped(data_root, eager_engine):
    """The nested `.source` is opened inside the trace, so it is opened where
    the run runs: `remote="local"` serializes the session and reads the same
    pattern."""
    raw = json.loads(KNOCKOUT.read_text())
    raw["sites"]["one_head"]["heads"] = [1]
    here = eager_engine.execute(plan.build_request(raw, data_root, eager_engine))
    shipped = eager_engine.execute(plan.build_request(raw, data_root, eager_engine), remote="local")
    for name in ("logit_diff", "clean.pattern"):
        assert torch.equal(here.result(name), shipped.result(name)), name


def test_the_pattern_is_at_the_same_address_on_gpt2(data_root):
    
    spec = model_block(REPO / "documents" / "v2" / "gpt2_reach.json")
    engine = NNterpEngine.load(spec, device_map="cpu", attn_implementation="eager")
    located = engine.locate("attention_probs", 0)
    # nnterp's row, which is one name over five per-family overrides, a sink
    # tag and a validator — where mini had one hardcoded operation
    assert located.accessor == "attention_probabilities"
    assert located.inside, "it is an operation inside the attention, not a module boundary"
    llama = NNterpEngine.load(
        model_block(REPO / "documents" / "v2" / "patching.json"),
        device_map="cpu", attn_implementation="eager",
    )
    other = llama.locate("attention_probs", 0)
    assert (located.accessor, located.io, located.inside) == (other.accessor, other.io, other.inside)


# --------------------------------------------------------------------- #
# units: the same mechanism at the grain of one feature
# --------------------------------------------------------------------- #


def test_units_are_single_features_of_any_site_with_a_width(data_root, model_engine):
    """Heads and units are one rule — view the features as groups, keep some
    — so a neuron patch is a site that names `units`, and compiles to the
    same `Selection` a head patch does."""
    raw = _at("mlp_activation")
    width = model_engine.width(model_engine.locate("mlp_activation", 0))
    raw["sites"]["target"]["units"] = [1, 5]
    built = plan.build_request(raw, data_root, model_engine)
    (tap,) = [t for t in of_kind(built, plan.Forward)[-1].taps if t.writes]
    assert (tap.writes[0].at.groups, tap.writes[0].at.take) == (width, (1, 5))
    assert "features=[1, 5]/" in __import__("causalab_mini.plan.explain", fromlist=["x"]).explain(built)

    some = model_engine.execute(built).result("logit_diff")
    every = _score({**raw, "sites": {**raw["sites"], "target": {**raw["sites"]["target"], "units": list(range(width))}}},
                   data_root, model_engine)
    assert torch.equal(every, _score(_at("mlp_activation"), data_root, model_engine))
    assert not torch.equal(some, every)


def test_the_mlps_width_is_the_down_projections_input_on_both_families(model_engine):
    """Llama says `intermediate_size`; GPT-2 says `n_inner`, and leaves it
    None to mean four times hidden. Checked against the module the activation
    feeds rather than the config it was read from — which is what caught the
    tiny GPT-2 carrying a stray `intermediate_size: 37` beside 128-wide MLPs."""
    
    gpt2 = NNterpEngine.load(model_block(REPO / "documents" / "v2" / "gpt2_reach.json"), dispatch=False)
    assert gpt2.width(gpt2.locate("mlp_activation", 0)) == gpt2.model.mlps[0].c_proj.weight.shape[0] == 128
    llama = model_engine.model.mlps[0].down_proj.in_features
    assert model_engine.width(model_engine.locate("mlp_neuron_output", 0)) == llama


def test_a_gate_over_neurons_is_a_gate_at_a_site_of_units(data_root, model_engine):
    """DBM over a chosen set of neurons: nothing but the site changed."""
    from causalab_mini.plan import sweep

    _, raw = sweep.points(json.loads((REPO / "documents" / "v2" / "dbm.json").read_text()))[0]
    raw["sites"]["target"] = {"component": "mlp_activation", "layers": 0, "units": [0, 2, 4, 6]}
    built = plan.build_request(raw, data_root, model_engine)
    (spec,) = built.step("featurizers", plan.Featurizers).specs
    assert spec.d == 4
    assert model_engine.execute(built).result("mask").shape == (4,)


def test_what_a_site_of_units_may_not_say(data_root, model_engine):
    with pytest.raises(ValidationError, match="heads or units, not both"):
        Spec.model_validate({**_at("attention_z", [0]), "sites": {
            **_at("attention_z", [0])["sites"],
            "target": {"component": "attention_z", "layers": 0, "heads": [0], "units": [1]}}})
    raw = _at("block_output")
    raw["sites"]["target"]["units"] = [16]
    with pytest.raises(plan.PlanError, match="unit 16 of a 16-unit tensor"):
        plan.build_request(raw, data_root, model_engine)


# --------------------------------------------------------------------- #
# swapping a counterfactual's pattern in
# --------------------------------------------------------------------- #

PATTERN = REPO / "documents" / "v2" / "attention_pattern_patching.json"


def _pattern_raw(heads):
    from causalab_mini.plan import sweep

    raw = sweep.points(json.loads(PATTERN.read_text()))[0][1]
    raw["sites"]["one_head"]["heads"] = heads
    return raw


def test_the_patched_head_looks_where_it_did_on_the_counterfactual(data_root, eager_engine):
    """Read the pattern back in the patched forward — a read sees its step's
    writes — for the patched head and for a bystander, which a clean forward
    over the same prompts reads too."""
    raw = _pattern_raw([2])
    raw["sites"]["bystander"] = {"component": "attention_probs", "layers": 0, "heads": [1]}
    clean = {"kind": "forward", "data": "pairs", "field": "input", "reads": {"bystander": {"site": "bystander", "pos": -1}}}
    raw["steps"] = {"clean": clean, **raw["steps"]}
    raw["steps"]["patched"]["reads"]["ours"] = {"site": "one_head", "pos": -1}
    raw["steps"]["patched"]["reads"]["bystander"] = {"site": "bystander", "pos": -1}
    for read in ("counterfactual.their_pattern", "patched.ours", "patched.bystander", "clean.bystander"):
        raw["steps"]["saves"][read] = f"{read}.safetensors"
    got = eager_engine.execute(plan.build_request(raw, data_root, eager_engine)).all_results()

    after = got["patched.ours"]
    assert torch.equal(after, got["counterfactual.their_pattern"]), "head 2 now attends as it did on the counterfactual"
    assert torch.equal(got["patched.bystander"], got["clean.bystander"]), "head 1 was not touched"
    assert torch.allclose(after.sum(-1), torch.ones(after.shape[:2]), atol=1e-6), "still a distribution"


def test_a_pattern_from_the_same_prompt_changes_nothing(data_root, eager_engine):
    raw = _pattern_raw([0, 1, 2, 3])
    patched = _score(raw, data_root, eager_engine)
    raw["steps"]["counterfactual"]["field"] = raw["steps"]["patched"]["field"]
    same = _score(raw, data_root, eager_engine)
    del raw["steps"]["patched"]["interventions"]
    clean = _score(raw, data_root, eager_engine)
    assert torch.allclose(same, clean, rtol=0, atol=1e-6)
    assert not torch.allclose(patched, clean, rtol=0, atol=1e-6)


def test_prompts_laid_out_differently_are_refused_before_any_forward(data_root, eager_engine):
    """A pattern is over *keys*. In `weekdays/train` the base and
    counterfactual days tokenize to different lengths, so key j there is a
    different word — or a pad — here, and every shape would still be right."""
    raw = _pattern_raw([0])
    raw["data"]["pairs"]["path"] = "weekdays/train"
    with pytest.raises(plan.PlanError, match="must tokenize to the same length, row by row"):
        plan.build_request(raw, data_root, eager_engine)


def test_a_pattern_with_no_prompts_behind_it_cannot_be_checked(data_root, eager_engine):
    """A mean of the pattern is a tensor with no prompts behind it, so there
    is no layout to check it against, and a write that swaps it in is
    refused."""
    raw = _pattern_raw([0])
    steps = raw["steps"]
    raw["steps"] = {"counterfactual": steps.pop("counterfactual"),
                    "kept": {"kind": "reduce", "reduce": "mean", "of": "counterfactual.their_pattern"}, **steps}
    steps["patched"]["interventions"]["writes"]["look_there"]["operand"] = "kept"
    with pytest.raises(plan.PlanError, match="has no prompts to check its layout against"):
        plan.build_request(raw, data_root, eager_engine)
