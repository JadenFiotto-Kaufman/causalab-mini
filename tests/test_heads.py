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

from causalab_mini import plan
from causalab_mini.address import AddressError
from causalab_mini.engine import NNterpEngine
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import intervene
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
PATCHING = REPO / "documents" / "v2" / "patching.json"
HEADS = REPO / "documents" / "v2" / "head_patching.json"
KNOCKOUT = REPO / "documents" / "v2" / "attention_knockout.json"
DAS = REPO / "documents" / "v2" / "das.json"


def _at(component, heads=None):
    raw = json.loads(PATCHING.read_text())
    raw["sites"]["target"] = {"component": component, "layers": [0]}
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

    got = intervene.gather(tensor, positions, 1, (heads, (1, 3)))
    assert got.shape == (2, 1, 2 * per)
    assert torch.equal(got[0, 0], torch.cat([two_axes[0, 4, 1], two_axes[0, 4, 3]]))

    written = intervene.scatter(tensor, positions, torch.zeros(2, 1, 2 * per), 1, (heads, (1, 3)))
    view = written.reshape(batch, seq, heads, per)
    assert view[0, 4, [1, 3]].abs().sum() == 0
    assert torch.equal(view[0, 4, [0, 2]], two_axes[0, 4, [0, 2]]), "the other heads are untouched"
    assert torch.equal(view[0, 3], two_axes[0, 3]), "and so is every other position"
    assert written.shape == tensor.shape


def test_a_ragged_window_splits_heads_too():
    tensor = torch.arange(2 * 5 * 8, dtype=torch.float32).reshape(2, 5, 8)
    got = intervene.gather(tensor, ((0, 1), (4,)), 1, (4, (2,)))
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
    raw["sites"]["rest"] = {"component": "attention_z", "layers": [0], "heads": [2, 3]}
    one = raw["interventions"]["patching"]
    one["reads"]["v_rest"] = {**one["reads"]["v_cf"], "site": "rest"}
    one["writes"]["patch_rest"] = {**one["writes"]["patch"], "site": "rest", "operand": "v_rest"}
    one["models"]["patched"]["writes"] = ["patch", "patch_rest"]

    built = plan.build_request(raw, data_root, model_engine)
    (tap,) = [t for t in built.step("score", plan.Observe).forwards[-1].taps if t.writes]
    assert [w.heads for w in tap.writes] == [(4, (0, 1)), (4, (2, 3))]
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
    raw["sites"]["target"] = {"component": "attention_z", "layers": [0], "heads": [0, 1]}
    raw["featurizers"]["rot"]["k"] = 4
    built = plan.build_request(raw, data_root, model_engine)
    (spec,) = built.step("featurizers", plan.Featurizers).specs
    assert (spec.d, spec.k) == (8, 4)
    assert torch.isfinite(model_engine.execute(built).step("fit", plan.Fit).results["train/loss"]).all()

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
    pattern = executed.step("score", plan.Observe).results["last_pattern"]

    rows, window, flat = pattern.shape
    assert (rows, window) == (4, 1) and flat % 4 == 0
    per_head = pattern.reshape(rows, 4, flat // 4)
    assert torch.allclose(per_head.sum(-1), torch.ones(rows, 4), atol=1e-6)


def test_the_scores_are_what_the_softmax_turns_into_the_pattern(data_root, eager_engine):
    raw = json.loads(KNOCKOUT.read_text())
    raw["sites"]["one_head"]["heads"] = [0]
    raw["sites"]["scores"] = {"component": "attention_scores", "layers": [0]}
    one = raw["interventions"]["knockout"]
    one["reads"]["scores"] = {**one["reads"]["pattern"], "site": "scores"}
    one["models"]["patched"]["writes"] = []
    raw["steps"]["score"]["outputs"]["last_scores"] = {"read": "scores"}
    results = eager_engine.execute(plan.build_request(raw, data_root, eager_engine)).step(
        "score", plan.Observe
    ).results

    rows = results["last_scores"].shape[0]
    scores = results["last_scores"].reshape(rows, 4, -1)
    assert torch.equal(scores.softmax(-1), results["last_pattern"].reshape(rows, 4, -1))


def test_knocking_out_a_head_is_zeroing_its_z(data_root, eager_engine):
    """The check that the write landed where it says: a head whose pattern is
    all zero at a position mixes nothing there, so its z is zero — the same
    intervention, stated one operation later."""
    knock = json.loads(KNOCKOUT.read_text())
    knock["sites"]["one_head"]["heads"] = [2]
    via_pattern = _score(knock, data_root, eager_engine)

    via_z = copy.deepcopy(knock)
    via_z["sites"]["one_head"] = {"component": "attention_z", "layers": [0], "heads": [2]}
    assert torch.allclose(via_pattern, _score(via_z, data_root, eager_engine), rtol=0, atol=1e-6)

    unpatched = copy.deepcopy(knock)
    unpatched["interventions"]["knockout"]["models"]["patched"]["writes"] = []
    assert not torch.equal(via_pattern, _score(unpatched, data_root, eager_engine))


def test_the_pattern_survives_being_shipped(data_root, eager_engine):
    """The nested `.source` is opened inside the trace, so it is opened where
    the run runs: `remote="local"` serializes the session and reads the same
    pattern."""
    raw = json.loads(KNOCKOUT.read_text())
    raw["sites"]["one_head"]["heads"] = [1]
    here = eager_engine.execute(plan.build_request(raw, data_root, eager_engine))
    shipped = eager_engine.execute(plan.build_request(raw, data_root, eager_engine), remote="local")
    for name in ("logit_diff", "last_pattern"):
        assert torch.equal(here.result(name), shipped.result(name)), name


def test_the_pattern_is_at_the_same_address_on_gpt2(data_root):
    from causalab_mini.plan import document

    spec = document.Document.load(REPO / "documents" / "gpt2_cpu.json").model
    engine = NNterpEngine.load(spec, device_map="cpu", attn_implementation="eager")
    located = engine.locate("attention_probs", 0)
    assert (located.op, located.inner) == ("attention_interface_1", "nn_functional_softmax_0")
