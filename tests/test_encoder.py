"""`sae` and `linear`: a loaded encoder/decoder pair, and the error term.

A sparse autoencoder explains part of an activation. The write seam has
carried an `err` slot since the first rotation, always zero; this is the
featurizer it was for:

    featurize(x) = (f, x − decode(f))        inverse(f′, err, x) = decode(f′) + err

so an intervention changes what the dictionary explains and returns the rest
untouched. With `features` on a write, the mechanism acts on named latents
and the others pass through.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save_file
from conftest import same_numbers

from causalab_mini import ops, plan
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import featurizer
from causalab_mini.plan import sweep
from causalab_mini.plan.spec_v2 import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "documents" / "v2" / "sae_feature_ablation.json"
BUNDLE = REPO / "documents" / "artifacts" / "sae.safetensors"
AT = ((2,), (2,))


@pytest.fixture
def sae():
    return featurizer.KINDS["sae"](**load_file(str(BUNDLE)))


@pytest.fixture
def raw():
    return sweep.points(json.loads(DOCUMENT.read_text()))[1][1]  # features = [1]


# --------------------------------------------------------------------- #
# the error term
# --------------------------------------------------------------------- #


def test_an_untouched_feature_vector_gives_the_activation_back(sae):
    """However bad the reconstruction is — this dictionary is random."""
    x = torch.randn(2, 1, 16)
    f, err = sae.featurize(x)
    assert err.norm() > 0.1 * x.norm(), "the dictionary does not explain x, and need not"
    assert torch.allclose(sae.inverse(f, err, x), x, atol=1e-6)
    assert (f >= 0).all() and f.shape == (2, 1, 32)


def test_ablating_one_latent_removes_exactly_its_decoder_direction(sae):
    x = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(3))
    latents = sae.featurize(ops.gather(x, AT))[0]
    live = int(latents[0, 0].argmax())

    written = ops.apply_write(x, AT, 0.0, "swap", sae, features=(live,))
    moved = ops.gather(x, AT) - ops.gather(written, AT)
    assert torch.allclose(moved, latents[..., live : live + 1] * sae.W_dec[live], atol=1e-5)
    # a latent that was already off is a write that does nothing
    dead = int((latents.sum(dim=(0, 1)) == 0).nonzero()[0])
    assert torch.allclose(ops.apply_write(x, AT, 0.0, "swap", sae, features=(dead,)), x, atol=1e-6)


def test_features_cut_an_operand_in_the_same_space_the_same_way(sae):
    """Swap two latents from a counterfactual's reading; the other thirty,
    and the error term, stay the base's."""
    base = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(1))
    other = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(2))
    theirs = sae.featurize(ops.gather(other, AT))[0]
    written = ops.apply_write(base, AT, theirs, "swap", sae, features=(4, 9))

    after = sae.featurize(ops.gather(base, AT))[0].clone()
    after[..., [4, 9]] = theirs[..., [4, 9]]
    err = sae.featurize(ops.gather(base, AT))[1]
    assert torch.allclose(ops.gather(written, AT), sae.decode(after) + err, atol=1e-5)


def test_linear_is_the_same_object_without_the_nonlinearity_and_may_be_tied():
    weight = torch.linalg.qr(torch.randn(8, 3))[0]
    tied = featurizer.KINDS["linear"](W_enc=weight)
    x = torch.randn(5, 8)
    f, err = tied.featurize(x)
    assert torch.allclose(f, x @ weight) and (f < 0).any(), "no relu"
    assert torch.equal(tied.W_dec, weight.T)
    assert torch.allclose(tied.inverse(f, err, x), x, atol=1e-6)


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


def test_the_bundle_travels_as_bytes_and_k_comes_from_it(raw, data_root, model_engine):
    built = plan.build_request(raw, data_root, model_engine)
    (spec,) = built.step("featurizers", plan.Featurizers).specs
    assert (spec.kind, spec.d, spec.k, spec.trained) == ("sae", 16, 32, False)
    assert isinstance(spec.weight, bytes) and len(spec.weight) < 8192  # ~4.4 KB of fp32
    assert "on its features [1]" in __import__("causalab_mini.plan.explain", fromlist=["x"]).explain(built)


@pytest.mark.parametrize("remote", [False, "local"])
def test_ablating_a_latent_on_the_model(remote, raw, data_root, model_engine):
    """The bundle's bytes ship with the plan, so `remote="local"` runs it too.

    The claim worth checking on a model: ablating a latent moves exactly the
    rows where that latent was on. Where it was already zero the write
    changes nothing, and the error term has carried everything the
    dictionary does not explain — so those rows are the un-intervened model
    (to rounding: decode(f) + (x − decode(f)) is x only to the last bit)."""
    write = raw["interventions"]["ablate"]["writes"]["ablate"]
    first = model_engine.execute(plan.build_request(raw, data_root, model_engine), remote=remote)
    active = first.result("latents")[:, 0, :] > 0
    assert active.shape == (4, 32)
    counts = active.sum(dim=0)
    mixed = ((counts > 0) & (counts < 4)).nonzero().flatten()
    latent = int(mixed[0]) if len(mixed) else int(counts.argmax())

    write["features"] = [latent]
    ablated = model_engine.execute(plan.build_request(raw, data_root, model_engine), remote=remote)
    raw["interventions"]["ablate"]["models"]["zeroed"]["writes"] = []
    clean = model_engine.execute(plan.build_request(raw, data_root, model_engine))

    moved = (ablated.result("logit_diff") - clean.result("logit_diff")).abs() > 1e-6
    assert torch.equal(moved, active[:, latent]), (latent, moved, active[:, latent])
    assert moved.any()


def test_the_hooks_engine_agrees(raw, data_root, model_engine):
    hooks = HooksEngine.load(Spec.model_validate(raw).model, device_map="cpu")
    assert same_numbers(
        model_engine.execute(plan.build_request(raw, data_root, model_engine)).result("logit_diff"),
        hooks.execute(plan.build_request(raw, data_root, hooks)).result("logit_diff"),
    )


def test_features_of_a_rotation_swap_part_of_a_subspace(data_root, model_engine):
    """Nothing about `features` is an SAE's: two of a rotation's eight
    directions, instead of all of them."""
    das = json.loads((REPO / "documents" / "v2" / "das.json").read_text())
    das["steps"] = {"score": das["steps"]["score"]}
    whole = model_engine.execute(plan.build_request(das, data_root, model_engine)).result("ce")
    das["interventions"]["das"]["writes"]["patch"]["features"] = [0, 1]
    part = model_engine.execute(plan.build_request(das, data_root, model_engine)).result("ce")
    assert not torch.equal(whole, part)


# --------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------- #


def test_what_a_bundle_must_hold(raw, data_root, model_engine, tmp_path):
    tensors = load_file(str(BUNDLE))

    def built_with(**kept):
        path = tmp_path / "one.safetensors"
        save_file({key: value.contiguous() for key, value in kept.items()}, str(path))
        raw["featurizers"]["dictionary"]["file_path"] = str(path)
        return plan.build_request(raw, data_root, model_engine)

    with pytest.raises(plan.PlanError, match=r"a sae bundle holds \['W_enc', 'W_dec'\]"):
        built_with(W_enc=tensors["W_enc"])
    with pytest.raises(plan.PlanError, match=r"W_enc \(8, 32\), not the \(16, 32\)"):
        built_with(W_enc=tensors["W_enc"][:8], W_dec=tensors["W_dec"])
    with pytest.raises(plan.PlanError, match=r"W_dec \(32, 8\)"):
        built_with(W_enc=tensors["W_enc"], W_dec=tensors["W_dec"][:, :8])
    built_with(W_enc=tensors["W_enc"], W_dec=tensors["W_dec"])  # the biases are optional

    raw["featurizers"]["dictionary"]["k"] = 64
    with pytest.raises(plan.PlanError, match=r"not the \((16, 64|64, 16)\)"):
        built_with(**tensors)


def test_what_a_document_may_not_ask_of_a_dictionary(raw, data_root, model_engine):
    bad = json.loads(json.dumps(raw))
    del bad["featurizers"]["dictionary"]["file_path"]
    with pytest.raises(ValidationError, match="loaded from a file"):
        Spec.model_validate(bad)

    bad = json.loads(json.dumps(raw))
    bad["steps"]["keep"] = {"kind": "weights", "names": ["dictionary"]}
    with pytest.raises(ValidationError, match="no fitted weight to publish"):
        Spec.model_validate(bad)

    raw["interventions"]["ablate"]["writes"]["ablate"]["features"] = [32]
    with pytest.raises(plan.PlanError, match="features \\[32\\] of a 32-dimensional feature space"):
        plan.build_request(raw, data_root, model_engine)
