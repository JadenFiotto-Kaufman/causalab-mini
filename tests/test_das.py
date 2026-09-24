"""DAS: the rotation, and the fit that turns it.

The first half is arithmetic — the defining property of a subspace swap, checked
on tensors with no model anywhere near them. The second half runs
`documents/das_cpu_reduction.json` end to end.
"""

import json

import pytest
import safetensors
import torch
from conftest import of_kind, tensors

from causalab_mini import cli, ops, plan
from causalab_mini.ops import featurizer
from causalab_mini.plan import document
from causalab_mini.engine import NNterpEngine


def rotation(d=16, k=8, seed=0):
    return featurizer.Subspace(featurizer.start_weight(d, k, seed))


@pytest.fixture
def das_plan(das_raw, data_root, model_engine):
    return plan.build_request(das_raw, data_root, model_engine)


# --------------------------------------------------------------------- #
# the rotation is orthonormal by construction
# --------------------------------------------------------------------- #


def test_the_cayley_transform_is_orthonormal_for_any_parameter():
    """Not "after training", not "within tolerance of the initialization": for
    every value the optimizer could possibly produce, including a wild one."""
    for k in (1, 8, 16):
        for scale in (0.0, 1e-3, 1.0, 50.0):
            basis = featurizer.cayley(featurizer.start_weight(16, k, seed=3) * scale)
            assert basis.shape == (16, k)
            deviation = (basis.T @ basis - torch.eye(k)).abs().max()
            assert deviation < 1e-5, f"k={k} scale={scale}: |QᵀQ − I| = {deviation}"


def test_the_start_basis_is_the_seeds_and_nothing_elses():
    assert torch.equal(featurizer.start_weight(16, 8, 0), featurizer.start_weight(16, 8, 0))
    assert not torch.equal(featurizer.start_weight(16, 8, 0), featurizer.start_weight(16, 8, 1))


def test_a_subspace_featurizer_satisfies_the_protocol_unchanged():
    assert isinstance(rotation(), ops.Featurizer)


# --------------------------------------------------------------------- #
# the defining property: the complement is untouched
# --------------------------------------------------------------------- #


def test_a_subspace_swap_takes_the_subspace_and_keeps_the_complement():
    """The whole method in one assertion, on the write seam as it already is.

    After the swap the activation's coordinates *in* the rotation are the
    counterfactual's, and its component *orthogonal* to the rotation is still
    the base's. Neither half is bit-identical, because both are reconstructed by
    a projection rather than copied — see FINDINGS §5.1.
    """
    subspace = rotation(k=4)
    basis = subspace.basis
    base = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1))
    # a window per row: (rows, w, width), here the unit window
    counterfactual = torch.randn(3, 1, 16, generator=torch.Generator().manual_seed(2))
    at = ((4,), (4,), (4,))
    first = ((0,), (0,), (0,))

    out = ops.apply_write(base, at, counterfactual @ basis, featurizer=subspace)
    written, before = ops.gather(out, at), ops.gather(base, at)

    # in the subspace: the counterfactual's coordinates, not the base's.
    assert torch.allclose(written @ basis, counterfactual @ basis, atol=1e-5)
    assert not torch.allclose(written @ basis, before @ basis, atol=1e-3)
    # orthogonal to it: the base's component, still.
    complement = lambda x: x - (x @ basis) @ basis.T
    assert torch.allclose(complement(written), complement(before), atol=1e-5)
    # and every other position of the tensor is untouched, bit for bit.
    assert torch.equal(ops.gather(out, first), ops.gather(base, first))


def test_a_full_width_subspace_swap_is_a_full_swap():
    """k = d is the bridge between the two methods: the complement is empty, so
    `inverse` returns the operand and DAS's write *is* activation patching's."""
    subspace = rotation(k=16)
    base = torch.randn(3, 5, 16, generator=torch.Generator().manual_seed(1))
    counterfactual = torch.randn(3, 1, 16, generator=torch.Generator().manual_seed(2))
    at = ((4,), (4,), (4,))

    rotated = ops.apply_write(
        base, at, counterfactual @ subspace.basis, featurizer=subspace
    )
    plain = ops.apply_write(base, at, counterfactual)
    assert torch.allclose(ops.gather(rotated, at), ops.gather(plain, at), atol=1e-5)


def test_the_read_and_the_write_are_one_parameter_set():
    """`featurize` and `inverse` are two methods over one tensor, so a gradient
    arriving through either reaches the same parameter — which is what "the same
    featurizer name at a read and at a write is one rotation" means."""
    subspace = rotation(k=4)
    subspace.weight.requires_grad_(True)
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(7))

    read, _ = subspace.featurize(x)
    read.sum().backward()
    assert subspace.weight.grad is not None
    from_the_read = subspace.weight.grad.clone()
    subspace.weight.grad = None

    subspace.inverse(torch.zeros(2, 4), None, x).sum().backward()
    assert subspace.weight.grad is not None
    assert not torch.equal(subspace.weight.grad, from_the_read)  # two paths, one tensor


# --------------------------------------------------------------------- #
# the plan: what the client decided before the session opened
# --------------------------------------------------------------------- #


def test_the_fit_is_compiled_into_the_plan_rows_and_all(das_plan):
    """Every update the fit will make is in the plan, as a plan: which rows,
    already tokenized, already in the order the seed put them in."""
    fit = das_plan.step("fit", plan.Fit)
    assert das_plan.step("featurizers", plan.Featurizers).specs == (
        plan.FeaturizerOp("rot", "subspace", k=8, d=16, parametrization="cayley", seed=0, trained=True),
    )
    # d is derived from (model, site) and is never authored: this model is 16 wide.
    assert len(fit.epochs) == 10
    # batch.pairs is 16 and the train split is 2 rows, so a batch is the whole
    # split and an epoch is one update.
    assert [len(epoch) for epoch in fit.epochs] == [1] * 10
    assert [len(f.input_ids) for f in of_kind(fit.epochs[0][0], plan.Forward)] == [2, 2]
    assert [len(f.input_ids) for f in of_kind(fit.evaluation, plan.Forward)] == [2, 2]
    assert (fit.objective, fit.params) == (((1.0, "ce"),), ("rot",))
    assert (fit.early_stop, fit.patience, fit.mode) == ("iia", 3, "max")


def test_the_rotation_reaches_the_read_and_the_write_and_not_the_head(das_plan):
    source, patched = of_kind(das_plan, plan.Forward)
    assert [read.featurizer for read in source.taps[0].reads] == ["rot"]
    assert [write.featurizer for write in patched.taps[0].writes] == ["rot"]
    # the metric's read is a *plain* lm_head read; the document refuses any other.
    assert [read.featurizer for read in patched.taps[1].reads] == ["identity"]


def test_a_k_wider_than_the_site_is_a_load_error(das_raw, data_root, model_engine):
    das_raw["method"]["featurizers"]["rot"]["k"] = 17
    with pytest.raises(plan.PlanError, match="not a subspace of the 16-wide site"):
        plan.build_request(das_raw, data_root, model_engine)


def test_an_eval_split_sharing_rows_with_the_fit_is_a_load_error(das_raw, data_root, model_engine):
    das_raw["method"]["train"]["eval"]["split"] = "weekdays/data"  # train + test
    with pytest.raises(plan.PlanError, match="endpoint-disjoint"):
        plan.build_request(das_raw, data_root, model_engine)


def test_the_same_ref_for_both_is_the_visible_train_equals_test_ablation(das_raw, data_root, model_engine):
    das_raw["method"]["train"]["eval"]["split"] = "weekdays/data#train"
    fitted = plan.build_request(das_raw, data_root, model_engine)
    assert of_kind(fitted.step("fit", plan.Fit).evaluation, plan.Forward)[0].input_ids == (
        of_kind(fitted, plan.Forward)[0].input_ids
    )


# --------------------------------------------------------------------- #
# the fit: one session, N executions of the same plan
# --------------------------------------------------------------------- #


@pytest.fixture
def fitted(model_engine, das_plan, model):
    return model_engine.execute(das_plan)


def test_the_fit_reduces_its_own_objective(fitted):
    """Measured at step 0 and at the end, on the thing the document said to
    minimize — `[[1.0, "ce"]]` — and not on anything else."""
    losses = fitted.result("train")["loss"]
    assert losses[-1] < losses[0]
    assert (losses[1:] < losses[:-1]).all(), losses


def test_the_fit_records_every_pass_where_it_happened(fitted):
    """A plan carries its own results, so a fit is not a black box that emits
    two curves: every update and every eval pass is still there, on the step
    that scored it. The flat search deliberately stops before them — six `iia`
    in one document would make every lookup ambiguous — so they are reached by
    saying where."""
    fit = fitted.step("fit", plan.Fit)
    assert sorted(fit.epochs[0][0].all_results()) == ["ce", "iia"]
    assert sorted(fit.evaluation.all_results()) == ["ce", "iia"]
    assert torch.equal(
        fitted.result("iia"), fitted.step("iia", plan.Metric).results["iia"]
    )


def test_the_rotation_is_still_orthonormal_after_training(fitted):
    """The Cayley parametrization's whole claim: no retraction step, no penalty
    term, no projection after the update, and `QᵀQ` is still `I`."""
    basis = featurizer.cayley(fitted.result("rot"))
    assert (basis.T @ basis - torch.eye(8)).abs().max() < 1e-5


def test_the_fit_moved_the_rotation_off_its_start(fitted):
    assert not torch.equal(fitted.result("rot"), featurizer.start_weight(16, 8, 0))


def test_early_stopping_ends_the_fit_before_its_epoch_budget(fitted):
    """`steps.epochs` is 10 and one epoch is one update here, so a fit that ran
    to the budget would leave 10 losses. The watched metric (`iia`, mode max)
    *falls* on every pass — the objective is `ce`, and on this model the two
    disagree — so the first pass is the best and patience 3 ends it at 4."""
    assert len(fitted.result("train")["loss"]) == 4
    evaluated = fitted.result("train")["eval"][:, 0]
    assert (evaluated[1:] < evaluated[:-1]).all(), evaluated


def test_the_same_seed_fits_the_same_rotation_and_another_seed_does_not(das_raw, data_root, model_engine):
    def fit(seed):
        das_raw["method"]["train"]["seed"] = seed
        built = plan.build_request(das_raw, data_root, model_engine)
        return model_engine.execute(built).result("rot")

    assert torch.equal(fit(0), fit(0))
    assert not torch.equal(fit(0), fit(1))


def test_a_full_width_rotation_is_a_plain_swap_end_to_end(das_raw, data_root, model_engine):
    """The bridge between the two methods, through the whole engine: at `k = d`
    the complement is empty, so DAS's write lands the counterfactual activation
    entire — whatever the rotation is, trained or not — and the run's metrics are
    the identity featurizer's to five decimals."""
    das_raw["method"]["featurizers"]["rot"]["k"] = 16
    rotated = model_engine.execute(plan.build_request(das_raw, data_root, model_engine))

    del das_raw["method"]["featurizers"], das_raw["method"]["train"]
    del das_raw["method"]["reads"]["v_cf"]["featurizer"]
    del das_raw["method"]["writes"]["patch"]["featurizer"]
    das_raw["method"]["save"] = das_raw["method"]["save"][:2]
    plain = model_engine.execute(plan.build_request(das_raw, data_root, model_engine))

    for name in ("iia", "ce"):
        assert torch.allclose(rotated.result(name), plain.result(name), atol=1e-5), name


def test_remote_local_fits_the_same_rotation_and_gets_the_same_numbers(model_engine, model, das_plan):
    """The single-path claim, under training: `remote="local"` serializes the
    session — the loop, the optimizer and the backward with it — and runs it with
    this project's modules hidden. Same plan, same numbers."""
    here = model_engine.execute(das_plan)
    shipped = model_engine.execute(das_plan, remote="local")
    here, shipped = tensors(here.all_results()), tensors(shipped.all_results())
    assert set(here) == set(shipped)
    for name, values in here.items():
        assert torch.equal(values, shipped[name]), name


# --------------------------------------------------------------------- #
# what leaves the run
# --------------------------------------------------------------------- #


def test_the_artifact_is_written_stamped_and_reloads_to_the_trained_values(model_engine, 
    tmp_path, data_root, model, das_plan
):
    results = model_engine.execute(das_plan)
    written = results.write(tmp_path)

    # the manifest, plus the two files every run carries: the document that
    # produced it and what ran it
    assert sorted(path.name for path in written) == [
        "ce.json", "document.json", "iia.json", "rot.safetensors", "run.json"
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "ce.json", "document.json", "iia.json", "rot.safetensors", "run.json"
    ]
    with safetensors.safe_open(tmp_path / "rot.safetensors", "pt") as bundle:
        assert bundle.keys() == ["weight"]  # one auto-declared slot, `rot.weight`
        assert torch.equal(bundle.get_tensor("weight"), results.result("rot"))
        stamp = bundle.metadata()
    # "A rotation fitted against bf16 weights is not the same artifact as one
    # fitted against fp32 weights, and the stamp is what says so."
    assert stamp["k"] == "8" and stamp["d"] == "16"
    assert stamp["model_dtype"] == "fp32"
    assert stamp["parametrization"] == "cayley"
    assert stamp["trained_on"] == "weekdays/data#train"
    assert stamp["produced_by"] == document.Document.load(data_root.parent / "das_cpu_reduction.json").digest
    assert [row["unit"] for row in json.loads((tmp_path / "ce.json").read_text())] == ["nat", "nat"]


def test_the_cli_runs_the_das_document_end_to_end(tmp_path, data_root):
    exit_code = cli.main(
        [
            "run",
            str(data_root.parent / "das_cpu_reduction.json"),
            "--data-root",
            str(data_root),
            "--out",
            str(tmp_path),
            "--device-map",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert len(json.loads((tmp_path / "iia.json").read_text())) == 2
    assert (tmp_path / "rot.safetensors").exists()


def test_a_fit_leaves_no_gradient_on_the_model(model_engine, das_plan):
    """FINDINGS §1.15, closed: the loaders freeze the model, so the backward
    stops at the featurizer and the model's weights carry no `.grad` — on a
    real model that is a second copy of it."""
    from causalab_mini.engine.engines.hooks import HooksEngine

    model_engine.execute(das_plan)
    module = model_engine.model._module
    assert not module.training
    assert all(not p.requires_grad and p.grad is None for p in module.parameters())

    hooks = HooksEngine.load(das_plan_model(), device_map="cpu")
    assert all(not p.requires_grad for p in hooks.model.parameters()) and not hooks.model.training


def das_plan_model():
    import json, pathlib

    from causalab_mini.plan.spec import Spec

    return Spec.model_validate(
        json.loads((pathlib.Path(__file__).resolve().parents[1] / "documents" / "v2" / "das.json").read_text())
    ).model
