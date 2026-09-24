"""Activation patching, end to end, against values derived independently."""

import copy
import json

import nnsight
import pytest
import torch
from conftest import of_kind

from causalab_mini import cli, ops, plan
from causalab_mini.plan import document
from causalab_mini.engine import NNterpEngine, steps
from causalab_mini.engine.engines.nnterp import engine as nnterp


def build(raw, data_root, engine):
    return plan.build_request(raw, data_root, engine)


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model_engine):
    return build(minimal_raw, data_root, model_engine)


def no_write(raw):
    """The same reads with nothing patched: `logits` in the un-intervened model."""
    raw = copy.deepcopy(raw)
    del raw["method"]["reads"]["v_cf"], raw["method"]["writes"], raw["method"]["intervened_models"]
    raw["method"]["reads"]["logits"]["model"] = "original"
    for entry in raw["method"]["save"]:
        entry["model"] = "original"
    return raw


def identity_write(raw):
    """The same swap, with the operand read off the base input at the same
    address — so the write puts back exactly what was already there."""
    raw = copy.deepcopy(raw)
    raw["method"]["reads"]["v_cf"]["input"] = "base"
    return raw


# --------------------------------------------------------------------- #
# exact values
# --------------------------------------------------------------------- #


def test_a_swap_lands_the_source_read_bit_for_bit(model, model_engine, minimal_plan):
    """The whole mechanism in one assertion: after the write, the activation at
    the address *is* the tensor the other forward read, to the bit."""
    source_forward, patched_forward = (
        steps.located(model_engine, one)[0]
        for one in of_kind(minimal_plan, plan.Forward)
    )
    read = source_forward.taps[0].reads[0]
    tap = patched_forward.taps[0]
    write = tap.writes[0]

    with model.session():
        landed = nnsight.save({})
        with model.trace(nnterp.batch(source_forward)):
            v_cf = ops.gather(nnterp.read(model, tap.address), read.at.positions).clone()
        with model.trace(nnterp.batch(patched_forward)):
            nnterp.write(
                model,
                tap.address,
                ops.apply_write(
                    nnterp.read(model, tap.address),
                    write.at.positions,
                    v_cf,
                    write.mechanism,
                    write.featurizer,
                ),
            )
            landed["after"] = ops.gather(
                nnterp.read(model, tap.address), write.at.positions
            ).clone()
            landed["source"] = v_cf

    assert landed["after"].shape == (4, 1, model.hidden_size)
    assert torch.equal(landed["after"], landed["source"])


def test_the_engines_metrics_equal_hand_computed_ones(model_engine, model, minimal_plan, data_root):
    """The same experiment written by hand with nnterp's own accessors — a
    different way to reach every tensor — down to the last bit."""
    results = model_engine.execute(minimal_plan)

    source, patched = of_kind(minimal_plan, plan.Forward)
    with model.trace(nnterp.batch(source)):
        v_cf = model.layers_output[0][:, -1, :].clone().save()
    with model.trace(nnterp.batch(patched)):
        model.layers_output[0][:, -1, :] = v_cf
        logits = model.logits[:, -1, :].clone().save()

    rows = torch.arange(4)
    cf_answer = torch.tensor([16340, 27822, 28728, 24211])  # " Sunday", " Monday", …
    base_answer = torch.tensor([28728, 24211, 16340, 27822])
    assert torch.equal(
        results.result("logit_diff"), logits[rows, cf_answer] - logits[rows, base_answer]
    )
    assert torch.equal(
        results.result("iia"), (logits.argmax(dim=-1) == cf_answer).to(torch.float32)
    )
    # And, for this random-weight model, `match` is 0 on every row.
    assert results.result("iia").tolist() == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------- #
# the write does what it claims, and only that
# --------------------------------------------------------------------- #


def test_a_swap_moves_the_logits_and_an_identity_write_does_not(minimal_raw, data_root, model_engine):
    clean = model_engine.execute(build(no_write(minimal_raw), data_root, model_engine))
    identity = model_engine.execute(build(identity_write(minimal_raw), data_root, model_engine))
    swapped = model_engine.execute(build(minimal_raw, data_root, model_engine))

    assert torch.equal(identity.result("logit_diff"), clean.result("logit_diff"))
    assert not torch.equal(swapped.result("logit_diff"), clean.result("logit_diff"))


def test_a_write_touches_only_the_position_it_declares(model, model_engine, minimal_plan):
    """`pos: -1` is one token per row. Every other position at the address comes
    out of the patched forward exactly as it went in."""
    source, patched = (
        steps.located(model_engine, one)[0]
        for one in of_kind(minimal_plan, plan.Forward)
    )
    tap = patched.taps[0]
    write = tap.writes[0]
    first_token = ((0,), (2,), (0,), (2,))  # the content start of each row, left-padded

    with model.session():
        seen = nnsight.save({})
        with model.trace(nnterp.batch(source)):
            v_cf = ops.gather(nnterp.read(model, tap.address), write.at.positions).clone()
            seen["source"] = v_cf
        with model.trace(nnterp.batch(patched)):
            seen["clean_first"] = ops.gather(
                nnterp.read(model, tap.address), first_token
            ).clone()
            seen["clean_last"] = ops.gather(
                nnterp.read(model, tap.address), write.at.positions
            ).clone()
        with model.trace(nnterp.batch(patched)):
            nnterp.write(
                model,
                tap.address,
                ops.apply_write(
                    nnterp.read(model, tap.address),
                    write.at.positions,
                    v_cf,
                    write.mechanism,
                    write.featurizer,
                ),
            )
            seen["patched_first"] = ops.gather(
                nnterp.read(model, tap.address), first_token
            ).clone()

    assert torch.equal(seen["clean_first"], seen["patched_first"])
    assert not torch.equal(seen["clean_last"], seen["source"])


# --------------------------------------------------------------------- #
# one code path
# --------------------------------------------------------------------- #


def test_remote_local_runs_the_same_plan_and_gets_the_same_numbers(model_engine, model, minimal_plan):
    """`remote="local"` serializes the session exactly as a remote run would and
    deserializes it with this project's modules hidden, then runs it. Same
    plan, same numbers, no second code path."""
    here = model_engine.execute(minimal_plan)
    shipped = model_engine.execute(minimal_plan, remote="local")
    assert set(here.all_results()) == set(shipped.all_results())
    for name, values in here.all_results().items():
        assert torch.equal(values, shipped.result(name))


# --------------------------------------------------------------------- #
# what leaves the run
# --------------------------------------------------------------------- #


def test_the_run_writes_the_save_manifest_and_nothing_else(model_engine, tmp_path, data_root, model, minimal_plan):
    results = model_engine.execute(minimal_plan)
    written = results.write(tmp_path)

    manifest = ["iia.json", "logit_diff.json"]
    carried = ["document.json", "run.json"]  # the experiment, and what ran it
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(manifest + carried)
    assert sorted(path.name for path in written) == sorted(manifest + carried)

    # the two carried files are enough to re-run and to say what ran
    assert json.loads((tmp_path / "document.json").read_text()) == minimal_plan.source
    ran = json.loads((tmp_path / "run.json").read_text())
    assert ran["engine"] == "NNterpEngine" and ran["remote"] is False
    assert set(ran["versions"]) >= {"torch", "nnsight", "nnterp", "transformers"}
    assert len(ran["causalab_mini"]) == 64

    rows = json.loads((tmp_path / "logit_diff.json").read_text())
    assert [row["example_id"] for row in rows] == ["0", "1", "2", "3"]
    assert {row["unit"] for row in rows} == {"logit"}
    assert {row["estimand_version"] for row in rows} == {"logit_diff/v1"}
    assert {row["produced_by"] for row in rows} == {document.Document.from_json(json.loads((data_root.parent / "minimal_cpu.json").read_text())).digest}
    assert [row["value"] for row in rows] == results.result("logit_diff").tolist()


def test_the_cli_runs_the_document_end_to_end(tmp_path, data_root, capsys):
    exit_code = cli.main(
        [
            "run",
            str(data_root.parent / "minimal_cpu.json"),
            "--data-root",
            str(data_root),
            "--out",
            str(tmp_path),
            "--device-map",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert len(json.loads((tmp_path / "iia.json").read_text())) == 4
    assert len(json.loads((tmp_path / "logit_diff.json").read_text())) == 4
