"""Activation patching, end to end, against values derived independently."""

import copy
import json

import nnsight
import pytest
import torch

from causalab_mini import address, cli, document, ops, output, plan, run


def build(raw, data_root, model):
    return plan.build(document.parse(raw), data_root, model)


@pytest.fixture
def minimal_plan(minimal_raw, data_root, model):
    return build(minimal_raw, data_root, model)


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


def test_a_swap_lands_the_source_read_bit_for_bit(model, minimal_plan):
    """The whole mechanism in one assertion: after the write, the activation at
    the address *is* the tensor the other forward read, to the bit."""
    source_forward, patched_forward = minimal_plan.forwards
    read = source_forward.taps[0].reads[0]
    tap = patched_forward.taps[0]
    write = tap.writes[0]

    with model.session():
        landed = nnsight.save({})
        with model.trace(run.batch(source_forward)):
            v_cf = ops.gather(
                address.read(model, tap.path, tap.side), read.positions
            ).clone()
        with model.trace(run.batch(patched_forward)):
            address.write(
                model,
                tap.path,
                tap.side,
                ops.apply_write(
                    address.read(model, tap.path, tap.side),
                    write.positions,
                    v_cf,
                    write.mechanism,
                    write.featurizer,
                ),
            )
            landed["after"] = ops.gather(
                address.read(model, tap.path, tap.side), write.positions
            ).clone()
            landed["source"] = v_cf

    assert landed["after"].shape == (4, model.hidden_size)
    assert torch.equal(landed["after"], landed["source"])


def test_the_engines_metrics_equal_hand_computed_ones(model, minimal_plan, data_root):
    """The same experiment written by hand with nnterp's own accessors — a
    different way to reach every tensor — down to the last bit."""
    results = run.execute(model, minimal_plan)

    source, patched = minimal_plan.forwards
    with model.trace(run.batch(source)):
        v_cf = model.layers_output[0][:, -1, :].clone().save()
    with model.trace(run.batch(patched)):
        model.layers_output[0][:, -1, :] = v_cf
        logits = model.logits[:, -1, :].clone().save()

    rows = torch.arange(4)
    cf_answer = torch.tensor([16340, 27822, 28728, 24211])  # " Sunday", " Monday", …
    base_answer = torch.tensor([28728, 24211, 16340, 27822])
    assert torch.equal(
        results["logit_diff"], logits[rows, cf_answer] - logits[rows, base_answer]
    )
    assert torch.equal(
        results["iia"], (logits.argmax(dim=-1) == cf_answer).to(torch.float32)
    )
    # And, for this random-weight model, `match` is 0 on every row.
    assert results["iia"].tolist() == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------- #
# the write does what it claims, and only that
# --------------------------------------------------------------------- #


def test_a_swap_moves_the_logits_and_an_identity_write_does_not(minimal_raw, data_root, model):
    clean = run.execute(model, build(no_write(minimal_raw), data_root, model))
    identity = run.execute(model, build(identity_write(minimal_raw), data_root, model))
    swapped = run.execute(model, build(minimal_raw, data_root, model))

    assert torch.equal(identity["logit_diff"], clean["logit_diff"])
    assert not torch.equal(swapped["logit_diff"], clean["logit_diff"])


def test_a_write_touches_only_the_position_it_declares(model, minimal_plan):
    """`pos: -1` is one token per row. Every other position at the address comes
    out of the patched forward exactly as it went in."""
    source, patched = minimal_plan.forwards
    tap = patched.taps[0]
    write = tap.writes[0]
    first_token = (0, 2, 0, 2)  # the content start of each row, left-padded

    with model.session():
        seen = nnsight.save({})
        with model.trace(run.batch(source)):
            v_cf = ops.gather(
                address.read(model, tap.path, tap.side), write.positions
            ).clone()
            seen["source"] = v_cf
        with model.trace(run.batch(patched)):
            seen["clean_first"] = ops.gather(
                address.read(model, tap.path, tap.side), first_token
            ).clone()
            seen["clean_last"] = ops.gather(
                address.read(model, tap.path, tap.side), write.positions
            ).clone()
        with model.trace(run.batch(patched)):
            address.write(
                model,
                tap.path,
                tap.side,
                ops.apply_write(
                    address.read(model, tap.path, tap.side),
                    write.positions,
                    v_cf,
                    write.mechanism,
                    write.featurizer,
                ),
            )
            seen["patched_first"] = ops.gather(
                address.read(model, tap.path, tap.side), first_token
            ).clone()

    assert torch.equal(seen["clean_first"], seen["patched_first"])
    assert not torch.equal(seen["clean_last"], seen["source"])


# --------------------------------------------------------------------- #
# one code path
# --------------------------------------------------------------------- #


def test_remote_local_runs_the_same_plan_and_gets_the_same_numbers(model, minimal_plan):
    """`remote="local"` serializes the session exactly as a remote run would and
    deserializes it with this project's modules hidden, then runs it. Same
    plan, same numbers, no second code path."""
    here = run.execute(model, minimal_plan)
    shipped = run.execute(model, minimal_plan, remote="local")
    assert set(here) == set(shipped)
    for name, values in here.items():
        assert torch.equal(values, shipped[name])


# --------------------------------------------------------------------- #
# what leaves the run
# --------------------------------------------------------------------- #


def test_the_run_writes_the_save_manifest_and_nothing_else(tmp_path, data_root, model, minimal_plan):
    results = run.execute(model, minimal_plan)
    written = output.write_results(tmp_path, minimal_plan, results)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["iia.json", "logit_diff.json"]
    assert sorted(path.name for path in written) == ["iia.json", "logit_diff.json"]

    rows = json.loads((tmp_path / "logit_diff.json").read_text())
    assert [row["example_id"] for row in rows] == ["0", "1", "2", "3"]
    assert {row["unit"] for row in rows} == {"logit"}
    assert {row["estimand_version"] for row in rows} == {"logit_diff/v1"}
    assert {row["produced_by"] for row in rows} == {document.parse(json.loads((data_root.parent / "minimal_cpu.json").read_text())).digest}
    assert [row["value"] for row in rows] == results["logit_diff"].tolist()


def test_the_cli_runs_the_document_end_to_end(tmp_path, data_root, capsys):
    exit_code = cli.main(
        [
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
