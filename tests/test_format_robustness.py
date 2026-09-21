"""A document is data an agent wrote: it must not quietly mean something else.

Found by exploratory testing (TESTING.md batches 1–2): coerced types ran a
different experiment, colliding sweep labels dropped points, a `column`
position matched inside a word, a metric could index a neuron by token id,
a save could leave `--out` or overwrite the run's own record.
"""

import copy
import json
import pathlib

import pytest
import torch
from pydantic import ValidationError

from causalab_mini import cli, plan
from causalab_mini.data import encoding, rows as rows_module
from causalab_mini.plan import sweep
from causalab_mini.plan.spec import Spec

REPO = pathlib.Path(__file__).resolve().parents[1]
V2 = REPO / "documents" / "v2"


def _doc(name="patching.json"):
    return sweep.points(json.loads((V2 / name).read_text()))[0][1]


# --------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw["interventions"]["patching"]["reads"]["v_cf"].update(pos=True),
        lambda raw: raw["interventions"]["patching"]["reads"]["v_cf"].update(pos="-1"),
        lambda raw: raw["interventions"]["patching"]["reads"]["v_cf"].update(pos=1.0),
        lambda raw: raw["sites"]["target"].update(layers=[True]),
        lambda raw: raw["sites"]["target"].update(layers=["0"]),
        lambda raw: raw["interventions"]["patching"].update(decode="3"),
    ],
    ids=["pos true", "pos '-1'", "pos 1.0", "layers [true]", "layers ['0']", "decode '3'"],
)
def test_an_integer_must_really_be_one(edit):
    """`"pos": true` validated, became position 1, and ran a different
    experiment from the `-1` its author meant."""
    raw = _doc()
    edit(raw)
    with pytest.raises(ValidationError):
        Spec.model_validate(raw)


def test_a_number_where_a_float_is_wanted_is_still_fine():
    raw = _doc("steer_renormalize.json")
    raw["interventions"]["steer"]["writes"]["steer"]["params"]["scale"] = 4
    Spec.model_validate(raw)
    raw["interventions"]["steer"]["writes"]["steer"]["params"]["scale"] = float("inf")
    with pytest.raises(ValidationError, match="finite"):
        Spec.model_validate(raw)


# --------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pos, why",
    [
        ({"span": "ab"}, "a span is"), ({"span": [1]}, "a span is"), ({"span": [3, 1]}, "empty on every row"),
        ({"all": False}, 'spelled {"all": true}'), ({"last": 0}, "at least 1"), ({"last": -1}, "at least 1"),
        ({"first": 2}, "not a position form"), ({"column": ""}, "non-empty string"), ({"index": 1, "last": 2}, "not a position form"),
    ],
)
def test_a_malformed_position_is_refused_by_name_and_told_the_forms(pos, why):
    """These were a TypeError from inside the resolver, `{"all": false}`
    meaning all, or "falls outside the row's content" for an empty window."""
    with pytest.raises(encoding.EncodingError, match=why) as refusal:
        encoding.check_form(pos)
    assert '{"span": [a, b]}' in str(refusal.value), "the message lists every form"
    raw = _doc()
    raw["interventions"]["patching"]["reads"]["v_cf"]["pos"] = pos
    with pytest.raises(ValidationError):
        Spec.model_validate(raw)


def test_a_column_position_names_one_place(model_engine):
    batch = encoding.encode(model_engine.tokenizer, ["Thursday is a day. If today is Thursday, tomorrow is"] * 3)
    with pytest.raises(encoding.EncodingError, match="occurs 2 times"):
        encoding.positions(batch, {"column": "x"}, ["Thursday"] * 3)
    found = encoding.positions(batch, {"column": "x"}, ["", "day", "tomorrow"])
    assert found[0] == (), "an empty value is an excluded row, not the first token"
    piece = model_engine.tokenizer.decode([batch.input_ids[1][i] for i in found[1]]).strip()
    assert piece == "day", "the word, not the tail of 'Thursday'"
    assert found[2] != ()


def test_a_ragged_read_that_finds_nothing_anywhere_still_gathers():
    """`[]` is a float tensor to torch; an all-empty window used to die on
    'tensors used as indices must be long'."""
    from causalab_mini.ops import intervene

    assert intervene.gather(torch.ones(2, 4, 3), ((), ())).shape == (2, 0, 3)
    assert intervene.gather(torch.ones(2, 4, 3), ((), (1,))).shape == (1, 3)


def test_a_position_found_in_no_row_at_all_is_a_typo(data_root, model_engine):
    raw = _doc("entity_mean_ablation.json")
    for one in raw["interventions"].values():
        for read in one["reads"].values():
            if isinstance(read["pos"], dict) and "column" in read["pos"]:
                read["pos"] = {"column": "base_answer"}  # a real column, whose text no prompt contains
    with pytest.raises(plan.PlanError, match="in none of these"):
        plan.build_request(raw, data_root, model_engine)


# --------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "edit",
    [
        lambda raw: raw["sites"]["head"].update(component="mlp_activation", layers=[0]),
        lambda raw: raw["sites"]["head"].update(units=[0, 1]),
    ],
    ids=["a neuron site", "units of the head"],
)
def test_a_metric_reads_the_vocabulary(edit):
    """A metric indexes its read by token id. Anywhere but the vocabulary
    that is a neuron's activation called a logit — an IndexError on a tiny
    model, a silent wrong number on a real one."""
    raw = _doc()
    edit(raw)
    with pytest.raises(ValidationError, match="not over the vocabulary"):
        Spec.model_validate(raw)


def test_a_wide_head_read_under_decode_is_refused():
    raw = _doc("generate_probe.json")
    one = next(iter(raw["interventions"].values()))
    head_site = next(name for name, site in raw["sites"].items() if site["component"] == "lm_head")
    one["reads"]["wide"] = {"site": head_site, "pos": {"last": 2}, "input": "base"}
    with pytest.raises(ValidationError, match="one position per pass"):
        Spec.model_validate(raw)


# --------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------- #


def test_colliding_sweep_labels_are_refused_not_dropped():
    """Three points, two labels, two plans: the third vanished. A six-axis
    case lost 4080 of 4096."""
    raw = _doc("zero_ablation.json")
    raw["sites"]["target"]["layers"] = {"sweep": [[0], [1], [0]]}
    with pytest.raises(sweep.SweepError, match="3 points but only 2 distinct labels"):
        sweep.points(raw)


def test_a_sweep_is_bounded_before_anything_is_copied():
    """`{"range": [0, 1e9]}` took a process past 110 GB. This must return at once."""
    raw = _doc()
    raw["sites"]["target"]["layers"] = [{"sweep": {"range": [0, 10**9]}}]
    with pytest.raises(sweep.SweepError, match="at most"):
        sweep.points(raw)
    raw["sites"]["target"]["layers"] = [{"sweep": {"range": [0, 100]}}]
    raw["interventions"]["patching"]["decode"] = {"sweep": {"range": [0, 100]}}
    with pytest.raises(sweep.SweepError, match="10000 points"):
        sweep.points(raw)


def test_a_label_is_one_directory_name():
    raw = _doc()
    raw["steps"]["score"]["rows"]["base"] = {"sweep": ["weekdays/train", "weekdays/data"]}
    assert [label for label, _ in sweep.points(raw)] == ["base=weekdays_train", "base=weekdays_data"]


# --------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["../escaped.json", "/tmp/escaped-by-causalab-mini.json", "a/../../escaped.json"])
def test_a_save_stays_inside_out(bad, data_root, model_engine, tmp_path):
    raw = _doc()
    raw["steps"]["score"]["saves"] = [{"value": "iia", "file_path": bad}]
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    with pytest.raises(plan.PlanError, match="outside the output directory"):
        executed.write(tmp_path / "out")
    assert not list(tmp_path.rglob("escaped*.json")), "and nothing was written on the way to refusing"


@pytest.mark.parametrize("paths", [("document.json",), ("run.json",), ("same.json", "same.json")])
def test_one_place_holds_one_thing(paths, data_root, model_engine, tmp_path):
    """A save named `document.json` used to replace the run's record with a
    metric table; two saves to one path kept the second."""
    raw = _doc()
    values = ["iia", "logit_diff"]
    raw["steps"]["score"]["saves"] = [{"value": values[i], "file_path": path} for i, path in enumerate(paths)]
    executed = model_engine.execute(plan.build_request(raw, data_root, model_engine))
    with pytest.raises(plan.PlanError, match="One place holds one thing"):
        executed.write(tmp_path)


def test_a_dataset_ref_stays_inside_the_data_root_and_says_what_is_there(data_root, tmp_path):
    with pytest.raises(rows_module.DataError, match="outside the data root"):
        rows_module.load(data_root, "../../etc/passwd")
    with pytest.raises(rows_module.DataError, match="weekdays/train"):
        rows_module.load(data_root, "weekdays")  # a directory, which is the natural first guess
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "empty.json").write_text("[]")
    with pytest.raises(rows_module.DataError, match="has no rows"):
        rows_module.load(tmp_path, "d/empty")
    (tmp_path / "d" / "twice.json").write_text(json.dumps([{"example_id": "a"}, {"example_id": "a"}]))
    with pytest.raises(rows_module.DataError, match="repeat"):
        rows_module.load(tmp_path, "d/twice")


def test_a_field_index_past_the_end_is_a_data_error():
    with pytest.raises(rows_module.DataError, match="index 3 of column 'counterfactual_inputs'"):
        rows_module.field_text({"counterfactual_inputs": ["only one"]}, "counterfactual_inputs[3]")


# --------------------------------------------------------------------- #
# the two formats, and the CLI's edges
# --------------------------------------------------------------------- #


def test_a_misspelt_steps_is_not_sent_to_the_other_format(data_root, model_engine):
    """It used to answer "missing top-level group 'data'" — a group of a
    format the author was not writing."""
    raw = _doc()
    raw["step"] = raw.pop("steps")
    with pytest.raises(plan.PlanError, match="neither `steps`") as refusal:
        plan.build_request(raw, data_root, model_engine)
    assert "'step'" in str(refusal.value), "and it shows the keys it did find"


def test_duplicate_json_keys_are_refused(tmp_path):
    path = tmp_path / "twice.json"
    path.write_text('{"model": 1, "model": 2}')
    with pytest.raises(plan.PlanError, match=r"\['model'\] appear twice"):
        cli._read(str(path))


def test_a_batch_size_is_positive(data_root, model_engine):
    with pytest.raises(SystemExit):
        cli.main(["run", str(V2 / "patching.json"), "--batch-size", "-1"])
    with pytest.raises(ValueError, match="positive number of rows"):
        model_engine.execute(plan.build_request(_doc(), data_root, model_engine), batch_size=-1)


def test_a_step_needs_rows_only_for_the_roles_it_runs_on():
    """A baseline that reads only `base` used to need a dataset for every
    declared role."""
    raw = _doc()
    one = raw["interventions"]["patching"]
    raw["interventions"]["clean"] = {
        "reads": {"logits": {**one["reads"]["logits"], "model": "original"}},
        "metrics": copy.deepcopy(one["metrics"]),
    }
    raw["steps"]["score"]["intervention"] = "patching"
    raw["steps"]["clean"] = {"kind": "observe", "intervention": "clean", "rows": {"base": raw["steps"]["score"]["rows"]["base"]}}
    Spec.model_validate(raw)
    del raw["steps"]["score"]["rows"]["counterfactual"]
    with pytest.raises(ValidationError, match="no rows for role 'counterfactual', which its intervention runs on"):
        Spec.model_validate(raw)
