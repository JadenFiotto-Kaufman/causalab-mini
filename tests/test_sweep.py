"""One document, three experiments.

The plan tree exists so that "the same experiment three times" needs no new
concept. This is the test of that claim: a swept document should compile to a
root plan with three ordinary children and run on an unchanged engine, and if
anything else had to be added, the nesting did not earn itself.
"""

import json
import pathlib

import pytest
import torch

from causalab_mini import plan
from causalab_mini.plan import document, sweep

REPO = pathlib.Path(__file__).resolve().parents[1]
DOCUMENT = REPO / "documents" / "pos_sweep_cpu.json"


@pytest.fixture
def swept_raw():
    return json.loads(DOCUMENT.read_text())


@pytest.fixture
def swept_plan(swept_raw, data_root, model_engine):
    return plan.build_request(swept_raw, data_root, model_engine)


# --------------------------------------------------------------------- #
# lowering: a sweep is three documents, decided on the client
# --------------------------------------------------------------------- #


def test_a_swept_document_is_n_documents(swept_raw):
    points = sweep.points(swept_raw)
    assert [label for label, _ in points] == ["pos=-1", "pos=-2", "pos=-3"]
    assert [one["method"]["writes"]["patch"]["pos"] for _, one in points] == [-1, -2, -3]
    # Every point is an ordinary document in every other way.
    for _, one in points:
        assert one["method"]["reads"] == swept_raw["method"]["reads"]


def test_an_unswept_document_is_one_unlabelled_point(minimal_raw):
    assert sweep.points(minimal_raw) == (("", minimal_raw),)


def test_each_point_has_its_own_digest(swept_raw):
    """A point is its own experiment, so `produced_by` distinguishes them —
    the three files are not three copies of one result."""
    digests = {document.Document.from_json(one).digest for _, one in sweep.points(swept_raw)}
    assert len(digests) == 3


def test_a_document_still_holding_a_wrapper_is_refused(swept_raw):
    with pytest.raises(document.DocumentError, match="build_request"):
        document.Document.from_json(swept_raw)


@pytest.mark.parametrize(
    "edit, message",
    [
        (
            lambda raw: (
                raw["method"]["writes"]["patch"].update(pos=-1),
                raw["model"].update(dtype={"sweep": ["fp32", "bf16"]}),
            ),
            "may not change the model",
        ),
        (lambda raw: raw["method"]["writes"]["patch"].update(pos={"sweep": {"range": [0, 3]}}), "literal list"),
        (lambda raw: raw["method"]["writes"]["patch"].update(pos={"sweep": []}), "sweep of nothing"),
    ],
    ids=["swept model", "range form", "empty"],
)
def test_the_sweep_forms_this_slice_does_not_run_are_refused_by_name(swept_raw, edit, message):
    edit(swept_raw)
    with pytest.raises(sweep.SweepError, match=message):
        sweep.points(swept_raw)


# --------------------------------------------------------------------- #
# the tree: three children, and nothing else new
# --------------------------------------------------------------------- #


def test_the_root_is_a_plan_with_three_child_plans(swept_plan, minimal_raw, data_root, model_engine):
    assert list(swept_plan.steps) == ["pos=-1", "pos=-2", "pos=-3"]
    for point in swept_plan.steps.values():
        assert isinstance(point, plan.Plan)
        # A point is shaped exactly like the unswept document's whole plan.
        assert list(point.steps) == list(
            plan.build_request(minimal_raw, data_root, model_engine).steps
        )
    # The root itself declares nothing: it is a list of experiments.
    assert swept_plan.saves == () and swept_plan.results == {}


def test_the_points_differ_in_the_swept_field_and_nothing_else(swept_plan):
    positions = [
        point.step("observe", plan.Observe).forwards[1].taps[0].writes[0].at.positions
        for point in swept_plan.steps.values()
    ]
    # Four rows, one position each. The rows are left-padded to a common
    # width of 11 here, so -1/-2/-3 are the absolute indices 10/9/8 — which is
    # the point of resolving a position on the client, against the padding the
    # tokenizer actually produced.
    assert positions == [((10,),) * 4, ((9,),) * 4, ((8,),) * 4]
    reads = {
        point.step("observe", plan.Observe).forwards[0].taps[0].reads[0].at.positions
        for point in swept_plan.steps.values()
    }
    assert len(reads) == 1


def test_a_swept_plan_is_still_pure_data(swept_plan):
    """Nesting did not make a plan less shippable: it is still strings and
    integers all the way down, and it still pickles."""
    import pickle

    assert pickle.loads(pickle.dumps(swept_plan)) == swept_plan


# --------------------------------------------------------------------- #
# running it: an unchanged engine, and results per point
# --------------------------------------------------------------------- #


def test_the_engine_runs_the_tree_it_was_given_without_knowing_about_sweeps(
    swept_plan, model_engine
):
    executed = model_engine.execute(swept_plan)

    for label, point in executed.steps.items():
        assert sorted(point.all_results()) == ["iia", "logit_diff"], label
        assert point.result("iia").shape == (4,)
    # Three experiments, three answers: patching at three positions does not
    # give one number three times.
    scores = [point.result("logit_diff") for point in executed.steps.values()]
    assert not torch.equal(scores[0], scores[1])


def test_a_flat_lookup_across_points_is_refused_rather_than_guessed(swept_plan, model_engine):
    """`iia` means three different things here, so asking the root for it is
    an error. Asking a point is not."""
    executed = model_engine.execute(swept_plan)
    with pytest.raises(plan.PlanError, match="3 in this plan"):
        executed.result("iia")
    assert executed.steps["pos=-2"].result("iia").shape == (4,)


def test_each_point_writes_its_own_files_in_its_own_directory(swept_plan, model_engine, tmp_path):
    """A plan's path in the tree is its path on disk, which is the whole of
    what nesting cost the writer."""
    written = model_engine.execute(swept_plan).write(tmp_path)

    # each point carries the lowered document it is; the root carries the
    # swept one and the run record
    assert sorted(str(path.relative_to(tmp_path)) for path in written) == [
        "document.json",
        "pos=-1/document.json",
        "pos=-1/iia.json",
        "pos=-1/logit_diff.json",
        "pos=-2/document.json",
        "pos=-2/iia.json",
        "pos=-2/logit_diff.json",
        "pos=-3/document.json",
        "pos=-3/iia.json",
        "pos=-3/logit_diff.json",
        "run.json",
    ]
    point = json.loads((tmp_path / "pos=-2" / "document.json").read_text())
    assert point["method"]["writes"]["patch"]["pos"] == -2
    rows = json.loads((tmp_path / "pos=-1" / "iia.json").read_text())
    assert [row["example_id"] for row in rows] == ["0", "1", "2", "3"]


def test_two_swept_fields_are_their_cross_product(swept_raw):
    """Points in the order the fields appear, labelled by both coordinates —
    the shape of the one shipped causalab pipeline (k x seed)."""
    swept_raw["method"]["reads"]["v_cf"]["pos"] = {"sweep": [-1, -2]}
    points = sweep.points(swept_raw)
    assert [label for label, _ in points] == [
        "pos=-1,pos=-1", "pos=-1,pos=-2", "pos=-1,pos=-3",
        "pos=-2,pos=-1", "pos=-2,pos=-2", "pos=-2,pos=-3",
    ]
    read, write = points[1][1]["method"]["reads"]["v_cf"]["pos"], points[1][1]["method"]["writes"]["patch"]["pos"]
    assert (read, write) == (-1, -2)
