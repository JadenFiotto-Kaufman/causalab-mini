"""Loading a document: what is accepted, and what is refused by name."""

import pytest

from causalab_mini import document

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_minimal_cpu_parses_to_the_declarations_notes_describes():
    doc = document.Document.load(REPO / "documents" / "minimal_cpu.json")

    assert doc.model.dtype == "fp32"
    assert doc.sites["target"] == document.SiteSpec("block_output", 0)
    assert doc.sites["lm_head"] == document.SiteSpec("lm_head", None)
    assert doc.reads["v_cf"] == document.ReadSpec("target", -1, "original", "counterfactual")
    assert doc.writes["patch"] == document.WriteSpec("target", -1, "swap", "v_cf")
    assert doc.intervened_models["patched"].writes == ("patch",)
    # The metric named `iia` is a `match` here and a `logit_diff` in das.json:
    # the name is the author's, the arithmetic is the kind's.
    assert doc.metrics["iia"].kind == "match"
    assert doc.metrics["iia"].columns == ("cf_answer",)
    assert doc.metrics["logit_diff"].columns == ("cf_answer", "base_answer")
    assert [entry.file_path for entry in doc.saves] == ["iia.json", "logit_diff.json"]


def test_the_das_document_is_refused_by_name_not_mis_run():
    with pytest.raises(document.DocumentError, match="method.featurizers"):
        document.Document.load(REPO / "documents" / "das_cpu_reduction.json")


def test_the_digest_ignores_authoring_metadata_and_nothing_else(minimal_raw):
    baseline = document.Document.from_json(minimal_raw).digest

    minimal_raw["header"]["description"] = "something else entirely"
    assert document.Document.from_json(minimal_raw).digest == baseline

    minimal_raw["method"]["sites"]["target"]["layers"] = [1]
    assert document.Document.from_json(minimal_raw).digest != baseline


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: raw["header"].update(protocol_version="2"), "protocol_version"),
        (lambda raw: raw["method"]["reads"]["logits"].update(input="counterfactual"), "contradicts"),
        (lambda raw: raw["method"]["save"][0].update(model="original"), "contradicts"),
        (lambda raw: raw["method"]["save"].pop(0), "never saved"),
        (lambda raw: raw["method"]["sites"]["target"].update(component="attention_probs"), "not implemented"),
        (lambda raw: raw["method"]["writes"]["patch"].update(do={"add_scaled": "v_cf"}), "not implemented"),
        (lambda raw: raw["method"]["writes"]["patch"].update(do={"swap": 0.0}), "must be a read name"),
        (lambda raw: raw["method"]["sites"]["target"].update(layers=[0, 1]), "one-element band"),
        (lambda raw: raw["method"]["reads"]["v_cf"].update(pos={"all": True}), "integer position"),
        (lambda raw: raw["method"]["metrics"]["iia"].pop("token_form"), "token_form"),
        (lambda raw: raw["method"]["reads"]["v_cf"].update(featurizer="rot"), "only site/pos/model/input"),
        (lambda raw: raw["method"]["intervened_models"].update(original={"input": "base"}), "reserved"),
        (lambda raw: raw["model"].update(dtype="int8"), "model.dtype"),
        (lambda raw: raw["method"].update(train={}), "method.train"),
    ],
)
def test_a_document_this_slice_cannot_run_is_a_load_error(minimal_raw, mutate, message):
    mutate(minimal_raw)
    with pytest.raises(document.DocumentError, match=message):
        document.Document.from_json(minimal_raw)


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: document.ModelSpec("k", "r", "int8"), "model.dtype"),
        (lambda: document.SiteSpec("attention_probs", 0), "not implemented"),
        (lambda: document.SiteSpec("lm_head", 0), "takes no layers"),
        (lambda: document.SiteSpec("block_output", None), "one layer"),
        (lambda: document.ReadSpec("target", -1, "original", "cf"), "read input"),
        (lambda: document.WriteSpec("target", -1, "add_scaled", "v_cf"), "not implemented"),
        (lambda: document.MetricSpec("match", "logits", "bare", ("cf_answer",)), "token_form"),
        (lambda: document.SaveSpec("iia", "iia.safetensors", "patched", "base"), ".json file"),
    ],
)
def test_a_piece_built_in_code_is_refused_exactly_like_one_built_from_json(build, message):
    """The value checks live on the piece, so they do not depend on having come
    through `from_json`."""
    with pytest.raises(document.DocumentError, match=message):
        build()
