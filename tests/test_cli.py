"""The command line: what an author can ask, and that none of it needs a GPU.

Every verb but `run` is exercised here on the tiny CPU models with
`weights=False` underneath, which is the point — the loop an agent lives in
(ask what exists, write, validate, explain) must run in seconds on a machine
that will never execute the document.
"""

import json
import pathlib

import pytest

from causalab_mini import cli

REPO = pathlib.Path(__file__).resolve().parents[1]
DAS = str(REPO / "documents" / "v2" / "das.json")
PATCHING = str(REPO / "documents" / "minimal_cpu.json")
DATA = str(REPO / "documents" / "data")
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


def _json(capsys, argv):
    assert cli.main(["--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_schema_is_the_models_own(capsys):
    out = _json(capsys, ["schema"])
    assert out["schema"]["required"] == ["model", "roles", "sites", "intervention", "steps"]
    assert "Fit" in out["schema"]["$defs"]


def test_vocab_is_the_tables_not_a_description_of_them(capsys):
    from causalab_mini import address
    from causalab_mini.ops import intervene

    out = _json(capsys, ["vocab"])
    assert set(out["components"]) == set(address.describe())
    assert out["mechanisms"] == sorted(intervene.MECHANISMS)
    assert out["metric_kinds"]["logit_diff"] == ["a", "b"]
    assert out["components"]["attention_query"]["interior"] is True
    assert out["components"]["block_output"]["width"] == "hidden_size"


def test_model_answers_without_weights(capsys):
    out = _json(capsys, ["model", TINY, "--revision", REVISION])
    assert out["num_layers"] == 2
    assert out["padding_side"] == "left"
    assert out["components"]["block_output"] == {"resolves": True, "width": 16, "op": None}
    assert out["components"]["lm_head"]["width"] == 32000
    assert out["components"]["attention_query"]["op"] == "attention_interface_1"


def test_model_reports_what_an_engine_refuses(capsys):
    """The hooks engine cannot reach an interior, and `model --engine hooks`
    says so per component rather than failing whole."""
    out = _json(capsys, ["model", TINY, "--revision", REVISION, "--engine", "hooks"])
    assert out["components"]["block_output"]["resolves"] is True
    assert out["components"]["attention_query"]["resolves"] is False
    assert "interior" in out["components"]["attention_query"]["why"]


def test_tokens_catches_the_multi_token_answer(capsys):
    """The commonest way a metric column is wrong: an answer that is two
    tokens on this tokenizer. `" Ottawa"` is; `" Paris"` is not."""
    out = _json(capsys, ["tokens", TINY, "--revision", REVISION, " Paris", " Ottawa"])
    assert out["texts"][" Paris"]["one_token"] is True
    assert out["texts"][" Ottawa"]["one_token"] is False
    assert out["texts"][" Ottawa"]["tokens"] == 2


def test_data_describes_a_ref(capsys):
    out = _json(capsys, ["data", "weekdays/train", "--data-root", DATA])
    assert out["rows"] == 4
    assert "cf_answer" in out["columns"]
    assert out["sample"][0]["input"].startswith("If today is")


def test_validate_reads_both_formats(capsys):
    assert _json(capsys, ["validate", DAS])["format"] == "plan-shaped"
    assert _json(capsys, ["validate", PATCHING])["format"] == "protocol"


def test_validate_refuses_with_a_path(tmp_path, capsys):
    broken = json.loads(pathlib.Path(DAS).read_text())
    broken["intervention"]["reads"]["v_cf"]["shuffle"] = {"seed": 1}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(Exception, match="intervention.reads.v_cf.shuffle"):
        cli.main(["validate", str(path)])


def test_explain_prints_the_compiled_plan_without_weights(capsys):
    out = _json(capsys, ["explain", DAS, "--data-root", DATA])
    assert out["steps"] == ["featurizers", "fit", "score", "weights"]
    text = out["text"]
    assert "rot: subspace k=8 d=16" in text  # d derived, never authored
    assert "pos=(10, 10)" in text and "pos=(8, 8)" in text  # one -1, two widths
    assert "saves=['held_out_iia.json']" in text


def test_run_is_the_same_pipeline_with_weights(tmp_path, capsys):
    out = _json(capsys, ["run", PATCHING, "--data-root", DATA, "--out", str(tmp_path), "--device-map", "cpu"])
    assert sorted(pathlib.Path(p).name for p in out["written"]) == [
        "document.json", "iia.json", "logit_diff.json", "run.json"
    ]
    ran = json.loads((tmp_path / "run.json").read_text())
    assert ran["engine"] == "NNterpEngine"


def test_ndif_is_the_nnterp_engine_with_no_local_weights_run_remotely():
    """The third `--engine` is not a third engine. It is the nnterp engine
    loaded as a meta shell — the weights are the server's — and executed with
    `remote=True`. Compiling is the same shell for every engine."""
    engine_class, loading, remote = cli.ENGINES["ndif"]
    assert engine_class is cli.ENGINES["nnterp"][0]
    assert loading == {"dispatch": False} and remote is True
    assert cli.SHAPE_ONLY == {"dispatch": False}
