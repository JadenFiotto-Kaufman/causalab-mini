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
PATCHING = str(REPO / "documents" / "v2" / "patching.json")
DATA = str(REPO / "documents" / "data")
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


def _json(capsys, argv):
    assert cli.main(["--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_schema_is_the_models_own(capsys):
    from causalab_mini.plan.spec import Spec

    out = _json(capsys, ["schema"])
    assert out["schema"] == Spec.model_json_schema()
    assert out["schema"]["required"] == ["model", "steps"]
    assert {"Forward", "Generate", "Reduce", "Fit"} <= set(out["schema"]["$defs"])


def test_vocab_is_the_tables_not_a_description_of_them(capsys):
    from causalab_mini import address
    from causalab_mini.ops import intervene

    out = _json(capsys, ["vocab"])
    assert set(out["step_kinds"]) == {"forward", "generate", "metric", "reduce", "fit"}
    assert out["reductions"] == ["mean", "pca"]
    assert set(out["components"]) == set(address.describe())
    assert out["mechanisms"] == sorted(intervene.MECHANISMS)
    from causalab_mini.ops.metrics import SIGNATURES

    assert set(out["metric_kinds"]) == set(SIGNATURES)
    assert out["metric_kinds"]["logit_diff"]["reads"] == ["of"] and out["metric_kinds"]["logit_diff"]["columns"] == ["a", "b"]
    assert out["metric_kinds"]["kl"]["reads"] == ["of", "against"]
    assert out["metric_kinds"]["cosine"]["logits"] is False
    assert out["metric_kinds"]["top_k"]["params"] == {"k": 5}
    assert out["metric_kinds"]["kl"]["doc"] == SIGNATURES["kl"].doc
    assert out["token_forms"] == ["space_prefixed", "bare", "id"]
    assert out["components"]["attention_query"]["accessor"] == "attention_queries"
    assert out["components"]["block_output"]["width"] == "hidden_size"


def test_model_answers_without_weights(capsys):
    out = _json(capsys, ["model", TINY, "--revision", REVISION])
    assert out["num_layers"] == 2
    assert out["padding_side"] == "left"
    # one band per run of layers that answer alike, so a model whose layers
    # are all the same says so in one entry
    assert out["components"]["block_output"] == [
        {"layers": "0-1", "resolves": True, "width": 16, "inside": False}
    ]
    assert out["components"]["lm_head"] == [
        {"layers": None, "resolves": True, "width": 32000, "inside": False}
    ]
    assert out["components"]["attention_query"][0]["inside"] is True


def test_model_takes_the_model_blocks_own_fields(capsys):
    """Two components exist only under eager attention, and a document says
    which implementation it runs in its `model` block — so this verb takes
    the same field, validated by the same model."""
    plain = _json(capsys, ["model", TINY, "--revision", REVISION])
    assert plain["attn_implementation"] == "sdpa"
    assert plain["components"]["attention_probs"][0]["resolves"] is False
    assert "eager" in plain["components"]["attention_probs"][0]["why"]

    eager = _json(capsys, ["model", TINY, "--revision", REVISION, "--attn-implementation", "eager"])
    assert eager["attn_implementation"] == "eager"
    assert eager["components"]["attention_probs"][0]["resolves"] is True


def test_model_reports_what_an_engine_refuses(capsys):
    """The hooks engine cannot reach inside a forward, and `model --engine hooks`
    says so per component rather than failing whole."""
    out = _json(capsys, ["model", TINY, "--revision", REVISION, "--engine", "hooks"])
    assert out["components"]["block_output"][0]["resolves"] is True
    assert out["components"]["attention_query"][0]["resolves"] is False
    assert "inside a module's forward" in out["components"]["attention_query"][0]["why"]
    assert out["components"]["logits"][0]["resolves"] is True, "it reaches this one now"


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


def test_validate_says_a_document_is_valid(capsys):
    out = _json(capsys, ["validate", DAS])
    assert out["ok"] is True and "fit" in out["steps"]


def test_a_protocol_document_is_refused_by_its_key(tmp_path, capsys):
    """The protocol's `method` shape has no reader any more: a document in it
    is refused as a document, naming the key it does not have."""
    old = {"header": {"protocol_version": "3"}, "model": json.loads(pathlib.Path(DAS).read_text())["model"], "method": {}}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    assert cli.main(["validate", str(path), "--data-root", DATA]) == 1
    assert "method" in capsys.readouterr().err


def test_validate_refuses_with_a_path(tmp_path, capsys):
    """A refusal is a message, not a traceback: the entry point catches this
    package's own error types, prints what they say, and exits 1."""
    broken = json.loads(pathlib.Path(DAS).read_text())
    broken["interventions"]["cf_read"]["reads"]["v_cf"]["shuffle"] = {"seed": 1}
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken))

    assert cli.main(["validate", str(path), "--data-root", DATA]) == 1
    said = capsys.readouterr()
    assert "interventions.cf_read.reads.v_cf.shuffle" in said.err
    assert "Traceback" not in said.err and said.out == ""

    with pytest.raises(Exception, match="interventions.cf_read.reads.v_cf.shuffle"):
        cli.main(["--traceback", "validate", str(path), "--data-root", DATA])


def test_validate_compiles_so_a_misspelled_component_fails_there(tmp_path, capsys):
    """It answers the same question `explain` does, against the same meta
    shell — so a name that only a model can refuse is refused here too,
    rather than validating and failing at the next verb."""
    broken = json.loads(pathlib.Path(DAS).read_text())
    broken["sites"]["target"]["component"] = "block_ouput"
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(broken))

    assert cli.main(["validate", str(path), "--data-root", DATA]) == 1
    assert "block_ouput" in capsys.readouterr().err


def test_explain_prints_the_compiled_plan_without_weights(capsys):
    out = _json(capsys, ["explain", DAS, "--data-root", DATA])
    assert out["steps"] == ["featurizers", "fit", "counterfactual", "patched", "iia", "ce"]
    text = out["text"]
    assert "rot: subspace k=8 d=16" in text  # d derived, never authored
    # the spec, not the rows: every pass of this document reads at the last
    # token, and the integer that is differs between passes of different width
    assert "pos={index:-1}" in text and "pos=(" not in text
    assert "saves=['held_out_iia.json']" in text
    # a step is its document's, and a forward says which rows it runs over
    assert "patched: forward on 'train'  (2 rows" in text


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
