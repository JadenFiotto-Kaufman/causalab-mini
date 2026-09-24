"""A chat prompt, and a position inside one of its turns.

A segment is a run of a row's prompt that the *frame* located, and its name
comes from the same place the run does. In the prompt frame that is a chat
turn named by the message's own `role`; in the continuation it is `eos`.
Nothing enumerates the names: a conversation says them.

What makes it a chat is the data. A role's field may hold a string or a list
of `{"role", "content"}` messages, and a list is rendered through the
checkpoint's own template — the same way an anchor's text is whatever the row
carries under that name. There is no flag on the document.
"""

import json
import pathlib

import pytest
import torch
from pydantic import ValidationError

from conftest import provenance

from causalab_mini import plan
from causalab_mini.data import rows as rows_module, tokens
from causalab_mini.engine.engines.hooks import HooksEngine
from causalab_mini.ops import locate
from causalab_mini.plan.spec_v2 import Spec
from causalab_mini.shapes import Anchor, Where

REPO = pathlib.Path(__file__).resolve().parents[1]
CHAT = REPO / "tests" / "fixtures" / "v2_old" / "chat_turn.json"


@pytest.fixture
def chat_raw():
    return json.loads(CHAT.read_text())


@pytest.fixture
def conversation(data_root):
    return [
        rows_module.field_value(row, "chat")
        for row in rows_module.load(data_root, "weekdays/chat")
    ]


# --------------------------------------------------------------------- #
# what the template rendered, and where each turn went
# --------------------------------------------------------------------- #


def test_a_turn_is_its_own_content_and_none_of_the_template(model, conversation):
    """The span is the message's content, found as itself — so the
    template's control tokens are between the turns and in none of them."""
    texts = [tokens.rendered(model.tokenizer, one, "row") for one in conversation]
    assert texts[0].startswith("<s>[INST]") and "[/INST]" in texts[0]

    ids, mask, _sample = tokens.encode(model.tokenizer, texts, add_special=False)
    spans = tokens.turns(model.tokenizer, ids, mask, conversation)
    _ids, _mask, frame = locate.frame_of_texts(model.tokenizer, texts, add_special=False)

    said = frame.texts[0]
    assert said[slice(*spans[0]["user[1]"])] == "If today is Thursday, tomorrow is"
    assert said[slice(*spans[0]["assistant[0]"])] == "Thursday."
    assert "[INST]" not in said[slice(*spans[0]["user[0]"])]


def test_a_role_that_speaks_twice_is_named_by_which_time(model, conversation):
    """`user[0]` and `user[1]`, the bracket a field path already uses for a
    list. The bare name is recorded only for a role that speaks once, and
    asking for it when two did is `alignment_ambiguous` — the rule a
    variable occurring twice gets, and the same fix: say which."""
    texts = [tokens.rendered(model.tokenizer, one, "row") for one in conversation]
    ids, mask, _ = tokens.encode(model.tokenizer, texts, add_special=False)
    spans = tokens.turns(model.tokenizer, ids, mask, conversation)
    _ids, _mask, frame = locate.frame_of_texts(model.tokenizer, texts, add_special=False)
    frame = _with_segments(frame, spans)

    assert set(spans[0]) == {"user[0]", "user[1]", "assistant[0]", "assistant"}
    windows, reasons = locate.locate(frame, Where(index=-1, scope=Anchor(segment="user")))
    assert windows == ((),) * 4 and set(reasons) == {"alignment_ambiguous"}
    windows, reasons = locate.locate(frame, Where(index=-1, scope=Anchor(segment="assistant")))
    assert set(reasons) == {""}, "one assistant turn, so the bare name is enough"
    windows, reasons = locate.locate(frame, Where(all=True, scope=Anchor(segment="system")))
    assert set(reasons) == {"alignment_missing"}, "a turn nobody took"


def test_a_turn_and_a_variable_compose(model, conversation, data_root):
    """The composition the anchor exists for: this row's entity, searched
    inside the turn that asked the question rather than in the whole
    conversation — which is what makes it unique, since the assistant said
    `Thursday.` first."""
    texts = [tokens.rendered(model.tokenizer, one, "row") for one in conversation]
    ids, mask, _ = tokens.encode(model.tokenizer, texts, add_special=False)
    spans = tokens.turns(model.tokenizer, ids, mask, conversation)
    _ids, _mask, frame = locate.frame_of_texts(model.tokenizer, texts, add_special=False)
    frame = _with_segments(frame, spans)
    anchors = ("Thursday", "Friday", "Saturday", "Sunday")

    scoped = Where(all=True, scope=Anchor(segment="user[1]", variable="entity"))
    windows, reasons = locate.locate(frame, scoped, anchors)
    assert set(reasons) == {""}
    assert [locate.tokens_of(frame, one, row) for row, one in enumerate(windows)][1:] == [
        "' Friday'", "' Saturday'", "' Sunday'"
    ]
    # and unscoped, row 0 says Thursday twice
    loose = Where(all=True, scope=Anchor(variable="entity"))
    assert locate.locate(frame, loose, anchors)[1][0] == "alignment_ambiguous"


def _with_segments(frame, spans):
    from dataclasses import replace

    return replace(frame, segments=spans)


# --------------------------------------------------------------------- #
# the document
# --------------------------------------------------------------------- #


def test_the_document_reads_the_question_and_not_the_template(chat_raw, data_root, model_engine):
    """`documents/v2/chat_turn.json`. The last token of the second user turn
    is the question's own last token; `-1` of the whole prompt is the
    template's closing `[/INST]`, which is what a document without segments
    would have had to read."""
    executed = model_engine.execute(plan.build_request(chat_raw, data_root, model_engine))
    where = executed.step("zeroed", plan.Forward).results["positions"]

    assert where["asked"]["tokens"] == ("' is'",) * 4
    assert where["logits"]["tokens"] == ("']'",) * 4, "the template's own last token"
    assert set(where["asked"]["reason"]) == {""}
    assert executed.result("p_answer").shape == (4,)


def test_the_two_engines_read_the_same_turn(chat_raw, data_root, model_engine):
    hooks = HooksEngine.load(Spec.model_validate(chat_raw).model, device_map="cpu")
    traced = model_engine.execute(plan.build_request(chat_raw, data_root, model_engine))
    hooked = hooks.execute(plan.build_request(chat_raw, data_root, hooks))
    assert provenance(traced) == provenance(hooked)
    assert torch.allclose(traced.result("p_answer"), hooked.result("p_answer"), rtol=0, atol=1e-7)


def test_the_spans_survive_the_serialized_path(chat_raw, data_root, model_engine):
    """They are the client's — only it saw the conversation — so they travel
    in the plan, and `remote="local"` is where that is proved."""
    here = model_engine.execute(plan.build_request(chat_raw, data_root, model_engine))
    shipped = model_engine.execute(
        plan.build_request(chat_raw, data_root, model_engine), remote="local"
    )
    assert provenance(here) == provenance(shipped)
    assert torch.equal(here.result("p_answer"), shipped.result("p_answer"))


# --------------------------------------------------------------------- #
# what is refused
# --------------------------------------------------------------------- #


def test_a_conversation_needs_a_template(chat_raw, data_root):
    """The tiny GPT-2 has none, and a row that is a conversation has no text
    without one."""
    from causalab_mini.engine import NNterpEngine
    from causalab_mini.plan.spec import Model

    raw = json.loads(json.dumps(chat_raw))
    raw["model"] = {"key": "hf-internal-testing/tiny-random-gpt2", "revision": "main", "dtype": "fp32"}
    engine = NNterpEngine.load(Model.model_validate(raw["model"]), dispatch=False)
    with pytest.raises(tokens.TokenError, match="no chat template"):
        plan.build_request(raw, data_root, engine)


def test_a_field_that_is_neither_a_string_nor_a_conversation_is_refused(model):
    with pytest.raises(tokens.TokenError, match="a list of"):
        tokens.rendered(model.tokenizer, [{"role": "user"}], "role 'base'")


@pytest.mark.parametrize(
    "pos, message",
    [
        ({"index": -1, "scope": {"segment": "eos"}}, "run of the generated frame"),
        ({"frame": "generated", "index": -1, "scope": {"segment": "user"}},
         "turn of the prompt; the one run the continuation frame locates is 'eos'"),
    ],
    ids=["eos in the prompt", "a turn in the continuation"],
)
def test_each_frame_locates_its_own_runs(chat_raw, pos, message):
    chat_raw["interventions"]["ask"]["reads"]["asked"]["pos"] = pos
    with pytest.raises(ValidationError, match=message):
        Spec.model_validate(chat_raw)
