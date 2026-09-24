# SPDX-License-Identifier: Apache-2.0
"""Tests for the System One API (POST /v1/systemone)."""

from __future__ import annotations

import contextlib
import math
import os
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.api import systemone_models as wire
from omlx.api.systemone import SystemOnePlan, _confidence
from omlx.api.systemone_models import SystemOneRequest
from omlx.engine import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.model_settings import ModelSettings
from omlx.server import ServerState, app

# TypeSafe's published contract, https://api.typesafe.ai/openapi.json
# (OpenAPI 0.2.0): each schema's (properties, required) field names.
TYPESAFE_SCHEMAS = {
    "SystemOneRequest": (
        {"state", "model", "questions"},
        {"state", "model", "questions"},
    ),
    "NoulQuestion": ({"type", "instructions", "criteria"}, {"type"}),
    "NoulCriteria": ({"true", "false"}, set()),
    "ChoiceQuestion": ({"type", "instructions", "criteria"}, {"type", "criteria"}),
    "ScoreQuestion": ({"type", "instructions", "criteria"}, {"type", "criteria"}),
    "SystemOneResponse": (
        {"model", "answers", "usage"},
        {"model", "answers", "usage"},
    ),
    "NoulAnswer": ({"type", "noul"}, {"type", "noul"}),
    "ChoiceAnswer": (
        {"type", "choice", "confidence", "probabilities"},
        {"type", "choice", "confidence", "probabilities"},
    ),
    "ScoreAnswer": (
        {"type", "score", "confidence", "legend", "probabilities"},
        {"type", "score", "confidence", "legend", "probabilities"},
    ),
    "Usage": ({"input_tokens", "output_tokens"}, {"input_tokens", "output_tokens"}),
}

# TypeSafe's quickstart request, verbatim apart from the model.
QUICKSTART = {
    "state": "Hi, I've been trying to connect my Stripe account but keep getting a 403 error.",
    "model": "dgemma",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity",
        },
    },
}


class WordTokenizer:
    """Stand-in tokenizer: each special marker, word, whitespace character
    and punctuation mark is one token, with ids assigned on first sight."""

    SPECIAL = ["<pad>", "<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>"]
    PATTERN = re.compile("|".join(re.escape(s) for s in SPECIAL) + r"|\w+|\s|[^\w\s]")
    pad_token_id = 0

    def __init__(self):
        self.vocab = {s: i for i, s in enumerate(self.SPECIAL)}

    def encode(self, text, add_special_tokens=True):
        return [
            self.vocab.setdefault(p, len(self.vocab))
            for p in self.PATTERN.findall(text)
        ]

    def decode(self, ids):
        pieces = {i: s for s, i in self.vocab.items()}
        return "".join(pieces[i] for i in ids)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def apply_chat_template(
        self, messages, tokenize, add_generation_prompt, enable_thinking
    ):
        assert tokenize and add_generation_prompt and enable_thinking is False
        text = "<bos>" + "".join(
            f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages
        )
        return self.encode(text + "<|turn>model\n")


def labels_first(first: float = 0.7):
    """A fake draw: each slot's first label gets ``first``, the rest share
    the remainder evenly, and no other token is returned."""

    def draw(slots, seed):
        read = []
        for _, labels in slots:
            n = len(labels)
            probs = [first] + [(1 - first) / (n - 1)] * (n - 1)
            read.append({label: math.log(p) for label, p in zip(labels, probs)})
        return read

    return draw


def make_engine(draw=None, *, model_type="diffusion_gemma", diffusion=True):
    """A diffusion engine stand-in whose sessions record each prompt and the
    seeds of every read against it."""
    engine = MagicMock(spec=VLMBatchedEngine)
    engine.is_diffusion_model = diffusion
    engine.model_type = model_type
    engine.tokenizer = WordTokenizer()
    engine.sessions = []
    draw = draw or labels_first()

    @contextlib.asynccontextmanager
    async def diffusion_read_session(prompt_ids):
        session = SimpleNamespace(prompt_ids=prompt_ids, reads=[])
        engine.sessions.append(session)

        async def read(canvas, slots, seeds, top_k):
            session.reads.append(
                SimpleNamespace(canvas=canvas, slots=slots, seeds=seeds)
            )
            return [draw(slots, seed) for seed in seeds]

        yield SimpleNamespace(read=read)

    engine.diffusion_read_session = diffusion_read_session
    return engine


def post(body, engine, *, max_context=None):
    with (
        patch("omlx.server._server_state", ServerState()),
        patch("omlx.server.get_engine_for_model", AsyncMock(return_value=engine)),
        patch("omlx.server.get_max_context_window", return_value=max_context),
        patch("omlx.server.get_server_metrics", return_value=MagicMock()),
    ):
        return TestClient(app).post("/v1/systemone", json=body)


def plan(body, tokenizer=None, **kwargs):
    return SystemOnePlan(
        tokenizer or WordTokenizer(), SystemOneRequest.model_validate(body), **kwargs
    )


# --- the API contract --------------------------------------------------------


@pytest.mark.parametrize(
    "ours, theirs, mode",
    [
        (wire.SystemOneRequest, "SystemOneRequest", "validation"),
        (wire.NoulQuestion, "NoulQuestion", "validation"),
        (wire.NoulCriteria, "NoulCriteria", "validation"),
        (wire.ChoiceQuestion, "ChoiceQuestion", "validation"),
        (wire.ScoreQuestion, "ScoreQuestion", "validation"),
        (wire.SystemOneResponse, "SystemOneResponse", "serialization"),
        (wire.NoulAnswer, "NoulAnswer", "serialization"),
        (wire.ChoiceAnswer, "ChoiceAnswer", "serialization"),
        (wire.ScoreAnswer, "ScoreAnswer", "serialization"),
        (wire.SystemOneUsage, "Usage", "serialization"),
    ],
)
def test_models_match_typesafe_openapi(ours, theirs, mode):
    """Every field TypeSafe defines exists here with the same requiredness,
    and nothing is added."""
    properties, required = TYPESAFE_SCHEMAS[theirs]
    schema = ours.model_json_schema(mode=mode)
    assert set(schema["properties"]) == properties
    assert set(schema.get("required", [])) == required


def test_openapi_constraints_are_enforced():
    """questions has minProperties 1 and score criteria minItems 1."""
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate(dict(QUICKSTART, questions={}))
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate(
            dict(QUICKSTART, questions={"q": {"type": "score", "criteria": []}})
        )


@pytest.mark.parametrize(
    "probs, documented",
    [
        # Worked examples from docs.typesafe.ai (rounded to two places there).
        ([0.84, 0.16, 0.0], 0.76),
        ([0.34, 0.40, 0.02, 0.24], 0.20),
        ([0.04, 0.35, 0.61], 0.42),
        ([0.0, 0.26, 0.0, 0.0, 0.74], 0.67),
        ([0.0, 0.57, 0.43], 0.35),
        ([0.0, 0.89, 0.11], 0.84),
        ([0.0, 0.95, 0.05], 0.92),
        ([0.0, 1.0, 0.0, 0.0, 0.0], 1.0),
    ],
)
def test_confidence_matches_typesafe_examples(probs, documented):
    assert _confidence(probs) == pytest.approx(documented, abs=0.01)


def test_confidence_bounds():
    assert _confidence([0.5, 0.5]) == 0.0
    assert _confidence([1 / 3] * 3) == pytest.approx(0.0)
    assert _confidence([1.0, 0.0]) == 1.0


# --- the endpoint --------------------------------------------------------------


def test_quickstart_answers():
    engine = make_engine()
    response = post(QUICKSTART, engine)
    assert response.status_code == 200, response.text
    body = response.json()
    assert list(body["answers"]) == ["department", "frustration", "is_urgent"]
    assert body["answers"]["department"] == {
        "type": "choice",
        "choice": "billing",
        "confidence": pytest.approx((3 * 0.7 - 1) / 2),
        "probabilities": {
            "billing": pytest.approx(0.7),
            "technical": pytest.approx(0.15),
            "sales": pytest.approx(0.15),
        },
    }
    frustration = body["answers"]["frustration"]
    assert frustration["score"] == pytest.approx(0.15 * 1 + 0.15 * 2)
    assert frustration["legend"] == {
        "0": "Calm, just stating facts",
        "1": "Frustrated but civil",
        "2": "Very angry, strong language",
    }
    assert body["answers"]["is_urgent"] == {"type": "noul", "noul": pytest.approx(0.7)}
    (session,) = engine.sessions
    canvas = session.reads[0].canvas
    rows = canvas.index(engine.tokenizer.convert_tokens_to_ids("<turn|>")) + 1
    assert body["usage"] == {
        "input_tokens": len(session.prompt_ids),
        "output_tokens": rows,
    }
    assert body["model"] == "dgemma"


def test_response_decodes_with_typesafe_sdk():
    sdk = pytest.importorskip("typesafe_sdk")
    response = post(QUICKSTART, make_engine())
    decoded = sdk.SystemOneResponse.model_validate_json(response.content)
    assert decoded.choices["department"].choice == "billing"
    assert decoded.scores["frustration"].legend[0] == "Calm, just stating facts"
    assert decoded.nouls["is_urgent"].noul == pytest.approx(0.7)


def test_a_confident_read_takes_one_draw():
    engine = make_engine(labels_first(0.9999))
    response = post(QUICKSTART, engine)
    assert response.status_code == 200, response.text
    (session,) = engine.sessions
    assert [read.seeds for read in session.reads] == [[42]]


def test_an_uncertain_read_averages_four_draws_over_one_prefill():
    """djev's auto policy: past 0.1 entropy on any slot, three more draws."""

    def draw(slots, seed):
        p = 0.9 if seed == 42 else 0.6
        return [
            {labels[0]: math.log(p), labels[1]: math.log(1 - p)} for _, labels in slots
        ]

    engine = make_engine(draw)
    body = dict(QUICKSTART, questions={"q": {"type": "noul", "instructions": "x"}})
    response = post(body, engine)
    assert response.status_code == 200, response.text
    (session,) = engine.sessions
    assert [read.seeds for read in session.reads] == [
        [42],
        [43, 43 + 7919, 43 + 2 * 7919],
    ]
    assert response.json()["answers"]["q"]["noul"] == pytest.approx((0.9 + 3 * 0.6) / 4)


def test_the_model_setting_turns_rereads_off():
    def uncertain(slots, seed):
        return [
            {labels[0]: math.log(0.6), labels[1]: math.log(0.4)} for _, labels in slots
        ]

    engine = make_engine(uncertain)
    body = dict(QUICKSTART, questions={"q": {"type": "noul", "instructions": "x"}})
    settings = ModelSettings(system_one_rereads_enabled=False)
    with patch("omlx.server.get_model_settings_for_request", return_value=settings):
        response = post(body, engine)
    assert response.status_code == 200, response.text
    (session,) = engine.sessions
    assert [read.seeds for read in session.reads] == [[42]]
    assert response.json()["answers"]["q"]["noul"] == pytest.approx(0.6)


def test_each_read_of_a_request_gets_its_own_noise():
    questions = {f"k{i}": {"type": "noul", "instructions": "x"} for i in range(40)}
    engine = make_engine(labels_first(0.9999))
    assert post(dict(QUICKSTART, questions=questions), engine).status_code == 200
    seeds = [session.reads[0].seeds for session in engine.sessions]
    assert seeds == [[42 + k * 104729] for k in range(len(engine.sessions))]


def test_one_possible_answer_needs_no_read():
    engine = make_engine()
    body = dict(
        QUICKSTART,
        questions={
            "team": {"type": "choice", "criteria": {"billing": None}},
            "level": {"type": "score", "criteria": [{"what": "fine"}]},
        },
    )
    response = post(body, engine)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "model": "dgemma",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "confidence": 1.0,
                "probabilities": {"billing": 1.0},
            },
            "level": {
                "type": "score",
                "score": 0.0,
                "confidence": 1.0,
                "legend": {"0": {"what": "fine"}},
                "probabilities": {"0": 1.0},
            },
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    assert engine.sessions == []


def test_forced_and_read_answers_keep_request_order():
    questions = dict(QUICKSTART["questions"])
    questions = {"only": {"type": "choice", "criteria": {"x": None}}, **questions}
    response = post(dict(QUICKSTART, questions=questions), make_engine())
    assert list(response.json()["answers"]) == list(questions)


@pytest.mark.parametrize(
    "questions, message",
    [
        (
            {"q": {"type": "choice", "criteria": {}}},
            "Choice question must have at least one choice: q",
        ),
        (
            {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}},
            "Too many choices. Must have at most 255 choices.",
        ),
        (
            {"q": {"type": "score", "criteria": [f"l{i}" for i in range(11)]}},
            "Too many score levels. Must have at most 10 levels.",
        ),
    ],
)
def test_unanswerable_questions_are_rejected(questions, message):
    engine = make_engine()
    response = post(dict(QUICKSTART, questions=questions), engine)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == message
    assert response.json()["error"]["param"] == "questions.q.criteria"
    assert engine.sessions == []


@pytest.mark.parametrize(
    "body",
    [
        {"model": "dgemma", "questions": {"a": {"type": "noul"}}},
        {"state": "x", "model": "dgemma", "questions": {}},
        {"state": "x", "model": "dgemma", "questions": {"a": {"type": "nope"}}},
        {"state": "x", "model": "dgemma", "questions": {"a": {"type": "choice"}}},
        {"state": 3, "model": "dgemma", "questions": {"a": {"type": "noul"}}},
    ],
)
def test_malformed_requests_are_422(body):
    assert post(body, make_engine()).status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_type": "gemma4"},
        {"diffusion": False},
    ],
)
def test_models_without_diffusion_reads_are_rejected(overrides):
    response = post(QUICKSTART, make_engine(**overrides))
    assert response.status_code == 400
    assert "does not support System One" in response.json()["error"]["message"]


def test_prompt_longer_than_the_context_window_is_rejected():
    engine = make_engine()
    response = post(QUICKSTART, engine, max_context=20)
    assert response.status_code == 400
    assert (
        "exceeds max context window of 20 tokens" in response.json()["error"]["message"]
    )
    assert engine.sessions == []


def test_structured_state_and_instructions_reach_the_prompt():
    engine = make_engine()
    body = {
        "model": "dgemma",
        "state": {"subject": "Duplicate charge", "message": "Please help."},
        "questions": {
            "dup": {
                "type": "noul",
                "instructions": {"question": "Is `subject` accurate?"},
                "criteria": {"true": "matches the message", "false": ["does not"]},
            }
        },
    }
    assert post(body, engine).status_code == 200
    prompt = engine.tokenizer.decode(engine.sessions[0].prompt_ids)
    assert '{"subject": "Duplicate charge", "message": "Please help."}' in prompt
    assert '{"question": "Is `subject` accurate?"}' in prompt
    assert "yes: matches the message" in prompt
    assert 'no: ["does not"]' in prompt
    assert "dup" not in prompt  # question names never reach the model


def test_missing_instructions_leave_the_question_line_empty():
    engine = make_engine()
    body = dict(QUICKSTART, questions={"q": {"type": "noul"}})
    assert post(body, engine).status_code == 200
    prompt = engine.tokenizer.decode(engine.sessions[0].prompt_ids)
    assert "\nQuestion q1: \n  yes\n  no\n" in prompt


# --- the read layout -----------------------------------------------------------


def test_choice_labels_cover_jevs_limit():
    body = dict(
        QUICKSTART,
        questions={
            "q": {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}}
        },
    )
    (read,) = plan(body)._reads
    (question,) = read.questions
    assert question.labels[:3] == ("A", "B", "C")
    assert len(set(question.labels)) == 255


def test_canvas_layout():
    tokenizer = WordTokenizer()
    (read,) = plan(QUICKSTART, tokenizer)._reads
    head = tokenizer.encode("<|channel>thought\n<channel|>")
    assert read.canvas[: len(head)] == head
    assert len(read.canvas) % 16 == 0 and len(read.canvas) <= 64
    close = read.canvas.index(tokenizer.convert_tokens_to_ids("<turn|>"))
    assert set(read.canvas[close + 1 :]) <= {tokenizer.pad_token_id}
    answer = tokenizer.decode(read.canvas[len(head) : close])
    assert answer == "q1: A\nq2: 1\nq3: yes"
    assert read.rows == close + 1
    for (position, label_ids), labels in zip(
        read.slots, [("A", "B", "C"), ("1", "2", "3"), ("yes", "no")]
    ):
        assert read.canvas[position] == label_ids[0]
        assert [tokenizer.decode([i]) for i in label_ids] == list(labels)


def test_long_question_lists_are_split_into_reads():
    questions = {
        f"k{i}": {"type": "noul", "instructions": f"question {i}"} for i in range(40)
    }
    body = dict(QUICKSTART, questions=questions)
    reads = plan(body)._reads
    assert len(reads) > 1
    assert [q.key for read in reads for q in read.questions] == list(questions)
    assert all(len(read.canvas) <= 64 for read in reads)
    engine = make_engine()
    response = post(body, engine)
    assert response.status_code == 200, response.text
    assert len(engine.sessions) == len(reads)
    assert list(response.json()["answers"]) == list(questions)
    assert response.json()["usage"] == {
        "input_tokens": max(len(read.prompt_ids) for read in reads),
        "output_tokens": sum(read.rows for read in reads),
    }
    for read, session in zip(reads, engine.sessions):
        # Each read's prompt lists only its own questions, as djev's chunks do.
        prompt = engine.tokenizer.decode(session.prompt_ids)
        ids = [f"Question {q.id}:" for q in read.questions]
        assert all(i in prompt for i in ids)
        assert prompt.count("\nQuestion q") == len(read.questions)


def test_indexed_format_with_mixed_types():
    """Past ten questions the template is "q1yes q2A q3 4 ..."; every label
    type must still change exactly one token."""
    questions = {}
    for i in range(12):
        if i % 3 == 0:
            questions[f"n{i}"] = {"type": "noul"}
        elif i % 3 == 1:
            questions[f"s{i}"] = {
                "type": "score",
                "criteria": [f"level {k}" for k in range(10)],
            }
        else:
            questions[f"c{i}"] = {
                "type": "choice",
                "criteria": {f"opt{k}": None for k in range(40)},
            }
    tokenizer = WordTokenizer()
    reads = plan(dict(QUICKSTART, questions=questions), tokenizer)._reads
    assert sum(len(read.questions) for read in reads) == 12
    assert tokenizer.decode(reads[0].canvas).startswith(
        "<|channel>thought\n<channel|>q1yes q2"
    )


def test_score_levels_past_nine_read_as_letters():
    body = dict(
        QUICKSTART, questions={"s": {"type": "score", "criteria": list("abcdefghij")}}
    )
    tokenizer = WordTokenizer()
    (read,) = plan(body, tokenizer)._reads
    assert read.questions[0].labels == tuple("ABCDEFGHIJ")
    assert [tokenizer.decode([i]) for i in read.slots[0][1]] == list("ABCDEFGHIJ")


def test_labels_that_tokenize_to_the_same_id_are_rejected():
    class CollidingTokenizer(WordTokenizer):
        """Reads score label 3 as 2."""

        def encode(self, text, add_special_tokens=True):
            return super().encode(text.replace("q1: 3", "q1: 2"))

    body = dict(
        QUICKSTART, questions={"s": {"type": "score", "criteria": ["a", "b", "c"]}}
    )
    with pytest.raises(InvalidRequestError, match="two labels tokenize to the same id"):
        plan(body, CollidingTokenizer())


def test_labels_that_do_not_share_a_slot_are_rejected():
    class MergingTokenizer(WordTokenizer):
        """Splits label C in two on a second line only, so C passes label
        discovery ("q1: C") but cannot share A's slot in the template."""

        def encode(self, text, add_special_tokens=True):
            return super().encode(text.replace("\nq2: C", "\nq2: C C"))

    questions = {
        "first": {"type": "noul"},
        "q": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}},
    }
    with pytest.raises(
        InvalidRequestError, match="'q': labels do not share one template slot"
    ):
        plan(dict(QUICKSTART, questions=questions), MergingTokenizer())


# --- a real DiffusionGemma checkpoint ---------------------------------------------
#
# OMLX_DIFFUSIONGEMMA_MODEL=/path/to/diffusiongemma-26B-A4B-it-4bit \
#     python -m pytest tests/test_systemone.py -m slow

REAL_MODEL = os.environ.get("OMLX_DIFFUSIONGEMMA_MODEL")
needs_real_model = pytest.mark.skipif(
    not REAL_MODEL,
    reason="set OMLX_DIFFUSIONGEMMA_MODEL to a DiffusionGemma checkpoint",
)


@pytest.fixture(scope="module")
def real_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REAL_MODEL)


@pytest.mark.slow
@needs_real_model
def test_real_tokenizer_gives_every_label_one_token(real_tokenizer):
    body = dict(
        QUICKSTART,
        questions={
            "c": {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}},
            "s": {"type": "score", "criteria": [str(i) for i in range(10)]},
            "n": {"type": "noul"},
        },
    )
    (read,) = plan(body, real_tokenizer)._reads
    assert len(set(read.questions[0].labels)) == 255
    head = real_tokenizer.encode(
        "<|channel>thought\n<channel|>", add_special_tokens=False
    )
    assert read.canvas[: len(head)] == head
    assert real_tokenizer.convert_tokens_to_ids("<turn|>") in read.canvas


@pytest.mark.slow
@needs_real_model
def test_real_tokenizer_keeps_one_slot_per_label_in_long_lists(real_tokenizer):
    questions = {}
    for i in range(40):
        if i % 3 == 0:
            questions[f"k{i}"] = {"type": "noul"}
        elif i % 3 == 1:
            questions[f"k{i}"] = {
                "type": "score",
                "criteria": [f"l{k}" for k in range(10)],
            }
        else:
            questions[f"k{i}"] = {
                "type": "choice",
                "criteria": {f"o{k}": None for k in range(255)},
            }
    reads = plan(dict(QUICKSTART, questions=questions), real_tokenizer)._reads
    assert [q.key for read in reads for q in read.questions] == list(questions)
    assert all(len(read.canvas) <= 64 for read in reads)


@pytest.mark.slow
@needs_real_model
async def test_real_model_answers_the_quickstart():
    engine = VLMBatchedEngine(model_name=REAL_MODEL)
    await engine.start()
    try:
        request = SystemOneRequest.model_validate(QUICKSTART)
        answers = await SystemOnePlan(engine.tokenizer, request).answer(engine)
    finally:
        await engine.stop()
    department = answers["department"]
    assert department.choice == "technical"
    assert sum(department.probabilities.values()) == pytest.approx(1.0)
    assert 0.0 <= department.confidence <= 1.0
    assert 0.0 <= answers["frustration"].score <= 2.0
    assert 0.0 <= answers["is_urgent"].noul <= 1.0
