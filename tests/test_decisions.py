# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI Decisions API readouts behind /v1/decisions."""

import asyncio
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from pydantic import ValidationError  # noqa: E402

from omlx.api import decisions  # noqa: E402
from omlx.api.decisions_models import DecisionRequest  # noqa: E402

VOCAB = 256
# Next-token logit of every non-letter token in the fake engine.
BACKGROUND = -30.0


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class _FakeChatEngine:
    """Chat engine whose next-token logits come from ``scores(prompt)``."""

    def __init__(self, scores, background=BACKGROUND):
        self.tokenizer = _CharTokenizer()
        self.scores = scores
        self.background = background
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        content = messages[0]["content"]
        text = (
            content
            if isinstance(content, str)
            else "".join(p.get("text", "") for p in content)
        )
        letters = re.findall(r"^([A-Z])\. ", text, flags=re.M)
        logits = np.full(VOCAB, self.background, dtype=np.float32)
        for letter, value in zip(letters, self.scores(text, letters), strict=True):
            logits[ord(letter)] = value
        kwargs["logits_observer"](None, mx.array(logits)[None])
        return SimpleNamespace(prompt_tokens=len(text), cached_tokens=1)


def _request(**body):
    return DecisionRequest.model_validate(
        {"model": "m", "input": "Checkout is down.", **body}
    )


def _run(engine, request, multimodal=False):
    return asyncio.run(decisions.decide(engine, request, multimodal))


def test_capture_reads_option_logprobs_and_keeps_logits():
    capture = decisions.OptionLogitsCapture([[1, 2], [3]])
    logits = mx.array([[0.0, 1.0, 1.0, 2.0]])
    out = capture(None, logits)
    assert out is logits
    total = math.log(sum(math.exp(v) for v in (0.0, 1.0, 1.0, 2.0)))
    assert capture.logprobs == pytest.approx(
        [math.log(2 * math.e) - total, 2.0 - total], rel=1e-5
    )
    capture(None, mx.array([[9.0, 0.0, 0.0, 0.0]]))
    assert capture.logprobs[1] == pytest.approx(2.0 - total, rel=1e-5)


def test_letter_ids_merge_spellings_and_reject_collisions():
    class Spaced:
        def encode(self, text, add_special_tokens=False):
            return [ord(text.strip()) + (1000 if text.startswith(" ") else 0)]

    ids = decisions.letter_token_ids(Spaced())
    assert ids[0] == [ord("A"), 1000 + ord("A")]

    class Collapsed:
        def encode(self, text, add_special_tokens=False):
            return [7]

    with pytest.raises(decisions.DecisionsRequestError):
        decisions.letter_token_ids(Collapsed())


def test_choice_answer_probabilities_and_confidence():
    engine = _FakeChatEngine(lambda text, letters: [2.0, 0.0, math.log(0.5)])
    request = _request(
        questions=[
            {
                "type": "choice",
                "name": "route",
                "instructions": "Which team?",
                "choices": [
                    {"value": "billing", "description": "Payments"},
                    "account",
                    {"value": "other"},
                ],
            }
        ]
    )
    result = _run(engine, request)
    (answer,) = result["answers"]
    weights = np.exp([2.0, 0.0, math.log(0.5)])
    probs = weights / weights.sum()
    assert answer["type"] == "choice" and answer["name"] == "route"
    assert answer["choice"] == "billing"
    assert [p["value"] for p in answer["probabilities"]] == [
        "billing",
        "account",
        "other",
    ]
    assert [p["probability"] for p in answer["probabilities"]] == pytest.approx(
        probs, abs=1e-4
    )
    assert answer["confidence"] == pytest.approx((probs[0] - 1 / 3) / (2 / 3), abs=1e-4)
    text = engine.calls[0]["messages"][0]["content"]
    assert text.startswith("Checkout is down.\n\nQuestion: Which team?")
    assert "A. billing: Payments\nB. account\nC. other" in text
    assert engine.calls[0]["messages"][1] == {
        "role": "assistant",
        "content": "Answer:",
    }
    assert engine.calls[0]["is_partial"] is True
    assert engine.calls[0]["max_tokens"] == 1
    assert engine.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_predicate_and_score_answers():
    def scores(text, letters):
        if "Yes" in text:
            return [math.log(0.8), math.log(0.2)]
        return [math.log(0.1), math.log(0.7), math.log(0.2)]

    request = _request(
        questions=[
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
            {
                "type": "score",
                "name": "severity",
                "instructions": "How severe?",
                "levels": [
                    {"label": "low"},
                    {"label": "medium", "description": "Some users"},
                    "high",
                ],
            },
        ]
    )
    result = _run(_FakeChatEngine(scores), request)
    predicate, score = result["answers"]
    assert predicate == {"type": "predicate", "name": "urgent", "probability": 0.8}
    assert score["score"] == pytest.approx(1.1, abs=1e-4)
    assert score["probabilities"] == [
        {"value": 0, "label": "low", "probability": 0.1},
        {"value": 1, "label": "medium", "probability": 0.7},
        {"value": 2, "label": "high", "probability": 0.2},
    ]
    assert 0.0 <= score["confidence"] <= 1.0
    assert result["input_tokens"] > 0
    assert result["cached_tokens"] == 2


def test_many_choices_compose_group_readouts():
    count = 30
    target = np.linspace(0.0, 3.0, count)

    def scores(text, letters):
        values = re.findall(r"^[A-Z]\. v(\d+)$", text, flags=re.M)
        return [target[int(v)] for v in values]

    request = _request(
        questions=[
            {
                "type": "choice",
                "name": "pick",
                "instructions": "Pick one",
                "choices": [f"v{i}" for i in range(count)],
            }
        ]
    )
    engine = _FakeChatEngine(scores)
    (answer,) = _run(engine, request)["answers"]
    # Two groups of 15 plus the readout over the two winners.
    assert len(engine.calls) == 3
    probs = [p["probability"] for p in answer["probabilities"]]
    full = np.exp(target - target.max())
    assert probs == pytest.approx(full / full.sum(), abs=1e-4)
    assert answer["choice"] == f"v{count - 1}"


def test_masked_option_letters_are_a_refusal():
    engine = _FakeChatEngine(lambda text, letters: [-np.inf] * len(letters))
    request = _request(
        questions=[{"type": "predicate", "name": "p", "instructions": "Is it?"}]
    )
    assert _run(engine, request)["answers"] == [{"type": "refusal", "name": "p"}]


def test_letters_without_the_next_token_mass_are_a_refusal():
    # The next token is something else (a reasoning or channel header), so
    # the letters' renormalized split is meaningless.
    engine = _FakeChatEngine(lambda text, letters: [0.0, -1.0], background=0.0)
    request = _request(
        questions=[{"type": "predicate", "name": "p", "instructions": "Is it?"}]
    )
    assert _run(engine, request)["answers"] == [{"type": "refusal", "name": "p"}]


def test_group_without_a_letter_answer_still_composes():
    # The group holding the right option is answered; the other group
    # rejects all of its options ("none of these"), which is not a refusal.
    count = 30

    def scores(text, letters):
        values = [int(v) for v in re.findall(r"^[A-Z]\. v(\d+)$", text, flags=re.M)]
        if count - 1 not in values:
            return [-8.0] * len(values)
        return [12.0 if v == count - 1 else 6.0 for v in values]

    request = _request(
        questions=[
            {
                "type": "choice",
                "name": "pick",
                "instructions": "Pick one",
                "choices": [f"v{i}" for i in range(count)],
            }
        ]
    )
    engine = _FakeChatEngine(scores, background=0.0)
    (answer,) = _run(engine, request)["answers"]
    assert answer["type"] == "choice"
    assert answer["choice"] == f"v{count - 1}"


def test_failed_question_cancels_the_other_readouts():
    started, finished, cancelled = [], [], []

    class Failing(_FakeChatEngine):
        async def chat(self, messages, **kwargs):
            name = re.search(r"Question: (Q\d+)", messages[0]["content"])[1]
            started.append(name)
            if name == "Q3":
                raise RuntimeError("memory guard")
            try:
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            finished.append(name)
            return await super().chat(messages, **kwargs)

    request = _request(
        questions=[
            {"type": "predicate", "name": f"q{i}", "instructions": f"Q{i}?"}
            for i in range(40)
        ]
    )

    async def run():
        with pytest.raises(RuntimeError):
            await decisions.decide(
                Failing(lambda text, letters: [0.0, 0.0]), request, False
            )
        count = len(started)
        await asyncio.sleep(0.2)
        return count

    count = asyncio.run(run())
    # Only the first readout, the first concurrent batch and the readout that
    # took the failed one's slot ever start, and nothing starts or keeps
    # running after the error is raised.
    assert len(started) == count <= 2 + decisions.READOUT_CONCURRENCY
    assert finished == ["Q0"]
    assert sorted(cancelled) == sorted(set(started) - {"Q0", "Q3"})


def test_boolean_choice_values():
    engine = _FakeChatEngine(lambda text, letters: [0.0, 2.0])
    request = _request(
        questions=[
            {
                "type": "choice",
                "name": "flag",
                "instructions": "Flag it?",
                "choices": [True, {"value": False, "description": "Leave it"}],
            }
        ]
    )
    (answer,) = _run(engine, request)["answers"]
    assert answer["choice"] is False
    assert [p["value"] for p in answer["probabilities"]] == [True, False]
    text = engine.calls[0]["messages"][0]["content"]
    assert "A. true\nB. false: Leave it" in text

    plan = decisions.to_systemone(request)
    assert plan["questions"]["flag"]["criteria"] == {"true": None, "false": "Leave it"}
    (answer,) = decisions.from_systemone(
        request,
        {
            "flag": {
                "type": "choice",
                "choice": "false",
                "probabilities": {"true": 0.3, "false": 0.7},
                "confidence": 0.4,
            }
        },
    )
    assert answer["choice"] is False
    assert answer["probabilities"] == [
        {"value": True, "probability": 0.3},
        {"value": False, "probability": 0.7},
    ]


def test_message_input_with_images():
    image = "data:image/png;base64,AAAA"
    request = DecisionRequest.model_validate(
        {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Look:"},
                        {"type": "input_image", "image_url": image},
                    ],
                }
            ],
            "questions": [{"type": "predicate", "name": "p", "instructions": "Cat?"}],
        }
    )
    engine = _FakeChatEngine(lambda text, letters: [0.0, 0.0])
    _run(engine, request, multimodal=True)
    content = engine.calls[0]["messages"][0]["content"]
    assert content[:2] == [
        {"type": "text", "text": "Look:"},
        {"type": "image_url", "image_url": {"url": image}},
    ]
    assert content[2]["text"].startswith("\n\nQuestion: Cat?")

    with pytest.raises(decisions.DecisionsRequestError):
        _run(engine, request, multimodal=False)


def test_engine_without_logits_is_rejected():
    class NoLogits(_FakeChatEngine):
        async def chat(self, messages, **kwargs):
            return SimpleNamespace(prompt_tokens=1)

    request = _request(
        questions=[{"type": "predicate", "name": "p", "instructions": "Is it?"}]
    )
    with pytest.raises(decisions.DecisionsRequestError):
        _run(NoLogits(None), request)


@pytest.mark.parametrize(
    "change",
    [
        {"questions": []},
        {"input": []},
        {"questions": [{"type": "rank", "name": "a", "instructions": "x"}]},
        {"questions": [{"type": "predicate", "name": "a", "instructions": "x"}] * 2},
        {
            "questions": [
                {"type": "choice", "name": "a", "instructions": "x", "choices": ["y"]}
            ]
        },
        {"input": [{"content": [{"type": "input_image", "image_url": "https://x"}]}]},
        {
            "questions": [
                {
                    "type": "choice",
                    "name": "a",
                    "instructions": "x",
                    "choices": [f"v{i}" for i in range(256)],
                }
            ]
        },
        {
            "questions": [
                {
                    "type": "score",
                    "name": "a",
                    "instructions": "x",
                    "levels": [f"l{i}" for i in range(11)],
                }
            ]
        },
        {
            "questions": [
                {"type": "choice", "name": "a", "instructions": "x", "choices": [1, 2]}
            ]
        },
        {
            "questions": [
                {
                    "type": "choice",
                    "name": "a",
                    "instructions": "x",
                    "choices": [True, "true"],
                }
            ]
        },
    ],
)
def test_schema_validation(change):
    body = {
        "model": "m",
        "input": "x",
        "questions": [{"type": "predicate", "name": "a", "instructions": "x"}],
        **change,
    }
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(body)


def test_systemone_translation_round_trip():
    request = _request(
        questions=[
            {"type": "predicate", "name": "p", "instructions": "Is it?"},
            {
                "type": "choice",
                "name": "c",
                "instructions": "Which?",
                "choices": [{"value": "x", "description": "X"}, "y"],
            },
            {
                "type": "score",
                "name": "s",
                "instructions": "How much?",
                "levels": ["low", {"label": "high", "description": "a lot"}],
            },
        ]
    )
    plan = decisions.to_systemone(request)
    assert plan["state"] == "Checkout is down."
    assert plan["questions"] == {
        "p": {"type": "noul", "instructions": "Is it?", "criteria": None},
        "c": {
            "type": "choice",
            "instructions": "Which?",
            "criteria": {"x": "X", "y": None},
        },
        "s": {
            "type": "score",
            "instructions": "How much?",
            "criteria": ["low", "high: a lot"],
        },
    }
    answers = decisions.from_systemone(
        request,
        {
            "p": {"type": "noul", "noul": 0.3},
            "c": {
                "type": "choice",
                "choice": "y",
                "probabilities": {"x": 0.4, "y": 0.6},
                "confidence": 0.2,
            },
            "s": {
                "type": "score",
                "score": 0.9,
                "probabilities": {"0": 0.1, "1": 0.9},
                "confidence": 0.8,
            },
        },
    )
    assert answers == [
        {"type": "predicate", "name": "p", "probability": 0.3},
        {
            "type": "choice",
            "name": "c",
            "choice": "y",
            "probabilities": [
                {"value": "x", "probability": 0.4},
                {"value": "y", "probability": 0.6},
            ],
            "confidence": 0.2,
        },
        {
            "type": "score",
            "name": "s",
            "score": 0.9,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1},
                {"value": 1, "label": "high", "probability": 0.9},
            ],
            "confidence": 0.8,
        },
    ]
