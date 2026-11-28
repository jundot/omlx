# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hindsight decision model behind /v1/systemone (hs-decide-1-flash)."""

import json

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.models import hindsight  # noqa: E402
from omlx.models.decision import (  # noqa: E402
    DecisionContextLengthError,
    DecisionRequestError,
)

PAIRS = [
    first + second
    for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
]
CODES = (
    list(hindsight.NOUL_CODES)
    + [str(d) for d in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + PAIRS
)


class FakeTokenizer:
    """A tokenizer where codes, punctuation and added tokens are single tokens."""

    def __init__(self, extra=()):
        words = set(CODES) | {":", "\n", "  ", "Q1", "Q2", "Q3"}
        words |= {" " + code for code in CODES}
        words |= set(extra)
        self.vocab = sorted(words, key=len, reverse=True)
        self._ids = {word: 1000 + index for index, word in enumerate(self.vocab)}
        self._next = 5000
        self._added = {}

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            for word in self.vocab:
                if text.startswith(word, index):
                    ids.append(self._ids[word])
                    index += len(word)
                    break
            else:
                char = text[index]
                if char not in self._ids:
                    self._ids[char] = self._next
                    self._next += 1
                ids.append(self._ids[char])
                index += 1
        return ids

    def get_added_vocab(self):
        return self._added


CHOICE = {
    "type": "choice",
    "instructions": "Which document gives the date?",
    "criteria": {"D00": None, "D01": None},
}
NOUL = {"type": "noul", "instructions": "Did the date change?"}
SCORE = {
    "type": "score",
    "instructions": "How well does D00 answer?",
    "criteria": ["about something else", "same topic", "helps", "answers"],
}


def request():
    return {
        "state": {"query": "When is the offsite?", "documents": {"D00": "moved to 17"}},
        "questions": {"latest": CHOICE, "moved": NOUL, "D00": SCORE},
    }


# --------------------------------------------------------------------------
# Detection


def _write(path, payload):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3_5_text", "architectures": ["Qwen3_5ForCausalLM"]}
        )
    )
    (path / "decision.json").write_text(json.dumps(payload))
    return path


def test_detect_hindsight_by_decision_json(tmp_path):
    from omlx.model_discovery import decision_kind, detect_model_type

    model = _write(
        tmp_path / "hs-decide-1-flash",
        {
            "prompt_version": 2,
            "typed": {
                "codes": "pairs",
                "plain_text": True,
                "max_request_tokens": 65536,
            },
        },
    )
    assert decision_kind(model) == "hindsight"
    assert detect_model_type(model) == "decision"


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt_version": 1, "typed": {"codes": "pairs", "plain_text": True}},
        {"prompt_version": 2, "typed": {"codes": "letters", "plain_text": True}},
        {"prompt_version": 2, "typed": {"codes": "pairs", "plain_text": False}},
        {"prompt_version": 2},
        {},
    ],
)
def test_other_formats_are_not_hindsight(tmp_path, payload):
    from omlx.model_discovery import decision_kind

    model = _write(tmp_path / "not-hindsight", payload)
    assert decision_kind(model) != "hindsight"


def test_plain_chat_checkpoint_is_not_a_decision_model(tmp_path):
    from omlx.model_discovery import decision_kind

    model = tmp_path / "chat"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    assert decision_kind(model) is None


# --------------------------------------------------------------------------
# Prompt layout


def test_prompt_layout_and_slots():
    tokenizer = FakeTokenizer()
    rendered = hindsight.render_request(tokenizer, request(), 65536)
    text = "".join(
        (
            next(word for word, token in tokenizer._ids.items() if token == i)
            if i in tokenizer._ids.values()
            else ""
        )
        for i in rendered.ids
    )
    assert text.startswith("<|im_start|>user\nState:\n")
    assert "[Q1] (choice) Which document gives the date?" in text
    assert "[Q2] (noul) Did the date change?" in text
    assert "[Q3] (score) How well does D00 answer?" in text
    # The assistant turn holds one answer line per question.
    assert text.endswith("<|im_end|>\n<|im_start|>assistant\nQ1:\nQ2:\nQ3:\n")
    assert [question.id for question in rendered.questions] == [
        "latest",
        "moved",
        "D00",
    ]


def test_slots_are_the_colons_of_the_answer_lines():
    tokenizer = FakeTokenizer()
    rendered = hindsight.render_request(tokenizer, request(), 65536)
    colon = tokenizer._ids[":"]
    for question in rendered.questions:
        assert rendered.ids[question.slot] == colon
    slots = [question.slot for question in rendered.questions]
    assert slots == sorted(slots)


def test_option_groups_follow_the_codes():
    tokenizer = FakeTokenizer()
    rendered = hindsight.render_request(tokenizer, request(), 65536)
    latest, moved, score = rendered.questions
    assert latest.codes == ["A", "B"]
    assert moved.codes == ["no", "yes"]
    assert score.codes == ["0", "1", "2", "3"]
    # Both spellings of a choice code are read, bare and after a space.
    assert len(latest.groups[0]) == 2
    # A noul code additionally has its capitalized spellings.
    assert len(moved.groups[1]) >= 2


def test_choice_codes_take_two_letter_forms_after_z():
    tokenizer = FakeTokenizer()
    table = hindsight.TokenTable(tokenizer)
    codes = table.pair_codes()
    assert list(codes[:26]) == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert codes[26] == "AA"  # a token both bare and after a space
    assert len(codes) == hindsight.MAX_CHOICE_OPTIONS


def test_long_request_is_refused():
    tokenizer = FakeTokenizer()
    with pytest.raises(DecisionContextLengthError):
        hindsight.render_request(tokenizer, request(), 32)


# --------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "x", "criteria": {"only": None}},
        {"type": "choice", "instructions": "x", "criteria": None},
        {"type": "score", "instructions": "x", "criteria": ["one"]},
        {"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]},
        {"type": "noul", "instructions": "x", "criteria": ["not", "an", "object"]},
        {"type": "ranking", "instructions": "x"},
    ],
)
def test_malformed_questions_are_refused(question):
    tokenizer = FakeTokenizer()
    with pytest.raises(DecisionRequestError):
        hindsight.render_request(
            tokenizer, {"state": "s", "questions": {"q": question}}, 65536
        )


def test_empty_request_is_refused():
    with pytest.raises(DecisionRequestError):
        hindsight.render_request(
            FakeTokenizer(), {"state": "s", "questions": {}}, 65536
        )


# --------------------------------------------------------------------------
# Answers


def test_answer_format():
    noul = hindsight.format_answer({"type": "noul"}, [0.1, 0.9])
    assert noul == {"type": "noul", "noul": 0.9}

    score = hindsight.format_answer(SCORE, [0.0182, 0.0324, 0.1462, 0.8032])
    assert score["type"] == "score"
    assert score["score"] == pytest.approx(2.7344, abs=1e-4)
    assert score["probabilities"]["3"] == 0.8032
    assert score["legend"]["2"] == "helps"
    assert 0.0 <= score["confidence"] <= 1.0

    choice = hindsight.format_answer(CHOICE, [0.75, 0.25])
    assert choice == {
        "type": "choice",
        "choice": "D00",
        "confidence": 0.5,
        "probabilities": {"D00": 0.75, "D01": 0.25},
    }


def test_confidence_bounds():
    assert hindsight.choice_confidence([1.0]) == 1.0
    assert hindsight.choice_confidence([1 / 3, 1 / 3, 1 / 3]) == 0.0
    assert hindsight.score_confidence([1 / 4] * 4) == 0.0
    assert hindsight.score_confidence([0.0, 0.0, 0.0, 1.0]) == 1.0


def test_softmax_is_a_distribution():
    probabilities = hindsight.softmax([0.0, 1.0, -1.0])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[1] == max(probabilities)


# --------------------------------------------------------------------------
# Format gate


def test_load_settings_reads_the_request_limit(tmp_path):
    _write(
        tmp_path / "m",
        {
            "prompt_version": 2,
            "typed": {"codes": "pairs", "plain_text": True, "max_request_tokens": 4096},
        },
    )
    assert hindsight.load_settings(tmp_path / "m") == 4096


def test_load_settings_refuses_another_format(tmp_path):
    _write(tmp_path / "m", {"prompt_version": 3, "typed": {"plain_text": True}})
    with pytest.raises(hindsight.HindsightFormatError):
        hindsight.load_settings(tmp_path / "m")
    with pytest.raises(hindsight.HindsightFormatError):
        hindsight.load_settings(tmp_path / "missing")


# --------------------------------------------------------------------------
# Readout


def test_readout_picks_the_code_with_the_largest_logit():
    tokenizer = FakeTokenizer()
    rendered = hindsight.render_request(
        tokenizer, {"state": "s", "questions": {"q": CHOICE}}, 65536
    )
    table = hindsight.TokenTable(tokenizer)
    codes = (
        list(hindsight.NOUL_CODES)
        + [str(d) for d in range(10)]
        + list(table.pair_codes())
    )
    classifier, column = [], {}
    for code in codes:
        for form in hindsight.spellings(code):
            token = table.single(form)
            if token is not None and token not in classifier:
                classifier.append(token)
    column = {token: index for index, token in enumerate(classifier)}

    class OneHotBackbone:
        def embed(self, ids):
            return mx.zeros((1, len(ids), 8)), None

        def make_cache(self):
            return []

        def prefill(self, ids, embeds, positions, cache, chunk_size, keep_all):
            yield len(ids)
            return mx.ones((len(ids), 8))

        def output_rows(self, ids):
            rows = mx.zeros((len(ids), 8))
            # 'B' scores highest, 'A' second: the second option should win.
            boost = {column[tokenizer._ids["B"]]: 2.0, column[tokenizer._ids["A"]]: 1.0}
            for row, value in boost.items():
                rows[row] = mx.array([value] + [0.0] * 7)
            return rows

    model = hindsight.HindsightModel.__new__(hindsight.HindsightModel)
    model.backbone = OneHotBackbone()
    model._classifier_ids = classifier
    model._column = column
    plan = hindsight.HindsightPlan(
        request={"questions": {"q": CHOICE}}, rendered=rendered
    )
    result = _drain(model.run(plan, lambda: 1024))
    answer = result["answers"]["q"]
    assert answer["choice"] == "D01"
    assert answer["probabilities"]["D01"] > answer["probabilities"]["D00"]
    assert result["input_tokens"] == len(rendered.ids)


def _drain(steps):
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value
