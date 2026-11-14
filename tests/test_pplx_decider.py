# SPDX-License-Identifier: Apache-2.0
"""
pplx-decider (Perplexity Decider v1.3) decision models.

Covers the ``joint_config.json`` detection (decision_kind / detect_model_type),
the wrapper-class selection in the decision engine, the checkpoint config
validation, and the chunked prompt protocol of the release runtime: context and
question chunks, the readout markers, the ``separate``/``isolated`` layouts and
the escaping of literal special tokens in user content.

No backbone weights are needed: the tests drive the model against a stub
backbone and a stub tokenizer.
"""

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.engine.decision import _MODEL_CLASSES  # noqa: E402
from omlx.model_discovery import decision_kind, detect_model_type  # noqa: E402
from omlx.models.decision import (  # noqa: E402
    DecisionContextLengthError,
    DecisionRequestError,
)
from omlx.models.pplx_decider import (  # noqa: E402
    CONTEXT_END,
    PplxDeciderModel,
)

MARKER = "<|fim_suffix|>"
SEPARATOR = "<|fim_middle|>"
CODES = ["A", "B", "C"]
HIDDEN = 8


class StubTokenizer:
    """Character-level tokenizer with the release's special tokens."""

    def __init__(self):
        self.specials = {
            "<|im_start|>": 502,
            "<|im_end|>": 503,
            "<|vision_start|>": 504,
            "<|vision_end|>": 505,
            "<|image_pad|>": 506,
            "<|video_pad|>": 507,
            MARKER: 500,
            SEPARATOR: 501,
        }
        self.all_special_tokens = list(self.specials)
        self._reverse = {value: key for key, value in self.specials.items()}
        self._ordered = sorted(self.specials.items(), key=lambda kv: -len(kv[0]))

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            for token, token_id in self._ordered:
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(100 + ord(text[index]) % 400)
                index += 1
        return ids

    def convert_ids_to_tokens(self, token_id):
        if token_id in self._reverse:
            return self._reverse[token_id]
        return "chr" + str(token_id)

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        out = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    if "image" in item
                    else item["text"]
                    for item in content
                )
            out += f"<|im_start|>{message['role']}\n{content.rstrip()}<|im_end|>\n"
        return out


class StubBackbone:
    """Stand-in for DecisionBackbone: tokenizer, config and tokenization only."""

    def __init__(self, tokenizer=None, limit=262144):
        self.tokenizer = tokenizer or StubTokenizer()
        self.config = {"text_config": {"hidden_size": HIDDEN}}
        self.image_processor = None
        self.max_position_embeddings = limit

    @property
    def has_vision(self):
        return False

    def load(self):
        return None

    def close(self):
        return None

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def _joint_config() -> dict:
    return {
        "version": 4,
        "template": "joint-chunked-readout-v1",
        "marker": MARKER,
        "readout_offset": 0,
        "separator": SEPARATOR,
    }


def _decision_config(**overrides) -> dict:
    config = {
        "format_version": 2,
        "model_type": "qwen3_5",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "temperature": 1.2263882168038833,
        "codes": CODES,
        "token_ids": [100 + ord(code) % 400 for code in CODES],
        "max_pixels": 1638400,
        "default_question_mode": "separate",
    }
    config.update(overrides)
    return config


def _write_model_dir(tmp_path, config=None, joint=None, readout=True, name="checkpoint"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (model_dir / "decision_config.json").write_text(json.dumps(config or _decision_config()))
    (model_dir / "joint_config.json").write_text(json.dumps(joint or _joint_config()))
    if readout:
        mx.save_safetensors(
            str(model_dir / "readout.safetensors"),
            {"weight": mx.zeros((len(CODES), HIDDEN))},
        )
    return model_dir


def _loaded_model(tmp_path, **config_overrides):
    """A model whose backbone is a stub, but whose load() ran for real."""
    path = _write_model_dir(tmp_path, config=_decision_config(**config_overrides))
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    model.load()
    return model


# --------------------------------------------------------------------------
# decision_kind / detect_model_type


def test_decision_kind_pplx_decider(tmp_path):
    assert decision_kind(_write_model_dir(tmp_path)) == "pplx_decider"


def test_decision_kind_needs_the_readout(tmp_path):
    assert decision_kind(_write_model_dir(tmp_path, readout=False)) is None


def test_decision_kind_ignores_a_foreign_template(tmp_path):
    joint = _joint_config() | {"template": "something-else"}
    assert decision_kind(_write_model_dir(tmp_path, joint=joint)) is None


def test_detect_model_type_pplx_decider_is_decision(tmp_path):
    assert detect_model_type(_write_model_dir(tmp_path)) == "decision"


def test_decision_classes_route_the_kind():
    from omlx.models.pplx_decider import PplxDeciderModel as Model

    assert _MODEL_CLASSES["pplx_decider"] is Model


# --------------------------------------------------------------------------
# config validation


def test_check_config_rejects_a_bad_attention_mode(tmp_path):
    path = _write_model_dir(tmp_path, config=_decision_config(attention_mode="causal"))
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    with pytest.raises(ValueError, match="attention_mode"):
        model.load()


def test_check_config_rejects_a_missing_temperature(tmp_path):
    config = _decision_config()
    config.pop("temperature")
    path = _write_model_dir(tmp_path, config=config)
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    with pytest.raises(ValueError, match="temperature"):
        model.load()


def test_load_reads_codes_temperature_and_marker(tmp_path):
    model = _loaded_model(tmp_path)
    assert model.codes == CODES
    assert model.temperature == pytest.approx(1.2263882168038833)
    assert model.marker_id == 500
    assert model.separator_id == 501
    assert model.mode == "separate"


def test_load_rejects_a_code_that_does_not_tokenize_to_its_id(tmp_path):
    path = _write_model_dir(tmp_path, config=_decision_config(token_ids=[1, 2, 3]))
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    with pytest.raises(ValueError, match="tokenize"):
        model.load()


# --------------------------------------------------------------------------
# prompt protocol


def test_state_text_and_escaping(tmp_path):
    model = _loaded_model(tmp_path)
    assert model.state_text("hello") == "State:\nhello"
    assert model.state_text("") == "State:\n(empty)"
    # A literal marker in the state must not read as a readout marker.
    escaped = model.state_text(f"a {MARKER} b")
    assert MARKER not in escaped
    assert "\uff1c|fim_suffix|\uff1e" in escaped


def test_question_text_lists_the_options_under_their_codes(tmp_path):
    model = _loaded_model(tmp_path)
    question = model._normalize("q", {"type": "choice", "instructions": "Pick", "criteria": {"one": "first", "two": None}})
    assert model.question_text(question) == (
        "Question:\nPick\nOptions:\nA: one: first\nB: two"
    )


def test_question_text_defaults_the_instructions(tmp_path):
    model = _loaded_model(tmp_path)
    question = model._normalize("q", {"type": "noul"})
    text = model.question_text(question)
    assert "Choose the best matching option." in text
    assert text.endswith("A: No / false\nB: Yes / true")


def test_context_text_keeps_the_state_and_leaves_the_turn_open(tmp_path):
    model = _loaded_model(tmp_path)
    state = "trailing spaces   "
    text = model.context_text(state, 0)
    assert text.startswith("<|im_start|>system\nClassify the shared state")
    assert text.endswith(f"<|im_start|>user\nState:\n{state}")
    assert CONTEXT_END not in text


# --------------------------------------------------------------------------
# request validation


def _request(**overrides):
    request = {
        "state": "the state",
        "questions": {"q": {"type": "noul", "instructions": "Is it true?"}},
    }
    request.update(overrides)
    return request


def test_encode_separate_gives_one_sequence_per_question(tmp_path):
    model = _loaded_model(tmp_path)
    request = _request(
        questions={
            "a": {"type": "noul"},
            "b": {"type": "choice", "criteria": {"x": None, "y": None}},
        }
    )
    plan = model.encode(request)
    assert [sequence.keys for sequence in plan.sequences] == [("a",), ("b",)]
    for sequence in plan.sequences:
        assert sequence.markers == [len(sequence.ids) - 1]
        assert sequence.ids[sequence.markers[0]] == model.marker_id
        assert sequence.ids[len(sequence.ids) - 2] != model.marker_id
    # The context (system prompt, state, separator) is shared by both sequences.
    head = list(plan.sequences[0].ids[: len(plan.sequences[1].ids)])
    assert head != list(plan.sequences[1].ids)
    assert list(plan.sequences[0].ids[:20]) == list(plan.sequences[1].ids[:20])


def test_encode_isolated_puts_every_question_in_one_sequence(tmp_path):
    model = _loaded_model(tmp_path)
    request = _request(
        mode="isolated",
        questions={"a": {"type": "noul"}, "b": {"type": "noul"}},
    )
    plan = model.encode(request)
    assert len(plan.sequences) == 1
    sequence = plan.sequences[0]
    assert sequence.keys == ("a", "b")
    assert len(sequence.markers) == 2
    assert [sequence.ids[marker] for marker in sequence.markers] == [model.marker_id] * 2


def test_encode_rejects_an_unknown_mode(tmp_path):
    model = _loaded_model(tmp_path)
    with pytest.raises(DecisionRequestError, match="mode"):
        model.encode(_request(mode="together"))


def test_encode_rejects_a_single_level_score(tmp_path):
    model = _loaded_model(tmp_path)
    with pytest.raises(DecisionRequestError, match="score"):
        model.encode(_request(questions={"q": {"type": "score", "criteria": ["only"]}}))


def test_encode_rejects_more_options_than_codes(tmp_path):
    model = _loaded_model(tmp_path)
    criteria = {f"option{index}": None for index in range(4)}
    with pytest.raises(DecisionRequestError, match="options"):
        model.encode(_request(questions={"q": {"type": "choice", "criteria": criteria}}))


def test_encode_rejects_a_question_over_the_context_limit(tmp_path):
    path = _write_model_dir(tmp_path)
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone(limit=32)
    model.load()
    with pytest.raises(DecisionContextLengthError):
        model.encode(_request())


def _png_data_uri() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_encode_rejects_images_without_a_vision_tower(tmp_path):
    model = _loaded_model(tmp_path)
    with pytest.raises(DecisionRequestError, match="vision"):
        model.encode(_request(images=[_png_data_uri()]))


# --------------------------------------------------------------------------
# answer formatting


def test_run_reads_out_probabilities_per_question(tmp_path, monkeypatch):
    """Drive run() with a stubbed encode/backbone forward."""
    model = _loaded_model(tmp_path)
    plan = model.encode(_request(questions={"a": {"type": "noul"}}))
    sequence = plan.sequences[0]

    class StubTextModel:
        class Norm:
            @staticmethod
            def __call__(hidden):
                return hidden

        def __init__(self):
            self.layers = []
            self.norm = StubTextModel.Norm()

    model.backbone._text_model = StubTextModel()
    model.backbone.embed_tokens = lambda ids: mx.zeros((1, len(ids), HIDDEN))

    def fake_encode_markers(embeds, positions, chunk_size, markers):
        hidden = mx.zeros((len(markers), HIDDEN))
        yield len(sequence.ids)
        return hidden

    monkeypatch.setattr(model, "_encode_markers", fake_encode_markers)
    steps = model.run(plan, lambda: 2048)
    with pytest.raises(StopIteration) as stop:
        while True:
            next(steps)
    result = stop.value.value
    assert set(result["answers"]) == {"a"}
    assert result["answers"]["a"]["type"] == "noul"
    assert result["answers"]["a"]["noul"] == pytest.approx(0.5)
    assert result["input_tokens"] == len(sequence.ids)


# The model exposes the question normalizer as a method for the tests above.
def _normalize(self, question_id, question):
    from omlx.models.pplx_decider import normalize_question

    return normalize_question(question_id, question)


PplxDeciderModel._normalize = _normalize


def test_normalize_question_shapes():
    from omlx.models.pplx_decider import normalize_question

    assert normalize_question("q", {"type": "noul"}).keys == ["false", "true"]
    choice = normalize_question("q", {"type": "choice", "criteria": {"k": "d"}})
    assert choice.texts == ["k: d"]
    score = normalize_question("q", {"type": "score", "criteria": ["low", "high"]})
    assert score.keys == ["0", "1"]
    assert score.levels == ["low", "high"]


def test_locate_adapter_dir_is_a_decision_sidecar():
    """The locate adapter is a directory; a quantizer has to copy it as a tree."""
    from omlx.model_discovery import DECISION_SIDECAR_DIRS, PPLX_LOCATE_DIR

    assert PPLX_LOCATE_DIR in DECISION_SIDECAR_DIRS
