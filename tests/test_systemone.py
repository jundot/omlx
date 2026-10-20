# SPDX-License-Identifier: Apache-2.0
"""Tests for the decision models behind /v1/systemone (Clef, OpenJev, d3)."""

import copy
import inspect
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from omlx.exceptions import InvalidRequestError  # noqa: E402
from omlx.models import clef, d3, decision, openjev  # noqa: E402
from omlx.models.decision import (  # noqa: E402
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
)


def _chars(text: str) -> list[int]:
    return [ord(c) for c in text]


def _text(ids) -> str:
    return "".join(chr(i) for i in ids)


def _drain(steps):
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


# --------------------------------------------------------------------------
# Clef prompt and answers


def test_clef_record_layout_and_spans():
    request = {
        "state": {"b": 1, "a": "é"},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"tech": "Outages", "billing": None},
            },
            "urgent": {"type": "noul", "instructions": None, "criteria": None},
        },
    }
    ids, fields = clef.encode_record(_chars, request)
    text = _text(ids)

    assert text.startswith(f"<|im_start|>system\n{clef.SYSTEM_PROMPT}<|im_end|>")
    assert 'STATE:\n{"a":"é","b":1}\n\nSCHEMA FIELDS:\n' in text
    assert text.endswith("<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")

    (team_id, team_span, team_options, team_ids), urgent = fields
    assert team_id == "team"
    assert _text(ids[slice(*team_span)]) == "Which team?"
    # Choice options are sorted by key; a null description is left out.
    assert team_ids == ["billing", "tech"]
    assert [_text(ids[s:e]) for s, e in team_options] == [
        '{"option_id":"billing"}',
        '{"description":"Outages","option_id":"tech"}',
    ]
    # Missing instructions fall back to the question id.
    assert _text(ids[slice(*urgent[1])]) == "urgent"
    assert urgent[3] == ["true", "false"]
    assert "The proposition is false" in _text(ids[slice(*urgent[2][1])])


def test_clef_truncates_state_unless_disabled():
    questions = {"q": {"type": "noul", "instructions": "Is it?"}}
    fixed = len(clef.encode_record(_chars, {"state": "", "questions": questions})[0])
    request = {"state": "x" * 500, "questions": questions}
    ids, _ = clef.encode_record(_chars, request, max_length=fixed + 100)
    assert len(ids) == fixed + 100

    with pytest.raises(DecisionContextLengthError):
        clef.encode_record(_chars, request, max_length=fixed + 100, truncate=False)
    with pytest.raises(DecisionContextLengthError):
        clef.encode_record(_chars, request, max_length=fixed - 1)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "criteria": ["a", "b"]},
        {"type": "choice", "criteria": {}},
        {"type": "score", "criteria": []},
        {"type": "noul", "criteria": ["yes"]},
    ],
)
def test_clef_rejects_malformed_criteria(question):
    with pytest.raises(DecisionRequestError):
        clef.question_options(question)


def test_clef_answer_format():
    choice = {"type": "choice", "criteria": {"z": None, "a": None}}
    answer = clef.format_answer(choice, {"a": 0.25, "z": 0.75})
    assert answer == {
        "type": "choice",
        "choice": "z",
        "confidence": 0.75,
        "probabilities": {"z": 0.75, "a": 0.25},
    }
    score = {"type": "score", "criteria": ["low", "mid", "high"]}
    answer = clef.format_answer(score, {"0": 0.2, "1": 0.3, "2": 0.5})
    assert answer["score"] == pytest.approx(1.3)
    assert answer["confidence"] == 0.5
    assert answer["legend"] == {"0": "low", "1": "mid", "2": "high"}
    noul = clef.format_answer({"type": "noul"}, {"true": 0.123456, "false": 0.876544})
    assert noul == {"type": "noul", "noul": 0.1235}


def test_joint_head_parameter_names_match_checkpoint_layout():
    head = clef.JointSchemaHead(
        hidden_size=16, width=8, routing_layers=2, layers=4, heads=2, feedforward=16
    )
    names = {name for name, _ in tree_flatten(head.parameters())}
    # The released joint_head.safetensors holds 122 tensors with torch names.
    assert len(names) == 122
    assert {
        "evidence_layers.1.attention.in_proj_weight",
        "evidence_layers.0.feedforward.0.weight",
        "evidence_layers.0.feedforward.3.bias",
        "layers.3.multihead_attn.out_proj.weight",
        "layers.0.norm3.bias",
        "residual_scorer.0.weight",
        "residual_scorer.3.bias",
        "type_embedding.weight",
        "prior_logit_scale",
        "residual_gate",
    } <= names


def test_joint_head_scores_each_question_over_its_own_options():
    mx.random.seed(0)
    head = clef.JointSchemaHead(
        hidden_size=16, width=8, routing_layers=2, layers=2, heads=2, feedforward=16
    )
    option_spans = [[(8, 10), (11, 15), (16, 17)], [(24, 28), (29, 31)], [(34, 36)]]
    plan = clef.ClefPlan(
        input_ids=np.zeros(40, dtype=np.int32),
        questions={},
        question_ids=["a", "b", "c"],
        type_ids=[1, 0, 2],
        question_spans=[(2, 6), (20, 23), (32, 33)],
        option_spans=option_spans,
        option_ids=[["x"] * len(spans) for spans in option_spans],
    )
    lexical = [mx.random.normal((len(spans), 16)) for spans in option_spans]

    logits = head(mx.random.normal((40, 16)), lexical, plan)

    assert [tuple(x.shape) for x in logits] == [(3,), (2,), (1,)]
    assert all(np.isfinite(np.array(x)).all() for x in logits)


# --------------------------------------------------------------------------
# OpenJev prompt and calibration


def test_openjev_text_prompt_layout():
    state = openjev.split_state({"msg": "charged twice"}, None)
    prompt = openjev.render_prompt(
        state, "Which team?", [("billing", ""), ("tech", "Outages")]
    )
    assert prompt == (
        'State:\n{"msg": "charged twice"}\n\nQuestion: Which team?\nOptions:\n'
        "[A] billing: \n[B] tech: Outages\n\n"
        "Answer with the letter of the best option only."
    )


def test_openjev_screenshot_task_layout():
    state = openjev.split_state(
        {
            "task": "Buy milk",
            "screenshot": "data:image/png;base64,AAAA",
            "history": ["click Search"],
            "url": "shop",
        },
        None,
    )
    assert state.image == "data:image/png;base64,AAAA"
    prompt = openjev.render_prompt(state, "Next element?", [("e1", "Add"), ("e2", "")])
    assert prompt == (
        "Task: Buy milk\nPrevious actions:\n- click Search\n"
        'Context: {"url": "shop"}\n\n'
        "The screenshot shows the current page with candidate elements marked by "
        "red letters.\n[A] Add\n[B] e2\n\nNext element? Answer with the letter only."
    )


def test_openjev_state_rendering_and_image_sources():
    assert openjev.split_state(["a", 1], None).text == "['a', 1]"
    raw = "A" * (openjev.RAW_IMAGE_MIN_CHARS + 1)
    state = openjev.split_state({"image": raw, "k": 1}, None)
    assert state.image == "data:image/png;base64," + raw
    assert state.text == '{"k": 1}'
    assert openjev.split_state("hi", ["data:image/png;base64,B"]).image is not None
    with pytest.raises(DecisionRequestError):
        openjev.split_state({"screenshot": "data:image/png;base64,A"}, ["x"])
    with pytest.raises(DecisionRequestError):
        openjev.split_state("hi", ["a", "b"])
    assert openjev.instruction_text({"instructions": {"goal": "x"}}) == "{'goal': 'x'}"


def test_openjev_group_composition_matches_full_softmax():
    logits = np.random.default_rng(0).normal(size=120)
    groups = openjev.option_groups(len(logits))
    assert [len(g) for g in groups] == [40, 40, 40]

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    parts = [softmax(logits[list(g)]).tolist() for g in groups]
    winners = [g[int(np.argmax(p))] for g, p in zip(groups, parts)]
    final = softmax(logits[winners]).tolist()
    composed = openjev.compose_groups(parts, final)
    assert np.allclose(composed, softmax(logits))


def test_openjev_calibration_and_confidence():
    assert openjev.noul_probability(0.5) == pytest.approx(0.5)
    z = math.log(0.9 / 0.1) / openjev.NOUL_TEMPERATURE
    assert openjev.noul_probability(0.9) == pytest.approx(1 / (1 + math.exp(-z)))
    assert openjev.noul_probability(1.0) < 1.0
    assert openjev.choice_confidence([0.5, 0.5]) == 0.0
    assert openjev.choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert openjev.score_confidence([0.0, 1.0, 0.0]) == 1.0
    assert openjev.score_confidence([0.5, 0.0, 0.5]) == 0.0


# --------------------------------------------------------------------------
# Backbone prefill on a tiny random Qwen3.5

_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 320,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 3,
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "rms_norm_eps": 1e-5,
    "head_dim": 32,
    "max_position_embeddings": 4096,
    "rope_parameters": {
        "rope_theta": 1000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [2, 3, 3],
        "mrope_interleaved": True,
        "rope_type": "default",
    },
}
_VISION_START, _VISION_END, _IMAGE = 302, 303, 300


@pytest.fixture(scope="module")
def tiny_backbone():
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    mx.random.seed(1)
    config = ModelConfig.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": _TEXT_CONFIG,
            "vision_config": {
                "model_type": "qwen3_5",
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 64,
                "patch_size": 4,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "num_position_embeddings": 64,
                "deepstack_visual_indexes": [],
            },
            "image_token_id": _IMAGE,
            "video_token_id": 301,
            "vision_start_token_id": _VISION_START,
            "vision_end_token_id": _VISION_END,
            "vocab_size": 320,
        }
    )
    model = Model(config)
    mx.eval(model.parameters())
    backbone = DecisionBackbone("tiny")
    backbone.model = model
    backbone.config = {"text_config": _TEXT_CONFIG}
    backbone.image_processor = SimpleNamespace(merge_size=2)
    backbone._text_model = model.language_model.model
    backbone._lm_head = model.language_model.lm_head
    return backbone


def _image_prompt(tail_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = [5, 6, 7, _VISION_START] + [_IMAGE] * 4 + [_VISION_END]
    ids += list(range(10, 10 + tail_length))
    pixels = np.random.default_rng(2).normal(size=(16, 96)).astype(np.float32)
    return np.asarray(ids, dtype=np.int32), pixels, np.array([[1, 4, 4]])


def test_chunked_prefill_matches_one_shot(tiny_backbone):
    ids, pixels, grid = _image_prompt(20)
    embeds, positions = tiny_backbone.embed(ids, pixels, grid)

    def hidden(chunk):
        return _drain(
            tiny_backbone.prefill(
                ids,
                embeds,
                positions,
                tiny_backbone.make_cache(),
                lambda: chunk,
                keep_all=True,
            )
        )

    whole = hidden(len(ids))
    chunked = hidden(5)
    assert chunked.shape == (len(ids), 64)
    assert np.allclose(np.array(chunked), np.array(whole), atol=1e-4)


def test_prefix_cache_copy_matches_full_prefill(tiny_backbone):
    ids, pixels, grid = _image_prompt(20)
    embeds, positions = tiny_backbone.embed(ids, pixels, grid)
    full_last = _drain(
        tiny_backbone.prefill(
            ids,
            embeds,
            positions,
            tiny_backbone.make_cache(),
            lambda: 7,
            keep_all=False,
        )
    )

    split = 14
    prefix_embeds, prefix_positions = tiny_backbone.embed(ids[:split], pixels, grid)
    prefix_cache = tiny_backbone.make_cache()
    _drain(
        tiny_backbone.prefill(
            ids[:split],
            prefix_embeds,
            prefix_positions,
            prefix_cache,
            lambda: 4,
            keep_all=False,
        )
    )
    results = []
    for _ in range(2):
        tail_embeds, tail_positions = tiny_backbone.embed_continuation(
            ids[split:], prefix_positions[:, :, -1]
        )
        results.append(
            _drain(
                tiny_backbone.prefill(
                    ids[split:],
                    tail_embeds,
                    tail_positions,
                    copy.deepcopy(prefix_cache),
                    lambda: 3,
                    keep_all=False,
                )
            )
        )
    # The second continuation proves the first one left the prefix untouched.
    for last in results:
        assert np.allclose(np.array(last), np.array(full_last), atol=1e-4)


class _CharTokenizer:
    """Character tokenizer with a minimal chat template for OpenJev runs."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 250 + 1 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return f"<u>{messages[0]['content']}<a>"

    def convert_tokens_to_ids(self, token):
        return _VISION_END


def test_openjev_shared_prefix_matches_independent_readouts(tiny_backbone, monkeypatch):
    tiny_backbone.tokenizer = _CharTokenizer()
    model = openjev.OpenJevModel("tiny")
    model.backbone = tiny_backbone
    model._letter_ids = [tiny_backbone.tokenize(c)[0] for c in openjev.LETTERS]
    request = {
        "state": {"ticket": "charged twice, nobody replied"},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {f"t{i}": None for i in range(60)},
            },
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["later", "today", "now"],
            },
        },
    }
    shared = _drain(model.run(model.encode(request), lambda: 16))

    monkeypatch.setattr(model, "_prefill_shared_prefix", lambda *a: (yield from ()))
    independent = _drain(model.run(model.encode(request), lambda: 16))

    assert shared["input_tokens"] == independent["input_tokens"]
    team = shared["answers"]["team"]
    assert len(team["probabilities"]) == 60
    assert sum(team["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    for key in ("team", "urgency"):
        a = shared["answers"][key]["probabilities"]
        b = independent["answers"][key]["probabilities"]
        assert np.allclose(list(a.values()), list(b.values()), atol=2e-4)
    assert shared["answers"]["angry"]["noul"] == pytest.approx(
        independent["answers"]["angry"]["noul"], abs=2e-4
    )
    json.dumps(shared)


def test_text_backbone_matches_patched_mlx_lm_logits():
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    # The MTP patch makes the inner text model return pre-norm hidden states.
    assert apply_mlx_lm_mtp_patch()
    mx.random.seed(3)
    model = Model(
        ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": _TEXT_CONFIG})
    )
    mx.eval(model.parameters())
    backbone = DecisionBackbone("tiny-text")
    backbone.model = model
    backbone._text_model = model.language_model.model
    backbone._lm_head = model.language_model.lm_head
    ids = np.arange(5, 25, dtype=np.int32)
    embeds, positions = backbone.embed(ids)
    hidden = _drain(
        backbone.prefill(
            ids, embeds, positions, backbone.make_cache(), lambda: 7, keep_all=False
        )
    )
    expected = model(mx.array(ids)[None])[0, -1]
    assert np.allclose(np.array(backbone.logits(hidden)), np.array(expected), atol=1e-4)


def test_openjev_requires_instructions():
    model = openjev.OpenJevModel("unused")
    model.backbone.tokenizer = _CharTokenizer()
    request = {"state": "s", "questions": {"q": {"type": "noul"}}}
    with pytest.raises(DecisionRequestError):
        model.encode(request)


def test_lm_head_rows_are_dequantized():
    backbone = DecisionBackbone("unused")
    linear = nn.Linear(64, 32, bias=False)
    backbone._lm_head = nn.QuantizedLinear.from_linear(linear, group_size=32, bits=8)
    rows = backbone.output_rows(mx.array([3, 7]))
    expected = linear.weight[mx.array([3, 7])]
    assert rows.shape == (2, 64)
    assert np.allclose(np.array(rows), np.array(expected), atol=2e-2)


def test_backbone_files_hide_a_head_beside_an_unindexed_shard(tmp_path):
    for name in ("model.safetensors", "readout.safetensors", "config.json"):
        (tmp_path / name).write_text(name)
    with decision._backbone_files(tmp_path) as view:
        assert view != tmp_path
        assert sorted(p.name for p in view.iterdir()) == [
            "config.json",
            "model.safetensors",
        ]
        assert (view / "model.safetensors").read_text() == "model.safetensors"
    assert not view.exists()

    # An index already names the backbone shards.
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    with decision._backbone_files(tmp_path) as view:
        assert view == tmp_path


# --------------------------------------------------------------------------
# Decision 3.0 (d3)

_ROUTE = {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {
        "returns": "Refunds, replacements and damaged deliveries",
        "billing": "Payments, invoices and charges",
        "technical": "Product setup and faults",
    },
}


def test_d3_user_prompt_matches_release_text():
    question = d3.normalize_question("route", _ROUTE)
    text = d3.user_prompt("The order arrived damaged.", question, ["A", "B", "C"])
    assert text == (
        "State:\nThe order arrived damaged.\n\n"
        "Question:\nWhich team should handle this request?\n\n"
        "Options:\n"
        "A: returns: Refunds, replacements and damaged deliveries\n"
        "B: billing: Payments, invoices and charges\n"
        "C: technical: Product setup and faults\n\n"
        "Reply with only the code of the best option."
    )

    # Noul shows descriptions only, false first; an empty description falls back.
    noul = d3.normalize_question(
        "n", {"type": "noul", "criteria": {"true": "Only some", "false": ""}}
    )
    assert noul.keys == ["false", "true"]
    lines = d3.user_prompt("", noul, ["A", "B"]).split("\n")
    assert lines[:5] == ["State:", "(empty)", "", "Question:", d3.DEFAULT_INSTRUCTIONS]
    assert lines[7:9] == ["A: No / false", "B: Only some"]

    # Non-string values are spaced JSON in the prompt and compact sorted JSON
    # in the legend; choice keys keep the request order.
    score = d3.normalize_question(
        "s",
        {
            "type": "score",
            "instructions": {"scale": "0-2", "metric": "risk"},
            "criteria": [{"b": 1, "a": 0}, "mid", "high"],
        },
    )
    text = d3.user_prompt({"z": 1, "a": [1, 2]}, score, ["A", "B", "C"])
    assert 'State:\n{"z": 1, "a": [1, 2]}' in text
    assert 'Question:\n{"scale": "0-2", "metric": "risk"}' in text
    assert 'A: {"b": 1, "a": 0}\nB: mid\nC: high' in text
    answer = d3.format_answer(score, [0.2, 0.3, 0.5])
    assert answer["legend"] == {"0": '{"a":0,"b":1}', "1": "mid", "2": "high"}


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "criteria": {}},
        {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        {"type": "choice", "criteria": {"": None}},
        {"type": "choice", "criteria": ["a", "b"]},
        {"type": "noul", "criteria": {"maybe": "x"}},
        {"type": "score", "criteria": ["only"]},
        {"type": "score", "criteria": [str(i) for i in range(11)]},
        {"type": "score", "criteria": ["same", "same"]},
        {"type": "score", "criteria": ["", "x"]},
        {"type": "noul", "instructions": float("nan")},
        {"type": "rank", "criteria": {"a": None}},
    ],
)
def test_d3_rejects_malformed_questions(question):
    with pytest.raises(DecisionRequestError):
        d3.normalize_question("q", question)


def test_d3_answer_format():
    route = d3.normalize_question("route", _ROUTE)
    # The first maximum wins a tie, as in the release.
    answer = d3.format_answer(route, [0.4, 0.4, 0.2])
    assert answer["choice"] == "returns"
    assert list(answer) == ["type", "choice", "probabilities", "confidence"]
    assert answer["confidence"] == pytest.approx(
        1 + sum(p * math.log(p) for p in (0.4, 0.4, 0.2)) / math.log(3), abs=1e-4
    )
    noul = d3.normalize_question("n", {"type": "noul"})
    assert d3.format_answer(noul, [0.25, 0.75]) == {"type": "noul", "noul": 0.75}
    score = d3.normalize_question("s", {"type": "score", "criteria": ["a", "b", "c"]})
    answer = d3.format_answer(score, [0.0, 0.5, 0.5])
    assert answer["score"] == 1.5
    assert answer["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
    single = d3.normalize_question("o", {"type": "choice", "criteria": {"x": None}})
    assert d3.format_answer(single, [1.0])["confidence"] == 1.0


def test_d3_mrope_positions_follow_transformers():
    t, image, video = 1, 300, 301
    ids = np.array(
        [t] * 3 + [image] * 4 + [t] * 2 + [video] * 6 + [t] + [video] * 6 + [t]
    )
    positions = d3.mrope_positions(
        ids, image, video, np.array([[1, 4, 4]]), np.array([[2, 4, 6]]), 2
    )
    expected = [
        [0, 1, 2] + [3] * 4 + [5, 6] + [7] * 6 + [10] + [11] * 6 + [14],
        [0, 1, 2]
        + [3, 3, 4, 4]
        + [5, 6]
        + [7, 7, 7, 8, 8, 8]
        + [10]
        + [11, 11, 11, 12, 12, 12]
        + [14],
        [0, 1, 2]
        + [3, 4, 3, 4]
        + [5, 6]
        + [7, 8, 9, 7, 8, 9]
        + [10]
        + [11, 12, 13, 11, 12, 13]
        + [14],
    ]
    assert positions.tolist() == expected


def test_d3_video_sampling_and_timestamps():
    # A 10 s clip at 30 fps is read at 2 fps; short clips keep at least 4 frames.
    assert (
        d3.sample_indices(300, 30.0)
        == np.linspace(0, 299, 20).round().astype(int).tolist()
    )
    assert d3.sample_indices(30, 30.0) == [0, 10, 19, 29]
    assert d3.sample_indices(2, 30.0) == [0, 1]
    assert len(d3.sample_indices(3000, 30.0)) == d3.VIDEO_MAX_FRAMES
    video = d3.D3Video(np.zeros((4, 8, 8, 3), np.uint8), 30.0, [0, 10, 19, 29])
    assert d3.frame_timestamps(video, 2) == pytest.approx([5 / 30, 24 / 30])


class _D3Tokenizer:
    """Character tokenizer with Qwen-style media placeholders."""

    special = {"<|image_pad|>": 300, "<|video_pad|>": 301, "<|vision_start|>": 302}

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            for token, token_id in self.special.items():
                if text.startswith(token):
                    ids.append(token_id)
                    text = text[len(token) :]
                    break
            else:
                ids.append(ord(text[0]) % 250 + 1)
                text = text[1:]
        return ids

    def apply_chat_template(self, conversation, **kwargs):
        content = conversation[1]["content"]
        if isinstance(content, str):
            return f"<s>{conversation[0]['content']}<u>{content}<a>"
        media = "".join(
            (
                "<|vision_start|><|image_pad|>"
                if part["type"] == "image"
                else "<|vision_start|><|video_pad|>"
            )
            for part in content[:-1]
        )
        return f"<u>{media}{content[-1]['text']}<a>"

    def convert_tokens_to_ids(self, token):
        return self.special[token]


@pytest.fixture
def tiny_d3(tiny_backbone):
    tiny_backbone.tokenizer = _D3Tokenizer()
    model = d3.D3Model("tiny")
    model.backbone = tiny_backbone
    model.codes = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
    model._image_token_id, model._video_token_id = 300, 301
    model.readout = mx.random.normal((d3.MAX_OPTIONS, 64), key=mx.random.key(4))
    return model


def test_d3_encode_matches_full_forward_without_causal_mask(tiny_d3, monkeypatch):
    from mlx_vlm.models.qwen3_5 import language

    backbone = tiny_d3.backbone
    ids = np.arange(5, 45, dtype=np.int32)
    positions = mx.array(np.tile(np.arange(len(ids), dtype=np.int32), (3, 1)))[:, None]

    def encode(chunk, causal):
        return _drain(
            backbone.encode(
                backbone.embed_tokens(ids), positions, lambda: chunk, causal
            )
        )

    tokens = mx.array(ids)[None]
    causal = backbone._text_model(tokens, position_ids=positions)[0, -1]
    assert np.allclose(np.array(encode(len(ids), True)), np.array(causal), atol=1e-4)

    monkeypatch.setattr(language, "_create_qwen3_5_attention_mask", lambda *a: None)
    full = backbone._text_model(tokens, position_ids=positions)[0, -1]
    for chunk in (len(ids), 7, 1):
        assert np.allclose(np.array(encode(chunk, False)), np.array(full), atol=1e-4)
    assert not np.allclose(np.array(full), np.array(causal), atol=1e-3)


def test_d3_run_answers_every_question(tiny_d3):
    request = {
        "state": {"ticket": "charged twice"},
        "questions": {
            "route": _ROUTE,
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "urgency": {"type": "score", "criteria": ["later", "today", "now"]},
        },
    }
    plan = tiny_d3.encode(request)
    steps = []

    def chunk():
        return 16

    run = tiny_d3.run(plan, chunk)
    try:
        while True:
            steps.append(next(run))
    except StopIteration as stop:
        result = stop.value
    lengths = [len(ids) for ids in plan.ids]
    assert sum(steps) == sum(lengths) == result["input_tokens"]
    assert len(steps) > len(lengths)
    answers = result["answers"]
    assert list(answers) == ["route", "angry", "urgency"]
    assert sum(answers["route"]["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert answers["route"]["choice"] in _ROUTE["criteria"]
    assert 0 <= answers["angry"]["noul"] <= 1
    assert answers["urgency"]["legend"] == {"0": "later", "1": "today", "2": "now"}
    whole = _drain(tiny_d3.run(tiny_d3.encode(request), lambda: 4096))
    assert whole["answers"]["route"] == answers["route"]
    json.dumps(result)


def test_d3_expands_media_placeholders(tiny_d3):
    ids = [1, 302, 300, 2, 302, 301, 3]
    assert tiny_d3._expand(ids, [3], [[7, 8, 301, 301]]) == [
        1,
        302,
        300,
        300,
        300,
        2,
        302,
        7,
        8,
        301,
        301,
        3,
    ]
    # Text requests keep a literal placeholder token as text.
    assert tiny_d3._expand(ids, [], []) == ids


def test_d3_rejects_literal_placeholder_with_media(tiny_d3, monkeypatch):
    monkeypatch.setattr(d3, "decode_images", lambda urls: [object()] if urls else [])
    monkeypatch.setattr(
        tiny_d3.backbone,
        "preprocess_images",
        lambda images: (np.zeros((16, 96), np.float32), np.array([[1, 4, 4]])),
    )
    tiny_d3.backbone.image_processor = SimpleNamespace(merge_size=2)
    request = {
        "state": "see <|image_pad|> here",
        "images": ["data:image/png;base64,AA=="],
        "questions": {"q": {"type": "noul"}},
    }
    with pytest.raises(DecisionRequestError, match="placeholder"):
        tiny_d3.encode(request)


def test_d3_video_encode_and_token_budget(tiny_d3, monkeypatch):
    decoded = []

    def decode(url):
        decoded.append(url)
        return d3.D3Video(np.zeros((4, 64, 64, 3), np.uint8), 2.0, [0, 1, 2, 3])

    monkeypatch.setattr(d3, "decode_video", decode)
    tiny_d3.backbone.image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    )
    request = {
        "state": "a clip",
        "videos": ["v0", "v1"],
        "questions": {"q": {"type": "noul"}},
    }
    # Four 64 x 64 frames are two temporal patches of 2 x 2 tokens.
    plan = tiny_d3.encode(request)
    assert plan.video_grids.tolist() == [[2, 4, 4], [2, 4, 4]]
    (ids,) = plan.ids
    assert int((ids == 301).sum()) == 16
    assert plan.positions[0].shape == (3, len(ids))

    # Decoding stops at the first video over the budget.
    decoded.clear()
    monkeypatch.setattr(d3, "VIDEO_MAX_TOKENS", 20)
    request["videos"] = [f"v{i}" for i in range(5)]
    with pytest.raises(DecisionRequestError, match="first 3 videos take 24"):
        tiny_d3.encode(request)
    assert decoded == ["v0", "v1", "v2"]


def test_d3_video_uri_errors_name_the_video(tiny_d3, monkeypatch):
    monkeypatch.setattr(d3, "require_opencv", lambda field: None)
    with pytest.raises(InvalidRequestError) as error:
        tiny_d3._decode_video("data:video/mp4,AAAA", 1)
    assert str(error.value) == "videos[1]: the video must use a base64 video data URI."
    assert error.value.field == "videos"


@pytest.mark.parametrize("model_class", [clef.ClefModel, openjev.OpenJevModel])
def test_clef_and_openjev_reject_videos(model_class):
    model = model_class("unused")
    request = {
        "state": "s",
        "videos": ["data:video/mp4;base64,AA=="],
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }
    with pytest.raises(DecisionRequestError, match="videos"):
        model.encode(request)


# --------------------------------------------------------------------------
# Qwen3.5 base-model checkpoints (d3 weights in mlx-vlm)


def _base_config(**overrides):
    text = dict(_TEXT_CONFIG, **overrides.pop("text", {}))
    return {
        "model_type": "qwen3_5",
        "text_config": text,
        "vision_config": {
            "model_type": "qwen3_5",
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [],
        },
        **overrides,
    }


@pytest.fixture
def base_compat(monkeypatch):
    """Install the base-model compat patch for one test, then restore mlx-vlm."""
    from mlx_vlm.models.qwen3_5 import config, language, qwen3_5

    from omlx.patches import mlx_vlm_qwen35_base_compat as compat

    for target, name in (
        (config.TextConfig, "from_dict"),
        (config.ModelConfig, "from_dict"),
        (language.Qwen3_5DecoderLayer, "__init__"),
        (language.Qwen3_5Model, "__init__"),
        (language.LanguageModel, "__init__"),
        (qwen3_5, "sanitize_key"),
    ):
        monkeypatch.setattr(target, name, inspect.getattr_static(target, name))
    monkeypatch.setattr(compat, "_APPLIED", False)
    assert compat.apply_mlx_vlm_qwen35_base_compat_patch()
    return compat


def test_layer_types_and_headless_base_model(base_compat):
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    schedule = [
        "linear_attention",
        "full_attention",
        "full_attention",
        "linear_attention",
    ]
    config = _base_config(
        text={"layer_types": schedule}, architectures=["Qwen3_5Model"]
    )
    assert base_compat.needs_base_compat(config)
    model = Model(ModelConfig.from_dict(config))
    layers = model.language_model.model.layers
    assert [layer.is_linear for layer in layers] == [True, False, False, True]
    assert "linear_attn" in layers[0] and "self_attn" in layers[2]
    assert "linear_attn" not in layers[2]
    assert model.language_model.model.fa_idx == 1
    assert "lm_head" not in model.language_model

    periodic = ["linear_attention", "full_attention"] * 2
    stock = _base_config(
        text={"layer_types": periodic},
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert not base_compat.needs_base_compat(stock)
    model = Model(ModelConfig.from_dict(stock))
    assert [layer.is_linear for layer in model.language_model.model.layers] == [
        True,
        False,
        True,
        False,
    ]
    assert model.language_model.model.fa_idx == 1
    assert "lm_head" in model.language_model


def test_base_layout_weights_sanitize_once():
    from mlx_vlm.models.qwen3_5 import Model, ModelConfig

    from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch

    apply_mlx_vlm_mtp_patch()
    model = Model(ModelConfig.from_dict(_base_config(architectures=["Qwen3_5Model"])))
    raw = {
        "language_model.embed_tokens.weight": mx.zeros((320, 64)),
        "language_model.norm.weight": mx.zeros((64,)),
        "language_model.layers.0.input_layernorm.weight": mx.zeros((64,)),
        "language_model.layers.0.linear_attn.conv1d.weight": mx.zeros((96, 1, 3)),
        "visual.blocks.0.norm1.weight": mx.zeros((32,)),
    }
    once = model.sanitize(dict(raw))
    assert set(once) == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        "vision_tower.blocks.0.norm1.weight",
    }
    assert once["language_model.model.norm.weight"].tolist() == [1.0] * 64
    assert once["language_model.model.layers.0.linear_attn.conv1d.weight"].shape == (
        96,
        3,
        1,
    )
    # A quantized output is sanitized again on load; norms must not shift twice.
    twice = model.sanitize(dict(once))
    assert set(twice) == set(once)
    assert twice["language_model.model.norm.weight"].tolist() == [1.0] * 64
