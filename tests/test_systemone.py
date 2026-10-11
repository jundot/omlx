# SPDX-License-Identifier: Apache-2.0
"""Tests for the decision models behind /v1/systemone (Clef and OpenJev)."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from omlx.models import clef, openjev  # noqa: E402
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
