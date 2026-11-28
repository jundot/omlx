# SPDX-License-Identifier: Apache-2.0
"""Tests for the decision models behind /v1/systemone (Clef, OpenJev, d3, d1,
autotrust JEV)."""

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
from omlx.models import autotrust_jev, clef, d1, d3, decision, openjev  # noqa: E402
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
    shared = _drain(model.run(model.encode(request), lambda: 64))

    monkeypatch.setattr(model, "_prefill_shared_prefix", lambda *a: (yield from ()))
    independent = _drain(model.run(model.encode(request), lambda: 64))

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


def test_backbone_files_hide_a_head_only_in_an_autotrust_jev_bundle(tmp_path):
    for name in ("model.safetensors", "head.safetensors", "config.json"):
        (tmp_path / name).write_text(name)
    # Without the autotrust JEV sidecars, head.safetensors is an ordinary file.
    with decision._backbone_files(tmp_path) as view:
        assert view == tmp_path

    (tmp_path / "judge_config.json").write_text(
        json.dumps(
            {
                "slots": {"ranges": {}, "template_version": "bare-v1"},
                "verbalizer_ids": [],
            }
        )
    )
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}")
    with decision._backbone_files(tmp_path) as view:
        assert view != tmp_path
        assert "head.safetensors" not in {p.name for p in view.iterdir()}
        assert "adapter" in {p.name for p in view.iterdir()}


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
# LiquidAI d1


@pytest.fixture(scope="module")
def tiny_d1():
    from mlx_vlm.models import d1_lfm3_vl

    config = d1_lfm3_vl.ModelConfig.from_dict(
        {
            "model_type": "lfm3_vl",
            "d1_attention": "set",
            "d1_set_size": 16,
            "projector_hidden_size": 64,
            "text_config": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 6,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "layer_types": [
                    "conv",
                    "full_attention",
                    "conv",
                    "sliding_attention",
                    "conv",
                    "full_attention",
                ],
                "sliding_window": 8,
            },
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "layer_types": ["conv", "full_attention"],
            },
        }
    )
    model = d1_lfm3_vl.Model(config)
    mx.eval(model.parameters())
    model.eval()
    wrapper = d1.D1Model("unused")
    wrapper.model = model
    return wrapper


def test_d1_layer_steps_match_mlx_vlm_forward(tiny_d1):
    ids = list(range(10, 40))
    question = [-1] * 18 + [1] * 6 + [2] * 6
    full = np.array(tiny_d1.model.d1_hidden(ids, 18, question))
    for chunk in (len(ids), 7, 1):
        steps = tiny_d1._hidden(ids, 18, question, [], lambda chunk=chunk: chunk)
        counts = []
        try:
            while True:
                counts.append(next(steps))
        except StopIteration as stop:
            hidden = np.array(stop.value)
        assert sum(counts) == len(ids)
        assert np.array_equal(hidden, full)


def test_d1_rejects_media_instructions_and_videos():
    from mlx_vlm.models.d1_lfm3_vl.api import SystemOneApi

    model = d1.D1Model("unused")
    model._requests = SystemOneApi()
    spoken = {"type": "noul", "instructions": {"type": "audio", "path": "/etc/hosts"}}
    with pytest.raises(DecisionRequestError, match="not media"):
        model.encode({"state": "s", "questions": {"q": spoken}})
    with pytest.raises(DecisionRequestError, match="videos"):
        model.encode(
            {
                "state": "s",
                "videos": ["data:video/mp4;base64,AA=="],
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            }
        )


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


# --------------------------------------------------------------------------
# autotrust JEV

_JEV_RANGES = {"noul": (0, 2), "score": (2, 8), "choice": (8, 24)}


class _FixedHiddenBackbone:
    """Backbone stand-in: character tokens, and a prefill that returns one
    fixed hidden state after yielding its token count."""

    def __init__(self, hidden=None, limit=None):
        self.hidden = hidden
        self.max_position_embeddings = limit

    def tokenize(self, text):
        return _chars(text)

    def make_cache(self):
        return []

    def embed(self, ids, *args):
        return mx.zeros((1, len(ids), 4)), None

    def embed_continuation(self, ids, last_position):
        return mx.zeros((1, len(ids), 4)), None

    def prefill(self, ids, embeds, positions, cache, chunk_size, keep_all):
        yield len(ids)
        return self.hidden


def _jev_model(backbone, weight=None, bias=None, temperatures=None):
    """An AutoTrustJevModel set up as load() would, without files."""
    model = autotrust_jev.AutoTrustJevModel("unused")
    model.backbone = backbone
    model._ranges = dict(_JEV_RANGES)
    model._temperatures = temperatures or {"noul": 1.0, "score": 1.0, "choice": 1.0}
    model._choice_labels = list("ABCDEFGHIJKLMNOP")
    model._head_weight = weight
    model._head_bias = bias
    return model


def _jev_prompts(request):
    model = _jev_model(_FixedHiddenBackbone())
    return [_text(r.passes[0]) for r in model.encode(request).readouts]


def test_autotrust_jev_prompts_match_the_release_format():
    prompts = _jev_prompts(
        {
            "state": {"user": "mara", "note": "größer"},
            "questions": {
                "allowed": {"type": "noul", "instructions": "Is mara allowed?"},
                "risk": {
                    "type": "score",
                    "instructions": "Rate the risk on a 0-5 scale.",
                    "criteria": ["0", "1", "2", "3", "4", "5"],
                },
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "Payments and\ninvoices", "tech": None},
                },
            },
        }
    )

    state = '{"user": "mara", "note": "größer"}'
    assert prompts == [
        f"[kind] noul\n[state] {state}\n[question] Is mara allowed?\n[options]\n"
        "false\ntrue\n[decision]:",
        f"[kind] score\n[state] {state}\n[question] Rate the risk on a 0-5 scale."
        "\n[options]\n0\n1\n2\n3\n4\n5\n[decision]:",
        f"[kind] choice\n[state] {state}\n[question] Which team?\n[options]\n"
        "A) billing: Payments and invoices\nB) tech\n[decision]:",
    ]


def test_autotrust_jev_readout_slices_the_kind_and_applies_its_temperature():
    rng = np.random.default_rng(4)
    hidden = mx.array(rng.normal(size=8).astype(np.float32))
    weight = mx.array(rng.normal(size=(24, 8)).astype(np.float32))
    bias = mx.array(rng.normal(size=24).astype(np.float32))
    temperatures = {"noul": 1.01, "score": 1.2, "choice": 0.98}
    model = _jev_model(_FixedHiddenBackbone(hidden), weight, bias, temperatures)
    request = {
        "state": "s",
        "questions": {
            "ok": {"type": "noul", "instructions": "ok?"},
            "level": {
                "type": "score",
                "instructions": "level?",
                "criteria": ["a", "b", "c", "d", "e", "f"],
            },
            "pick": {
                "type": "choice",
                "instructions": "pick?",
                "criteria": {"x": None, "y": None, "z": None},
            },
        },
    }

    answers = _drain(model.run(model.encode(request), lambda: 64))["answers"]

    logits = np.asarray(weight @ hidden + bias, dtype=np.float64)

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    noul = softmax(logits[0:2] / 1.01)
    score = softmax(logits[2:8] / 1.2)
    choice = softmax(logits[8:11] / 0.98)
    # Slot order is false, true: the answer is P(true).
    assert answers["ok"] == {"type": "noul", "noul": decision.round4(noul[1])}
    assert answers["level"]["score"] == decision.round4(float(np.arange(6) @ score))
    assert answers["level"]["legend"] == dict(zip("012345", "abcdef", strict=True))
    assert list(answers["level"]["probabilities"].values()) == [
        decision.round4(p) for p in score
    ]
    assert answers["pick"]["probabilities"] == {
        k: decision.round4(p) for k, p in zip("xyz", choice, strict=True)
    }
    assert answers["pick"]["choice"] == "xyz"[int(np.argmax(choice))]
    json.dumps(answers)


def test_autotrust_jev_shared_prefix_matches_independent_readouts(
    tiny_backbone, monkeypatch
):
    tiny_backbone.tokenizer = _CharTokenizer()
    rng = np.random.default_rng(5)
    model = _jev_model(
        tiny_backbone,
        mx.array(rng.normal(size=(24, 64)).astype(np.float32)),
        mx.array(rng.normal(size=24).astype(np.float32)),
        {"noul": 1.01, "score": 1.0, "choice": 0.98},
    )
    request = {
        "state": {"ticket": "charged twice, nobody replied"},
        "questions": {
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {f"t{i}": None for i in range(16)},
            },
            "refund": {"type": "noul", "instructions": "Is a refund due?"},
            "channel": {
                "type": "choice",
                "instructions": "Which channel?",
                "criteria": {"mail": "e-mail", "phone": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["0", "1", "2", "3", "4", "5"],
            },
        },
    }

    def run_counting():
        steps, tokens = model.run(model.encode(request), lambda: 16), 0
        while True:
            try:
                tokens += next(steps)
            except StopIteration as stop:
                return stop.value, tokens

    shared, shared_tokens = run_counting()
    monkeypatch.setattr(model, "_prefill_shared_prefix", lambda *a: (yield from ()))
    independent, independent_tokens = run_counting()

    assert shared_tokens < independent_tokens
    assert shared["input_tokens"] == independent["input_tokens"]
    for key in ("team", "channel", "urgency"):
        a = shared["answers"][key]["probabilities"]
        b = independent["answers"][key]["probabilities"]
        assert np.allclose(list(a.values()), list(b.values()), atol=2e-4)
    for key in ("angry", "refund"):
        assert shared["answers"][key]["noul"] == pytest.approx(
            independent["answers"][key]["noul"], abs=2e-4
        )


@pytest.mark.parametrize(
    ("request_patch", "message"),
    [
        ({"images": ["data:image/png;base64,AA"]}, "text only"),
        ({"videos": ["data:video/mp4;base64,AA"]}, "text only"),
        (
            {
                "questions": {
                    "q": {"type": "choice", "instructions": "?", "criteria": {}}
                }
            },
            "at least two",
        ),
        ({"questions": {"q": {"type": "noul"}}}, "instructions"),
        (
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "?",
                        "criteria": {"a": None},
                    }
                }
            },
            "at least two",
        ),
        (
            {
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "?",
                        "criteria": {str(i): None for i in range(257)},
                    }
                }
            },
            "at most 256",
        ),
        (
            {
                "questions": {
                    "q": {"type": "score", "instructions": "?", "criteria": ["a"]}
                }
            },
            "at least two levels",
        ),
        (
            {
                "questions": {
                    "q": {"type": "noul", "instructions": "?", "criteria": ["a"]}
                }
            },
            "object",
        ),
    ],
)
def test_autotrust_jev_rejects_requests_outside_the_native_kinds(
    request_patch, message
):
    model = _jev_model(_FixedHiddenBackbone())
    request = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}

    with pytest.raises(DecisionRequestError, match=message):
        model.encode({**request, **request_patch})


def test_autotrust_jev_prompt_over_the_context_limit_is_an_error():
    model = _jev_model(_FixedHiddenBackbone(limit=40))
    request = {
        "state": "x" * 40,
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }

    with pytest.raises(DecisionContextLengthError, match="limit is 40"):
        model.encode(request)


def _judge(**changes):
    judge = {
        "hidden_size": 64,
        "weights_mode": "unmerged",
        "softcap": None,
        "slots": {
            "ranges": {kind: list(r) for kind, r in _JEV_RANGES.items()},
            "verbalizers": ["false", "true", *"012345", *"ABCDEFGHIJKLMNOP"],
            "template_version": "bare-v1",
        },
        "verbalizer_ids": list(range(24)),
    }
    judge.update(changes)
    return judge


def test_autotrust_jev_judge_config_is_checked():
    ranges, verbalizers, ids = autotrust_jev._check_judge(_judge())
    assert ranges == _JEV_RANGES and verbalizers[8] == "A" and ids == list(range(24))

    for changes, message in [
        ({"weights_mode": "merged"}, "weights_mode"),
        ({"softcap": 30.0}, "softcap"),
        ({"verbalizer_ids": list(range(23))}, "same length"),
    ]:
        with pytest.raises(ValueError, match=message):
            autotrust_jev._check_judge(_judge(**changes))
    bad_score = _judge()
    bad_score["slots"]["ranges"]["score"] = [2, 7]
    with pytest.raises(ValueError, match="score slot range"):
        autotrust_jev._check_judge(bad_score)


def test_autotrust_jev_temperatures_default_to_one(tmp_path):
    path = tmp_path / "calibration.json"
    assert autotrust_jev._read_temperatures(path) == {
        "noul": 1.0,
        "score": 1.0,
        "choice": 1.0,
    }
    path.write_text(json.dumps({"per_kind": {"noul": 1.01, "choice": 0.98}}))
    assert autotrust_jev._read_temperatures(path) == {
        "noul": 1.01,
        "score": 1.0,
        "choice": 0.98,
    }
    path.write_text(json.dumps({"per_kind": {"score": 0}}))
    with pytest.raises(ValueError, match="score temperature"):
        autotrust_jev._read_temperatures(path)


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _release_tournament(count, in_group_of, final_of):
    """The release's tournament (``s1_dist`` in ``serve_decide.py``),
    transcribed without its async calls. ``in_group_of(group)`` and
    ``final_of(finalists)`` return the probabilities of one pass."""
    g = math.ceil(count / 16)
    base, extra = divmod(count, g)
    groups, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0)
        groups.append(list(range(i, i + size)))
        i += size
    parts_g = [in_group_of(group) for group in groups]
    in_group = {o: p for g, ps in zip(groups, parts_g) for o, p in zip(g, ps)}
    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
    rest = sorted(
        (o for g in groups for o in g if o not in set(chosen)),
        key=lambda o: (-in_group[o], o),
    )
    fin = sorted(chosen + rest[: max(0, 16 - len(chosen))])
    final = dict(zip(fin, final_of(fin)))
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    share = [0.0] * len(groups)
    cap = [0.0] * len(groups)
    for f in fin:
        share[group_of[f]] += final[f]
        cap[group_of[f]] += in_group[f]
    among = sum(a * b for a, b in zip(share, cap))
    p = [
        final[o] * among if o in final else share[group_of[o]] * in_group[o]
        for o in range(count)
    ]
    total = sum(p)
    return groups, in_group, fin, final, [x / total for x in p]


@pytest.mark.parametrize("count", [17, 40, 64, 255, 256])
@pytest.mark.parametrize("ties", [False, True])
def test_autotrust_jev_tournament_follows_the_release(count, ties):
    rng = np.random.default_rng(count)

    def probabilities(n):
        values = np.ones(n) if ties else rng.random(n)
        return list(values / values.sum())

    groups, in_group, finalists, final, expected = _release_tournament(
        count, lambda g: probabilities(len(g)), lambda f: probabilities(len(f))
    )

    assert autotrust_jev.tournament_groups(count, 16) == groups
    assert autotrust_jev.tournament_finalists(groups, in_group, 16) == finalists
    got = autotrust_jev.tournament_probabilities(groups, in_group, final)
    assert np.allclose(got, expected, rtol=0, atol=1e-12)


def _choice_run(count, monkeypatch, temperature=0.98, kind="choice"):
    """Run one question with ``count`` options; record every pass's prompt."""
    rng = np.random.default_rng(6)
    hidden = mx.array(rng.normal(size=8).astype(np.float32))
    weight = mx.array(rng.normal(size=(24, 8)).astype(np.float32))
    bias = mx.array(rng.normal(size=24).astype(np.float32))
    temperatures = {"noul": 1, "score": 1, "choice": temperature}
    model = _jev_model(_FixedHiddenBackbone(hidden), weight, bias, temperatures)
    if kind == "choice":
        criteria = {f"o{i}": None for i in range(count)}
    else:
        criteria = [f"level {i}" for i in range(count)]
    request = {
        "state": "s",
        "questions": {"q": {"type": kind, "instructions": "?", "criteria": criteria}},
    }
    prompts = []
    readout = model._readout

    def spy(ids, *args):
        prompts.append(_text(ids))
        return (yield from readout(ids, *args))

    monkeypatch.setattr(model, "_readout", spy)
    result = _drain(model.run(model.encode(request), lambda: 64))
    head = np.asarray(weight @ hidden + bias, dtype=np.float64)[8:24] / temperature
    return result, prompts, head


def _option_lines(prompt):
    return prompt.split("[options]\n")[1].split("\n[decision]:")[0].split("\n")


def test_autotrust_jev_sixteen_options_are_one_pass(monkeypatch):
    result, prompts, head = _choice_run(16, monkeypatch)

    assert len(prompts) == 1
    assert _option_lines(prompts[0])[-1] == "P) o15"
    assert list(result["answers"]["q"]["probabilities"].values()) == [
        decision.round4(p) for p in _softmax(head)
    ]


def test_autotrust_jev_more_options_run_a_tournament(monkeypatch):
    result, prompts, head = _choice_run(20, monkeypatch)

    # Two groups of ten, then a final of 16, each labelled from A.
    assert len(prompts) == 3
    assert _option_lines(prompts[0]) == [f"{'ABCDEFGHIJ'[i]}) o{i}" for i in range(10)]
    assert _option_lines(prompts[1]) == [
        f"{'ABCDEFGHIJ'[i]}) o{10 + i}" for i in range(10)
    ]
    final_lines = _option_lines(prompts[2])
    assert [line[0] for line in final_lines] == list("ABCDEFGHIJKLMNOP")
    # The same hidden state every pass, so a pass's probabilities depend on
    # the option count only: the release's tournament over those values.
    *_, expected = _release_tournament(
        20,
        lambda g: list(_softmax(head[: len(g)])),
        lambda f: list(_softmax(head[: len(f)])),
    )
    answer = result["answers"]["q"]
    assert list(answer["probabilities"].values()) == [
        decision.round4(p) for p in expected
    ]
    assert result["input_tokens"] == sum(len(prompt) for prompt in prompts)


def test_autotrust_jev_score_with_many_levels_is_a_tournament(monkeypatch):
    result, prompts, _ = _choice_run(20, monkeypatch, kind="score")

    answer = result["answers"]["q"]
    assert len(prompts) == 3 and prompts[0].startswith("[kind] choice\n")
    assert answer["type"] == "score" and len(answer["probabilities"]) == 20
    assert answer["legend"]["19"] == "level 19"


def test_autotrust_jev_tournament_final_must_fit_the_context():
    # Each group of nine or eight long options fits; sixteen of them do not.
    long_options = {f"o{i}-" + "x" * 30: None for i in range(17)}
    request = {
        "state": "s",
        "questions": {
            "q": {"type": "choice", "instructions": "?", "criteria": long_options}
        },
    }
    model = _jev_model(_FixedHiddenBackbone(limit=400))

    with pytest.raises(DecisionContextLengthError):
        model.encode(request)
    model = _jev_model(_FixedHiddenBackbone(limit=700))
    assert len(model.encode(request).readouts[0].passes) == 2


def test_autotrust_jev_score_without_six_levels_is_asked_as_a_choice():
    rng = np.random.default_rng(7)
    hidden = mx.array(rng.normal(size=8).astype(np.float32))
    weight = mx.array(rng.normal(size=(24, 8)).astype(np.float32))
    bias = mx.array(rng.normal(size=24).astype(np.float32))
    model = _jev_model(
        _FixedHiddenBackbone(hidden),
        weight,
        bias,
        {"noul": 1, "score": 3, "choice": 0.9},
    )
    request = {
        "state": "s",
        "questions": {
            "risk": {
                "type": "score",
                "instructions": "How risky?",
                "criteria": ["low", "1: some", "high: \nvery"],
            }
        },
    }

    plan = model.encode(request)
    answer = _drain(model.run(plan, lambda: 64))["answers"]["risk"]

    assert _text(plan.readouts[0].passes[0]).endswith(
        "[options]\nA) low\nB) some\nC) high: very\n[decision]:"
    )
    assert _text(plan.readouts[0].passes[0]).startswith("[kind] choice\n")
    # The choice slots and the choice temperature, not the score ones.
    probs = _softmax(np.asarray(weight @ hidden + bias, dtype=np.float64)[8:11] / 0.9)
    assert answer["type"] == "score"
    assert answer["legend"] == {"0": "low", "1": "1: some", "2": "high: \nvery"}
    assert list(answer["probabilities"].values()) == [decision.round4(p) for p in probs]
    assert answer["score"] == decision.round4(float(np.arange(3) @ probs))


def test_autotrust_jev_rubric_text_follows_the_question():
    def question_line(question):
        prompt = _jev_prompts({"state": "s", "questions": {"q": question}})[0]
        return prompt.split("\n")[2]

    assert (
        question_line(
            {
                "type": "noul",
                "instructions": "Is mara allowed?",
                "criteria": {"true": "she may act", "false": "Not allowed."},
            }
        )
        == "[question] Is mara allowed? True means: she may act. "
        "False means: Not allowed."
    )
    assert question_line({"type": "noul", "instructions": "Ok?", "criteria": {}}) == (
        "[question] Ok?"
    )
    six = {"type": "score", "instructions": "Rate the risk."}
    assert (
        question_line({**six, "criteria": ["0: none", "1", "2", "3", "4", "5: severe"]})
        == "[question] Rate the risk. Levels: 0 = none; 5 = severe."
    )
    assert (
        question_line({**six, "criteria": ["0", "1", "2", "3", "4", "5"]})
        == "[question] Rate the risk."
    )


def test_autotrust_jev_decisions_and_systemone_give_the_same_prompts():
    from omlx.api.decisions import to_systemone
    from omlx.api.decisions_models import DecisionRequest

    decisions = DecisionRequest.model_validate(
        {
            "model": "m",
            "input": "The invoice was paid twice.",
            "questions": [
                {"name": "ok", "type": "predicate", "instructions": "Is it ok?"},
                {
                    "name": "team",
                    "type": "choice",
                    "instructions": "Which team?",
                    "choices": [
                        {"value": "billing", "description": "Payments"},
                        {"value": "tech"},
                    ],
                },
                {
                    "name": "risk",
                    "type": "score",
                    "instructions": "Rate the risk.",
                    "levels": [
                        {"label": "low", "description": "minor"},
                        {"label": "mid"},
                        {"label": "high"},
                    ],
                },
            ],
        }
    )
    systemone = {
        "state": "The invoice was paid twice.",
        "questions": {
            "ok": {"type": "noul", "instructions": "Is it ok?"},
            "team": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "tech": None},
            },
            "risk": {
                "type": "score",
                "instructions": "Rate the risk.",
                "criteria": ["low: minor", "mid", "high"],
            },
        },
    }

    assert _jev_prompts(to_systemone(decisions)) == _jev_prompts(systemone)


# A tiny autotrust JEV bundle on disk: Qwen3.5 text backbone with both layer
# kinds, a word-level tokenizer that knows the verbalizers, a random PEFT
# adapter, a random head, judge config and calibration.
_JEV_BUNDLE_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_value_heads": 4,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "vocab_size": 64,
    "full_attention_interval": 2,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}
_JEV_WORDS = [
    "<unk>",
    "false",
    "true",
    *"012345",
    *"ABCDEFGHIJKLMNOP",
    *"[]):\n xy",
    *("kind", "state", "question", "options", "decision", "noul", "score"),
    *("choice", "Is", "it", "ok", "Which", "team", "Rate", "the", "risk", "?"),
]
_JEV_TARGETS = {
    "linear_attn": ("in_proj_qkv", "in_proj_z", "out_proj"),
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}
_JEV_BUNDLE_TEMPERATURES = {"noul": 1.01, "score": 1.2, "choice": 0.98}


def _module(root, path):
    for part in path.split("."):
        root = root[int(part)] if isinstance(root, list) else root[part]
    return root


def _write_jev_bundle(path, judge_changes=None, adapter_config_changes=None):
    """Write a tiny autotrust JEV bundle; return (adapter tensors, head)."""
    from mlx_lm.models import qwen3_5
    from tokenizers import Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    path.mkdir(parents=True)
    mx.random.seed(11)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(_JEV_BUNDLE_CONFIG))
    mx.eval(model.parameters())
    weights = {
        name.removeprefix("language_model."): value
        for name, value in tree_flatten(model.parameters())
    }
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(_JEV_BUNDLE_CONFIG))
    vocab = {word: i for i, word in enumerate(_JEV_WORDS)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"\w+|[^\w\s]|\s"), behavior="isolated"
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>"
    ).save_pretrained(path)

    text_model = model.language_model.model
    tensors = {}
    for i, layer in enumerate(text_model.layers):
        for block, names in _JEV_TARGETS.items():
            if block not in layer:
                continue
            for name in names:
                module_path = f"layers.{i}.{block}.{name}"
                out_dims, in_dims = _module(text_model, module_path).weight.shape
                tensors[module_path] = (
                    mx.random.normal((4, in_dims)),
                    mx.random.normal((out_dims, 4)) * 0.05,
                )
    (path / "adapter").mkdir()
    adapter_config = {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "bias": "none"}
    adapter_config.update(adapter_config_changes or {})
    (path / "adapter" / "adapter_config.json").write_text(json.dumps(adapter_config))
    mx.save_safetensors(
        str(path / "adapter" / "adapter_model.safetensors"),
        {
            f"base_model.model.model.{p}.lora_{name}.weight": t
            for p, (a, b) in tensors.items()
            for name, t in (("A", a), ("B", b))
        },
    )

    head = (mx.random.normal((24, 64)), mx.random.normal((24,)))
    mx.save_safetensors(
        str(path / "head.safetensors"), {"proj.weight": head[0], "proj.bias": head[1]}
    )
    verbalizers = ["false", "true", *"012345", *"ABCDEFGHIJKLMNOP"]
    judge = {
        "adapter_subfolder": "adapter",
        "hidden_size": 64,
        "weights_mode": "unmerged",
        "softcap": None,
        "slots": {
            "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
            "verbalizers": verbalizers,
            "template_version": "bare-v1",
        },
        "verbalizer_ids": [vocab[word] for word in verbalizers],
    }
    judge.update(judge_changes or {})
    (path / "judge_config.json").write_text(json.dumps(judge))
    (path / "calibration.json").write_text(
        json.dumps({"per_kind": _JEV_BUNDLE_TEMPERATURES})
    )
    return tensors, head


_JEV_BUNDLE_REQUEST = {
    "state": {"note": "the team is ok"},
    "questions": {
        "ok": {"type": "noul", "instructions": "Is it ok?"},
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"A": None, "B": "the B team", "C": None},
        },
        "risk": {
            "type": "score",
            "instructions": "Rate the risk.",
            "criteria": ["0", "1", "2", "3", "4", "5"],
        },
    },
}


def _probabilities(answer):
    if answer["type"] == "noul":
        return [1.0 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def test_autotrust_jev_bundle_loads_and_matches_merged_weights(tmp_path):
    bundle = tmp_path / "jev"
    tensors, (head_weight, head_bias) = _write_jev_bundle(bundle)
    model = autotrust_jev.AutoTrustJevModel(str(bundle))
    model.load()

    plan = model.encode(_JEV_BUNDLE_REQUEST)
    answers = _drain(model.run(plan, lambda: 7))["answers"]

    # Reference: a fresh load with the adapter merged into the weights, and
    # the head readout done by hand.
    reference = DecisionBackbone(str(bundle))
    reference.load()
    for module_path, (a, b) in tensors.items():
        leaf = _module(reference._text_model, module_path)
        leaf.weight = leaf.weight + 2.0 * (b @ a)
    slots = {"noul": slice(0, 2), "score": slice(2, 8), "choice": slice(8, 11)}
    for readout in plan.readouts:
        embeds, positions = reference.embed(readout.passes[0])
        hidden = _drain(
            reference.prefill(
                readout.passes[0],
                embeds,
                positions,
                reference.make_cache(),
                lambda: 1024,
                keep_all=False,
            )
        )
        logits = np.asarray(head_weight @ hidden + head_bias, dtype=np.float64)
        z = logits[slots[readout.kind]] / _JEV_BUNDLE_TEMPERATURES[readout.kind]
        expected = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        got = _probabilities(answers[readout.question_id])
        assert np.allclose(got, expected, atol=2e-4), readout.question_id

    # Asked one at a time, every question gets the same answer.
    for question_id, question in _JEV_BUNDLE_REQUEST["questions"].items():
        alone = {**_JEV_BUNDLE_REQUEST, "questions": {question_id: question}}
        single = _drain(model.run(model.encode(alone), lambda: 7))["answers"]
        assert np.allclose(
            _probabilities(single[question_id]),
            _probabilities(answers[question_id]),
            atol=2e-4,
        )
    model.close()


@pytest.mark.parametrize(
    ("judge_changes", "adapter_changes", "remove", "message"),
    [
        (
            # The head is sized for the wrong hidden size too, so the check
            # that fails is the backbone's.
            {"hidden_size": 32},
            None,
            None,
            "hidden size 64 is not 32",
        ),
        ({"verbalizer_ids": list(range(30, 54))}, None, None, "does not encode"),
        (None, {"use_dora": True}, None, "DoRA"),
        (None, None, "adapter/adapter_config.json", "not an autotrust JEV bundle"),
        ({"weights_mode": "merged"}, None, None, "weights_mode"),
    ],
)
def test_autotrust_jev_load_refuses_bundles_that_do_not_fit(
    tmp_path, judge_changes, adapter_changes, remove, message
):
    bundle = tmp_path / "jev"
    _write_jev_bundle(bundle, judge_changes, adapter_changes)
    if judge_changes and judge_changes.get("hidden_size") == 32:
        mx.save_safetensors(
            str(bundle / "head.safetensors"),
            {"proj.weight": mx.zeros((24, 32)), "proj.bias": mx.zeros((24,))},
        )
    if remove:
        (bundle / remove).unlink()

    with pytest.raises(ValueError, match=message):
        autotrust_jev.AutoTrustJevModel(str(bundle)).load()


def test_autotrust_jev_option_limit_follows_the_choice_slots(monkeypatch):
    model = _jev_model(
        _FixedHiddenBackbone(mx.ones((8,))), mx.ones((24, 8)), mx.zeros((24,))
    )
    model._choice_labels = list("ABCDEFGHIJ")

    def request(count):
        criteria = {f"o{i}": None for i in range(count)}
        return {
            "state": "s",
            "questions": {
                "q": {"type": "choice", "instructions": "?", "criteria": criteria}
            },
        }

    # Ten groups of ten, then a final of ten: the most ten slots can finish.
    plan = model.encode(request(100))
    assert len(plan.readouts[0].passes) == 10
    answer = _drain(model.run(plan, lambda: 64))["answers"]["q"]
    assert len(answer["probabilities"]) == 100
    with pytest.raises(DecisionRequestError, match="at most 100 options"):
        model.encode(request(101))


class _CostlyEuroBackbone(_FixedHiddenBackbone):
    """Character tokens, except that a euro sign takes 20 tokens."""

    def tokenize(self, text):
        return [t for c in text for t in ([0] * 20 if c == "€" else [ord(c)])]


def test_autotrust_jev_tournament_final_check_counts_tokens():
    # Sixteen long ASCII options and four short ones that cost many tokens:
    # by characters the ASCII ones look like the worst final, by tokens the
    # euro options are.
    options = [f"o{i:02d}" + "x" * 26 for i in range(16)]
    options += [f"e{i}€€" for i in range(4)]
    backbone = _CostlyEuroBackbone()
    lines_tokens = {o: len(backbone.tokenize(f"A) {o}")) for o in options}
    assert min(lines_tokens[o] for o in options[16:]) > max(
        lines_tokens[o] for o in options[:16]
    )
    request = {
        "state": "s",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "?",
                "criteria": dict.fromkeys(options),
            }
        },
    }
    model = _jev_model(backbone)

    def final_tokens(chosen):
        lines = model._choice_lines(chosen)
        return len(model._prompt_ids("choice", "s", "?", lines))

    by_characters = final_tokens(sorted(options, key=len, reverse=True)[:16])
    by_tokens = final_tokens(options[16:] + options[:12])
    assert by_characters < by_tokens
    backbone.max_position_embeddings = (by_characters + by_tokens) // 2

    with pytest.raises(DecisionContextLengthError):
        model.encode(request)
