# SPDX-License-Identifier: Apache-2.0
"""Tests for MiniCPM-OCR-Jev decision adapter and required Qwen3.5 runtime patches."""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

from omlx.model_discovery import decision_kind, detect_model_type  # noqa: E402
from omlx.models import minicpm_jev  # noqa: E402
from omlx.models.decision import DecisionRequestError  # noqa: E402
from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_patch  # noqa: E402


def _integration_path(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value) if value else None


_MINICPM_PATH = _integration_path("OMLX_MINICPM_JEV_MODEL")
_MINICPM_EXAMPLES = _integration_path("OMLX_MINICPM_JEV_EXAMPLES")


def _png_data_url(width: int = 32, height: int = 32, color=(255, 255, 255)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _drain(steps):
    while True:
        try:
            next(steps)
        except StopIteration as stop:
            return stop.value


_STUB_TOKEN_IDS = {
    "<|image_pad|>": 999,
    "<|video_pad|>": 998,
    "<|vision_start|>": 997,
    "<|vision_end|>": 996,
}


def _stub_prompt_tokens(text):
    return [
        _STUB_TOKEN_IDS[token]
        for token in re.findall(
            r"<\|(?:image_pad|video_pad|vision_start|vision_end)\|>", text
        )
    ] + [minicpm_jev.EXPECTED_LAST_TOKEN]


def _stub_model():
    model = minicpm_jev.MiniCPMJevModel("stub")
    model._image_token_id = 999
    model._video_token_id = 998
    model._vision_start_token_id = 997
    model._vision_end_token_id = 996

    def encode(embeds, positions, chunk_size, causal):
        yield embeds.shape[1]
        return mx.ones((4,), dtype=mx.float32)

    model.backbone = SimpleNamespace(
        has_vision=True,
        max_position_embeddings=2048,
        image_processor=SimpleNamespace(merge_size=2),
        tokenize=_stub_prompt_tokens,
        tokenizer=SimpleNamespace(
            get_vocab=lambda: dict(_STUB_TOKEN_IDS),
            convert_tokens_to_ids=_STUB_TOKEN_IDS.get,
        ),
        preprocess_images=lambda imgs: (
            np.zeros((8, 4), dtype=np.float32),
            np.array([[1, 2, 2], [1, 2, 2]]),
        ),
        embed=lambda ids, pv, gr: (
            mx.zeros((1, len(ids), 4), dtype=mx.float32),
            mx.zeros((3, 1, len(ids)), dtype=mx.int32),
        ),
        encode=encode,
        model=SimpleNamespace(score=lambda x: mx.array([[0.0, 2.0]], dtype=mx.float32)),
        load=lambda: None,
        close=lambda: None,
    )
    return model


# --------------------------------------------------------------------------
# Model discovery


def test_decision_kind_minicpm_jev_no_collision(tmp_path):
    # MiniCPM-OCR-Jev should not be misclassified as OpenJev even with "jev" or
    # "openjev" in its directory name.
    jev_dir = tmp_path / "openjev-minicpm-copy"
    jev_dir.mkdir()
    (jev_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "problem_type": "single_label_classification",
                "vision_config": {"depth": 24, "hidden_size": 1024},
            }
        )
    )
    (jev_dir / "decision_config.json").write_text(
        json.dumps(
            {
                "format_id": "minicpm-ocr-jev-v1",
                "prompt_template_md5": "9aab68e8ba8ab4da17ba9752bddb199c",
                "images_per_prompt": 2,
            }
        )
    )
    assert decision_kind(jev_dir) == "minicpm_jev"
    assert detect_model_type(jev_dir) == "decision"

    jev_native = tmp_path / "custom-classifier"
    jev_native.mkdir()
    (jev_native / "config.json").write_text(
        json.dumps({"model_type": "minicpm_ocr_jev"})
    )
    assert decision_kind(jev_native) == "minicpm_jev"

    # Plain OpenJev still routes to openjev.
    oj = tmp_path / "openjev-mlx"
    oj.mkdir()
    (oj / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {"depth": 24}})
    )
    assert decision_kind(oj) == "openjev"

    # d3 keeps its own classification.
    d3_dir = tmp_path / "d3-model"
    d3_dir.mkdir()
    (d3_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (d3_dir / "readout.safetensors").write_bytes(b"")
    (d3_dir / "decision_config.json").write_text(
        json.dumps({"format_id": "d3-code-readout-v1"})
    )
    assert decision_kind(d3_dir) == "d3"


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3_5ForConditionalGeneration", "Qwen3_5ForSequenceClassification"],
)
def test_minicpm_discovery_requires_explicit_ocr_identity(tmp_path, architecture):
    # A generic Qwen binary classifier does not imply the OCR training contract.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": [architecture],
                "problem_type": "single_label_classification",
                "vision_config": {"depth": 24},
            }
        )
    )
    assert decision_kind(tmp_path) is None
    marker = {
        "format_id": "minicpm-ocr-jev-v1",
        "prompt_template_md5": "9aab68e8ba8ab4da17ba9752bddb199c",
        "images_per_prompt": 2,
    }
    (tmp_path / "decision_config.json").write_text(json.dumps(marker))
    assert decision_kind(tmp_path) == "minicpm_jev"
    for field, value in (
        ("format_id", "other"),
        ("prompt_template_md5", "unknown"),
        ("images_per_prompt", 1),
    ):
        (tmp_path / "decision_config.json").write_text(
            json.dumps({**marker, field: value})
        )
        assert decision_kind(tmp_path) is None


# --------------------------------------------------------------------------
# Conv3D transpose & skip_logits runtime patch


def test_minicpm_classifier_sanitizer_preserves_quantized_score_and_drops_mtp():
    from mlx_vlm.models.qwen3_5 import qwen3_5

    assert apply_mlx_vlm_mtp_patch()
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=True, num_hidden_layers=0),
            vision_config=SimpleNamespace(in_channels=3),
        ),
        score=True,
    )
    weights = {
        "mtp.layers.0.weight": mx.zeros((4,)),
        "score.weight": mx.zeros((2, 4), dtype=mx.uint32),
        "score.scales": mx.ones((2, 1)),
        "score.biases": mx.zeros((2, 1)),
    }
    result = qwen3_5.Model.sanitize(model, weights)
    assert not any("mtp." in key for key in result)
    for key in ("score.weight", "score.scales", "score.biases"):
        assert result[key] is weights[key]


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe"])
def test_qwen35_runtime_skip_logits_bypasses_speculative_verify(monkeypatch, family):
    import importlib

    ql = importlib.import_module(f"mlx_vlm.models.{family}.language")
    runtime = importlib.import_module(
        "omlx.patches.mlx_vlm_mtp."
        + ("qwen35_vlm_runtime" if family == "qwen3_5" else "qwen35_moe_vlm_runtime")
    )
    runtime.apply()
    recorded = {}

    lm = ql.LanguageModel.__new__(ql.LanguageModel)
    nn.Module.__init__(lm)

    def _forbid_verify(*args, **kwargs):
        raise AssertionError("speculative_verify must not run when skip_logits=True")

    from mlx_vlm.models.qwen3_5 import language as qwen_language

    # Patch the defining class: restoring an inherited bound method on the
    # singleton instance would leave an attribute shadowing later class spies.
    monkeypatch.setattr(
        type(qwen_language._EXACT_SPECULATIVE_VERIFIER), "verify", _forbid_verify
    )
    monkeypatch.setattr(
        lm,
        "get_rope_index",
        lambda *a, **kw: (
            mx.zeros((1, 1), dtype=mx.int32),
            mx.zeros((1, 1), dtype=mx.int32),
        ),
    )

    class _Inner:
        fa_idx = 0
        ssm_idx = 1
        layers = [SimpleNamespace()]

        def __call__(
            self, inputs, cache=None, inputs_embeds=None, position_ids=None, **kw
        ):
            recorded["model_called"] = True
            return mx.zeros((1, 2, 4))

    lm.model = _Inner()
    lm._rope_deltas = None
    lm._position_ids = None
    lm.args = SimpleNamespace(tie_word_embeddings=False)
    out = lm(
        mx.array([[1, 2]], dtype=mx.int32),
        return_hidden=True,
        skip_logits=True,
    )
    assert recorded.get("model_called") is True
    assert out.logits is None
    assert len(out.hidden_states) == 1


# --------------------------------------------------------------------------
# MiniCPM-OCR-Jev unit tests


@pytest.mark.parametrize(
    "text",
    ["Candidate OCR text:\n  Hello  \n", " \nCandidate OCR text:\nHello\t ", "  Raw\n"],
)
def test_minicpm_jev_preserves_literal_candidate_text(text):
    assert minicpm_jev.extract_ocr_text(text) == text
    for key in ("ocr_text", "candidate_text", "text"):
        assert minicpm_jev.extract_ocr_text({key: text}) == text


def test_minicpm_jev_prompt_padding_and_validation():
    import hashlib

    assert (
        hashlib.md5(minicpm_jev.build_prompt("{text}").encode("utf-8")).hexdigest()
        == minicpm_jev.PROMPT_TEMPLATE_MD5
    )
    assert minicpm_jev.extract_ocr_text("Raw\n") == "Raw\n"

    # Aspect ratio padding preserves <=199:1 and white-pads >199:1.
    normal = Image.new("RGB", (200, 40), (10, 20, 30))
    assert minicpm_jev.pad_aspect_ratio(normal).size == (200, 40)
    extreme = Image.new("RGB", (600, 2), (10, 20, 30))
    padded = minicpm_jev.pad_aspect_ratio(extreme)
    w, h = padded.size
    assert w == 600 and w / h <= minicpm_jev.MAX_ASPECT_RATIO
    assert padded.getpixel((0, 0)) == (255, 255, 255)

    # Validation & binary head scoring with stub backbone.
    model = _stub_model()

    valid_imgs = [_png_data_url(), _png_data_url()]
    with pytest.raises(DecisionRequestError, match="2 images"):
        model.encode(
            {
                "state": "abc",
                "images": [valid_imgs[0]],
                "questions": {"q": {"type": "noul"}},
            }
        )
    with pytest.raises(DecisionRequestError, match="videos"):
        model.encode(
            {
                "state": "abc",
                "images": valid_imgs,
                "videos": ["data:video/mp4;base64,AA=="],
                "questions": {"q": {"type": "noul"}},
            }
        )
    with pytest.raises(DecisionRequestError, match="match/mismatch"):
        model.encode(
            {
                "state": "abc",
                "images": valid_imgs,
                "questions": {
                    "q": {"type": "choice", "criteria": {"fruit": "F", "vehicle": "V"}}
                },
            }
        )

    from omlx.api.decisions import from_systemone, to_systemone
    from omlx.api.decisions_models import DecisionRequest

    request = DecisionRequest.model_validate(
        {
            "model": "stub",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": url} for url in valid_imgs
                    ]
                    + [{"type": "input_text", "text": "DATE: 2026"}],
                }
            ],
            "questions": [
                {"name": "noul_q", "type": "predicate", "instructions": "Matches?"},
                {
                    "name": "choice_q",
                    "type": "choice",
                    "instructions": "Matches?",
                    "choices": ["incorrect", "correct"],
                },
                {
                    "name": "score_q",
                    "type": "score",
                    "instructions": "Matches?",
                    "levels": ["bad", "good"],
                },
            ],
        }
    )
    plan = model.encode(to_systemone(request))
    out = _drain(model.run(plan, lambda: 1024))
    p_yes = 1.0 / (1.0 + np.exp(-2.0))
    assert out["answers"]["noul_q"]["noul"] == pytest.approx(round(float(p_yes), 4))
    assert out["answers"]["choice_q"]["choice"] == "correct"
    assert list(out["answers"]["choice_q"]["probabilities"]) == ["incorrect", "correct"]
    score = out["answers"]["score_q"]
    assert score["score"] == pytest.approx(0.8808, abs=1e-4)
    assert score["legend"] == {"0": "bad", "1": "good"}
    assert "expected_level" not in score
    assert "confidence" not in out["answers"]["noul_q"]
    bridged = from_systemone(request, out["answers"])[2]
    assert bridged["score"] == pytest.approx(0.8808, abs=1e-4)
    assert bridged["probabilities"] == [
        {"value": 0, "label": "bad", "probability": 0.1192},
        {"value": 1, "label": "good", "probability": 0.8808},
    ]


@pytest.mark.parametrize("image_token_count", [1, 3])
def test_minicpm_jev_rejects_tokenized_image_slot_mismatch(image_token_count):
    # Literal image-pad special tokens in OCR text create extra tokenizer slots,
    # although the two full vision placeholders still pass the raw-text guard.
    model = _stub_model()
    model.backbone.tokenize = lambda s: (
        [997, 996] * 2 + [999] * image_token_count + [271]
    )
    with pytest.raises(DecisionRequestError, match="image.*(slot|token|placeholder)"):
        model.encode(
            {
                "state": "literal <|image_pad|> text",
                "images": [_png_data_url(), _png_data_url()],
                "questions": {"q": {"type": "noul"}},
            }
        )


@pytest.mark.slow
@pytest.mark.skipif(
    not (
        _MINICPM_PATH
        and _MINICPM_PATH.exists()
        and _MINICPM_EXAMPLES
        and _MINICPM_EXAMPLES.exists()
    ),
    reason="set OMLX_MINICPM_JEV_MODEL and OMLX_MINICPM_JEV_EXAMPLES for local parity",
)
def test_real_minicpm_jev_oq8_openbmb_reference_parity():
    assert decision_kind(_MINICPM_PATH) == "minicpm_jev"
    model = minicpm_jev.MiniCPMJevModel(str(_MINICPM_PATH))
    model.load()
    try:
        for name in ("01_text_putnam", "02_formula_putnam", "03_table_gb_standard"):
            c = json.loads((_MINICPM_EXAMPLES / name / "case.json").read_text())
            role = c["reference_render_of"]
            text = next(x["text"] for x in c["candidates"] if x["role"] == role)
            ref = c["observed_prob_yes"][role]
            img1 = "data:image/png;base64," + base64.b64encode(
                (_MINICPM_EXAMPLES / name / c["block_image"]).read_bytes()
            ).decode("ascii")
            img2 = "data:image/png;base64," + base64.b64encode(
                (_MINICPM_EXAMPLES / name / c["reference_render"]).read_bytes()
            ).decode("ascii")
            if name == "01_text_putnam":
                for token in ("<|video_pad|>", "<|vision_start|>", "<|vision_end|>"):
                    with pytest.raises(
                        DecisionRequestError, match="(video|vision|reserved)"
                    ):
                        model.encode(
                            {
                                "state": text + token,
                                "images": [img1, img2],
                                "questions": {"match": {"type": "noul"}},
                            }
                        )
            plan = model.encode(
                {
                    "state": text,
                    "images": [img1, img2],
                    "questions": {
                        "match": {"type": "noul", "instructions": "Consistent?"},
                        "cls": {
                            "type": "choice",
                            "instructions": "Consistent?",
                            "criteria": {"correct": "Match", "incorrect": "Mismatch"},
                        },
                    },
                }
            )
            res = _drain(model.run(plan, lambda: 512))
            p_yes = res["answers"]["match"]["noul"]
            assert abs(p_yes - ref) <= 0.02, (name, p_yes, ref)
            expected_cls = "correct" if ref >= 0.5 else "incorrect"
            assert res["answers"]["cls"]["choice"] == expected_cls
    finally:
        model.close()


def test_minicpm_jev_tiny_vision_backbone_matches_classifier():
    """Two real tiny vision inputs and layer chunking match the native head."""
    from mlx_vlm.models.minicpm_ocr_jev.config import ModelConfig
    from mlx_vlm.models.minicpm_ocr_jev.minicpm_ocr_jev import Model

    from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_runtime_patch

    assert apply_mlx_vlm_mtp_patch()
    assert apply_mlx_vlm_mtp_runtime_patch("qwen3_5")
    cfg = ModelConfig.from_dict(
        {
            "model_type": "minicpm_ocr_jev",
            "image_token_id": 319,
            "vision_start_token_id": 317,
            "vision_end_token_id": 318,
            "pad_token_id": 0,
            "vocab_size": 320,
            "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 320,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 16,
                "max_position_embeddings": 2048,
                "rope_parameters": {
                    "rope_theta": 1000.0,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [1, 1, 2],
                    "type": "default",
                },
            },
            "vision_config": {
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "out_hidden_size": 32,
                "num_heads": 2,
                "patch_size": 2,
                "spatial_patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [],
            },
        }
    )
    native = Model(cfg)
    model = minicpm_jev.MiniCPMJevModel("tiny")
    backbone = model.backbone
    backbone.model = native
    backbone._text_model = native.language_model.model
    backbone.config = {"text_config": {"max_position_embeddings": 2048}}
    backbone.image_processor = SimpleNamespace(merge_size=2)
    backbone.tokenize = lambda text: [317, 319, 318, 10, 317, 319, 318, 271]
    pixels = np.linspace(-1, 1, 8 * 24, dtype=np.float32).reshape(8, 24)
    grids = np.array([[1, 2, 2], [1, 2, 2]], dtype=np.int32)
    backbone.preprocess_images = lambda images: (pixels, grids)
    model._image_token_id = 319
    model._video_token_id = 316
    model._vision_start_token_id = 317
    model._vision_end_token_id = 318
    plan = model.encode(
        {
            "state": "text",
            "images": [_png_data_url()] * 2,
            "questions": {"match": {"type": "noul"}},
        }
    )
    reference = native(
        mx.array(plan.ids)[None], mx.array(pixels), image_grid_thw=mx.array(grids)
    ).logits
    expected = round(
        float(mx.softmax(reference.astype(mx.float32), axis=-1)[0, 1].item()), 4
    )
    for chunk in (1, 1024):
        result = _drain(model.run(plan, lambda c=chunk: c))
        assert result["answers"]["match"]["noul"] == pytest.approx(expected, abs=1e-4)
        assert result["input_tokens"] == len(plan.ids)


def test_minicpm_identity_survives_common_conversion_sidecar_copy(tmp_path):
    from omlx.oq import _copy_model_sidecars

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    marker = json.dumps(
        {
            "format_id": "minicpm-ocr-jev-v1",
            "prompt_template_md5": minicpm_jev.PROMPT_TEMPLATE_MD5,
            "images_per_prompt": 2,
        }
    )
    (source / "decision_config.json").write_text(marker)
    _copy_model_sidecars(source, output)
    assert (output / "decision_config.json").read_text() == marker
    assert (
        decision_kind(
            output,
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForSequenceClassification"],
            },
        )
        == "minicpm_jev"
    )


def test_official_shape_unmarked_classifier_warns_about_adapter_setup(tmp_path, caplog):
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "problem_type": "single_label_classification",
        "vision_config": {"depth": 24},
        "num_labels": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert decision_kind(tmp_path) is None
    assert detect_model_type(tmp_path) == "vlm"
    assert "classifier" in caplog.text
    assert "decision_config.json" in caplog.text
    assert "docs/minicpm-ocr-jev.md" in caplog.text
    caplog.clear()
    (tmp_path / "decision_config.json").write_text(
        json.dumps(
            {
                "format_id": "minicpm-ocr-jev-v1",
                "prompt_template_md5": minicpm_jev.PROMPT_TEMPLATE_MD5,
                "images_per_prompt": 2,
            }
        )
    )
    assert detect_model_type(tmp_path) == "decision"
    assert not caplog.text


@pytest.mark.parametrize(
    "token", ["<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
)
def test_minicpm_rejects_reserved_video_and_boundary_tokens_during_encode(token):
    model = _stub_model()
    with pytest.raises(DecisionRequestError, match="(video|vision|reserved)"):
        model.encode(
            {
                "state": f"literal {token} text",
                "images": [_png_data_url()] * 2,
                "questions": {"q": {"type": "noul"}},
            }
        )


@pytest.mark.parametrize("incompatibility", ["missing_video_token", "prompt_suffix"])
def test_minicpm_load_validates_tokenizer_contract(incompatibility):
    model = _stub_model()
    if incompatibility == "missing_video_token":
        model.backbone.tokenizer.get_vocab = lambda: {
            key: value
            for key, value in _STUB_TOKEN_IDS.items()
            if key != "<|video_pad|>"
        }
    else:
        model.backbone.tokenize = lambda text: _stub_prompt_tokens(text)[:-1] + [270]
    with pytest.raises(ValueError, match="(tokenizer|prompt|token)"):
        model.load()


@pytest.mark.parametrize("native_identity", [False, True])
def test_minicpm_quantization_rejects_text_only_before_output_creation(
    tmp_path, native_identity
):
    from omlx.oq import quantize_oq_streaming

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    config = {
        "model_type": "qwen3_5",
        "problem_type": "single_label_classification",
        "vision_config": {"depth": 24},
    }
    if native_identity:
        config["model_type"] = "minicpm_ocr_jev"
    else:
        (source / "decision_config.json").write_text(
            json.dumps(
                {
                    "format_id": "minicpm-ocr-jev-v1",
                    "prompt_template_md5": minicpm_jev.PROMPT_TEMPLATE_MD5,
                    "images_per_prompt": 2,
                }
            )
        )
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="MiniCPM-OCR-Jev.*vision tower"):
        quantize_oq_streaming(str(source), str(output), oq_level=4, text_only=True)
    assert not output.exists()


@pytest.mark.parametrize("texts", [[], ["one", "two"], [""], [" \nraw\t "]])
def test_minicpm_bridge_requires_one_exact_candidate_text_part(texts):
    from omlx.api.decisions import to_systemone
    from omlx.api.decisions_models import DecisionRequest

    request = DecisionRequest.model_validate(
        {
            "model": "stub",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": _png_data_url()},
                        {"type": "input_image", "image_url": _png_data_url()},
                        *({"type": "input_text", "text": text} for text in texts),
                    ],
                }
            ],
            "questions": [
                {"type": "predicate", "name": "match", "instructions": "Match?"}
            ],
        }
    )
    bridged = to_systemone(request)
    model = _stub_model()
    if len(texts) != 1:

        def forbidden_preprocess(images):
            raise AssertionError(
                "invalid text parts must fail before image preprocessing"
            )

        model.backbone.preprocess_images = forbidden_preprocess
        with pytest.raises(DecisionRequestError, match="exactly one input_text"):
            model.encode(bridged)
    else:
        prompts = []
        model.backbone.tokenize = lambda text: prompts.append(
            text
        ) or _stub_prompt_tokens(text)
        model.encode(bridged)
        assert bridged["_text_parts"] == texts
        assert prompts == [minicpm_jev.build_prompt(texts[0])]


def test_minicpm_prompt_fingerprint_is_checked_before_loading_weights(monkeypatch):
    model = _stub_model()
    monkeypatch.setattr(minicpm_jev, "PROMPT_TEMPLATE_MD5", "incorrect")

    def forbidden_load():
        raise AssertionError("a prompt mismatch must fail before loading weights")

    model.backbone.load = forbidden_load
    with pytest.raises(RuntimeError, match="prompt MD5 mismatch"):
        model.load()


def _tiny_minicpm_conversion_source(tmp_path, native_identity):
    from safetensors.numpy import save_file

    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model_type": "minicpm_ocr_jev" if native_identity else "qwen3_5",
        "problem_type": "single_label_classification",
        "vision_config": {"depth": 1},
        "hidden_size": 64,
        "num_hidden_layers": 1,
    }
    (source / "config.json").write_text(json.dumps(config))
    if not native_identity:
        (source / "decision_config.json").write_text(
            json.dumps(
                {
                    "format_id": "minicpm-ocr-jev-v1",
                    "prompt_template_md5": minicpm_jev.PROMPT_TEMPLATE_MD5,
                    "images_per_prompt": 2,
                }
            )
        )
    score = np.arange(128, dtype=np.float32).reshape(2, 64) / 16
    save_file(
        {
            "score.weight": score,
            "model.layers.0.self_attn.q_proj.weight": np.ones(
                (64, 64), dtype=np.float32
            ),
        },
        str(source / "model.safetensors"),
    )
    return source, score


def _assert_float_head_and_quantized_backbone(output, expected_score):
    weights = mx.load(str(output / "model.safetensors"))
    assert mx.issubdtype(weights["score.weight"].dtype, mx.floating)
    assert weights["score.weight"].shape == (2, 64)
    assert np.array_equal(
        np.asarray(weights["score.weight"].astype(mx.float32)), expected_score
    )
    assert "score.scales" not in weights
    assert "score.biases" not in weights
    assert "model.layers.0.self_attn.q_proj.scales" in weights
    config = json.loads((output / "config.json").read_text())
    assert "score" not in config["quantization"]
    assert "_oq_non_quantizable" not in config


@pytest.mark.parametrize("native_identity", [False, True])
@pytest.mark.parametrize("oq_level", [3, 8])
def test_minicpm_streaming_conversion_keeps_float_score_head(
    tmp_path, monkeypatch, native_identity, oq_level
):
    from omlx import oq

    source, score = _tiny_minicpm_conversion_source(tmp_path, native_identity)
    monkeypatch.setattr(oq, "_build_model_sanitizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(oq, "_build_non_quantizable_set", lambda config: set())
    output = tmp_path / "output"
    oq.quantize_oq_streaming(
        str(source), str(output), oq_level=oq_level, sensitivity_map_override={0: 1.0}
    )
    _assert_float_head_and_quantized_backbone(output, score)
    seen = []
    original = oq.universal_quant_predicate

    def observe(path, module, config, level=4):
        result = original(path, module, config, level)
        if path in {"score", "score.weight"}:
            seen.append(result)
        return result

    monkeypatch.setattr(oq, "universal_quant_predicate", observe)
    oq.estimate_bpw_and_size(str(source), oq_level=oq_level)
    assert seen and all(result is False for result in seen)


@pytest.mark.parametrize("native_identity", [False, True])
def test_minicpm_calibration_proxy_keeps_float_score_head(
    tmp_path, monkeypatch, native_identity
):
    from omlx import oq

    source, score = _tiny_minicpm_conversion_source(tmp_path, native_identity)
    monkeypatch.setattr(oq, "_build_model_sanitizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(oq, "_build_non_quantizable_set", lambda config: set())
    output = tmp_path / "proxy"
    oq._build_streaming_proxy_for_sensitivity(str(source), output, dtype="bfloat16")
    _assert_float_head_and_quantized_backbone(output, score)
