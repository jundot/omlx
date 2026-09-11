# SPDX-License-Identifier: Apache-2.0
"""SentenceTransformers ModernBERT CrossEncoder reranker coverage."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from omlx.model_discovery import detect_model_type
from omlx.models.reranker import (
    MLXRerankerModel,
    ModernBertSentenceTransformersCrossEncoder,
    SentenceTransformersCrossEncoderHead,
    _sanitize_sentence_transformers_modernbert_weights,
)
from omlx.models.sentence_transformers import (
    ModernBertCrossEncoderPipeline,
    parse_modernbert_cross_encoder_pipeline,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_pipeline(tmp_path: Path, hidden_size: int = 4) -> Path:
    _write_json(
        tmp_path / "config.json",
        {
            "architectures": ["ModernBertModel"],
            "model_type": "modernbert",
            "hidden_size": hidden_size,
        },
    )
    _write_json(
        tmp_path / "config_sentence_transformers.json",
        {
            "activation_fn": "torch.nn.modules.linear.Identity",
            "model_type": "CrossEncoder",
        },
    )
    _write_json(
        tmp_path / "modules.json",
        [
            {
                "idx": 0,
                "name": "0",
                "path": "",
                "type": "sentence_transformers.base.modules.transformer.Transformer",
            },
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
            },
            {
                "idx": 2,
                "name": "2",
                "path": "2_Dense",
                "type": "sentence_transformers.base.modules.dense.Dense",
            },
            {
                "idx": 3,
                "name": "3",
                "path": "3_LayerNorm",
                "type": "sentence_transformers.sentence_transformer.modules.layer_norm.LayerNorm",
            },
            {
                "idx": 4,
                "name": "4",
                "path": "4_Dense",
                "type": "sentence_transformers.base.modules.dense.Dense",
            },
        ],
    )
    _write_json(
        tmp_path / "1_Pooling/config.json",
        {
            "embedding_dimension": hidden_size,
            "include_prompt": True,
            "pooling_mode": "cls",
        },
    )
    _write_json(
        tmp_path / "2_Dense/config.json",
        {
            "activation_function": "torch.nn.modules.activation.GELU",
            "bias": False,
            "in_features": hidden_size,
            "module_input_name": "sentence_embedding",
            "module_output_name": "sentence_embedding",
            "out_features": hidden_size,
        },
    )
    _write_json(
        tmp_path / "3_LayerNorm/config.json",
        {"dimension": hidden_size},
    )
    _write_json(
        tmp_path / "4_Dense/config.json",
        {
            "activation_function": "torch.nn.modules.linear.Identity",
            "bias": True,
            "in_features": hidden_size,
            "module_input_name": "sentence_embedding",
            "module_output_name": "scores",
            "out_features": 1,
        },
    )
    return tmp_path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parse_and_discover_modernbert_cross_encoder(tmp_path) -> None:
    model_path = _write_pipeline(tmp_path)

    pipeline = parse_modernbert_cross_encoder_pipeline(model_path)

    assert pipeline == ModernBertCrossEncoderPipeline(
        hidden_size=4,
        pooling_path="1_Pooling",
        dense_path="2_Dense",
        layer_norm_path="3_LayerNorm",
        output_path="4_Dense",
        layer_norm_eps=1e-5,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
    )
    assert detect_model_type(model_path) == "reranker"
    MLXRerankerModel(str(model_path))._validate_architecture()


def test_hub_id_resolves_before_modular_loader_dispatch(tmp_path, monkeypatch) -> None:
    model_path = _write_pipeline(tmp_path)
    loader = MLXRerankerModel("example/ettin-cross-encoder")
    loaded_model = MagicMock()
    loaded_model.config = MagicMock()
    seen_paths = []
    monkeypatch.setattr(
        "mlx_embeddings.utils.get_model_path",
        lambda model_name: model_path,
    )
    monkeypatch.setattr(
        loader,
        "_load_modernbert_sentence_transformers",
        lambda resolved: seen_paths.append(resolved) or (loaded_model, MagicMock()),
    )
    monkeypatch.setattr(loader, "_try_compile", lambda: False)

    loader.load()

    assert seen_paths == [model_path]
    assert loader._resolved_model_path == model_path
    assert loader._is_sentence_transformers_cross_encoder is True


def test_parser_translates_nested_modernbert_rope_parameters(tmp_path) -> None:
    model_path = _write_pipeline(tmp_path)
    config = _load_json(model_path / "config.json")
    config["rope_parameters"] = {
        "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 160000.0},
    }
    _write_json(model_path / "config.json", config)

    pipeline = parse_modernbert_cross_encoder_pipeline(model_path)

    assert pipeline.global_rope_theta == 160000.0
    assert pipeline.local_rope_theta == 160000.0


def test_parser_accepts_legacy_cls_pooling_config(tmp_path) -> None:
    model_path = _write_pipeline(tmp_path)
    pooling_path = model_path / "1_Pooling/config.json"
    pooling = _load_json(pooling_path)
    pooling.pop("embedding_dimension")
    pooling.pop("pooling_mode")
    pooling["word_embedding_dimension"] = 4
    pooling["pooling_mode_cls_token"] = True
    pooling["pooling_mode_mean_tokens"] = False
    _write_json(pooling_path, pooling)

    pipeline = parse_modernbert_cross_encoder_pipeline(model_path)

    assert pipeline.hidden_size == 4
    assert detect_model_type(model_path) == "reranker"


def test_parser_rejects_unrepresentable_layer_topology(tmp_path) -> None:
    model_path = _write_pipeline(tmp_path)
    config = _load_json(model_path / "config.json")
    config.update(
        {
            "num_hidden_layers": 4,
            "global_attn_every_n_layers": 3,
            "layer_types": ["full_attention"] * 4,
        }
    )
    _write_json(model_path / "config.json", config)

    with pytest.raises(ValueError, match="layer_types"):
        parse_modernbert_cross_encoder_pipeline(model_path)
    assert detect_model_type(model_path) == "reranker"


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("1_Pooling/config.json", "pooling_mode", "mean", "CLS pooling"),
        (
            "2_Dense/config.json",
            "activation_function",
            "torch.nn.modules.activation.ReLU",
            "GELU",
        ),
        ("2_Dense/config.json", "bias", True, "bias=false"),
        ("2_Dense/config.json", "out_features", 3, "hidden_size"),
        ("3_LayerNorm/config.json", "dimension", 3, "hidden_size"),
        (
            "4_Dense/config.json",
            "activation_function",
            "torch.nn.modules.activation.Sigmoid",
            "Identity",
        ),
        ("4_Dense/config.json", "bias", False, "bias=true"),
        ("4_Dense/config.json", "out_features", 2, "out_features=1"),
        (
            "config_sentence_transformers.json",
            "activation_fn",
            "torch.nn.modules.activation.Sigmoid",
            "outer activation",
        ),
    ],
)
def test_parser_rejects_incompatible_pipeline(
    tmp_path, target, field, value, match
) -> None:
    model_path = _write_pipeline(tmp_path)
    config_path = model_path / target
    config = _load_json(config_path)
    config[field] = value
    _write_json(config_path, config)

    with pytest.raises(ValueError, match=match):
        parse_modernbert_cross_encoder_pipeline(model_path)
    assert detect_model_type(model_path) == "reranker"


def test_parser_rejects_wrong_module_order(tmp_path) -> None:
    model_path = _write_pipeline(tmp_path)
    modules = _load_json(model_path / "modules.json")
    modules[2], modules[3] = modules[3], modules[2]
    modules[2]["idx"] = 2
    modules[3]["idx"] = 3
    _write_json(model_path / "modules.json", modules)

    with pytest.raises(ValueError, match="module chain"):
        parse_modernbert_cross_encoder_pipeline(model_path)
    assert detect_model_type(model_path) == "reranker"


def test_parser_rejects_module_path_escape(tmp_path) -> None:
    model_path = _write_pipeline(tmp_path)
    modules = _load_json(model_path / "modules.json")
    modules[2]["path"] = "../outside"
    _write_json(model_path / "modules.json", modules)

    with pytest.raises(ValueError, match="escapes model directory"):
        parse_modernbert_cross_encoder_pipeline(model_path)


def test_head_returns_raw_signed_scores() -> None:
    import mlx.core as mx

    head = SentenceTransformersCrossEncoderHead(hidden_size=2, layer_norm_eps=1e-5)
    head.load_weights(
        [
            ("dense.weight", mx.eye(2)),
            ("norm.weight", mx.ones((2,))),
            ("norm.bias", mx.zeros((2,))),
            ("output.weight", mx.array([[1.0, -1.0]])),
            ("output.bias", mx.array([-0.5])),
        ]
    )
    scores = head(mx.array([[1.0, 2.0], [2.0, 1.0]])).squeeze(-1)
    mx.eval(scores)

    assert scores[0].item() < -1.0
    assert scores[1].item() > 0.0
    assert scores[0].item() < 0.0  # proves no sigmoid/clamp


def test_sanitize_maps_nested_head_weights_and_validates_shapes(tmp_path) -> None:
    pipeline = parse_modernbert_cross_encoder_pipeline(_write_pipeline(tmp_path))
    weights = {
        "embeddings.tok_embeddings.weight": np.zeros((16, 4), dtype=np.float32),
        "2_Dense.linear.weight": np.zeros((4, 4), dtype=np.float32),
        "3_LayerNorm.norm.weight": np.ones((4,), dtype=np.float32),
        "3_LayerNorm.norm.bias": np.zeros((4,), dtype=np.float32),
        "4_Dense.linear.weight": np.zeros((1, 4), dtype=np.float32),
        "4_Dense.linear.bias": np.zeros((1,), dtype=np.float32),
    }

    mapped = _sanitize_sentence_transformers_modernbert_weights(weights, pipeline)

    assert set(mapped) == {
        "model.embeddings.tok_embeddings.weight",
        "head.dense.weight",
        "head.norm.weight",
        "head.norm.bias",
        "head.output.weight",
        "head.output.bias",
    }
    assert "head.dense.bias" not in mapped

    weights["2_Dense.linear.weight"] = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="2_Dense.linear.weight.*expected"):
        _sanitize_sentence_transformers_modernbert_weights(weights, pipeline)


def test_sanitize_rejects_missing_or_unexpected_head_tensors(tmp_path) -> None:
    pipeline = parse_modernbert_cross_encoder_pipeline(_write_pipeline(tmp_path))
    valid = {
        "2_Dense.linear.weight": np.zeros((4, 4), dtype=np.float32),
        "3_LayerNorm.norm.weight": np.ones((4,), dtype=np.float32),
        "3_LayerNorm.norm.bias": np.zeros((4,), dtype=np.float32),
        "4_Dense.linear.weight": np.zeros((1, 4), dtype=np.float32),
        "4_Dense.linear.bias": np.zeros((1,), dtype=np.float32),
    }
    missing = dict(valid)
    missing.pop("4_Dense.linear.bias")
    with pytest.raises(ValueError, match="missing head tensor"):
        _sanitize_sentence_transformers_modernbert_weights(missing, pipeline)

    unexpected = dict(valid)
    unexpected["2_Dense.linear.bias"] = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected head tensor"):
        _sanitize_sentence_transformers_modernbert_weights(unexpected, pipeline)


def test_wrapper_uses_cls_backbone_output_and_raw_head_score() -> None:
    import mlx.core as mx
    from mlx_embeddings.models.modernbert import ModelArgs

    pipeline = ModernBertCrossEncoderPipeline(
        hidden_size=2,
        pooling_path="1_Pooling",
        dense_path="2_Dense",
        layer_norm_path="3_LayerNorm",
        output_path="4_Dense",
        layer_norm_eps=1e-5,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
    )
    config = ModelArgs(
        model_type="modernbert",
        vocab_size=16,
        hidden_size=2,
        num_hidden_layers=1,
        intermediate_size=4,
        num_attention_heads=1,
        max_position_embeddings=8,
        local_attention=4,
        architectures=["ModernBertModel"],
    )
    model = ModernBertSentenceTransformersCrossEncoder(config, pipeline)
    model.model = MagicMock(
        return_value={
            "last_hidden_state": mx.array(
                [[[1.0, 2.0], [99.0, 99.0]], [[2.0, 1.0], [99.0, 99.0]]]
            ),
            "hidden_states": None,
        }
    )
    model.head.load_weights(
        [
            ("dense.weight", mx.eye(2)),
            ("norm.weight", mx.ones((2,))),
            ("norm.bias", mx.zeros((2,))),
            ("output.weight", mx.array([[1.0, -1.0]])),
            ("output.bias", mx.array([-0.5])),
        ]
    )

    output = model(
        input_ids=mx.array([[1, 2], [1, 2]]),
        attention_mask=mx.ones((2, 2)),
    )
    mx.eval(output.pooler_output)

    assert output.pooler_output.shape == (2, 1)
    assert output.pooler_output[0, 0].item() < -1.0
    assert output.pooler_output[1, 0].item() > 0.0
