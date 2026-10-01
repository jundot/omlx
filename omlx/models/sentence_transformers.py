# SPDX-License-Identifier: Apache-2.0
"""Validation for supported SentenceTransformers model pipelines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModernBertCrossEncoderPipeline:
    """Validated module layout for a modular ModernBERT CrossEncoder."""

    hidden_size: int
    pooling_path: str
    dense_path: str
    layer_norm_path: str
    output_path: str
    layer_norm_eps: float
    global_rope_theta: float
    local_rope_theta: float


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _module_path(model_path: Path, value: Any, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} module path must be a non-empty string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.parts[0]
        in {
            ".",
            "..",
        }
    ):
        raise ValueError(f"{label} module path escapes model directory: {value!r}")
    root = model_path.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"{label} module path escapes model directory: {value!r}"
        ) from error
    if not resolved.is_dir():
        raise ValueError(f"{label} module directory does not exist: {resolved}")
    return value, resolved


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _activation_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value.rsplit(".", 1)[-1]


def _expect_field(
    config: dict[str, Any], field: str, expected: Any, label: str
) -> None:
    if config.get(field) != expected:
        raise ValueError(
            f"{label} requires {field}={expected!r}; got {config.get(field)!r}"
        )


def is_modernbert_cross_encoder(model_path: str | Path) -> bool:
    """Detect the explicit CrossEncoder marker without claiming compatibility."""

    root = Path(model_path)
    try:
        config = _json_object(root / "config.json", "model config")
        cross_encoder = _json_object(
            root / "config_sentence_transformers.json",
            "SentenceTransformers config",
        )
    except ValueError:
        return False
    return (
        config.get("architectures") == ["ModernBertModel"]
        and config.get("model_type") == "modernbert"
        and cross_encoder.get("model_type") == "CrossEncoder"
    )


def parse_modernbert_cross_encoder_pipeline(
    model_path: str | Path,
) -> ModernBertCrossEncoderPipeline:
    """Parse the one modular ModernBERT CrossEncoder layout oMLX can execute.

    The narrow contract prevents a generic embedding export from being admitted as
    a reranker merely because it contains a Dense module.
    """

    root = Path(model_path)
    config = _json_object(root / "config.json", "model config")
    if config.get("architectures") != ["ModernBertModel"]:
        raise ValueError(
            "SentenceTransformers CrossEncoder requires architectures="
            "['ModernBertModel']"
        )
    if config.get("model_type") != "modernbert":
        raise ValueError(
            "SentenceTransformers CrossEncoder requires model_type='modernbert'"
        )
    hidden_size = _positive_int(config.get("hidden_size"), "hidden_size")
    global_rope_theta = _positive_float(
        config.get("global_rope_theta", 160000.0),
        "global_rope_theta",
    )
    local_rope_theta = _positive_float(
        config.get("local_rope_theta", 10000.0),
        "local_rope_theta",
    )
    rope_parameters = config.get("rope_parameters")
    if rope_parameters is not None:
        if not isinstance(rope_parameters, dict):
            raise ValueError("rope_parameters must be a JSON object")
        full_attention = rope_parameters.get("full_attention")
        sliding_attention = rope_parameters.get("sliding_attention")
        if not isinstance(full_attention, dict) or not isinstance(
            sliding_attention, dict
        ):
            raise ValueError(
                "rope_parameters must declare full_attention and sliding_attention"
            )
        if (
            full_attention.get("rope_type") != "default"
            or sliding_attention.get("rope_type") != "default"
        ):
            raise ValueError("Only default ModernBERT RoPE parameters are supported")
        global_rope_theta = _positive_float(
            full_attention.get("rope_theta"),
            "full_attention rope_theta",
        )
        local_rope_theta = _positive_float(
            sliding_attention.get("rope_theta"),
            "sliding_attention rope_theta",
        )

    layer_types = config.get("layer_types")
    if layer_types is not None:
        num_hidden_layers = _positive_int(
            config.get("num_hidden_layers"),
            "num_hidden_layers",
        )
        global_attn_every = _positive_int(
            config.get("global_attn_every_n_layers", 3),
            "global_attn_every_n_layers",
        )
        expected_layer_types = [
            "full_attention" if index % global_attn_every == 0 else "sliding_attention"
            for index in range(num_hidden_layers)
        ]
        if layer_types != expected_layer_types:
            raise ValueError(
                "ModernBERT layer_types do not match the supported "
                "global_attn_every_n_layers topology"
            )

    cross_encoder = _json_object(
        root / "config_sentence_transformers.json",
        "SentenceTransformers config",
    )
    _expect_field(cross_encoder, "model_type", "CrossEncoder", "CrossEncoder config")
    if (
        _activation_name(
            cross_encoder.get("activation_fn"), "CrossEncoder outer activation"
        )
        != "Identity"
    ):
        raise ValueError(
            "CrossEncoder outer activation must be Identity for raw scores"
        )

    modules_path = root / "modules.json"
    try:
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read SentenceTransformers modules: {modules_path}"
        ) from error
    expected_types = ["Transformer", "Pooling", "Dense", "LayerNorm", "Dense"]
    if not isinstance(modules, list) or len(modules) != len(expected_types):
        raise ValueError(
            "Unsupported SentenceTransformers module chain; expected "
            "Transformer -> Pooling -> Dense -> LayerNorm -> Dense"
        )
    for index, (module, expected_type) in enumerate(zip(modules, expected_types)):
        if not isinstance(module, dict):
            raise ValueError(f"SentenceTransformers module {index} must be an object")
        module_type = module.get("type")
        if (
            module.get("idx") != index
            or not isinstance(module_type, str)
            or not module_type.startswith("sentence_transformers.")
            or module_type.rsplit(".", 1)[-1] != expected_type
        ):
            raise ValueError(
                "Unsupported SentenceTransformers module chain; expected "
                "Transformer -> Pooling -> Dense -> LayerNorm -> Dense"
            )
    if modules[0].get("path") not in {"", "."}:
        raise ValueError("Transformer module must load from the model root")

    pooling_path, pooling_dir = _module_path(root, modules[1].get("path"), "Pooling")
    dense_path, dense_dir = _module_path(root, modules[2].get("path"), "Dense")
    layer_norm_path, layer_norm_dir = _module_path(
        root, modules[3].get("path"), "LayerNorm"
    )
    output_path, output_dir = _module_path(root, modules[4].get("path"), "output Dense")

    pooling = _json_object(pooling_dir / "config.json", "Pooling config")
    pooling_mode = pooling.get("pooling_mode")
    if pooling_mode is None and pooling.get("pooling_mode_cls_token") is True:
        conflicting_modes = (
            "pooling_mode_lasttoken",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_mean_tokens",
            "pooling_mode_weightedmean_tokens",
        )
        if not any(pooling.get(field) is True for field in conflicting_modes):
            pooling_mode = "cls"
    if pooling_mode != "cls":
        raise ValueError("SentenceTransformers reranker requires CLS pooling")
    pooling_dimension = pooling.get(
        "embedding_dimension",
        pooling.get("word_embedding_dimension"),
    )
    if pooling_dimension != hidden_size:
        raise ValueError(
            "Pooling config requires embedding_dimension or "
            f"word_embedding_dimension={hidden_size}; got {pooling_dimension!r}"
        )

    dense = _json_object(dense_dir / "config.json", "Dense config")
    _expect_field(dense, "in_features", hidden_size, "Dense config hidden_size")
    _expect_field(dense, "out_features", hidden_size, "Dense config hidden_size")
    _expect_field(dense, "bias", False, "Dense config bias=false")
    _expect_field(
        dense,
        "module_input_name",
        "sentence_embedding",
        "Dense config",
    )
    _expect_field(
        dense,
        "module_output_name",
        "sentence_embedding",
        "Dense config",
    )
    if _activation_name(dense.get("activation_function"), "Dense activation") != "GELU":
        raise ValueError("SentenceTransformers hidden Dense activation must be GELU")

    layer_norm = _json_object(layer_norm_dir / "config.json", "LayerNorm config")
    _expect_field(
        layer_norm,
        "dimension",
        hidden_size,
        "LayerNorm config hidden_size",
    )
    layer_norm_eps = layer_norm.get("eps", 1e-5)
    if (
        isinstance(layer_norm_eps, bool)
        or not isinstance(layer_norm_eps, (int, float))
        or not math.isfinite(layer_norm_eps)
        or layer_norm_eps <= 0
    ):
        raise ValueError("LayerNorm eps must be a finite positive number")

    output = _json_object(output_dir / "config.json", "output Dense config")
    _expect_field(output, "in_features", hidden_size, "output Dense config")
    _expect_field(output, "out_features", 1, "output Dense config out_features=1")
    _expect_field(output, "bias", True, "output Dense config bias=true")
    _expect_field(
        output,
        "module_input_name",
        "sentence_embedding",
        "output Dense config",
    )
    _expect_field(output, "module_output_name", "scores", "output Dense config")
    if (
        _activation_name(output.get("activation_function"), "output Dense activation")
        != "Identity"
    ):
        raise ValueError(
            "SentenceTransformers output Dense activation must be Identity"
        )

    return ModernBertCrossEncoderPipeline(
        hidden_size=hidden_size,
        pooling_path=pooling_path,
        dense_path=dense_path,
        layer_norm_path=layer_norm_path,
        output_path=output_path,
        layer_norm_eps=float(layer_norm_eps),
        global_rope_theta=global_rope_theta,
        local_rope_theta=local_rope_theta,
    )


def is_modernbert_cross_encoder_pipeline(model_path: str | Path) -> bool:
    """Return whether ``model_path`` satisfies the strict supported contract."""

    try:
        parse_modernbert_cross_encoder_pipeline(model_path)
    except (OSError, TypeError, ValueError):
        return False
    return True
