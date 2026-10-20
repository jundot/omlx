# SPDX-License-Identifier: Apache-2.0
"""Tests for serving text-only Gemma 4 checkpoints on the VLM engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm.utils")

import mlx.core as mx  # noqa: E402
import mlx_vlm.utils as _vu  # noqa: E402

from omlx.engine.vlm import _strip_vision_config_if_orphaned  # noqa: E402
from omlx.model_discovery import discover_models  # noqa: E402

MERGED_HEAD = {"mtp_assistant_config": {"num_hidden_layers": 4}}


def _write_safetensors(path: Path, keys: list[str]) -> None:
    mx.save_safetensors(str(path), {key: mx.zeros((1,)) for key in keys})


def _model_dir(
    tmp_path: Path,
    *,
    name: str,
    model_type: str = "gemma4",
    vision_config: bool = False,
    merged_head: bool = False,
    vision_weights: bool = False,
) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    text_config: dict = {"hidden_size": 32, "num_hidden_layers": 1}
    if merged_head:
        text_config.update(MERGED_HEAD)
    config: dict = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": model_type,
        "text_config": text_config,
    }
    if vision_config:
        config["vision_config"] = {"hidden_size": 16}
    (model_dir / "config.json").write_text(json.dumps(config))

    keys = ["language_model.model.layers.0.self_attn.q_proj.weight"]
    if vision_weights:
        keys.append("vision_embedder.patch_dense.weight")
    _write_safetensors(model_dir / "model.safetensors", keys)
    return model_dir


def _config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


@pytest.mark.parametrize(
    "model_type, merged_head, vision, expected_engine, expected_type",
    [
        ("gemma4", False, False, "batched", "llm"),
        ("gemma4", True, False, "vlm", "llm"),
        ("gemma4", True, True, "vlm", "vlm"),
        ("gemma4_unified", False, False, "batched", "llm"),
        ("gemma4_unified", True, False, "vlm", "vlm"),
        ("gemma4_unified", False, True, "vlm", "vlm"),
    ],
)
def test_discovery_routes_gemma4(
    tmp_path, model_type, merged_head, vision, expected_engine, expected_type
):
    _model_dir(
        tmp_path,
        name="gemma4",
        model_type=model_type,
        merged_head=merged_head,
        vision_config=vision,
        vision_weights=vision,
    )
    found = discover_models(tmp_path)["gemma4"]
    assert (found.engine_type, found.model_type) == (expected_engine, expected_type)
    if expected_type == "llm":
        assert found.text_only_size == 0


def test_unified_audio_without_vision_stays_on_vlm(tmp_path: Path):
    d = _model_dir(tmp_path, name="unified_audio", model_type="gemma4_unified")
    config = _config(d)
    config["audio_config"] = {"output_proj_dims": 8}
    (d / "config.json").write_text(json.dumps(config))
    _write_safetensors(d / "model.safetensors", ["embed_audio.embedding_projection.weight"])

    found = discover_models(tmp_path)["unified_audio"]
    assert found.engine_type == "vlm"
    assert found.model_type == "vlm"


def test_text_only_gemma4_loads_as_unified(tmp_path: Path):
    # mlx-vlm's gemma4 builds a vision tower unconditionally.
    d = _model_dir(tmp_path, name="textonly_head", merged_head=True)
    with _strip_vision_config_if_orphaned(d):
        assert _vu.load_config(d)["model_type"] == "gemma4_unified"
    assert _vu.load_config(d)["model_type"] == "gemma4"


def test_vision_checkpoint_keeps_model_type(tmp_path: Path):
    d = _model_dir(tmp_path, name="vision", vision_config=True, vision_weights=True)
    with _strip_vision_config_if_orphaned(d):
        assert _vu.load_config(d)["model_type"] == "gemma4"


def test_unified_sanitize_keeps_mtp_head():
    # gemma4_unified's sanitize would move the head under language_model.model.
    from types import SimpleNamespace

    from mlx_vlm.models.gemma4_unified import Model

    from omlx.patches.mlx_vlm_mtp import gemma4_vlm_runtime

    assert gemma4_vlm_runtime.apply()
    head_key = "language_model.mtp.pre_projection.weight"
    out = Model.sanitize(
        SimpleNamespace(embed_audio=None),
        {head_key: mx.zeros((1,)), "language_model.norm.weight": mx.zeros((1,))},
    )
    assert head_key in out
    assert "language_model.model.norm.weight" in out
