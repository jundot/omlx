# SPDX-License-Identifier: Apache-2.0
"""Tests for serving text-only Gemma 4 checkpoints on the VLM engine."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm.utils")

import mlx_vlm.utils as _vu  # noqa: E402

from omlx.engine.vlm import _strip_vision_config_if_orphaned  # noqa: E402
from omlx.model_discovery import (  # noqa: E402
    _gemma4_text_only_prefers_llm_engine,
    _gemma4_text_only_wants_vlm_engine,
    discover_models,
)

MERGED_HEAD = {"mtp_assistant_config": {"num_hidden_layers": 4}}


def _write_safetensors(path: Path, keys: list[str]) -> None:
    header: dict = {}
    offset = 0
    for key in keys:
        header[key] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00\x00\x80?" * len(keys))


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


def test_text_only_gemma4_with_head_wants_vlm(tmp_path: Path):
    d = _model_dir(tmp_path, name="textonly_head", merged_head=True)
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is True


def test_vision_checkpoint_is_left_alone(tmp_path: Path):
    d = _model_dir(tmp_path, name="vision", vision_config=True, merged_head=True)
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is False


def test_text_only_gemma4_without_head_stays_on_llm(tmp_path: Path):
    d = _model_dir(tmp_path, name="textonly_bare")
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is False


def test_unified_needs_no_rerouting(tmp_path: Path):
    # gemma4_unified already routes to mlx-vlm on its model_type.
    d = _model_dir(
        tmp_path, name="unified_head", model_type="gemma4_unified", merged_head=True
    )
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is False


def test_text_only_unified_without_head_prefers_llm(tmp_path: Path):
    d = _model_dir(tmp_path, name="unified_bare", model_type="gemma4_unified")
    assert _gemma4_text_only_prefers_llm_engine(_config(d)) is True


def test_unified_vision_checkpoint_stays_on_vlm(tmp_path: Path):
    d = _model_dir(
        tmp_path,
        name="unified_vision",
        model_type="gemma4_unified",
        vision_config=True,
    )
    assert _gemma4_text_only_prefers_llm_engine(_config(d)) is False


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


def test_llm_route_also_reports_llm(tmp_path: Path):
    # supports_images is model_type == "vlm", so it moves with the engine.
    _model_dir(tmp_path, name="unified_textonly", model_type="gemma4_unified")
    found = discover_models(tmp_path)["unified_textonly"]
    assert found.engine_type == "batched"
    assert found.model_type == "llm"
    assert found.text_only_size == 0


def test_vlm_route_still_reports_llm(tmp_path: Path):
    _model_dir(tmp_path, name="gemma4_textonly_head", merged_head=True)
    found = discover_models(tmp_path)["gemma4_textonly_head"]
    assert found.engine_type == "vlm"
    assert found.model_type == "llm"


def test_vision_checkpoint_keeps_both_fields(tmp_path: Path):
    _model_dir(tmp_path, name="gemma4_vision", vision_config=True, vision_weights=True)
    found = discover_models(tmp_path)["gemma4_vision"]
    assert found.engine_type == "vlm"
    assert found.model_type == "vlm"


def test_unified_sanitize_keeps_mtp_head():
    # gemma4_unified's sanitize would move the head under language_model.model.
    from types import SimpleNamespace

    import mlx.core as mx
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
