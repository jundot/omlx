# SPDX-License-Identifier: Apache-2.0
"""Tests for serving text-only Gemma 4 checkpoints on the VLM engine.

Two decisions are covered. Which engine a checkpoint is routed to, which is
made from the config at discovery time; and what the mlx-vlm loader is handed,
which is decided from the weights so that an absent sub-config and an orphaned
one behave the same way.

Tiny fixtures throughout — configs and one-tensor shards, no checkpoint.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("mlx_vlm.utils")

import mlx_vlm.utils as _vu  # noqa: E402

from omlx.engine.vlm import (  # noqa: E402
    _has_vision_weights,
    _strip_audio_config_if_orphaned,
)
from omlx.model_discovery import (  # noqa: E402
    _gemma4_text_only_prefers_llm_engine,
    _gemma4_text_only_wants_vlm_engine,
)

MERGED_HEAD = {"mtp_assistant_config": {"num_hidden_layers": 4}}


def _write_safetensors(path: Path, keys: list[str]) -> None:
    """A shard with the given key names and one f32 scalar each."""
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
    path.write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x00\x00\x80?" * len(keys)
    )


def _model_dir(
    tmp_path: Path,
    *,
    name: str,
    model_type: str = "gemma4",
    vision_config: bool = False,
    audio_config: bool = False,
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
    if audio_config:
        config["audio_config"] = {"hidden_size": 16}
    (model_dir / "config.json").write_text(json.dumps(config))

    keys = ["language_model.model.layers.0.self_attn.q_proj.weight"]
    if vision_weights:
        keys.append("vision_embedder.patch_dense.weight")
    _write_safetensors(model_dir / "model.safetensors", keys)
    return model_dir


def _config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


# ---------------------------------------------------------------------------
# Which engine
# ---------------------------------------------------------------------------


def test_text_only_gemma4_with_a_head_wants_the_vlm_engine(tmp_path: Path):
    # The case #3536 is about: no vision sub-config, so it is classed
    # text-only and served by mlx-lm, where the merged head has no binding
    # site and MTP is advertised without ever drafting.
    d = _model_dir(tmp_path, name="textonly_head", merged_head=True)
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is True


def test_a_vision_checkpoint_is_left_alone(tmp_path: Path):
    d = _model_dir(tmp_path, name="vision", vision_config=True, merged_head=True)
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is False


def test_text_only_gemma4_without_a_head_is_left_on_mlx_lm(tmp_path: Path):
    # Reaching the VLM engine costs memory, so a checkpoint with no head to
    # drive has nothing to gain by moving.
    d = _model_dir(tmp_path, name="textonly_bare")
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is False


def test_unified_needs_no_rerouting(tmp_path: Path):
    # gemma4_unified already routes to mlx-vlm on its model_type alone.
    d = _model_dir(
        tmp_path, name="unified_head", model_type="gemma4_unified", merged_head=True
    )
    assert _gemma4_text_only_wants_vlm_engine(_config(d)) is False


def test_text_only_unified_without_a_head_prefers_mlx_lm(tmp_path: Path):
    # It is routed to mlx-vlm by model_type, and used to fail there and fall
    # back. Now that the load succeeds it would stay and pay the overhead, so
    # send it to mlx-lm deliberately instead of by way of a failed load.
    d = _model_dir(tmp_path, name="unified_bare", model_type="gemma4_unified")
    assert _gemma4_text_only_prefers_llm_engine(_config(d)) is True


def test_a_unified_vision_checkpoint_stays_on_the_vlm_engine(tmp_path: Path):
    d = _model_dir(
        tmp_path, name="unified_vision", model_type="gemma4_unified",
        vision_config=True,
    )
    assert _gemma4_text_only_prefers_llm_engine(_config(d)) is False


# ---------------------------------------------------------------------------
# What the loader is handed
# ---------------------------------------------------------------------------


def test_vision_weights_are_detected_by_embedder_prefix(tmp_path: Path):
    # A unified checkpoint has no vision tower at all -- the single
    # transformer does the work -- so the embedder prefixes are what decide it.
    present = _model_dir(tmp_path, name="has_vision", vision_weights=True)
    absent = _model_dir(tmp_path, name="no_vision")
    assert _has_vision_weights(present) is True
    assert _has_vision_weights(absent) is False


def test_loader_is_pointed_at_the_text_capable_module(tmp_path: Path):
    # mlx-vlm resolves the model class from model_type, and its `gemma4`
    # module builds a vision tower unconditionally; only `gemma4_unified`
    # carries the text-only branch.
    d = _model_dir(tmp_path, name="textonly_head2", merged_head=True)
    with _strip_audio_config_if_orphaned(d):
        cfg = _vu.load_config(d)
    assert cfg["model_type"] == "gemma4_unified"
    assert cfg["_omlx_absent_modalities"] == ["vision"]


def test_a_vision_checkpoint_keeps_its_model_type(tmp_path: Path):
    d = _model_dir(
        tmp_path, name="vision2", vision_config=True, vision_weights=True
    )
    with _strip_audio_config_if_orphaned(d):
        cfg = _vu.load_config(d)
    assert cfg["model_type"] == "gemma4"
    assert "_omlx_absent_modalities" not in cfg


def test_absent_vision_config_is_marked_rather_than_nulled(tmp_path: Path):
    # load_model reads `config.get("vision_config", {}).get("skip_vision")`,
    # so the key has to keep a dict shape; nulling it raises AttributeError.
    d = _model_dir(tmp_path, name="textonly_head3", merged_head=True)
    with _strip_audio_config_if_orphaned(d):
        cfg = _vu.load_config(d)
    # A None here is an AttributeError inside load_model, which is exactly how
    # the first attempt at this failed.
    assert cfg.get("vision_config", {}) is not None
    assert isinstance(cfg.get("vision_config", {}), dict)
    assert "vision" in cfg["_omlx_absent_modalities"]

# ---------------------------------------------------------------------------
# The two fields have to move together
# ---------------------------------------------------------------------------


def test_rerouting_to_the_llm_engine_also_reports_it_as_an_llm(tmp_path: Path):
    """`model_type` is what advertises capability, so it moves with the engine.

    `supports_images` is `model_type == "vlm"`, and `detect_model_type` classes
    every gemma4_unified checkpoint as a VLM whether or not it carries vision.
    Moving only `engine_type` left a text-only checkpoint on the LLM engine
    still telling clients it accepts images -- they would send one, and the
    engine that received it cannot process image content at all.
    """
    from omlx.model_discovery import discover_models

    root = tmp_path / "models"
    root.mkdir()
    _model_dir(root, name="unified_textonly", model_type="gemma4_unified")

    found = discover_models(root)["unified_textonly"]
    assert found.engine_type == "batched"
    assert found.model_type == "llm"
    assert found.text_only_size == 0


def test_rerouting_to_the_vlm_engine_keeps_reporting_it_as_an_llm(tmp_path: Path):
    # The other direction: served by the VLM engine so its head can be driven,
    # but still a text-only model, so it must not advertise images.
    from omlx.model_discovery import discover_models

    root = tmp_path / "models"
    root.mkdir()
    _model_dir(root, name="gemma4_textonly_head", merged_head=True)

    found = discover_models(root)["gemma4_textonly_head"]
    assert found.engine_type == "vlm"
    assert found.model_type == "llm"


def test_a_vision_checkpoint_keeps_both_fields(tmp_path: Path):
    from omlx.model_discovery import discover_models

    root = tmp_path / "models"
    root.mkdir()
    _model_dir(
        root, name="gemma4_vision", vision_config=True, vision_weights=True
    )

    found = discover_models(root)["gemma4_vision"]
    assert found.engine_type == "vlm"
    assert found.model_type == "vlm"


def test_one_unreadable_shard_does_not_decide_the_modality(tmp_path: Path):
    # A per-shard read error used to answer for the whole checkpoint, which
    # for a real multimodal model drops the modality and turns a transient
    # error into a strict-load failure on its unclaimed tensors.
    model_dir = _model_dir(tmp_path, name="torn", vision_config=True)
    _write_safetensors(
        model_dir / "model-00002.safetensors",
        ["vision_embedder.patch_dense.weight"],
    )
    (model_dir / "model-00003.safetensors").write_bytes(b"not safetensors")

    assert _has_vision_weights(model_dir) is True
