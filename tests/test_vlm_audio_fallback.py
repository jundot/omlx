"""Tests for the audio_tower fallback in VLM loading.

Background: oQ-quantized multimodal Gemma 4 checkpoints sometimes ship with
`audio_config` in `config.json` but no `audio_tower.*` weights in the
safetensors. Loading them via `mlx_vlm.utils.load(...)` then crashes with
"Missing 752 parameters" because mlx-vlm instantiates `AudioEncoder` based
on `audio_config`. The `_strip_audio_config_if_orphaned` context manager
swaps `mlx_vlm.utils.load_config` for the duration of the call so that the
config is read with `audio_config = None` when audio weights are absent,
letting the model load without audio support.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import mlx_vlm.utils as _vu

from omlx.engine.vlm import (
    _AUDIO_CONFIG_KEYS,
    _has_audio_weights,
    _strip_audio_config_if_orphaned,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_safetensors(path: Path, keys: list[str]) -> None:
    """Write a tiny safetensors file with the given parameter keys."""
    from safetensors.numpy import save_file
    import numpy as np

    payload = {k: np.zeros((1,), dtype=np.float32) for k in keys}
    save_file(payload, str(path))


def _build_model_dir(
    tmp_path: Path,
    *,
    name: str,
    has_audio_config: bool,
    has_audio_weights: bool,
) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()

    config: dict = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {"hidden_size": 32, "num_hidden_layers": 1},
        "vision_config": {"hidden_size": 16},
    }
    if has_audio_config:
        config["audio_config"] = {"hidden_size": 16}
        config["audio_token_id"] = 258881
        config["boa_token_id"] = 256000
        config["eoa_token_id"] = 258883
        config["eoa_token_index"] = 258883
    (model_dir / "config.json").write_text(json.dumps(config))

    keys = ["language_model.model.layers.0.self_attn.q_proj.weight"]
    if has_audio_weights:
        keys.append("audio_tower.layers.0.feed_forward1.linear.weight")
        keys.append("embed_audio.embedding_projection.weight")
    _write_safetensors(model_dir / "model.safetensors", keys)

    return model_dir


# ---------------------------------------------------------------------------
# _has_audio_weights
# ---------------------------------------------------------------------------


class TestHasAudioWeights:
    def test_returns_true_when_audio_tower_key_present(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="m1", has_audio_config=True, has_audio_weights=True,
        )
        assert _has_audio_weights(model_dir) is True

    def test_returns_false_when_no_audio_keys(self, tmp_path: Path):
        model_dir = _build_model_dir(
            tmp_path, name="m2", has_audio_config=True, has_audio_weights=False,
        )
        assert _has_audio_weights(model_dir) is False

    def test_returns_false_for_empty_dir(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _has_audio_weights(empty) is False


# ---------------------------------------------------------------------------
# _strip_audio_config_if_orphaned
# ---------------------------------------------------------------------------


class TestStripAudioConfigIfOrphaned:
    def test_passthrough_when_config_has_no_audio(self, tmp_path: Path):
        # Config with no audio_config — patch must leave the dict untouched.
        model_dir = _build_model_dir(
            tmp_path, name="vision_only",
            has_audio_config=False, has_audio_weights=False,
        )
        with _strip_audio_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert "audio_config" not in cfg

    def test_passthrough_when_audio_weights_present(self, tmp_path: Path):
        # Healthy multimodal model — audio_config must remain in the dict.
        model_dir = _build_model_dir(
            tmp_path, name="full",
            has_audio_config=True, has_audio_weights=True,
        )
        with _strip_audio_config_if_orphaned(model_dir):
            cfg = _vu.load_config(model_dir)
        assert cfg.get("audio_config") is not None

    def test_strips_audio_when_weights_missing(self, tmp_path: Path, caplog):
        # Defective oQ-style checkpoint: audio_config present, audio weights absent.
        model_dir = _build_model_dir(
            tmp_path, name="defective",
            has_audio_config=True, has_audio_weights=False,
        )
        with caplog.at_level("WARNING"):
            with _strip_audio_config_if_orphaned(model_dir):
                cfg = _vu.load_config(model_dir)
        # audio_config must be explicitly None (not popped) so mlx-vlm's
        # `setdefault("audio_config", {})` does not repopulate it.
        assert "audio_config" in cfg
        assert cfg["audio_config"] is None
        # Other audio-related keys are popped.
        for k in _AUDIO_CONFIG_KEYS:
            if k != "audio_config":
                assert k not in cfg
        # WARN log fired.
        assert any(
            "audio_tower weights missing" in rec.message
            for rec in caplog.records
        )

    def test_warning_only_logged_once_per_path(self, tmp_path: Path, caplog):
        model_dir = _build_model_dir(
            tmp_path, name="def2",
            has_audio_config=True, has_audio_weights=False,
        )
        with caplog.at_level("WARNING"):
            with _strip_audio_config_if_orphaned(model_dir):
                _vu.load_config(model_dir)
                _vu.load_config(model_dir)
                _vu.load_config(model_dir)
        warnings = [
            rec for rec in caplog.records
            if "audio_tower weights missing" in rec.message
        ]
        assert len(warnings) == 1

    def test_load_config_restored_on_normal_exit(self, tmp_path: Path):
        original = _vu.load_config
        model_dir = _build_model_dir(
            tmp_path, name="r1",
            has_audio_config=True, has_audio_weights=False,
        )
        with _strip_audio_config_if_orphaned(model_dir):
            assert _vu.load_config is not original
        assert _vu.load_config is original

    def test_load_config_restored_on_exception(self, tmp_path: Path):
        original = _vu.load_config
        model_dir = _build_model_dir(
            tmp_path, name="r2",
            has_audio_config=True, has_audio_weights=False,
        )
        with pytest.raises(RuntimeError, match="boom"):
            with _strip_audio_config_if_orphaned(model_dir):
                raise RuntimeError("boom")
        assert _vu.load_config is original

    def test_skips_when_path_is_not_directory(self, tmp_path: Path):
        # When the patched loader is called with a non-directory path (e.g.
        # an HF repo ID before download), the audio_config branch must defer
        # to mlx-vlm's normal flow rather than error out.
        nonexistent = tmp_path / "nonexistent-repo"
        sentinel = {
            "audio_config": {"hidden_size": 99},
            "audio_token_id": 12345,
        }
        with patch.object(_vu, "load_config", return_value=sentinel):
            with _strip_audio_config_if_orphaned(nonexistent):
                cfg = _vu.load_config(nonexistent)
        # cfg returned unchanged — audio_config still a dict, not None.
        assert cfg["audio_config"] == {"hidden_size": 99}
        assert cfg["audio_token_id"] == 12345
