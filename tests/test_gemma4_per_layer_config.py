"""Gemma4 checkpoints that keep the full-attention head overrides in
``text_config.per_layer_config`` instead of the legacy global fields (#3537)."""

import copy
import json
from pathlib import Path

import mlx_vlm.utils as mlx_vlm_utils
import pytest

from omlx.engine.vlm import (
    _derive_gemma4_global_kv_on_load,
    _gemma4_global_kv_from_per_layer_config,
)

FULL_ATTENTION_LAYERS = (5, 11, 17, 23, 29)


def _config(**text_overrides) -> dict:
    layer_types = ["sliding_attention"] * 30
    for idx in FULL_ATTENTION_LAYERS:
        layer_types[idx] = "full_attention"
    text_config = {
        "model_type": "gemma4_text",
        "num_hidden_layers": 30,
        "head_dim": 256,
        "num_key_value_heads": 8,
        "layer_types": layer_types,
        "per_layer_config": {
            f"{idx:02d}": {"head_dim": 512, "num_key_value_heads": 2}
            for idx in FULL_ATTENTION_LAYERS
        },
    }
    text_config.update(text_overrides)
    return {"model_type": "gemma4", "text_config": text_config}


def _write_model_dir(tmp_path: Path, name: str, config: dict) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def test_derives_legacy_global_fields_from_per_layer_config():
    assert _gemma4_global_kv_from_per_layer_config(_config()) == {
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
    }


def test_explicit_legacy_fields_win():
    assert _gemma4_global_kv_from_per_layer_config(_config(global_head_dim=512)) == {
        "num_global_key_value_heads": 2,
    }
    assert (
        _gemma4_global_kv_from_per_layer_config(
            _config(global_head_dim=512, num_global_key_value_heads=2)
        )
        == {}
    )


def test_only_full_attention_overrides_count():
    config = _config()
    config["text_config"]["per_layer_config"]["00"] = {
        "head_dim": 128,
        "num_key_value_heads": 4,
    }
    assert _gemma4_global_kv_from_per_layer_config(config) == {
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
    }


def test_without_layer_types_every_override_counts():
    config = _config()
    del config["text_config"]["layer_types"]
    assert _gemma4_global_kv_from_per_layer_config(config) == {
        "global_head_dim": 512,
        "num_global_key_value_heads": 2,
    }


def test_disagreeing_overrides_are_not_derived():
    config = _config()
    config["text_config"]["per_layer_config"]["11"]["num_key_value_heads"] = 4
    assert _gemma4_global_kv_from_per_layer_config(config) == {"global_head_dim": 512}


def test_non_gemma4_and_missing_per_layer_config_are_ignored():
    config = _config()
    config["model_type"] = "qwen3_vl"
    config["text_config"]["model_type"] = "qwen3_vl_text"
    assert _gemma4_global_kv_from_per_layer_config(config) == {}

    config = _config()
    del config["text_config"]["per_layer_config"]
    assert _gemma4_global_kv_from_per_layer_config(config) == {}

    assert _gemma4_global_kv_from_per_layer_config({"model_type": "gemma4"}) == {}


def test_on_load_hands_derived_fields_to_the_loader(tmp_path, monkeypatch):
    config = _config()
    model_dir = _write_model_dir(tmp_path, "model", config)
    seen = []

    def fake_load_config(model_path, **kwargs):
        seen.append((model_path, kwargs))
        return copy.deepcopy(config)

    monkeypatch.setattr(mlx_vlm_utils, "load_config", fake_load_config)

    with _derive_gemma4_global_kv_on_load(model_dir):
        assert mlx_vlm_utils.load_config is not fake_load_config
        loaded = mlx_vlm_utils.load_config(model_dir, trust_remote_code=True)

    text_config = loaded["text_config"]
    assert text_config["global_head_dim"] == 512
    assert text_config["num_global_key_value_heads"] == 2
    # The source layout stays in place; only the missing legacy fields are added.
    assert text_config["per_layer_config"] == config["text_config"]["per_layer_config"]
    assert text_config["head_dim"] == 256
    assert text_config["num_key_value_heads"] == 8
    assert seen == [(model_dir, {"trust_remote_code": True})]
    # The wrapper is removed again once loading is over.
    assert mlx_vlm_utils.load_config is fake_load_config


def test_on_load_keeps_explicit_legacy_fields(tmp_path, monkeypatch):
    config = _config(global_head_dim=512, num_global_key_value_heads=4)
    model_dir = _write_model_dir(tmp_path, "model", config)
    monkeypatch.setattr(
        mlx_vlm_utils, "load_config", lambda model_path, **kw: copy.deepcopy(config)
    )

    with _derive_gemma4_global_kv_on_load(model_dir):
        loaded = mlx_vlm_utils.load_config(model_dir)

    assert loaded["text_config"]["num_global_key_value_heads"] == 4


@pytest.mark.parametrize(
    "config",
    [
        _config(global_head_dim=512, num_global_key_value_heads=2),
        {"model_type": "qwen3_vl", "text_config": {"per_layer_config": {"0": {}}}},
        {"model_type": "gemma4", "text_config": {"head_dim": 256}},
    ],
)
def test_on_load_is_a_noop_without_anything_to_derive(tmp_path, monkeypatch, config):
    model_dir = _write_model_dir(tmp_path, "model", config)

    def fake_load_config(model_path, **kwargs):
        return {"text_config": {}}

    monkeypatch.setattr(mlx_vlm_utils, "load_config", fake_load_config)

    with _derive_gemma4_global_kv_on_load(model_dir):
        assert mlx_vlm_utils.load_config is fake_load_config


def test_on_load_tolerates_a_missing_config(tmp_path, monkeypatch):
    def fake_load_config(model_path, **kwargs):
        return {"text_config": {}}

    monkeypatch.setattr(mlx_vlm_utils, "load_config", fake_load_config)

    with _derive_gemma4_global_kv_on_load(tmp_path / "missing"):
        assert mlx_vlm_utils.load_config is fake_load_config
