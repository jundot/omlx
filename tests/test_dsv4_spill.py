"""Unit tests for per-expert DeepSeek-V4 spill-stacking."""

import mlx.core as mx

from omlx.patches.deepseek_v4 import spill as S


def _fake_weights(n_layers=2, n_experts=2):
    weights = {}
    for i in range(n_layers):
        for e in range(n_experts):
            for src, shp in (("w1", (8, 4)), ("w2", (4, 8)), ("w3", (8, 4))):
                weights[f"model.layers.{i}.ffn.experts.{e}.{src}.weight"] = mx.zeros(shp)
                weights[f"model.layers.{i}.ffn.experts.{e}.{src}.scales"] = mx.zeros((shp[0], 2))
                weights[f"model.layers.{i}.ffn.experts.{e}.{src}.biases"] = mx.zeros((shp[0], 2))
    return weights


def test_stack_layer_to_spill_roundtrip(tmp_path):
    weights = _fake_weights()
    out = S.stack_layer_to_spill(weights, layer_idx=0, n_experts=2, spill_dir=tmp_path)
    # per-expert keys popped, stacked keys returned
    assert not any(".ffn.experts." in k and k.startswith("model.layers.0.") for k in weights)
    assert len(out) == 9
    assert out["model.layers.0.ffn.switch_mlp.gate_proj.weight"].shape == (2, 8, 4)
    assert out["model.layers.0.ffn.switch_mlp.down_proj.scales"].shape == (2, 4, 2)
    assert S.spill_layer_ok(tmp_path, 0)
    assert not S.spill_layer_ok(tmp_path, 1)


def test_spill_validity_roundtrip(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 64)
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    (spill_dir / "spill_layer_00.safetensors").write_bytes(b"y" * 64)
    S.write_manifest(spill_dir, model_dir, ["spill_layer_00.safetensors"], {"k": "spill_layer_00.safetensors"})
    assert S.spill_is_valid(model_dir) is None  # spill_dir_for(model) != spill_dir
    # validity is anchored at the conventional location
    conv = S.spill_dir_for(model_dir)
    conv.mkdir(parents=True)
    (conv / "spill_layer_00.safetensors").write_bytes(b"y" * 64)
    S.write_manifest(conv, model_dir, ["spill_layer_00.safetensors"], {"k": "spill_layer_00.safetensors"})
    assert S.spill_is_valid(model_dir) == conv
    # source change invalidates
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 65)
    assert S.spill_is_valid(model_dir) is None


def test_spill_key_to_file():
    m = {"key_to_file": {"a": "f.safetensors"}}
    assert S.spill_key_to_file(m) == {"a": "f.safetensors"}
    assert S.spill_key_to_file({}) == {}


def test_spill_disabled_env(monkeypatch):
    monkeypatch.setenv("OMLX_DSV4_SPILL", "0")
    assert S.spill_disabled()
    monkeypatch.setenv("OMLX_DSV4_SPILL", "1")
    assert not S.spill_disabled()
