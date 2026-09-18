"""Unified MoE backend: legacy keys route to expert_streaming where owned."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx.model_settings import moe_offload_requested
from omlx.patches import moe_expert_offload as legado


def _cfg(tmp_path: Path, model_type: str) -> str:
    d = tmp_path / ("m-" + model_type)
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def test_moe_offload_requested():
    assert moe_offload_requested(None) is False
    assert moe_offload_requested({}) is False
    assert moe_offload_requested(SimpleNamespace()) is False
    assert moe_offload_requested({"moe_expert_offload_enabled": True}) is True
    assert moe_offload_requested(SimpleNamespace(moe_expert_offload_enabled=True)) is True
    assert moe_offload_requested({"expert_streaming_enabled": True}) is True
    assert moe_offload_requested(SimpleNamespace(expert_streaming_enabled=True)) is True


def test_streaming_owns_model(tmp_path):
    assert legado._streaming_owns_model(_cfg(tmp_path, "qwen4_exp")) is True
    assert legado._streaming_owns_model(_cfg(tmp_path, "gemma4")) is False
    assert legado._streaming_owns_model(_cfg(tmp_path, "deepseek_v41")) is False
    assert legado._streaming_owns_model(str(tmp_path / "nope")) is False


def _fake_est(**kw):
    base = dict(supported=True, expert_bytes=8 * 1024**3, num_moe_layers=4)
    base.update(kw)
    return SimpleNamespace(**base)


class TestViaStreaming:
    EST = "omlx.patches.expert_streaming.residency.expert_streaming_estimate"
    CONV = "omlx.patches.expert_streaming.convert_model_to_streaming"

    def test_already_converted_not_reconverted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.EST, lambda *_a, **_k: _fake_est())

        def _boom(*_a, **_k):
            raise AssertionError("must not reconvert")

        monkeypatch.setattr(self.CONV, _boom)
        model = SimpleNamespace(_expert_streaming_backing=object())
        assert legado._apply_via_streaming(model, _cfg(tmp_path, "qwen4_exp"), 0.25) == 4

    def test_unsupported_estimate_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.EST, lambda *_a, **_k: _fake_est(supported=False))
        out = legado._apply_via_streaming(SimpleNamespace(), _cfg(tmp_path, "qwen4_exp"), 0.25)
        assert out == 0


def _wrappable_model(tmp_path):
    """A one-layer model + checkpoint the LEGACY adapter can wrap, in a
    dir whose config claims a streaming-owned model type."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.models.switch_layers import SwitchGLU

    mx.random.seed(0)
    glu = SwitchGLU(64, 32, 32)
    nn.quantize(glu, group_size=32, bits=4)
    tensors = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(glu, proj)
        for field in ("weight", "scales", "biases"):
            if lin.get(field) is not None:
                tensors[f"layers.0.experts.switch_glu.{proj}.{field}"] = lin[field]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp"})
    )

    class _Experts(nn.Module):
        def __init__(self, g):
            super().__init__()
            self.switch_glu = g

    class _Layer(nn.Module):
        def __init__(self, g):
            super().__init__()
            self.experts = _Experts(g)

    class _Mini(nn.Module):
        def __init__(self, g):
            super().__init__()
            self.layers = [_Layer(g)]

    return _Mini(glu)


class TestDowngradeSemantics:
    """B8: the alias path's fallback must be loud and stamped, and a
    converter crash on a streaming-owned type must fail hard."""

    EST = "omlx.patches.expert_streaming.residency.expert_streaming_estimate"
    CONV = "omlx.patches.expert_streaming.convert_model_to_streaming"

    def test_converter_crash_on_supported_type_propagates(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(self.EST, lambda *_a, **_k: _fake_est())

        def _boom(*_a, **_k):
            raise RuntimeError("converter exploded")

        monkeypatch.setattr(self.CONV, _boom)
        with pytest.raises(RuntimeError, match="converter exploded"):
            legado.apply_moe_expert_offload(
                SimpleNamespace(), _cfg(tmp_path, "qwen4_exp"), 0.25
            )

    def test_converts_nothing_falls_back_stamped(self, tmp_path, monkeypatch):
        """backing=None -> the legacy adapter still serves, but the state
        and the store carry the downgrade marker into the summary."""
        model = _wrappable_model(tmp_path)
        monkeypatch.setattr(self.EST, lambda *_a, **_k: _fake_est())
        monkeypatch.setattr(self.CONV, lambda *_a, **_k: (None, None))
        try:
            n = legado.apply_moe_expert_offload(model, tmp_path, 0.25)
            assert n == 1
            state = model._moe_offload_legacy_state
            assert state.streaming_fallback_reason
            assert (
                state.summary()["streaming_fallback_reason"]
                == state.streaming_fallback_reason
            )
            store = model._expert_streaming_backing
            assert (
                store.streaming_fallback_reason
                == state.streaming_fallback_reason
            )
        finally:
            store = getattr(model, "_expert_streaming_backing", None)
            if store is not None:
                store.close()

    def test_estimate_failure_falls_back_stamped(self, tmp_path, monkeypatch):
        """A raise while support is unproven (the estimate itself broke)
        degrades to legacy with the marker instead of crashing the load."""
        model = _wrappable_model(tmp_path)

        def _boom(*_a, **_k):
            raise OSError("estimate unreadable")

        monkeypatch.setattr(self.EST, _boom)
        try:
            n = legado.apply_moe_expert_offload(model, tmp_path, 0.25)
            assert n == 1
            state = model._moe_offload_legacy_state
            assert state.streaming_fallback_reason
        finally:
            store = getattr(model, "_expert_streaming_backing", None)
            if store is not None:
                store.close()
