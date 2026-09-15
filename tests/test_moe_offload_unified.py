"""Unified MoE backend: legacy keys route to expert_streaming where owned."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


from omlx.model_settings import moe_offload_requested
from omlx.patches import moe_expert_offload as legado


def _cfg(tmp_path: Path, model_type: str) -> str:
    d = tmp_path / ("m-" + model_type)
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


class TestRequested:
    def test_none(self):
        assert moe_offload_requested(None) is False

    def test_empty(self):
        assert moe_offload_requested({}) is False
        assert moe_offload_requested(SimpleNamespace()) is False

    def test_legacy(self):
        assert moe_offload_requested({"moe_expert_offload_enabled": True}) is True
        ns = SimpleNamespace(moe_expert_offload_enabled=True)
        assert moe_offload_requested(ns) is True

    def test_canonical(self):
        assert moe_offload_requested({"expert_streaming_enabled": True}) is True
        ns = SimpleNamespace(expert_streaming_enabled=True)
        assert moe_offload_requested(ns) is True


class TestOwnership:
    def test_owned(self, tmp_path):
        assert legado._streaming_owns_model(_cfg(tmp_path, "qwen4_exp")) is True

    def test_legacy_types(self, tmp_path):
        assert legado._streaming_owns_model(_cfg(tmp_path, "gemma4")) is False
        assert legado._streaming_owns_model(_cfg(tmp_path, "deepseek_v41")) is False

    def test_missing(self, tmp_path):
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
