# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.model_loading.maybe_load_custom_quantization."""

import sys
import types
from unittest.mock import MagicMock

import pytest

from omlx.utils import model_loading
from omlx.utils.model_loading import (
    maybe_apply_pre_load_patches,
    maybe_load_custom_quantization,
)


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


class TestNoDispatch:
    """Cases where the dispatcher should return None and let the caller
    fall back to the standard mlx-lm/mlx-vlm load path."""

    def test_missing_config_returns_none(self, tmp_path):
        # tmp_path has no config.json
        assert maybe_load_custom_quantization(str(tmp_path), is_vlm=False) is None

    def test_malformed_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, "{not valid json")
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_no_quantization_config_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"model_type": "llama"}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_empty_quant_method_returns_none(self, tmp_path):
        path = _write_config(tmp_path, '{"quantization_config": {}}')
        assert maybe_load_custom_quantization(path, is_vlm=False) is None

    def test_unknown_quant_method_returns_none(self, tmp_path):
        # Methods we don't dispatch on (mlx-lm may or may not handle them
        # natively; either way the dispatcher stays out of the way).
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "awq"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) is None


def _install_paroquant_stub(monkeypatch, load_impl):
    """Register a minimal paroquant.inference.backends.mlx.load stub."""
    names = [
        "paroquant",
        "paroquant.inference",
        "paroquant.inference.backends",
        "paroquant.inference.backends.mlx",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    load_mod = types.ModuleType("paroquant.inference.backends.mlx.load")
    load_mod.load = load_impl
    monkeypatch.setitem(sys.modules, "paroquant.inference.backends.mlx.load", load_mod)


class TestParoquantDispatch:
    """Cases where quant_method == 'paroquant'."""

    def test_paroquant_missing_raises_install_hint(self, tmp_path, monkeypatch):
        # Force the import to fail: shadow the package with a sentinel that
        # blocks submodule resolution. setitem(..., None) makes `import X`
        # raise ImportError in the standard machinery.
        for name in [
            "paroquant",
            "paroquant.inference",
            "paroquant.inference.backends",
            "paroquant.inference.backends.mlx",
            "paroquant.inference.backends.mlx.load",
        ]:
            monkeypatch.setitem(sys.modules, name, None)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ImportError, match="paroquant"):
            maybe_load_custom_quantization(path, is_vlm=False)

    def test_paroquant_text_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is True
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=False)
        assert result == ("MODEL", "PROC")

    def test_paroquant_vlm_load_returns_tuple(self, tmp_path, monkeypatch):
        def fake_load(model_path, force_text):
            assert force_text is False
            return "VLM_MODEL", "VLM_PROC", True

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        result = maybe_load_custom_quantization(path, is_vlm=True)
        assert result == ("VLM_MODEL", "VLM_PROC")

    def test_paroquant_text_only_for_vlm_load_raises(self, tmp_path, monkeypatch):
        # is_vlm=True but the loader returned (..., loaded_is_vlm=False).
        def fake_load(model_path, force_text):
            return "MODEL", "PROC", False

        _install_paroquant_stub(monkeypatch, fake_load)

        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "paroquant"}}',
        )
        with pytest.raises(ValueError, match="text-only"):
            maybe_load_custom_quantization(path, is_vlm=True)

    def test_quant_method_case_insensitive(self, tmp_path, monkeypatch):
        # The dispatcher lowercases quant_method, so mixed-case configs
        # (e.g. produced by other tooling) still hit the paroquant path.
        captured = {}

        def fake_load(model_path, force_text):
            captured["called"] = True
            return "M", "P", False

        _install_paroquant_stub(monkeypatch, fake_load)
        path = _write_config(
            tmp_path,
            '{"quantization_config": {"quant_method": "ParoQuant"}}',
        )
        assert maybe_load_custom_quantization(path, is_vlm=False) == ("M", "P")
        assert captured["called"] is True


class TestVlmMtpPreLoadDispatch:
    """maybe_apply_pre_load_patches must wire the mlx-vlm MTP sanitize
    patch alongside the runtime patch for MTP-capable VLM checkpoints.

    The dense Qwen3.5/3.6 VLM runtime patch does not touch Model.sanitize;
    it relies on apply_mlx_vlm_mtp_patch having installed the mtp.*
    preservation first. If only the runtime patch runs, stock mlx-vlm
    sanitize strips every mtp.* key and the MTP head loads at random
    init (PR #1320)."""

    def _stub_patches(self, monkeypatch):
        """Replace the patch modules with mocks that record call order.

        Returns the recorded-order list plus the sanitize/runtime mocks."""
        calls: list[str] = []
        sanitize_mock = MagicMock(side_effect=lambda: calls.append("sanitize") or True)
        runtime_mock = MagicMock(side_effect=lambda: calls.append("runtime") or True)
        # Side-step the real mlx-lm load_config monkey-patch.
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(
                set_mtp_active=MagicMock(),
                apply_mlx_lm_mtp_patch=MagicMock(return_value=True),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_vlm_mtp",
            MagicMock(
                apply_mlx_vlm_mtp_patch=sanitize_mock,
                apply_mlx_vlm_mtp_runtime_patch=runtime_mock,
            ),
        )
        return calls, sanitize_mock, runtime_mock

    def test_sanitize_patch_runs_before_runtime_for_vlm_mtp(
        self, tmp_path, monkeypatch
    ):
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        # qwen3_5 (dense VLM) declaring an MTP head under text_config.
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=True)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_called_once()
        runtime_mock.assert_called_once()
        # Ordering matters: the dense runtime patch assumes sanitize was
        # already installed by apply_mlx_vlm_mtp_patch.
        assert calls == ["sanitize", "runtime"]

    def test_sanitize_patch_skipped_when_mtp_disabled(self, tmp_path, monkeypatch):
        calls, sanitize_mock, runtime_mock = self._stub_patches(monkeypatch)
        path = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "vision_config": {}, '
            '"text_config": {"mtp_num_hidden_layers": 1}}',
        )
        settings = types.SimpleNamespace(mtp_enabled=False)

        maybe_apply_pre_load_patches(path, model_settings=settings)

        sanitize_mock.assert_not_called()
        runtime_mock.assert_not_called()
        assert calls == []
