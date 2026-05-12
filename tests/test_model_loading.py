# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.model_loading.maybe_load_custom_quantization."""

import sys
import types

import pytest

from omlx.utils.model_loading import maybe_load_custom_quantization


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
    monkeypatch.setitem(
        sys.modules, "paroquant.inference.backends.mlx.load", load_mod
    )


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

    def test_paroquant_text_only_for_vlm_load_raises(
        self, tmp_path, monkeypatch
    ):
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
