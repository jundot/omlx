# SPDX-License-Identifier: Apache-2.0
"""Tests for DDTree integration (rides on top of DFlash)."""

import pytest

from omlx.model_settings import ModelSettings


class TestDDTreeModelSettings:
    """DDTree fields in ModelSettings."""

    def test_default_values(self):
        settings = ModelSettings()
        assert settings.ddtree_enabled is False
        assert settings.ddtree_budget == 4
        assert settings.ddtree_exact_commit is False

    def test_to_dict_includes_ddtree_fields_when_enabled(self):
        settings = ModelSettings(
            dflash_enabled=True,
            ddtree_enabled=True,
            ddtree_budget=6,
            ddtree_exact_commit=True,
        )
        d = settings.to_dict()
        assert d["ddtree_enabled"] is True
        assert d["ddtree_budget"] == 6
        assert d["ddtree_exact_commit"] is True

    def test_to_dict_budget_always_present(self):
        """ddtree_budget has a non-None default so it always serializes."""
        settings = ModelSettings()
        d = settings.to_dict()
        assert d["ddtree_budget"] == 4

    def test_from_dict_with_ddtree_fields(self):
        data = {
            "dflash_enabled": True,
            "ddtree_enabled": True,
            "ddtree_budget": 8,
            "ddtree_exact_commit": True,
        }
        settings = ModelSettings.from_dict(data)
        assert settings.ddtree_enabled is True
        assert settings.ddtree_budget == 8
        assert settings.ddtree_exact_commit is True

    def test_from_dict_ignores_unknown_ddtree_fields(self):
        data = {"ddtree_enabled": True, "ddtree_legacy_field": "ignored"}
        settings = ModelSettings.from_dict(data)
        assert settings.ddtree_enabled is True

    def test_roundtrip_serialization(self):
        original = ModelSettings(
            dflash_enabled=True,
            ddtree_enabled=True,
            ddtree_budget=2,
            ddtree_exact_commit=False,
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.ddtree_enabled == original.ddtree_enabled
        assert restored.ddtree_budget == original.ddtree_budget
        assert restored.ddtree_exact_commit == original.ddtree_exact_commit


class TestDDTreeEngineInit:
    """DFlashEngine picks up DDTree settings from model_settings."""

    def _engine(self, **settings_kwargs):
        from omlx.engine.dflash import DFlashEngine

        s = ModelSettings(
            dflash_enabled=True,
            dflash_draft_model="z-lab/Qwen3.5-4B-DFlash",
            **settings_kwargs,
        )
        return DFlashEngine(
            model_name="test-model",
            draft_model_path="z-lab/Qwen3.5-4B-DFlash",
            model_settings=s,
        )

    def test_ddtree_disabled_by_default(self):
        e = self._engine()
        assert e._ddtree_enabled is False
        assert e._ddtree_budget == 4
        assert e._ddtree_exact_commit is False
        assert e._use_ddtree() is False

    def test_ddtree_enabled_reads_settings(self):
        e = self._engine(ddtree_enabled=True, ddtree_budget=6, ddtree_exact_commit=True)
        assert e._ddtree_enabled is True
        assert e._ddtree_budget == 6
        assert e._ddtree_exact_commit is True

    def test_use_ddtree_true_when_importable(self):
        pytest.importorskip("ddtree_mlx")
        e = self._engine(ddtree_enabled=True)
        assert e._use_ddtree() is True

    def test_use_ddtree_false_in_fallback_mode(self):
        pytest.importorskip("ddtree_mlx")
        e = self._engine(ddtree_enabled=True)
        e._in_fallback_mode = True
        assert e._use_ddtree() is False

    def test_import_failure_is_cached_and_logged_once(self, monkeypatch, caplog):
        """If ddtree_mlx import fails, _use_ddtree returns False and warns once."""
        import builtins
        import importlib
        import logging

        e = self._engine(ddtree_enabled=True)

        real_import = builtins.__import__

        def _raising_import(name, *args, **kwargs):
            if name.startswith("ddtree_mlx"):
                raise ImportError("simulated missing ddtree_mlx")
            return real_import(name, *args, **kwargs)

        # Force the engine's import cache to re-resolve under our mock.
        e._ddtree_import_ok = None
        # Evict real ddtree_mlx from sys.modules so the import actually runs.
        import sys
        for mod in list(sys.modules):
            if mod.startswith("ddtree_mlx"):
                sys.modules.pop(mod, None)

        monkeypatch.setattr(builtins, "__import__", _raising_import)
        caplog.set_level(logging.WARNING, logger="omlx.engine.dflash")
        try:
            assert e._use_ddtree() is False
            assert e._ddtree_import_ok is False
            # Second call must not re-warn (cached).
            caplog.clear()
            assert e._use_ddtree() is False
            assert not any(
                "ddtree-mlx is not installed" in rec.message
                for rec in caplog.records
            )
        finally:
            monkeypatch.setattr(builtins, "__import__", real_import)
            # Re-import ddtree_mlx so other tests see it again.
            try:
                importlib.import_module("ddtree_mlx")
            except ImportError:
                pass

    def test_get_stats_includes_ddtree_fields(self):
        e = self._engine(ddtree_enabled=True, ddtree_budget=6)
        stats = e.get_stats()
        assert stats["engine_type"] == "dflash"
        assert stats["ddtree_enabled"] is True
        assert stats["ddtree_budget"] == 6
        assert stats["ddtree_exact_commit"] is False
        # ddtree_active is computed; depends on ddtree_mlx importability.
        assert "ddtree_active" in stats


class TestDDTreeAdminRequest:
    """ModelSettingsRequest accepts DDTree fields and enforces DFlash dependency."""

    def test_accepts_ddtree_fields(self):
        from omlx.admin.routes import ModelSettingsRequest

        req = ModelSettingsRequest.model_validate(
            {"ddtree_enabled": True, "ddtree_budget": 6, "ddtree_exact_commit": True}
        )
        assert req.ddtree_enabled is True
        assert req.ddtree_budget == 6
        assert req.ddtree_exact_commit is True

    def test_none_defaults(self):
        from omlx.admin.routes import ModelSettingsRequest

        req = ModelSettingsRequest.model_validate({})
        assert req.ddtree_enabled is None
        assert req.ddtree_budget is None
        assert req.ddtree_exact_commit is None
