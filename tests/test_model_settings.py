# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.model_settings module."""

import json
import tempfile
from pathlib import Path

import pytest

from omlx.model_settings import ModelSettings, ModelSettingsManager


class TestModelSettings:
    """Tests for ModelSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ModelSettings()
        assert settings.max_context_window is None
        assert settings.max_tokens is None
        assert settings.temperature is None
        assert settings.top_p is None
        assert settings.top_k is None
        assert settings.repetition_penalty is None
        assert settings.force_sampling is False
        assert settings.is_pinned is False
        assert settings.is_default is False

    def test_max_context_window(self):
        """Test max_context_window field."""
        settings = ModelSettings(max_context_window=4096)
        assert settings.max_context_window == 4096
        d = settings.to_dict()
        assert d["max_context_window"] == 4096

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        settings = ModelSettings(temperature=0.7, is_pinned=True)
        d = settings.to_dict()
        assert "temperature" in d
        assert "is_pinned" in d
        assert "max_tokens" not in d  # None should be excluded
        assert "max_context_window" not in d  # None should be excluded
        assert "repetition_penalty" not in d  # None should be excluded

    def test_to_dict_preserves_zero_values(self):
        """Test to_dict preserves zero values (not treated as None)."""
        settings = ModelSettings(temperature=0.0, top_p=0.0, top_k=0)
        d = settings.to_dict()
        assert "temperature" in d
        assert d["temperature"] == 0.0
        assert "top_p" in d
        assert d["top_p"] == 0.0
        assert "top_k" in d
        assert d["top_k"] == 0

    def test_zero_values_roundtrip(self):
        """Test zero values survive to_dict -> from_dict roundtrip."""
        original = ModelSettings(temperature=0.0, top_p=0.0, top_k=0)
        restored = ModelSettings.from_dict(original.to_dict())
        assert restored.temperature == 0.0
        assert restored.top_p == 0.0
        assert restored.top_k == 0

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "temperature": 0.8,
            "repetition_penalty": 1.3,
            "is_pinned": True,
            "invalid_key": "should be ignored"
        }
        settings = ModelSettings.from_dict(data)
        assert settings.temperature == 0.8
        assert settings.repetition_penalty == 1.3
        assert settings.is_pinned is True
        assert not hasattr(settings, "invalid_key")

    def test_repetition_penalty_roundtrip(self):
        """Test repetition_penalty survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(repetition_penalty=1.5)
        d = original.to_dict()
        assert d["repetition_penalty"] == 1.5
        restored = ModelSettings.from_dict(d)
        assert restored.repetition_penalty == 1.5

    def test_chat_template_kwargs_default(self):
        """Test chat_template_kwargs defaults to None."""
        settings = ModelSettings()
        assert settings.chat_template_kwargs is None

    def test_chat_template_kwargs_to_dict(self):
        """Test chat_template_kwargs included in to_dict when set."""
        settings = ModelSettings(
            chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}
        )
        d = settings.to_dict()
        assert "chat_template_kwargs" in d
        assert d["chat_template_kwargs"]["enable_thinking"] is False
        assert d["chat_template_kwargs"]["reasoning_effort"] == "low"

    def test_chat_template_kwargs_excluded_when_none(self):
        """Test chat_template_kwargs excluded from to_dict when None."""
        settings = ModelSettings()
        d = settings.to_dict()
        assert "chat_template_kwargs" not in d

    def test_chat_template_kwargs_roundtrip(self):
        """Test chat_template_kwargs survives to_dict -> from_dict roundtrip."""
        original = ModelSettings(
            chat_template_kwargs={"enable_thinking": True, "custom_key": 42}
        )
        d = original.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.chat_template_kwargs == {"enable_thinking": True, "custom_key": 42}

    def test_chat_template_kwargs_from_dict(self):
        """Test chat_template_kwargs created from dict."""
        data = {
            "temperature": 0.8,
            "chat_template_kwargs": {"reasoning_effort": "high"},
        }
        settings = ModelSettings.from_dict(data)
        assert settings.temperature == 0.8
        assert settings.chat_template_kwargs == {"reasoning_effort": "high"}


class TestModelSettingsManager:
    """Tests for ModelSettingsManager class."""

    def test_empty_settings(self):
        """Test with no settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = manager.get_settings("nonexistent")
            assert settings.is_pinned is False
            assert settings.is_default is False

    def test_load_existing_file(self):
        """Test loading from existing settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file
            settings_file = Path(tmpdir) / "model_settings.json"
            settings_file.write_text(json.dumps({
                "version": 1,
                "models": {
                    "llama-3b": {
                        "temperature": 0.7,
                        "is_pinned": True,
                        "is_default": True
                    }
                }
            }))

            manager = ModelSettingsManager(Path(tmpdir))
            settings = manager.get_settings("llama-3b")
            assert settings.temperature == 0.7
            assert settings.is_pinned is True
            assert settings.is_default is True

    def test_set_settings(self):
        """Test setting and saving settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(temperature=0.9, is_pinned=True)
            manager.set_settings("test-model", settings)

            # Verify saved
            loaded = manager.get_settings("test-model")
            assert loaded.temperature == 0.9
            assert loaded.is_pinned is True

            # Verify file was created
            settings_file = Path(tmpdir) / "model_settings.json"
            assert settings_file.exists()

    def test_zero_values_persist(self):
        """Test zero sampling values survive save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(temperature=0.0, top_p=0.0, top_k=0)
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.temperature == 0.0
            assert loaded.top_p == 0.0
            assert loaded.top_k == 0

    def test_repetition_penalty_persist(self):
        """Test repetition_penalty survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(repetition_penalty=1.3)
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.repetition_penalty == 1.3

    def test_exclusive_default(self):
        """Test only one model can be default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            # Set first model as default
            settings1 = ModelSettings(is_default=True)
            manager.set_settings("model-1", settings1)
            assert manager.get_default_model_id() == "model-1"

            # Set second model as default
            settings2 = ModelSettings(is_default=True)
            manager.set_settings("model-2", settings2)

            # model-2 should be default, model-1 should not
            assert manager.get_default_model_id() == "model-2"
            assert manager.get_settings("model-1").is_default is False
            assert manager.get_settings("model-2").is_default is True

    def test_multiple_pinned(self):
        """Test multiple models can be pinned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            manager.set_settings("model-1", ModelSettings(is_pinned=True))
            manager.set_settings("model-2", ModelSettings(is_pinned=True))
            manager.set_settings("model-3", ModelSettings(is_pinned=False))

            pinned = manager.get_pinned_model_ids()
            assert "model-1" in pinned
            assert "model-2" in pinned
            assert "model-3" not in pinned

    def test_get_all_settings(self):
        """Test getting all settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            manager.set_settings("model-1", ModelSettings(temperature=0.5))
            manager.set_settings("model-2", ModelSettings(temperature=0.9))

            all_settings = manager.get_all_settings()
            assert len(all_settings) == 2
            assert "model-1" in all_settings
            assert "model-2" in all_settings

    def test_chat_template_kwargs_persist(self):
        """Test chat_template_kwargs survives save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            settings = ModelSettings(
                chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "medium"}
            )
            manager.set_settings("test-model", settings)

            # Reload from file
            manager2 = ModelSettingsManager(Path(tmpdir))
            loaded = manager2.get_settings("test-model")
            assert loaded.chat_template_kwargs == {
                "enable_thinking": False,
                "reasoning_effort": "medium",
            }

    def test_chat_template_kwargs_clear(self):
        """Test clearing chat_template_kwargs by setting to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))

            # Set kwargs
            settings = ModelSettings(
                chat_template_kwargs={"enable_thinking": True}
            )
            manager.set_settings("test-model", settings)
            assert manager.get_settings("test-model").chat_template_kwargs is not None

            # Clear kwargs
            settings = ModelSettings(chat_template_kwargs=None)
            manager.set_settings("test-model", settings)
            loaded = manager.get_settings("test-model")
            assert loaded.chat_template_kwargs is None

    def test_thread_safety(self):
        """Test thread-safe access."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            errors = []

            def worker(model_id):
                try:
                    for i in range(10):
                        manager.set_settings(model_id, ModelSettings(temperature=i/10))
                        _ = manager.get_settings(model_id)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(f"model-{i}",)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
