# SPDX-License-Identifier: Apache-2.0
"""Tests for ImageSettings serialization in omlx.settings.

Guards the to_dict/from_dict contract the admin settings UI and
~/.fmlx/settings.json persistence depend on, plus the GlobalSettings
integration points (the "image" section in to_dict and _load_from_file).
Mirrors tests/test_video_settings.py.
"""
import json
from pathlib import Path

from omlx.settings import GlobalSettings, ImageSettings


class TestImageSettings:
    """Tests for the ImageSettings dataclass."""

    def test_defaults(self):
        settings = ImageSettings()
        assert settings.enabled is False
        assert settings.worker_python == ""
        assert settings.memory_lease_gb == 0.0  # 0 = auto by model kind
        assert settings.max_queued_jobs == 8
        assert settings.job_timeout_seconds == 1800
        assert settings.progress_stall_timeout_seconds == 300
        assert settings.default_steps == 0  # 0 = per-alias default
        assert settings.default_size == "1024x1024"
        assert settings.max_steps == 60
        assert settings.max_pixels == 2048 * 2048
        assert settings.max_n == 4
        assert settings.sync_timeout_seconds == 900
        assert settings.artifacts_max_count == 200
        assert settings.artifacts_max_gb == 10.0

    def test_to_dict_emits_every_field(self):
        result = ImageSettings().to_dict()
        expected_keys = {
            "enabled", "worker_python", "memory_lease_gb", "max_queued_jobs",
            "job_timeout_seconds", "progress_stall_timeout_seconds",
            "default_steps", "default_size", "max_steps", "max_pixels",
            "max_n", "sync_timeout_seconds", "artifacts_max_count",
            "artifacts_max_gb",
        }
        assert set(result.keys()) == expected_keys

    def test_from_dict(self):
        data = {
            "enabled": True,
            "worker_python": "/opt/venv/bin/python",
            "memory_lease_gb": 12.0,
            "max_n": 2,
        }
        settings = ImageSettings.from_dict(data)
        assert settings.enabled is True
        assert settings.worker_python == "/opt/venv/bin/python"
        assert settings.memory_lease_gb == 12.0
        assert settings.max_n == 2

    def test_from_dict_partial_uses_defaults(self):
        settings = ImageSettings.from_dict({"enabled": True})
        assert settings.enabled is True
        assert settings.worker_python == ""  # default
        assert settings.default_size == "1024x1024"  # default
        assert settings.sync_timeout_seconds == 900  # default

    def test_from_dict_ignores_unknown_keys(self):
        """Forward compat: a settings.json written by a newer fmlx must not
        crash an older from_dict."""
        settings = ImageSettings.from_dict(
            {"enabled": True, "some_future_field": 42}
        )
        assert settings.enabled is True
        assert not hasattr(settings, "some_future_field")

    def test_to_dict_from_dict_roundtrip(self):
        """Every field must survive a to_dict -> from_dict roundtrip."""
        settings = ImageSettings(
            enabled=True,
            worker_python="/opt/venv/bin/python",
            memory_lease_gb=24.0,
            max_queued_jobs=2,
            job_timeout_seconds=3600,
            progress_stall_timeout_seconds=120,
            default_steps=12,
            default_size="768x768",
            max_steps=40,
            max_pixels=1024 * 1024,
            max_n=2,
            sync_timeout_seconds=300,
            artifacts_max_count=10,
            artifacts_max_gb=20.0,
        )
        restored = ImageSettings.from_dict(settings.to_dict())
        assert restored == settings
        assert restored.to_dict() == settings.to_dict()

    def test_get_worker_python_explicit_path(self):
        settings = ImageSettings(worker_python="/opt/venv/bin/python")
        assert settings.get_worker_python(Path("/base")) == Path(
            "/opt/venv/bin/python"
        )

    def test_get_worker_python_defaults_to_video_venv(self):
        """Empty worker_python resolves to the SHARED video venv python."""
        settings = ImageSettings()
        assert settings.get_worker_python(Path("/base")) == Path(
            "/base/venvs/video/bin/python"
        )


class TestGlobalSettingsImageSection:
    """GlobalSettings integration: the image section round-trips through
    to_dict and _load_from_file."""

    def test_global_defaults_include_image(self, tmp_path):
        settings = GlobalSettings(base_path=tmp_path)
        assert settings.image == ImageSettings()
        assert settings.image.enabled is False

    def test_to_dict_includes_image_section(self, tmp_path):
        settings = GlobalSettings(base_path=tmp_path)
        settings.image.enabled = True
        settings.image.max_n = 2
        result = settings.to_dict()
        assert "image" in result
        assert result["image"]["enabled"] is True
        assert result["image"]["max_n"] == 2
        assert result["image"] == settings.image.to_dict()

    def test_load_from_file_parses_image_section(self, tmp_path):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "image": {
                        "enabled": True,
                        "memory_lease_gb": 12.0,
                        "default_size": "768x768",
                        "max_n": 2,
                    },
                }
            )
        )

        settings = GlobalSettings.load(base_path=tmp_path)
        assert settings.image.enabled is True
        assert settings.image.memory_lease_gb == 12.0
        assert settings.image.default_size == "768x768"
        assert settings.image.max_n == 2
        # Unspecified fields keep defaults
        assert settings.image.sync_timeout_seconds == 900

    def test_load_from_file_without_image_section_keeps_defaults(
        self, tmp_path
    ):
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"version": "1.0"}))

        settings = GlobalSettings.load(base_path=tmp_path)
        assert settings.image == ImageSettings()
