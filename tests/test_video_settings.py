# SPDX-License-Identifier: Apache-2.0
"""Tests for VideoSettings serialization in omlx.settings.

VideoSettings had no dict-roundtrip coverage before the SeedVR2 upscaler
fields (upscaler_model_path, max_upscale_resolution) were added, so this
file guards the full to_dict/from_dict contract the admin settings UI and
~/.fmlx/settings.json persistence depend on.
"""
from pathlib import Path

from omlx.settings import VideoSettings


class TestVideoSettings:
    """Tests for VideoSettings dataclass."""

    def test_defaults(self):
        """Test default values, including the new upscaler fields."""
        settings = VideoSettings()
        assert settings.enabled is False
        assert settings.worker_python == ""
        assert settings.upscaler_model_path == ""
        assert settings.max_upscale_resolution == 2160

    def test_to_dict_includes_upscaler_fields(self):
        """to_dict must emit the upscaler fields or the admin settings
        page silently drops them on save."""
        settings = VideoSettings(
            upscaler_model_path="/models/seedvr2", max_upscale_resolution=1080
        )
        result = settings.to_dict()
        assert result["upscaler_model_path"] == "/models/seedvr2"
        assert result["max_upscale_resolution"] == 1080

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "enabled": True,
            "upscaler_model_path": "/models/seedvr2",
            "max_upscale_resolution": 1440,
        }
        settings = VideoSettings.from_dict(data)
        assert settings.enabled is True
        assert settings.upscaler_model_path == "/models/seedvr2"
        assert settings.max_upscale_resolution == 1440

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        settings = VideoSettings.from_dict({"enabled": True})
        assert settings.upscaler_model_path == ""  # default
        assert settings.max_upscale_resolution == 2160  # default

    def test_to_dict_from_dict_roundtrip(self):
        """Every field must survive a to_dict -> from_dict roundtrip."""
        settings = VideoSettings(
            enabled=True,
            worker_python="/opt/venv/bin/python",
            memory_lease_gb=24.0,
            max_queued_jobs=2,
            job_timeout_seconds=3600,
            progress_stall_timeout_seconds=300,
            default_steps=12,
            default_fps=24,
            max_frames=97,
            max_steps=40,
            max_pixels_per_frame=640 * 480,
            artifacts_max_count=10,
            artifacts_max_gb=20.0,
            upscaler_model_path="/models/seedvr2",
            max_upscale_resolution=1080,
        )
        restored = VideoSettings.from_dict(settings.to_dict())
        assert restored == settings
        assert restored.to_dict() == settings.to_dict()

    def test_get_upscaler_dir_explicit_path(self):
        """An explicit upscaler_model_path wins over the convention dir."""
        settings = VideoSettings(upscaler_model_path="/models/seedvr2")
        assert settings.get_upscaler_dir(Path("/base")) == Path("/models/seedvr2")

    def test_get_upscaler_dir_convention_default(self):
        """Empty path resolves to the SeedVR2 convention dir under base_path."""
        settings = VideoSettings()
        assert settings.get_upscaler_dir(Path("/base")) == Path(
            "/base/models/AbstractFramework/seedvr2-3b-8bit"
        )
