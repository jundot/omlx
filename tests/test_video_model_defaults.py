# SPDX-License-Identifier: Apache-2.0
"""Per-model video generation defaults: persistence + admin UI surface.

Route resolution behavior lives in test_video_routes.py::TestModelDefaults;
this file covers the ModelSettings dataclass roundtrip and the settings
modal template (the video section the per-model modal shows for
model_type == "video", which used to be empty for video models).
"""
import json
from pathlib import Path

from omlx.model_settings import ModelSettings

REPO = Path(__file__).parent.parent
MODAL = REPO / "omlx" / "admin" / "templates" / "dashboard" / "_modal_model_settings.html"
I18N_DIR = REPO / "omlx" / "admin" / "i18n"

VIDEO_FIELDS = {
    "video_default_steps": 12,
    "video_default_fps": 24,
    "video_default_size": "384x224",
    "video_default_seconds": 2.0,
    "video_default_upscale_resolution": 1080,
}


class TestModelSettingsVideoFields:
    def test_defaults_are_none(self):
        ms = ModelSettings()
        for field in VIDEO_FIELDS:
            assert getattr(ms, field) is None

    def test_roundtrip(self):
        ms = ModelSettings(**VIDEO_FIELDS)
        restored = ModelSettings.from_dict(ms.to_dict())
        for field, value in VIDEO_FIELDS.items():
            assert getattr(restored, field) == value

    def test_from_dict_ignores_unknown_and_missing(self):
        restored = ModelSettings.from_dict({"video_default_steps": 8})
        assert restored.video_default_steps == 8
        assert restored.video_default_upscale_resolution is None


class TestSettingsModalVideoSection:
    def test_video_section_gated_on_video_type(self):
        source = MODAL.read_text(encoding="utf-8")
        assert "selectedModel?.model_type === 'video'" in source, (
            "the settings modal must show a video section for video models "
            "(it used to render empty for them)"
        )
        for field in VIDEO_FIELDS:
            assert f"modelSettings.{field}" in source, f"modal must bind {field}"

    def test_modal_i18n_keys_flat(self):
        keys = [
            "modal.model_settings.video_section",
            "modal.model_settings.video_hint",
            "modal.model_settings.video_steps",
            "modal.model_settings.video_fps",
            "modal.model_settings.video_seconds",
            "modal.model_settings.video_size",
            "modal.model_settings.video_upscale",
            "modal.model_settings.video_upscale_off",
        ]
        for lang in ("en", "zh"):
            data = json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))
            for key in keys:
                assert key in data and data[key].strip(), f"{lang}.json missing {key}"
