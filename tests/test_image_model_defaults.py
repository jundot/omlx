# SPDX-License-Identifier: Apache-2.0
"""Per-model image generation defaults: persistence + admin UI surface.

Route resolution behavior lives in test_image_routes.py; this file covers
the ModelSettings dataclass roundtrip and the settings modal template (the
image section the per-model modal shows for model_type == "image").
"""
import json
from pathlib import Path

from omlx.model_settings import ModelSettings

REPO = Path(__file__).parent.parent
MODAL = REPO / "omlx" / "admin" / "templates" / "dashboard" / "_modal_model_settings.html"
I18N_DIR = REPO / "omlx" / "admin" / "i18n"

IMAGE_FIELDS = {
    "image_default_steps": 9,
    "image_default_size": "768x1024",
}


class TestModelSettingsImageFields:
    def test_defaults_are_none(self):
        ms = ModelSettings()
        for field in IMAGE_FIELDS:
            assert getattr(ms, field) is None

    def test_roundtrip(self):
        ms = ModelSettings(**IMAGE_FIELDS)
        restored = ModelSettings.from_dict(ms.to_dict())
        for field, value in IMAGE_FIELDS.items():
            assert getattr(restored, field) == value

    def test_from_dict_ignores_unknown_and_missing(self):
        restored = ModelSettings.from_dict({"image_default_steps": 12})
        assert restored.image_default_steps == 12
        assert restored.image_default_size is None


class TestSettingsModalImageSection:
    def test_image_section_gated_on_image_type(self):
        source = MODAL.read_text(encoding="utf-8")
        assert "selectedModel?.model_type === 'image'" in source, (
            "the settings modal must show an image section for image models "
            "(its own template block, not a widening of the video one)"
        )
        for field in IMAGE_FIELDS:
            assert f"modelSettings.{field}" in source, f"modal must bind {field}"

    def test_modal_i18n_keys_flat(self):
        keys = [
            "modal.model_settings.image_section",
            "modal.model_settings.image_hint",
            "modal.model_settings.image_steps",
            "modal.model_settings.image_size",
            "modal.model_settings.image_pipeline",
            "modal.model_settings.image_alias",
            "modal.model_settings.image_pipeline_hint",
        ]
        for lang in ("en", "zh"):
            data = json.loads((I18N_DIR / f"{lang}.json").read_text(encoding="utf-8"))
            for key in keys:
                assert key in data and data[key].strip(), f"{lang}.json missing {key}"
