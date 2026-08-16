"""Regression tests for the bundled model-settings presets."""

import json
from pathlib import Path


def _bundled_presets() -> dict[str, dict]:
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "omlx/admin/static/omlx_preset.json").read_text())
    presets = bundle["presets"]
    assert len({preset["name"] for preset in presets}) == len(presets)
    return {preset["name"]: preset for preset in presets}


def test_minimax_m3_replaces_m27_without_top_k():
    presets = _bundled_presets()

    assert "minimax-m27" not in presets
    assert presets["minimax-m3"]["settings"] == {
        "temperature": 1.0,
        "top_p": 0.95,
    }


def test_deepseek_v4_defaults():
    presets = _bundled_presets()

    assert presets["deepseek-v4"]["settings"] == {
        "temperature": 1.0,
        "top_p": 1.0,
    }


def test_glm_5x_replaces_version_specific_preset():
    presets = _bundled_presets()

    assert "glm-5-2" not in presets
    assert presets["glm-5-x"] == {
        "name": "glm-5-x",
        "display_name": "GLM-5.x",
        "description": "GLM-5.x (Zhipu default)",
        "settings": {
            "temperature": 1.0,
            "top_p": 0.95,
        },
    }
