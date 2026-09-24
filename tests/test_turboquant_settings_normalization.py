# SPDX-License-Identifier: Apache-2.0
"""TurboQuant settings must not depend on JSON numeric spelling."""

import json

import pytest

from omlx.model_settings import ModelSettings, ModelSettingsManager


@pytest.mark.parametrize("value", [8, 8.0, "8", "8.0", 2.5, "2.5", 3.5, "3.5"])
def test_settings_normalize_turboquant_bits_on_load(tmp_path, value):
    settings_file = tmp_path / "model_settings.json"
    original = json.dumps(
        {"version": 1, "models": {"model-a": {"turboquant_kv_bits": value}}}
    )
    settings_file.write_text(original)

    manager = ModelSettingsManager(tmp_path)
    settings = manager.get_settings("model-a")

    assert type(settings.turboquant_kv_bits) is float
    assert settings.turboquant_kv_bits == float(value)
    assert type(ModelSettings(turboquant_kv_bits=value).turboquant_kv_bits) is float
    assert settings_file.read_text() == original  # Loading is not a migration.

    manager.set_settings("model-a", settings)
    stored = json.loads(settings_file.read_text())["models"]["model-a"]
    assert type(stored["turboquant_kv_bits"]) is float
    assert stored["turboquant_kv_bits"] == float(value)


@pytest.mark.parametrize("value", [None, "", "invalid"])
def test_normalization_does_not_drop_other_model_settings(tmp_path, value):
    (tmp_path / "model_settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "model-a": {
                        "turboquant_kv_bits": value,
                        "model_alias": "test-alias",
                        "temperature": 0.7,
                    }
                },
            }
        )
    )
    settings = ModelSettingsManager(tmp_path).get_settings("model-a")
    assert settings.model_alias == "test-alias"
    assert settings.temperature == 0.7


@pytest.mark.parametrize("value", [8, 8.0, "8", "8.0", 2.5, "2.5", 3.5, "3.5"])
def test_profile_saves_canonical_turboquant_bits(tmp_path, value):
    manager = ModelSettingsManager(tmp_path)
    manager.save_profile("model-a", "test", "Test", None, {"turboquant_kv_bits": value})
    stored = json.loads(manager.profiles_file.read_text())
    bits = stored["profiles"]["model-a"]["test"]["settings"]["turboquant_kv_bits"]
    assert type(bits) is float
    assert bits == float(value)

    manager.update_profile("model-a", "test", settings={"turboquant_kv_bits": value})
    stored = json.loads(manager.profiles_file.read_text())
    bits = stored["profiles"]["model-a"]["test"]["settings"]["turboquant_kv_bits"]
    assert type(bits) is float
    assert bits == float(value)


@pytest.mark.parametrize("value", [8, "8", "8.0", "2.5"])
def test_legacy_profile_is_normalized_without_rewriting_on_load(tmp_path, value):
    profiles_file = tmp_path / "model_profiles.json"
    original = json.dumps(
        {
            "version": 1,
            "profiles": {
                "model-a": {
                    "test": {
                        "name": "test",
                        "display_name": "Test",
                        "api_name": "test",
                        "settings": {"turboquant_kv_bits": value, "temperature": None},
                    }
                }
            },
        }
    )
    profiles_file.write_text(original)
    manager = ModelSettingsManager(tmp_path)
    profile = manager.get_profile("model-a", "test")
    assert profile is not None
    assert type(profile["settings"]["turboquant_kv_bits"]) is float
    assert profiles_file.read_text() == original
    # A metadata-only save must also canonicalize old settings on disk.
    manager.update_profile("model-a", "test", display_name="Renamed")
    stored = json.loads(profiles_file.read_text())
    settings = stored["profiles"]["model-a"]["test"]["settings"]
    assert type(settings["turboquant_kv_bits"]) is float
    assert settings["turboquant_kv_bits"] == float(value)
    assert settings["temperature"] is None


@pytest.mark.parametrize("value", [None, ""])
def test_profile_unset_bits_remain_absent(tmp_path, value):
    manager = ModelSettingsManager(tmp_path)
    profile = manager.save_profile(
        "model-a", "test", "Test", None, {"turboquant_kv_bits": value}
    )
    assert "turboquant_kv_bits" not in profile["settings"]


def test_settings_serialization_normalizes_assignment():
    settings = ModelSettings()
    settings.turboquant_kv_bits = "8"
    assert type(settings.to_dict()["turboquant_kv_bits"]) is float


def test_default_turboquant_bits_are_float():
    assert type(ModelSettings().turboquant_kv_bits) is float
    assert ModelSettings().turboquant_kv_bits == 4.0
