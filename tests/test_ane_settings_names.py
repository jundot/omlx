# SPDX-License-Identifier: Apache-2.0
"""One model-neutral schema, with legacy names translated only on input."""

import json

import pytest

from omlx.model_profiles import (
    ANE_PREFILL_ALIASES,
    filter_profile_fields,
    normalize_ane_prefill_settings,
)
from omlx.model_settings import ModelSettings, ModelSettingsManager


@pytest.mark.parametrize("alias,name", ANE_PREFILL_ALIASES.items())
def test_every_legacy_name_reads_into_one_canonical_field(alias, name):
    value = getattr(ModelSettings(), name)
    settings = ModelSettings.from_dict({alias: value})
    assert getattr(settings, name) == value
    assert alias not in settings.to_dict()
    assert alias not in settings.__dict__


def test_explicit_neutral_false_and_zero_win_without_changing_input():
    original = {
        "qwen35_ane_prefill_enabled": True,
        "clef_flash_ane_enabled": True,
        "ane_prefill_enabled": False,
        "qwen35_ane_prefill_cpu_fraction": 0.135,
        "clef_flash_cpu_fraction": 0.125,
        "ane_prefill_cpu_fraction": 0,
        "temperature": 0.4,
    }
    expected = {
        "ane_prefill_enabled": False,
        "ane_prefill_cpu_fraction": 0,
        "temperature": 0.4,
    }
    assert normalize_ane_prefill_settings(original) == expected
    assert original["qwen35_ane_prefill_enabled"] is True


def test_interim_clef_values_override_qwen_values_and_unset_clef_values_fall_back():
    assert normalize_ane_prefill_settings(
        {
            "qwen35_ane_prefill_enabled": True,
            "clef_flash_ane_enabled": False,
            "qwen35_ane_prefill_fraction": 0.25,
            "clef_flash_ane_fraction": None,
        }
    ) == {"ane_prefill_enabled": False, "ane_prefill_fraction": 0.25}


def test_legacy_profile_overrides_a_neutral_base():
    base = ModelSettings(
        ane_prefill_enabled=True, ane_prefill_cpu_fraction=0.135
    ).to_dict()
    profile = filter_profile_fields(
        {
            "qwen35_ane_prefill_enabled": False,
            "clef_flash_cpu_fraction": 0,
        }
    )
    merged = ModelSettings.from_dict({**base, **profile})
    assert merged.ane_prefill_enabled is False
    assert merged.ane_prefill_cpu_fraction == 0


def test_legacy_ane_and_a8_settings_migrate_together():
    legacy = {
        "qwen35_ane_prefill_enabled": False,
        "qwen35_ane_prefill_shared_fraction": 0,
        "qwen35_oq_a8_enabled": True,
        "qwen35_oq_a8_min_tokens": 256,
    }
    expected = {
        "ane_prefill_enabled": False,
        "ane_prefill_shared_fraction": 0,
        "a8_prefill_enabled": True,
        "a8_prefill_min_tokens": 256,
    }
    assert filter_profile_fields(legacy) == expected
    settings = ModelSettings.from_dict(legacy).to_dict()
    assert {name: settings[name] for name in expected} == expected


def test_saved_records_migrate_independently_by_model_id(tmp_path):
    target = tmp_path / "model_settings.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "clef-flash-oQ4": {
                        "clef_flash_ane_enabled": True,
                        "clef_flash_ane_fraction": 0.25,
                    },
                    "qwen": {
                        "qwen35_ane_prefill_enabled": False,
                        "qwen35_ane_prefill_fraction": 0.53,
                    },
                },
            }
        )
    )
    manager = ModelSettingsManager(tmp_path)
    clef = manager.get_settings("clef-flash-oQ4")
    qwen = manager.get_settings("qwen")
    assert clef.ane_prefill_enabled is True
    assert qwen.ane_prefill_enabled is False
    clef.ane_prefill_fraction = 0.3
    manager.set_settings("clef-flash-oQ4", clef)
    saved = json.loads(target.read_text())["models"]
    assert saved["clef-flash-oQ4"]["ane_prefill_fraction"] == 0.3
    assert saved["qwen"]["ane_prefill_fraction"] == 0.53
    assert not any(
        key in ANE_PREFILL_ALIASES for record in saved.values() for key in record
    )


def test_loaded_legacy_profile_keeps_engine_overrides_and_returns_neutral_keys(
    tmp_path,
):
    (tmp_path / "model_settings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "clef": {"ane_prefill_enabled": True},
                },
            }
        )
    )
    (tmp_path / "model_profiles.json").write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "clef": {
                        "gpu": {
                            "name": "gpu",
                            "api_name": "gpu",
                            "expose_as_model": True,
                            "settings": {"qwen35_ane_prefill_enabled": False},
                        }
                    },
                },
            }
        )
    )
    manager = ModelSettingsManager(tmp_path)
    profile = manager.list_profiles("clef")[0]
    assert profile["has_engine_fields"] is True
    assert profile["settings"] == {"ane_prefill_enabled": False}
    _, runtime = manager.get_exposed_profile_runtime_settings_for_request("clef:gpu")
    assert runtime.ane_prefill_enabled is False
    saved = json.loads((tmp_path / "model_profiles.json").read_text())
    assert saved["profiles"]["clef"]["gpu"]["settings"] == {
        "ane_prefill_enabled": False
    }
