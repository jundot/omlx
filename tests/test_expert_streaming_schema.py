# SPDX-License-Identifier: Apache-2.0
"""STREAMING_SETTING_SCHEMA: one bounds table for the expert_streaming_*
settings family — the schema covers every ModelSettings field,
``EXPERT_STREAMING_TUNABLE_KEYS`` derives from it, and
``validate_streaming_setting`` applies the same bounds the runtime
coercion and the admin write path enforce.
"""

import dataclasses

from omlx.model_profiles import (
    EXPERT_STREAMING_TUNABLE_KEYS,
    STREAMING_SETTING_SCHEMA,
)
from omlx.model_settings import ModelSettings
from omlx.patches.expert_streaming import (
    _io_overrides,
    validate_streaming_setting,
)


def test_schema_covers_every_expert_streaming_field():
    fields = {
        f.name
        for f in dataclasses.fields(ModelSettings)
        if f.name.startswith("expert_streaming_")
    }
    assert fields, "expected expert_streaming_* fields on ModelSettings"
    assert set(STREAMING_SETTING_SCHEMA) == fields


def test_tunable_keys_derive_from_schema():
    assert (
        tuple(
            key
            for key, spec in STREAMING_SETTING_SCHEMA.items()
            if spec.get("tunable", True)
        )
        == EXPERT_STREAMING_TUNABLE_KEYS
    )
    # The routing switch itself is not a tunable.
    assert "expert_streaming_enabled" not in EXPERT_STREAMING_TUNABLE_KEYS
    assert STREAMING_SETTING_SCHEMA["expert_streaming_enabled"]["tunable"] is False


def test_schema_defaults_match_model_settings():
    defaults = {
        f.name: f.default
        for f in dataclasses.fields(ModelSettings)
        if f.name in STREAMING_SETTING_SCHEMA
    }
    for key, spec in STREAMING_SETTING_SCHEMA.items():
        assert spec["default"] == defaults[key], key


def test_unknown_key_and_none_validate_to_none():
    assert validate_streaming_setting("nope", 1) is None
    for key in STREAMING_SETTING_SCHEMA:
        # None is the unset marker: it always passes through as None.
        assert validate_streaming_setting(key, None) is None


def test_bool_fields_are_strict():
    for key, spec in STREAMING_SETTING_SCHEMA.items():
        if spec["type"] != "bool":
            continue
        assert validate_streaming_setting(key, True) is True
        assert validate_streaming_setting(key, False) is False
        # The admin contract: truthy non-bools are rejected, not coerced.
        assert validate_streaming_setting(key, "true") is None
        assert validate_streaming_setting(key, 1) is None


def test_numeric_bounds_match_schema():
    for key, spec in STREAMING_SETTING_SCHEMA.items():
        lo_hi = spec.get("bounds")
        if lo_hi is None:
            continue
        lo, hi, lo_open = lo_hi
        if spec["type"] == "int":
            # Below lo is invalid; above hi clamps (the _clamped_int
            # contract the io-overrides coercion uses).
            if lo > 0:
                assert validate_streaming_setting(key, lo - 1) is None
            if hi is not None:
                assert validate_streaming_setting(key, hi + 1) == hi
                assert validate_streaming_setting(key, hi) == hi
        else:
            # Floats reject outside the band, no clamping.
            if lo_open:
                assert validate_streaming_setting(key, lo) is None
            else:
                assert validate_streaming_setting(key, lo) == lo
            if lo > 0:
                assert validate_streaming_setting(key, lo / 2) is None
            if hi is not None:
                assert validate_streaming_setting(key, hi) == hi
                assert validate_streaming_setting(key, hi * 2) is None
        # Bools never coerce into numbers.
        assert validate_streaming_setting(key, True) is None


def test_choice_fields_normalize_and_reject():
    for key, spec in STREAMING_SETTING_SCHEMA.items():
        if spec["type"] != "choice":
            continue
        for choice in spec["choices"]:
            assert validate_streaming_setting(key, choice) == choice
            if choice:
                normalized = validate_streaming_setting(key, f" {choice.upper()} ")
                assert normalized == choice
        assert validate_streaming_setting(key, "bogus") is None


def test_cold_tier_accepts_2_to_8():
    # The runtime accepts any "2".."8" digit label
    # (conversion._resolve_cold_tier_root); "" stays legal = off.
    assert validate_streaming_setting("expert_streaming_cold_tier", "") == ""
    for bits in range(2, 9):
        assert validate_streaming_setting(
            "expert_streaming_cold_tier", str(bits)
        ) == str(bits)
    assert validate_streaming_setting("expert_streaming_cold_tier", "1") is None
    assert validate_streaming_setting("expert_streaming_cold_tier", "9") is None


def test_topk_threshold_floor_is_runtime_min():
    # adaptive_topk._MIN_THRESHOLD = 0.05: below it the runtime drops to
    # exact routing, so the schema rejects instead of persisting a knob
    # that never engages.
    from omlx.patches.expert_streaming.adaptive_topk import _MIN_THRESHOLD

    lo = STREAMING_SETTING_SCHEMA["expert_streaming_topk_threshold"]["bounds"][0]
    assert lo == _MIN_THRESHOLD
    assert validate_streaming_setting("expert_streaming_topk_threshold", 0.04) is None
    assert validate_streaming_setting("expert_streaming_topk_threshold", 0.05) == 0.05


def test_io_overrides_share_schema_bounds():
    # The io-override coercion delegates to the schema: same clamp/reject
    # outcomes as the previous hand-tabled validators.
    settings = ModelSettings(
        expert_streaming_io_depth=4096,  # clamps to 64
        expert_streaming_dynamic_max_gib=0,  # lo-open: rejected -> None
        expert_streaming_pin_regime="PREFILL",  # normalized lowercase
        expert_streaming_cache_policy="bogus",  # rejected -> None
    )
    ov = _io_overrides(settings)
    assert ov["expert_streaming_io_depth"] == 64
    assert ov["expert_streaming_dynamic_max_gib"] is None
    assert ov["expert_streaming_pin_regime"] == "prefill"
    assert ov["expert_streaming_cache_policy"] is None
    # Bool io keys keep their raw pass-through (None = unset).
    assert ov["expert_streaming_seed"] is None


def test_io_overrides_all_none_by_default():
    ov = _io_overrides(ModelSettings())
    assert all(v is None for v in ov.values())


def test_streaming_owns_model_and_config_type(tmp_path):
    import json

    from omlx.patches.expert_streaming import (
        load_config_model_type,
        streaming_owns_model,
    )

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
    assert load_config_model_type(tmp_path) == "glm_moe_dsa"
    assert streaming_owns_model(tmp_path) is True
    assert streaming_owns_model("glm4_moe") is True
    assert streaming_owns_model("llama") is False
    assert streaming_owns_model(None) is False
    assert streaming_owns_model(tmp_path / "missing") is False
    assert load_config_model_type(tmp_path / "missing") == ""
