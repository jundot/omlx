# SPDX-License-Identifier: Apache-2.0
"""mtp_enabled is tri-state: None = auto (on for embedded-DSpark checkpoints).

V4-Flash-0731 ships DSpark drafting stages in its config, yet a plain
``mtp_enabled: bool = False`` default served it without them. None now
means "use the checkpoint's intent"; an explicit False still wins, and
auto yields to any explicitly chosen speculative path.
"""

import pytest

from omlx.model_settings import ModelSettings

DSPARK_CONFIG = {
    "model_type": "deepseek_v4",
    "num_nextn_predict_layers": 1,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": [40, 41, 42],
}
PLAIN_MTP_CONFIG = {
    "model_type": "deepseek_v4",
    "num_nextn_predict_layers": 1,
}


def test_default_is_auto_and_serializes_as_absent():
    settings = ModelSettings()
    assert settings.mtp_enabled is None
    assert "mtp_enabled" not in settings.to_dict()
    assert ModelSettings.from_dict({}).mtp_enabled is None


def test_explicit_values_round_trip():
    assert ModelSettings.from_dict({"mtp_enabled": False}).mtp_enabled is False
    assert ModelSettings.from_dict({"mtp_enabled": True}).mtp_enabled is True
    assert ModelSettings(mtp_enabled=False).to_dict()["mtp_enabled"] is False


def test_auto_does_not_trip_the_dflash_exclusivity_check():
    # None must behave as "not explicitly on" for the mutual-exclusion
    # validator — dflash alone is a valid configuration.
    settings = ModelSettings(dflash_enabled=True)
    assert settings.mtp_enabled is None
    with pytest.raises(ValueError):
        ModelSettings(mtp_enabled=True, dflash_enabled=True)


def test_embedded_dspark_detection():
    pytest.importorskip("mlx.core")
    from omlx.utils.model_loading import _has_dspark_heads

    assert _has_dspark_heads(DSPARK_CONFIG) is True
    assert _has_dspark_heads(PLAIN_MTP_CONFIG) is False
    assert _has_dspark_heads(dict(DSPARK_CONFIG, dspark_block_size=0)) is False
    assert (
        _has_dspark_heads(dict(DSPARK_CONFIG, dspark_target_layer_ids=[]))
        is False
    )
    # VLM exports nest the declarations under text_config.
    nested = {
        "model_type": "deepseek_v4",
        "text_config": {
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [40, 41, 42],
        },
    }
    assert _has_dspark_heads(nested) is True


def test_auto_resolution_end_to_end_through_pre_load_patches(tmp_path):
    """The tri-state must resolve through the real pre-load dispatch:
    None + embedded-DSpark config -> MTP active; None + plain -> inactive;
    explicit False always wins."""
    pytest.importorskip("mlx.core")
    import json

    from omlx.patches.mlx_lm_mtp import is_mtp_active, set_mtp_active
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    def resolve(config, settings):
        model_dir = tmp_path / f"m{config.get('dspark_block_size', 0)}"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "config.json").write_text(json.dumps(config))
        set_mtp_active(False)
        maybe_apply_pre_load_patches(str(model_dir), model_settings=settings)
        return is_mtp_active()

    try:
        assert resolve(DSPARK_CONFIG, None) is True
        assert resolve(DSPARK_CONFIG, ModelSettings()) is True
        assert resolve(PLAIN_MTP_CONFIG, ModelSettings()) is False
        assert resolve(DSPARK_CONFIG, ModelSettings(mtp_enabled=False)) is False
        # Auto yields to an explicitly chosen speculative path — the
        # exclusivity validator only sees explicit True, never a resolved
        # auto, so the resolution itself must not create the conflict.
        assert (
            resolve(DSPARK_CONFIG, ModelSettings(dflash_enabled=True)) is False
        )
    finally:
        set_mtp_active(False)


def test_variant_signature_distinguishes_auto_from_explicit_off():
    """Explicitly disabling MTP on an auto-enabled checkpoint must reload."""
    pytest.importorskip("mlx.core")
    from types import SimpleNamespace

    from omlx.engine_pool import EnginePool

    fake_pool = SimpleNamespace(
        _settings_manager=None,
        _entries={},
        _entry_is_diffusion_model=lambda entry: False,
        _canonical_signature_value=EnginePool._canonical_signature_value,
    )

    def signature(settings):
        return EnginePool._engine_runtime_signature(fake_pool, "m", settings)

    auto = signature(ModelSettings())
    off = signature(ModelSettings(mtp_enabled=False))
    on = signature(ModelSettings(mtp_enabled=True))
    assert auto != off
    assert auto != on
    assert off != on
