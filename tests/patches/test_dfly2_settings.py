"""The DFly2 flag: off by default, and never alongside the embedded drafter."""

import pytest

from omlx.model_settings import ModelSettings, validate_dfly2_settings


def test_dfly2_is_off_by_default():
    assert ModelSettings().dfly2_checkpoint is None


def test_setting_a_checkpoint_alone_is_allowed():
    settings = ModelSettings(dfly2_checkpoint="/models/dfly2")
    assert settings.dfly2_checkpoint == "/models/dfly2"


def test_dfly2_and_mtp_cannot_share_a_model():
    with pytest.raises(ValueError, match="dfly2_checkpoint replaces"):
        ModelSettings(dfly2_checkpoint="/models/dfly2", mtp_enabled=True)


def test_validator_accepts_either_drafter_alone():
    validate_dfly2_settings({"dfly2_checkpoint": "/models/dfly2"})
    validate_dfly2_settings({"mtp_enabled": True})
