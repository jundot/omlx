"""The DFly2 flag: off by default, and never alongside the embedded drafter."""

import pytest

from omlx.model_settings import ModelSettings, validate_dfly2_settings


def test_dfly2_is_off_by_default():
    assert ModelSettings().dfly2_checkpoint is None


def test_setting_a_checkpoint_alone_is_allowed():
    settings = ModelSettings(dfly2_checkpoint="/models/dfly2")
    assert settings.dfly2_checkpoint == "/models/dfly2"


def test_dfly2_may_run_with_the_mtp_machinery_armed():
    """The draft is called from the MTP cycle, so the cycle must stay enabled."""
    settings = ModelSettings(dfly2_checkpoint="/models/dfly2", mtp_enabled=True)
    assert settings.dfly2_checkpoint == "/models/dfly2"
    assert settings.mtp_enabled is True


def test_empty_checkpoint_path_is_rejected():
    with pytest.raises(ValueError, match="must be a path or None"):
        ModelSettings(dfly2_checkpoint="   ")


def test_validator_accepts_either_drafter_alone():
    validate_dfly2_settings({"dfly2_checkpoint": "/models/dfly2"})
    validate_dfly2_settings({"mtp_enabled": True})
