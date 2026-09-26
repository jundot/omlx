# SPDX-License-Identifier: Apache-2.0
"""Canonical state recovery settings plumbing.

A model chooses two things: whether to recover, and how large an execution
slice it does it in. It does not choose a ceiling. The share of the machine
recovery may have is one server-level number, because every engine in the pool
shares one accelerator and the pool rewrites this same config object before
every load — a ceiling read off it would be whichever model loaded last.
"""

import pytest

from omlx.model_settings import ModelSettings
from omlx.scheduler import SchedulerConfig
from omlx.settings import SchedulerSettings
from omlx.canonical_recovery import apply_canonical_recovery_settings


class TestModelSettingsRoundTrip:
    def test_defaults(self):
        settings = ModelSettings()
        assert settings.canonical_state_recovery_enabled is False
        assert settings.canonical_state_recovery_slice_tokens == 0

    def test_round_trip_through_to_dict_and_from_dict(self):
        settings = ModelSettings(
            canonical_state_recovery_enabled=True,
            canonical_state_recovery_slice_tokens=512,
        )
        data = settings.to_dict()
        assert data["canonical_state_recovery_enabled"] is True
        assert data["canonical_state_recovery_slice_tokens"] == 512

        restored = ModelSettings.from_dict(data)
        assert restored.canonical_state_recovery_enabled is True
        assert restored.canonical_state_recovery_slice_tokens == 512

    def test_a_negative_slice_is_refused(self):
        with pytest.raises(ValueError, match="canonical_state_recovery_slice_tokens"):
            ModelSettings(canonical_state_recovery_slice_tokens=-1)


class TestThereIsNoPerModelCeiling:
    """The knob that used to be here granted nothing and said otherwise.

    It was read only when a Scheduler built its own budget, which under a pool
    never happens — the pool always supplies one. An operator who set it got no
    ceiling and no error, so it is gone rather than documented.
    """

    def test_model_settings_has_no_budget_percentage(self):
        assert not hasattr(ModelSettings(), "canonical_state_recovery_budget_pct")

    def test_the_ceiling_is_a_server_level_setting(self):
        assert SchedulerSettings().canonical_state_recovery_global_budget_pct == 0.0
        assert SchedulerSettings.from_dict(
            {"canonical_state_recovery_global_budget_pct": 12.5}
        ).canonical_state_recovery_global_budget_pct == 12.5

    def test_the_server_level_ceiling_is_clamped_to_a_percentage(self):
        for sent, expected in ((-5.0, 0.0), (250.0, 100.0)):
            assert SchedulerSettings.from_dict(
                {"canonical_state_recovery_global_budget_pct": sent}
            ).canonical_state_recovery_global_budget_pct == expected


class TestReachesSchedulerConfig:
    def test_enabled_settings_land_on_scheduler_config(self):
        config = SchedulerConfig()
        apply_canonical_recovery_settings(
            config,
            ModelSettings(canonical_state_recovery_enabled=True, canonical_state_recovery_slice_tokens=512),
        )
        assert config.canonical_state_recovery_enabled is True
        assert config.canonical_state_recovery_slice_tokens == 512

    def test_default_settings_leave_scheduler_config_disabled(self):
        config = SchedulerConfig()
        apply_canonical_recovery_settings(config, ModelSettings())
        assert config.canonical_state_recovery_enabled is False
        assert config.canonical_state_recovery_slice_tokens == 0

    def test_missing_settings_object_defaults_safely(self):
        # A model with no ModelSettings entry (getattr falls back).
        config = SchedulerConfig()
        apply_canonical_recovery_settings(config, object())
        assert config.canonical_state_recovery_enabled is False
        assert config.canonical_state_recovery_slice_tokens == 0

    def test_a_bare_scheduler_reads_the_same_server_level_ceiling(self):
        """No pool, so the Scheduler builds its own budget — from the one
        ceiling there is, not from a second per-model knob."""
        from unittest.mock import MagicMock

        from omlx.scheduler import Scheduler

        model = MagicMock()
        model.layers = []
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2
        scheduler = Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(
                canonical_state_recovery_enabled=True,
                canonical_state_recovery_global_budget_pct=7.5,
            ),
        )
        assert scheduler._canonical_recovery_budget.pct == 7.5
        assert scheduler._canonical_recovery_budget.shared is False


class TestEnabledAndUnschedulableIsDiagnosable:
    """Enabled with a zero ceiling is reachable, and used to be silent.

    The per-model switch is the documented one, the ceiling is server-level and
    defaults to zero, and at zero `CanonicalRecoveryBudget.allows` is False in every
    window, so every candidate is declined before it runs. Nothing downstream
    reported it: the counters live on the usage object of requests the job
    never touched, and the decline itself is not logged. An operator who set
    the switch saw a feature that was on and did nothing.
    """

    def test_a_zero_server_ceiling_with_the_switch_on_is_logged(self, caplog):
        config = SchedulerConfig()
        config.canonical_state_recovery_global_budget_pct = 0.0
        with caplog.at_level("WARNING", logger="omlx.canonical_recovery"):
            apply_canonical_recovery_settings(
                config, ModelSettings(canonical_state_recovery_enabled=True)
            )
        assert "canonical_state_recovery_global_budget_pct" in caplog.text
        assert "will never be scheduled" in caplog.text

    def test_a_non_zero_ceiling_says_nothing(self, caplog):
        config = SchedulerConfig()
        config.canonical_state_recovery_global_budget_pct = 10.0
        with caplog.at_level("WARNING", logger="omlx.canonical_recovery"):
            apply_canonical_recovery_settings(
                config, ModelSettings(canonical_state_recovery_enabled=True)
            )
        assert caplog.text == ""

    def test_the_switch_off_says_nothing_whatever_the_ceiling(self, caplog):
        config = SchedulerConfig()
        config.canonical_state_recovery_global_budget_pct = 0.0
        with caplog.at_level("WARNING", logger="omlx.canonical_recovery"):
            apply_canonical_recovery_settings(
                config, ModelSettings(canonical_state_recovery_enabled=False)
            )
        assert caplog.text == ""
