# SPDX-License-Identifier: Apache-2.0
"""Tests for shadow prefill (EXP-003) settings plumbing and usage telemetry."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from omlx.engine.base import BaseEngine
from omlx.model_settings import ModelSettings
from omlx.shadow_prefill import apply_shadow_prefill_settings
from omlx.scheduler import SchedulerConfig


class TestModelSettingsRoundTrip:
    def test_defaults(self):
        settings = ModelSettings()
        assert settings.shadow_prefill_enabled is False
        assert settings.shadow_prefill_budget_pct == 0.0
        assert settings.shadow_prefill_publish_mode == "progressive"

    def test_round_trip_through_to_dict_and_from_dict(self):
        settings = ModelSettings(
            shadow_prefill_enabled=True,
            shadow_prefill_budget_pct=15.0,
            shadow_prefill_publish_mode="terminal",
        )
        data = settings.to_dict()
        assert data["shadow_prefill_enabled"] is True
        assert data["shadow_prefill_budget_pct"] == 15.0
        assert data["shadow_prefill_publish_mode"] == "terminal"

        restored = ModelSettings.from_dict(data)
        assert restored.shadow_prefill_enabled is True
        assert restored.shadow_prefill_budget_pct == 15.0
        assert restored.shadow_prefill_publish_mode == "terminal"


class TestModelSettingsValidation:
    def test_invalid_publish_mode_rejected(self):
        with pytest.raises(ValueError, match="shadow_prefill_publish_mode"):
            ModelSettings(shadow_prefill_publish_mode="sideways")

    def test_budget_pct_below_zero_rejected(self):
        with pytest.raises(ValueError, match="shadow_prefill_budget_pct"):
            ModelSettings(shadow_prefill_budget_pct=-1.0)

    def test_budget_pct_above_hundred_rejected(self):
        with pytest.raises(ValueError, match="shadow_prefill_budget_pct"):
            ModelSettings(shadow_prefill_budget_pct=101.0)

    def test_budget_pct_boundaries_accepted(self):
        ModelSettings(shadow_prefill_budget_pct=0.0)
        ModelSettings(shadow_prefill_budget_pct=100.0)

    def test_terminal_mode_accepted(self):
        settings = ModelSettings(shadow_prefill_publish_mode="terminal")
        assert settings.shadow_prefill_publish_mode == "terminal"


class TestReachesSchedulerConfig:
    def test_enabled_settings_land_on_scheduler_config(self):
        scheduler_config = SchedulerConfig()
        settings = ModelSettings(
            shadow_prefill_enabled=True,
            shadow_prefill_budget_pct=7.5,
            shadow_prefill_publish_mode="terminal",
        )
        apply_shadow_prefill_settings(scheduler_config, settings)
        assert scheduler_config.shadow_prefill_enabled is True
        assert scheduler_config.shadow_prefill_budget_pct == 7.5
        assert scheduler_config.shadow_prefill_publish_mode == "terminal"

    def test_default_settings_leave_scheduler_config_disabled(self):
        scheduler_config = SchedulerConfig()
        settings = ModelSettings()
        apply_shadow_prefill_settings(scheduler_config, settings)
        assert scheduler_config.shadow_prefill_enabled is False
        assert scheduler_config.shadow_prefill_budget_pct == 0.0
        assert scheduler_config.shadow_prefill_publish_mode == "progressive"

    def test_missing_settings_object_defaults_safely(self):
        # A model with no ModelSettings entry (getattr falls back).
        scheduler_config = SchedulerConfig()
        apply_shadow_prefill_settings(scheduler_config, object())
        assert scheduler_config.shadow_prefill_enabled is False
        assert scheduler_config.shadow_prefill_budget_pct == 0.0
        assert scheduler_config.shadow_prefill_publish_mode == "progressive"
