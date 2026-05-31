# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfiguredModel — the (model, configuration) resolution object.

These pin the two-input resolutions: the *effective* thinking state reported to
clients (per-model override > template default) and the *override-only* kwargs
injected into the chat template at inference time (an unset toggle defers to the
template's own default).
"""

from dataclasses import dataclass

import pytest

from omlx.configured_model import ConfiguredModel, new_configured_model
from omlx.model_settings import ModelSettings


@dataclass
class FakeEntry:
    """Duck-typed stand-in for EngineEntry's relevant fields."""

    model_id: str = ""
    model_path: str = ""
    model_type: str = "llm"
    engine_type: str = "batched"
    estimated_size: int = 0
    thinking_default: bool | None = None
    preserve_thinking_default: bool | None = None
    model_context_length: int | None = None


@dataclass
class FakeSampling:
    """Duck-typed stand-in for SamplingDefaults' relevant fields."""

    max_context_window: int = 32768
    max_tokens: int = 32768


class TestEnableThinking:
    def test_override_true_wins(self):
        cm = new_configured_model(ModelSettings(enable_thinking=True), FakeEntry(thinking_default=False))
        assert cm.enable_thinking is True

    def test_override_false_wins(self):
        cm = new_configured_model(ModelSettings(enable_thinking=False), FakeEntry(thinking_default=True))
        assert cm.enable_thinking is False

    def test_falls_back_to_template_default(self):
        # Qwen-style: thinks by default; user left the toggle on auto.
        cm = new_configured_model(ModelSettings(enable_thinking=None), FakeEntry(thinking_default=True))
        assert cm.enable_thinking is True

    def test_no_toggle_is_none(self):
        # Model exposes no thinking toggle and the user set no override.
        cm = new_configured_model(ModelSettings(enable_thinking=None), FakeEntry(thinking_default=None))
        assert cm.enable_thinking is None

    def test_no_entry_uses_override_only(self):
        assert new_configured_model(ModelSettings(enable_thinking=True)).enable_thinking is True
        assert new_configured_model(ModelSettings()).enable_thinking is None


class TestPreserveThinking:
    def test_override_wins(self):
        cm = new_configured_model(
            ModelSettings(preserve_thinking=False), FakeEntry(preserve_thinking_default=True)
        )
        assert cm.preserve_thinking is False

    def test_falls_back_to_default(self):
        cm = new_configured_model(
            ModelSettings(preserve_thinking=None), FakeEntry(preserve_thinking_default=True)
        )
        assert cm.preserve_thinking is True


class TestThinkingTemplateOverrides:
    """The inference-time contract: inject only explicit toggles, never the
    template default — so an unset toggle defers to the template."""

    def test_empty_when_no_overrides(self):
        cm = new_configured_model(ModelSettings(), FakeEntry(thinking_default=True))
        assert cm.thinking_template_overrides() == {}

    def test_includes_only_set_toggles(self):
        cm = new_configured_model(ModelSettings(enable_thinking=True))
        assert cm.thinking_template_overrides() == {"enable_thinking": True}

    def test_includes_both_when_set(self):
        cm = new_configured_model(ModelSettings(enable_thinking=False, preserve_thinking=True))
        assert cm.thinking_template_overrides() == {
            "enable_thinking": False,
            "preserve_thinking": True,
        }

    def test_template_default_never_leaks_into_overrides(self):
        # thinking_default=True must NOT force enable_thinking into the kwargs;
        # that would change behaviour for models whose template already defaults
        # on. This is the equivalence that makes the server refactor safe.
        cm = new_configured_model(ModelSettings(enable_thinking=None), FakeEntry(thinking_default=True))
        assert cm.thinking_template_overrides() == {}
        assert cm.enable_thinking is True  # but the *reported* state is still True


class TestMaxContextWindow:
    """Precedence: per-model setting > discovered context > sampling default."""

    def test_global_default_when_nothing_set(self):
        cm = new_configured_model(ModelSettings(), sampling=FakeSampling(max_context_window=32768))
        assert cm.max_context_window == 32768

    def test_discovered_context_wins_over_global(self):
        cm = new_configured_model(
            ModelSettings(),
            FakeEntry(model_context_length=262144),
            FakeSampling(max_context_window=32768),
        )
        assert cm.max_context_window == 262144

    def test_per_model_setting_wins_over_discovery(self):
        cm = new_configured_model(
            ModelSettings(max_context_window=16384),
            FakeEntry(model_context_length=262144),
            FakeSampling(max_context_window=32768),
        )
        assert cm.max_context_window == 16384

    def test_per_model_setting_wins_over_global(self):
        cm = new_configured_model(
            ModelSettings(max_context_window=8192),
            sampling=FakeSampling(max_context_window=32768),
        )
        assert cm.max_context_window == 8192

    def test_no_entry_falls_to_sampling(self):
        cm = new_configured_model(ModelSettings(), sampling=FakeSampling(max_context_window=65536))
        assert cm.max_context_window == 65536


class TestMaxTokens:
    def test_settings_wins_over_sampling(self):
        cm = new_configured_model(ModelSettings(max_tokens=4096), sampling=FakeSampling(max_tokens=32768))
        assert cm.max_tokens == 4096

    def test_falls_to_sampling(self):
        cm = new_configured_model(ModelSettings(), sampling=FakeSampling(max_tokens=8192))
        assert cm.max_tokens == 8192


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
