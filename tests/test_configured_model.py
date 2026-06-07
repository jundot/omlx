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
from omlx.engine_pool import EngineEntry
from omlx.model_settings import ModelSettings
from omlx.server import SamplingDefaults


def entry(**kwargs) -> EngineEntry:
    """Build a real EngineEntry, defaulting the identity boilerplate so a test
    only spells out the field under test (e.g. ``model_context_length``)."""
    return EngineEntry(
        model_id=kwargs.pop("model_id", "test-model"),
        model_path=kwargs.pop("model_path", ""),
        model_type=kwargs.pop("model_type", "llm"),
        engine_type=kwargs.pop("engine_type", "batched"),
        estimated_size=kwargs.pop("estimated_size", 0),
        **kwargs,
    )


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
    """Precedence: per-model setting > discovered context > sampling default.

    The discovered-context tier is additionally subject to an operator policy
    cap (``max_context_window_policy``): when set, the native context is clamped
    to ``min(native, policy)``. The per-model setting and the sampling fallback
    are deliberately *not* clamped — those are explicit operator choices, so the
    policy is a ceiling on auto-discovered values only.
    """

    def test_global_default_when_nothing_set(self):
        cm = new_configured_model(
            ModelSettings(), sampling=SamplingDefaults(max_context_window=32768)
        )
        assert cm.max_context_window == 32768

    def test_discovered_context_wins_over_global(self):
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=262144),
            SamplingDefaults(max_context_window=32768),
        )
        assert cm.max_context_window == 262144

    def test_per_model_setting_wins_over_discovery(self):
        cm = new_configured_model(
            ModelSettings(max_context_window=16384),
            entry(model_context_length=262144),
            SamplingDefaults(max_context_window=32768),
        )
        assert cm.max_context_window == 16384

    def test_per_model_setting_wins_over_global(self):
        cm = new_configured_model(
            ModelSettings(max_context_window=8192),
            sampling=SamplingDefaults(max_context_window=32768),
        )
        assert cm.max_context_window == 8192

    def test_no_entry_falls_to_sampling(self):
        cm = new_configured_model(
            ModelSettings(), sampling=SamplingDefaults(max_context_window=65536)
        )
        assert cm.max_context_window == 65536

    # --- operator policy cap (max_context_window_policy) ---

    def test_policy_unset_native_wins_unchanged(self):
        # With no policy, the model's native context is returned verbatim —
        # existing installs see no behavior change.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=262_144),
            SamplingDefaults(
                max_context_window=32768, max_context_window_policy=None
            ),
        )
        assert cm.max_context_window == 262_144

    def test_policy_clamps_native(self):
        # Policy below native: native is clamped down to the policy.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=262_144),
            SamplingDefaults(
                max_context_window=32768, max_context_window_policy=128_000
            ),
        )
        assert cm.max_context_window == 128_000, (
            "Policy of 128k must clamp a model that natively declares 256k"
        )

    def test_policy_native_below_policy_wins(self):
        # Policy is a ceiling, not a floor: a native length already under it wins.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=32_768),
            SamplingDefaults(
                max_context_window=32768, max_context_window_policy=128_000
            ),
        )
        assert cm.max_context_window == 32_768

    def test_per_model_override_escapes_policy(self):
        # A per-model override is the operator's explicit per-model choice; the
        # policy cap does NOT clamp it. Escape hatch for individual models that
        # should exceed the policy.
        cm = new_configured_model(
            ModelSettings(max_context_window=200_000),
            entry(model_context_length=100_000),
            SamplingDefaults(
                max_context_window=32768, max_context_window_policy=64_000
            ),
        )
        assert cm.max_context_window == 200_000, (
            "Per-model override must escape the policy clamp"
        )

    def test_policy_does_not_apply_to_fallback_path(self):
        # No native context and no per-model override: the sampling fallback
        # applies and the policy does NOT clamp it. The policy caps the *native*
        # path only, so existing settings.json files keep working unchanged even
        # after a policy is later added to the install.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=None),
            SamplingDefaults(
                max_context_window=32_768, max_context_window_policy=16_000
            ),
        )
        assert cm.max_context_window == 32_768


class TestEmbeddingMaxLength:
    """embedding_max_length precedence: explicit request length > the resolved
    context window > a defensive 512 floor."""

    def test_request_length_wins(self):
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=4096),
            SamplingDefaults(max_context_window=32768),
        )
        assert cm.embedding_max_length(256) == 256

    def test_falls_back_to_context_window(self):
        # No explicit request length: defer to the model's resolved context.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=4096),
            SamplingDefaults(max_context_window=32768),
        )
        assert cm.embedding_max_length() == 4096

    def test_respects_policy_cap(self):
        # The resolved context window already reflects the policy clamp, so the
        # embedding length inherits it.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=262_144),
            SamplingDefaults(
                max_context_window=32768, max_context_window_policy=128_000
            ),
        )
        assert cm.embedding_max_length() == 128_000

    def test_floor_when_no_context_known(self):
        # Degenerate config: no per-model setting, no native context, and no
        # sampling default either -> the defensive 512 floor applies.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=None),
            SamplingDefaults(max_context_window=None),
        )
        assert cm.embedding_max_length() == 512


class TestMaxTokens:
    def test_settings_wins_over_sampling(self):
        cm = new_configured_model(ModelSettings(max_tokens=4096), sampling=FakeSampling(max_tokens=32768))
        assert cm.max_tokens == 4096

    def test_falls_to_sampling(self):
        cm = new_configured_model(ModelSettings(), sampling=FakeSampling(max_tokens=8192))
        assert cm.max_tokens == 8192


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
