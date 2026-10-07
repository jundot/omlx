# SPDX-License-Identifier: Apache-2.0
"""Tests for ConfiguredModel — the (model, configuration) resolution object.

These pin the layered resolutions: the *effective* thinking state reported to
clients (per-model override > template default) and the context/token limits
(per-model override > policy-clamped discovered value > global default).
"""


from omlx.configured_model import new_configured_model
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


class TestEnableThinking:
    """Mirrors the request path: toggle > chat_template_kwargs > thinking
    budget > template default."""

    def test_override_true_wins(self):
        cm = new_configured_model(
            ModelSettings(enable_thinking=True), entry(thinking_default=False)
        )
        assert cm.enable_thinking is True

    def test_override_false_wins(self):
        cm = new_configured_model(
            ModelSettings(enable_thinking=False), entry(thinking_default=True)
        )
        assert cm.enable_thinking is False

    def test_falls_back_to_template_default(self):
        # Qwen-style: thinks by default; user left the toggle on auto.
        cm = new_configured_model(
            ModelSettings(enable_thinking=None), entry(thinking_default=True)
        )
        assert cm.enable_thinking is True

    def test_no_toggle_is_none(self):
        # Model exposes no thinking toggle and the user set no override.
        cm = new_configured_model(
            ModelSettings(enable_thinking=None), entry(thinking_default=None)
        )
        assert cm.enable_thinking is None

    def test_no_entry_uses_override_only(self):
        assert new_configured_model(ModelSettings(enable_thinking=True)).enable_thinking is True
        assert new_configured_model(ModelSettings()).enable_thinking is None

    def test_chat_template_kwargs_beat_template_default(self):
        # The admin UI can also set enable_thinking as a raw template kwarg;
        # that is what gets rendered, so it must be what gets reported.
        cm = new_configured_model(
            ModelSettings(chat_template_kwargs={"enable_thinking": False}),
            entry(thinking_default=True),
        )
        assert cm.enable_thinking is False

    def test_toggle_beats_chat_template_kwargs(self):
        cm = new_configured_model(
            ModelSettings(
                enable_thinking=True,
                chat_template_kwargs={"enable_thinking": False},
            ),
            entry(thinking_default=None),
        )
        assert cm.enable_thinking is True

    def test_thinking_budget_switches_thinking_on(self):
        # A positive budget forces enable_thinking=True at render time when
        # nothing else set it (Gemma 4 templates suppress thinking otherwise).
        cm = new_configured_model(
            ModelSettings(thinking_budget_enabled=True, thinking_budget_tokens=512),
            entry(thinking_default=False),
        )
        assert cm.enable_thinking is True

    def test_disabled_budget_does_not_switch_thinking_on(self):
        cm = new_configured_model(
            ModelSettings(thinking_budget_enabled=False, thinking_budget_tokens=512),
            entry(thinking_default=False),
        )
        assert cm.enable_thinking is False


class TestPreserveThinking:
    """Mirrors the request path: toggle > chat_template_kwargs > template
    support, the last only while thinking is not switched off."""

    def test_override_wins(self):
        cm = new_configured_model(
            ModelSettings(preserve_thinking=False), entry(preserve_thinking_default=True)
        )
        assert cm.preserve_thinking is False

    def test_falls_back_to_default(self):
        cm = new_configured_model(
            ModelSettings(preserve_thinking=None), entry(preserve_thinking_default=True)
        )
        assert cm.preserve_thinking is True

    def test_unsupported_template_is_none(self):
        cm = new_configured_model(ModelSettings(), entry(preserve_thinking_default=None))
        assert cm.preserve_thinking is None

    def test_default_not_applied_when_thinking_disabled(self):
        # preserve_thinking_default only means "the template has the flag";
        # with thinking off the request path does not send it.
        cm = new_configured_model(
            ModelSettings(enable_thinking=False),
            entry(thinking_default=True, preserve_thinking_default=True),
        )
        assert cm.preserve_thinking is None

    def test_chat_template_kwargs_layer(self):
        cm = new_configured_model(
            ModelSettings(chat_template_kwargs={"preserve_thinking": False}),
            entry(preserve_thinking_default=True),
        )
        assert cm.preserve_thinking is False


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

    def test_policy_zero_is_unset(self):
        # ``0`` is the "no policy" sentinel from settings.json; it must not
        # clamp the native context to nothing.
        cm = new_configured_model(
            ModelSettings(),
            entry(model_context_length=262_144),
            SamplingDefaults(max_context_window=32768, max_context_window_policy=0),
        )
        assert cm.max_context_window == 262_144

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


class TestMaxTokens:
    def test_settings_wins_over_sampling(self):
        cm = new_configured_model(
            ModelSettings(max_tokens=4096), sampling=SamplingDefaults(max_tokens=32768)
        )
        assert cm.max_tokens == 4096

    def test_falls_to_sampling(self):
        cm = new_configured_model(ModelSettings(), sampling=SamplingDefaults(max_tokens=8192))
        assert cm.max_tokens == 8192
