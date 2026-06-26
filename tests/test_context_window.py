# SPDX-License-Identifier: Apache-2.0
"""Tests for context window validation feature."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx.model_settings import ModelSettings


class TestGetMaxContextWindow:
    """Tests for get_max_context_window() priority logic."""

    def _make_server_state(
        self, global_max_ctx=32768, policy_cap=None
    ):
        """Create a mock server state with given global
        ``max_context_window`` fallback and optional
        ``max_context_window_policy`` cap."""
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(
            max_context_window=global_max_ctx,
            max_context_window_policy=policy_cap,
        )
        state.settings_manager = None
        # Discovery-tier (#1308) lookups are exercised in TestGetMaxContextWindow
        # in test_server.py; nulling the pool here keeps these focused on the
        # per-model-setting → global fallback path.
        state.engine_pool = None
        return state

    def test_returns_global_default(self):
        """Test returns global default when no model settings."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window()
            assert result == 32768

    def test_model_setting_overrides_global(self):
        """Test model-specific setting takes priority over global."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=4096
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == 4096

    def test_falls_back_to_global_when_model_not_set(self):
        """Test falls back to global when model has no max_context_window."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=65536)
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=None
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            result = get_max_context_window("test-model")
            assert result == 65536

    def test_no_model_id_returns_global(self):
        """Test returns global when model_id is None."""
        from omlx.server import get_max_context_window

        state = self._make_server_state(global_max_ctx=16384)
        with patch("omlx.server._server_state", state):
            result = get_max_context_window(None)
            assert result == 16384

    def _mount_native_and_policy(
        self, native_ctx: int | None, policy_cap: int | None
    ):
        """Mount a server state with a model that has the given native
        context length, the policy field set to ``policy_cap``, and no
        per-model override."""
        state = self._make_server_state(
            global_max_ctx=32768, policy_cap=policy_cap
        )
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=None
        )
        state.settings_manager = mock_manager

        mock_pool = MagicMock()
        mock_entry = MagicMock()
        mock_entry.model_context_length = native_ctx
        mock_pool.get_entry.return_value = mock_entry
        state.engine_pool = mock_pool
        return state

    def test_policy_unset_native_wins_unchanged(self):
        """With ``max_context_window_policy`` unset, the model's
        native context length is returned verbatim — existing
        installs see no behavior change after this PR."""
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(
            native_ctx=262_144, policy_cap=None
        )
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("big-model") == 262_144

    def test_policy_set_clamps_native(self):
        """With ``max_context_window_policy=128_000`` and a model that
        natively declares 256 K, the effective cap is the policy."""
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(
            native_ctx=262_144, policy_cap=128_000
        )
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("big-model") == 128_000, (
                "Policy of 128k must clamp a model that natively declares 256k"
            )

    def test_policy_set_native_below_policy_wins(self):
        """When the model's native length is already below the policy,
        the native value wins — policy is a ceiling, not a floor."""
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(
            native_ctx=32_768, policy_cap=128_000
        )
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("small-model") == 32_768

    def test_per_model_override_escapes_policy(self):
        """A per-model override is the operator's explicit per-model
        choice; the global policy cap does NOT clamp it. This is the
        operator's escape hatch for individual models that should
        exceed the policy."""
        from omlx.server import get_max_context_window

        state = self._mount_native_and_policy(
            native_ctx=100_000, policy_cap=64_000
        )
        # Add a per-model override above both native and policy
        state.settings_manager = MagicMock()
        state.settings_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=200_000
        )
        with patch("omlx.server._server_state", state):
            assert get_max_context_window("override-model") == 200_000, (
                "Per-model override must escape the policy clamp"
            )

    def test_policy_does_not_apply_to_fallback_path(self):
        """When the model has no discoverable native context AND no
        per-model override, the fallback default applies — the policy
        is documented as clamping the *native* path only. Existing
        ``settings.json`` files with the historical 32768 fallback
        therefore keep working unchanged even when a policy is later
        added to the install."""
        from omlx.server import get_max_context_window

        # native_ctx=None: model config doesn't expose a context length
        state = self._mount_native_and_policy(
            native_ctx=None, policy_cap=16_000
        )
        with patch("omlx.server._server_state", state):
            # Fallback (32768) returned, not the policy (16_000).
            assert get_max_context_window("no-native-model") == 32_768


class TestValidateContextWindow:
    """Tests for validate_context_window()."""

    def _make_server_state(self, global_max_ctx=32768):
        from omlx.server import SamplingDefaults

        state = MagicMock()
        state.sampling = SamplingDefaults(max_context_window=global_max_ctx)
        state.settings_manager = None
        return state

    def test_passes_when_under_limit(self):
        """Test no exception when token count is under limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            # Should not raise
            validate_context_window(500)

    def test_passes_at_exact_limit(self):
        """Test no exception when token count equals limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            # Should not raise (equal is OK)
            validate_context_window(1000)

    def test_raises_when_over_limit(self):
        """Test HTTPException raised when token count exceeds limit."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=1000)
        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(1001)
            assert exc_info.value.status_code == 400
            assert "1001 tokens" in exc_info.value.detail
            assert "1000 tokens" in exc_info.value.detail

    def test_raises_with_model_specific_limit(self):
        """Test uses model-specific limit when available."""
        from omlx.server import validate_context_window

        state = self._make_server_state(global_max_ctx=32768)
        mock_manager = MagicMock()
        mock_manager.get_settings_for_request.return_value = ModelSettings(
            max_context_window=100
        )
        state.settings_manager = mock_manager

        with patch("omlx.server._server_state", state):
            with pytest.raises(HTTPException) as exc_info:
                validate_context_window(200, "test-model")
            assert exc_info.value.status_code == 400
            assert "200 tokens" in exc_info.value.detail
            assert "100 tokens" in exc_info.value.detail


class TestCountChatTokens:
    """Tests for BatchedEngine.count_chat_tokens()."""

    def test_count_chat_tokens(self):
        """Test token counting with mocked tokenizer."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer = mock_tokenizer

        # Mock model (not gpt_oss)
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        count = engine.count_chat_tokens(messages)

        assert count == 5
        mock_tokenizer.apply_chat_template.assert_called_once()
        mock_tokenizer.encode.assert_called_once_with("formatted prompt")

    def test_count_chat_tokens_with_tools(self):
        """Test token counting includes tools in template."""
        from omlx.engine.batched import BatchedEngine

        engine = BatchedEngine.__new__(BatchedEngine)
        engine._loaded = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt with tools"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7]
        engine._tokenizer = mock_tokenizer
        engine._model = MagicMock(spec=[])
        engine._enable_thinking = None

        messages = [{"role": "user", "content": "Call a tool"}]
        tools = [{"type": "function", "function": {"name": "test"}}]
        count = engine.count_chat_tokens(messages, tools)

        assert count == 7
