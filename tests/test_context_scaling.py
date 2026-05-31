# SPDX-License-Identifier: Apache-2.0
"""Tests for Anthropic context scaling functionality."""

from unittest.mock import MagicMock, patch

import pytest

from omlx.server import SamplingDefaults, ServerState, scale_anthropic_tokens
from omlx.settings import ClaudeCodeSettings, GlobalSettings


class TestScaleAnthropicTokens:
    """Tests for scale_anthropic_tokens function."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def _configure(self, enabled: bool, target: int, max_context: int):
        gs = GlobalSettings.__new__(GlobalSettings)
        gs.claude_code = ClaudeCodeSettings(
            context_scaling_enabled=enabled,
            target_context_size=target,
        )
        self._state.global_settings = gs
        self._state.sampling = SamplingDefaults(max_context_window=max_context)

    def test_scaling_disabled_returns_original(self):
        """When scaling is disabled, return original token count."""
        self._configure(enabled=False, target=200000, max_context=32768)
        assert scale_anthropic_tokens(1000) == 1000

    def test_scaling_enabled_basic(self):
        """Test basic scaling: 100k model, 200k target."""
        self._configure(enabled=True, target=200000, max_context=100000)
        # 50000 * (200000 / 100000) = 100000
        assert scale_anthropic_tokens(50000, "test-model") == 100000

    def test_scaling_preserves_ratio(self):
        """Test that scaling preserves usage ratio."""
        self._configure(enabled=True, target=200000, max_context=131072)
        # 96000 * (200000 / 131072)
        expected = int(96000 * 200000 / 131072)
        assert scale_anthropic_tokens(96000, "test-model") == expected

    def test_no_scaling_when_actual_ge_target(self):
        """When actual context >= target, no scaling needed."""
        self._configure(enabled=True, target=100000, max_context=200000)
        assert scale_anthropic_tokens(50000, "test-model") == 50000

    def test_no_scaling_when_actual_equals_target(self):
        """When actual context == target, no scaling."""
        self._configure(enabled=True, target=200000, max_context=250000)
        assert scale_anthropic_tokens(100000, "test-model") == 100000

    def test_global_settings_none_returns_original(self):
        """When global_settings is None, return original."""
        assert scale_anthropic_tokens(5000, "test-model") == 5000

    def test_actual_context_none_returns_original(self):
        """When max_context_window resolves to None, no scaling occurs."""
        gs = GlobalSettings.__new__(GlobalSettings)
        gs.claude_code = ClaudeCodeSettings(
            context_scaling_enabled=True,
            target_context_size=200000,
        )
        self._state.global_settings = gs
        mock_config = MagicMock()
        mock_config.max_context_window = None
        with patch("omlx.server.model_config", return_value=mock_config):
            assert scale_anthropic_tokens(5000, "test-model") == 5000

    def test_zero_tokens(self):
        """Test with zero token count."""
        self._configure(enabled=True, target=200000, max_context=32768)
        assert scale_anthropic_tokens(0, "test-model") == 0

    def test_scaling_returns_int(self):
        """Test that scaling always returns integer."""
        self._configure(enabled=True, target=200000, max_context=131072)
        result = scale_anthropic_tokens(12345, "test-model")
        assert isinstance(result, int)
