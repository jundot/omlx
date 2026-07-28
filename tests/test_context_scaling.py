# SPDX-License-Identifier: Apache-2.0
"""Regression tests confirming the Anthropic token-scaling mechanism is gone.

See openspec/changes/fix-claude-code-autocompact-threshold: reported token
usage must always reflect the real token count. The old
`scale_anthropic_tokens()` ratio-scaling (target_context_size /
actual_context_size) inflated numerator while `CLAUDE_CODE_AUTO_COMPACT_WINDOW`
kept reporting the real, un-scaled denominator, causing auto-compact to fire
far too early or, when it didn't fire in time, allowing the real prompt to
exceed the real context window before Claude Code intervened.
"""

import omlx.server as server_module
from omlx.settings import ClaudeCodeSettings


class TestScalingMechanismRemoved:
    """The ratio-scaling function and its settings must no longer exist."""

    def test_scale_anthropic_tokens_no_longer_exists(self):
        assert not hasattr(server_module, "scale_anthropic_tokens")

    def test_claude_code_settings_has_no_scaling_fields(self):
        settings = ClaudeCodeSettings()
        assert not hasattr(settings, "context_scaling_enabled")
        assert not hasattr(settings, "target_context_size")

    def test_claude_code_settings_has_threshold_pct(self):
        settings = ClaudeCodeSettings()
        assert settings.autocompact_threshold_pct == 85
