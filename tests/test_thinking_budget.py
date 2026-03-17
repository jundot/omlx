# SPDX-License-Identifier: Apache-2.0
"""Tests for ThinkingBudgetProcessor logits processor."""

from unittest.mock import MagicMock, patch

import pytest

# Lazy-import mlx.core — tests skip gracefully if unavailable.
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.model_settings import ModelSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logits(vocab_size: int = 100):
    """Create a dummy logits tensor [1, vocab_size]."""
    return mx.zeros((1, vocab_size))


def _make_tokens(*token_ids: int):
    """Create a tokens tensor from a list of token IDs."""
    return mx.array(list(token_ids))


# ---------------------------------------------------------------------------
# ThinkingBudgetProcessor unit tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="mlx not available")
class TestThinkingBudgetProcessor:
    """Unit tests for the ThinkingBudgetProcessor."""

    THINK_END_ID = 42  # Dummy </think> token ID
    THINK_START_ID = 41  # Dummy <think> token ID

    NEWLINE_ID = 99  # Dummy \n token ID

    def _make_processor(self, budget: int = 5, end_ids=None, trailing_ids=None):
        return ThinkingBudgetProcessor(
            think_end_token_ids=end_ids or [self.THINK_END_ID],
            budget=budget,
            think_start_token_id=self.THINK_START_ID,
            trailing_token_ids=trailing_ids,
        )

    # --- Budget enforcement ---

    def test_forces_end_token_when_budget_exceeded(self):
        """After budget tokens, logits should force the end-think token."""
        proc = self._make_processor(budget=3)

        # First call (first_call flag skips state update)
        logits = proc(_make_tokens(10), _make_logits())
        assert not proc._forcing

        # Simulate token generation: each call = one decode step
        logits = proc(_make_tokens(10, 20), _make_logits())
        assert not proc._forcing

        logits = proc(_make_tokens(10, 20, 30), _make_logits())
        # Budget=3, third token should trigger forcing
        assert proc._forcing or proc._done

        # The forced logits should have -inf everywhere except target
        target_logit = logits[0, self.THINK_END_ID].item()
        other_logit = logits[0, 0].item()
        assert target_logit == 0.0
        assert other_logit == float("-inf")

    def test_enters_suppression_after_forcing(self):
        """After forcing </think>, processor should suppress duplicate end tokens."""
        proc = self._make_processor(budget=1)

        # Call 1 (first_call): budget=1, forcing starts → forces THINK_END_ID
        forced_logits = proc(_make_tokens(10), _make_logits())
        assert proc._forcing
        assert forced_logits[0, self.THINK_END_ID].item() == 0.0

        # Call 2: force_sequence has only [THINK_END_ID] (no trailing), so
        # _force_idx advances to 1 == len([42]) → enters suppression
        logits = proc(_make_tokens(10, self.THINK_END_ID), _make_logits())
        assert proc._suppress_end
        assert not proc._done
        # The end token should be suppressed (-inf)
        assert logits[0, self.THINK_END_ID].item() == float("-inf")

    def test_trailing_tokens_forced_after_end(self):
        """Trailing tokens (e.g. \\n) should be forced after </think>."""
        trailing = [self.NEWLINE_ID]
        proc = self._make_processor(budget=1, trailing_ids=trailing)
        # _force_sequence = [THINK_END_ID, NEWLINE_ID]

        # Call 1: budget hit, forces THINK_END_ID
        logits0 = proc(_make_tokens(10), _make_logits())
        assert logits0[0, self.THINK_END_ID].item() == 0.0

        # Call 2: _force_idx advances to 1, forces NEWLINE_ID
        logits1 = proc(_make_tokens(10, self.THINK_END_ID), _make_logits())
        assert proc._forcing
        assert logits1[0, self.NEWLINE_ID].item() == 0.0

        # Call 3: _force_idx advances to 2 == len([42, 99]) → suppression
        logits2 = proc(_make_tokens(10, self.THINK_END_ID, self.NEWLINE_ID), _make_logits())
        assert proc._suppress_end
        assert logits2[0, self.THINK_END_ID].item() == float("-inf")

    def test_natural_end_before_budget(self):
        """If model produces </think> naturally, processor becomes no-op."""
        proc = self._make_processor(budget=100)

        # First call
        proc(_make_tokens(10), _make_logits())

        # Second call — model naturally produced </think>
        proc(_make_tokens(10, self.THINK_END_ID), _make_logits())
        assert proc._done

        # Subsequent call should be no-op
        original = _make_logits()
        result = proc(_make_tokens(10, self.THINK_END_ID, 50), original)
        assert mx.array_equal(result, original)

    def test_first_call_skips_state_update(self):
        """First call should not check tokens[-1] for state transitions."""
        proc = self._make_processor(budget=100)

        # Simulate prompt ending with </think> token (shouldn't happen but edge case)
        proc(_make_tokens(self.THINK_END_ID), _make_logits())

        # Should still be in thinking mode (first call skipped state update)
        assert proc._in_thinking
        assert not proc._done

    # --- Multi-token end sequence ---

    def test_multi_token_forcing(self):
        """Multi-token </think> should be forced one token at a time."""
        end_ids = [50, 51, 52]  # e.g. "</" + "think" + ">"
        proc = self._make_processor(budget=1, end_ids=end_ids)

        # Call 1 (first_call): budget hit, forcing starts at _force_idx=0 → token 50
        logits0 = proc(_make_tokens(10), _make_logits())
        assert proc._forcing
        assert logits0[0, 50].item() == 0.0

        # Call 2: _update_state advances _force_idx to 1 → forces token 51
        logits1 = proc(_make_tokens(10, 50), _make_logits())
        assert proc._forcing
        assert logits1[0, 51].item() == 0.0

        # Call 3: _force_idx advances to 2 → forces token 52
        logits2 = proc(_make_tokens(10, 50, 51), _make_logits())
        assert proc._forcing
        assert logits2[0, 52].item() == 0.0

        # Call 4: _force_idx advances to 3 == len(end_ids), enters suppression
        logits3 = proc(_make_tokens(10, 50, 51, 52), _make_logits())
        assert proc._suppress_end
        assert not proc._done
        # All end tokens suppressed
        for tid in end_ids:
            assert logits3[0, tid].item() == float("-inf")

    def test_multi_token_natural_detection(self):
        """Sliding window should detect multi-token </think> naturally."""
        end_ids = [50, 51]
        proc = self._make_processor(budget=100, end_ids=end_ids)

        proc(_make_tokens(10), _make_logits())  # First call

        # Generate tokens that match the end sequence
        proc(_make_tokens(10, 50), _make_logits())
        assert not proc._done

        proc(_make_tokens(10, 50, 51), _make_logits())
        assert proc._done

    # --- Edge cases ---

    def test_zero_budget(self):
        """Budget=0 should force on the very first thinking token."""
        proc = self._make_processor(budget=0)

        # First call — budget is 0, so _thinking_tokens (0) >= budget (0)
        logits = proc(_make_tokens(10), _make_logits())
        assert proc._forcing
        assert logits[0, self.THINK_END_ID].item() == 0.0

    def test_large_budget_no_forcing(self):
        """With a very large budget, no forcing should happen."""
        proc = self._make_processor(budget=10000)

        # Use token IDs 100+ to avoid colliding with THINK_END_ID (42) or THINK_START_ID (41)
        for i in range(50):
            proc(_make_tokens(*range(100, 100 + i + 1)), _make_logits())

        assert not proc._forcing
        assert not proc._done
        assert proc._in_thinking


# ---------------------------------------------------------------------------
# ModelSettings serialization
# ---------------------------------------------------------------------------


class TestModelSettingsThinkingBudget:
    """Test thinking_budget fields in ModelSettings."""

    def test_to_dict_includes_thinking_budget(self):
        settings = ModelSettings(thinking_budget_enabled=True, thinking_budget_tokens=4096)
        d = settings.to_dict()
        assert d["thinking_budget_enabled"] is True
        assert d["thinking_budget_tokens"] == 4096

    def test_from_dict_with_thinking_budget(self):
        data = {"thinking_budget_enabled": True, "thinking_budget_tokens": 2048}
        settings = ModelSettings.from_dict(data)
        assert settings.thinking_budget_enabled is True
        assert settings.thinking_budget_tokens == 2048

    def test_defaults(self):
        settings = ModelSettings()
        assert settings.thinking_budget_enabled is False
        assert settings.thinking_budget_tokens is None

    def test_to_dict_excludes_none(self):
        settings = ModelSettings()
        d = settings.to_dict()
        assert "thinking_budget_tokens" not in d


# ---------------------------------------------------------------------------
# _resolve_thinking_budget (server.py helper)
# ---------------------------------------------------------------------------


class TestResolveThinkingBudget:
    """Test the _resolve_thinking_budget helper function."""

    def _import_resolve(self):
        from omlx.server import _resolve_thinking_budget
        return _resolve_thinking_budget

    def test_request_override_takes_priority(self):
        resolve = self._import_resolve()
        req = MagicMock(spec=[])
        req.thinking_budget = 1024
        result = resolve(req, None)
        assert result == 1024

    def test_anthropic_budget_tokens(self):
        resolve = self._import_resolve()
        req = MagicMock(spec=[])
        thinking = MagicMock(spec=[])
        thinking.budget_tokens = 2048
        req.thinking = thinking
        result = resolve(req, None)
        assert result == 2048

    def test_returns_none_when_disabled(self):
        resolve = self._import_resolve()
        req = MagicMock(spec=[])
        result = resolve(req, None)
        assert result is None
