# SPDX-License-Identifier: Apache-2.0
"""Unit tests for context compaction strategies (Task B5).

Covers CompactStrategy ABC, NoCompact, SlidingWindowCompact, and
TieredCompact (3-phase priority compaction with protection rules).
"""
from __future__ import annotations

import pytest

from omlx.context.compaction import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    TRUNCATED_MARKER,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: str, content: str, **extra) -> dict:
    """Build a chat message dict with optional extra keys."""
    m: dict = {"role": role, "content": content}
    m.update(extra)
    return m


def _system(content: str = "system prompt") -> dict:
    return _msg("system", content)


def _user(content: str = "user input") -> dict:
    return _msg("user", content)


def _assistant(content: str = "assistant text") -> dict:
    return _msg("assistant", content)


def _tool(content: str = "tool result", tool_call_id: str = "tc_1") -> dict:
    return _msg("tool", content, tool_call_id=tool_call_id)


def _assistant_tool_call(tool_calls: list[dict] | None = None) -> dict:
    """Assistant message with tool_calls (no text content)."""
    if tool_calls is None:
        tool_calls = [{"id": "tc_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _assistant_reasoning(reasoning: str = "thinking...", content: str = "answer") -> dict:
    """Assistant message with reasoning_content (reasoning models)."""
    return {"role": "assistant", "reasoning_content": reasoning, "content": content}


# A very rough token estimator that counts 1 token per 4 chars.
def _est_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        for v in m.values():
            if isinstance(v, str):
                total += len(v) // 4 + 1
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for iv in item.values():
                            total += len(str(iv)) // 4 + 1
    return max(total, 1)


# ---------------------------------------------------------------------------
# CompactStrategy ABC
# ---------------------------------------------------------------------------

class TestCompactStrategyABC:
    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            CompactStrategy()  # type: ignore[abstract]

    def test_compact_signature_returns_tuple(self):
        """Subclass must return (list, int)."""
        strat = NoCompact()
        msgs = [_system(), _user()]
        result = strat.compact(msgs, budget_tokens=9999)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], int)


# ---------------------------------------------------------------------------
# NoCompact
# ---------------------------------------------------------------------------

class TestNoCompact:
    def test_passthrough_returns_same_messages(self):
        msgs = [_system(), _user(), _assistant()]
        strat = NoCompact()
        out, tokens = strat.compact(msgs, budget_tokens=100)
        assert out == msgs
        assert tokens == _est_tokens(msgs)

    def test_empty_messages(self):
        strat = NoCompact()
        out, tokens = strat.compact([], budget_tokens=100)
        assert out == []
        assert tokens == 0

    def test_budget_ignored(self):
        """NoCompact ignores budget — always returns full list."""
        msgs = [_system("x" * 1000)]
        strat = NoCompact()
        out, _ = strat.compact(msgs, budget_tokens=1)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# SlidingWindowCompact
# ---------------------------------------------------------------------------

class TestSlidingWindowCompact:
    def test_keeps_last_n_messages(self):
        msgs = [_system(), _user(), _assistant("a1"), _user("u2"), _assistant("a2")]
        strat = SlidingWindowCompact(max_messages=3)
        out, _ = strat.compact(msgs, budget_tokens=99999)
        assert len(out) == 3
        assert out[-1] is msgs[-1]

    def test_never_drops_system_prompt(self):
        """System prompt must survive even if it's outside the window."""
        msgs = [_system("important"), _user("u1"), _assistant("a1"), _user("u2")]
        strat = SlidingWindowCompact(max_messages=2)
        out, _ = strat.compact(msgs, budget_tokens=99999)
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "important"

    def test_all_messages_kept_when_under_window(self):
        msgs = [_system(), _user()]
        strat = SlidingWindowCompact(max_messages=10)
        out, _ = strat.compact(msgs, budget_tokens=99999)
        assert len(out) == 2

    def test_returns_estimated_tokens(self):
        msgs = [_system(), _user(), _assistant()]
        strat = SlidingWindowCompact(max_messages=5)
        _, tokens = strat.compact(msgs, budget_tokens=99999)
        assert tokens == _est_tokens(msgs)

    def test_budget_enforced_when_window_still_too_large(self):
        """If the windowed messages still exceed budget, drop from the front
        (preserving system + user as long as possible)."""
        big = "x" * 400  # ~100 tokens each
        msgs = [
            _system(big),
            _user(big),
            _assistant(big),
            _assistant(big),
            _assistant(big),
        ]
        strat = SlidingWindowCompact(max_messages=10)
        out, tokens = strat.compact(msgs, budget_tokens=150)
        assert tokens <= 150


# ---------------------------------------------------------------------------
# TieredCompact — Phase behavior
# ---------------------------------------------------------------------------

class TestTieredCompactPhases:
    def test_no_compaction_when_under_budget(self):
        msgs = [_system(), _user(), _assistant()]
        strat = TieredCompact()
        out, tokens = strat.compact(msgs, budget_tokens=99999)
        assert out == msgs
        assert tokens == _est_tokens(msgs)

    def test_phase1_drops_nudge_messages(self):
        """Phase 1 drops messages whose content matches a nudge / retry pattern."""
        nudge_content = (
            "You provided a text response instead of making a tool call. "
            "Please use the available tools to answer the request."
        )
        msgs = [
            _system(),
            _user(),
            _assistant(nudge_content),  # nudge — should be dropped
            _assistant("real answer"),
        ]
        strat = TieredCompact()
        # Tight budget so phase 1 kicks in.
        out, _ = strat.compact(msgs, budget_tokens=_est_tokens(msgs) - 1)
        contents = [m.get("content", "") for m in out]
        assert nudge_content not in contents

    def test_phase1_truncates_tool_results(self):
        """Phase 1 truncates long tool results to a skeleton with a marker."""
        long_result = "data: " + "x" * 800  # ~200 tokens
        msgs = [
            _system(),
            _user(),
            _assistant_tool_call(),
            _tool(long_result),
        ]
        strat = TieredCompact()
        out, _ = strat.compact(msgs, budget_tokens=_est_tokens(msgs) // 2)
        tool_msgs = [m for m in out if m["role"] == "tool"]
        for tm in tool_msgs:
            assert TRUNCATED_MARKER in tm["content"]
            assert len(tm["content"]) < len(long_result)

    def test_phase2_drops_tool_results_entirely(self):
        """Phase 2 drops tool result messages (role='tool')."""
        msgs = [
            _system(),
            _user(),
            _assistant_tool_call(),
            _tool("result A"),
            _assistant_tool_call(),
            _tool("result B" + "x" * 400),
        ]
        strat = TieredCompact()
        # Very tight budget to push into phase 2.
        budget = _est_tokens([_system(), _user()])
        out, _ = strat.compact(msgs, budget_tokens=budget)
        assert all(m["role"] != "tool" for m in out)

    def test_phase3_drops_reasoning_and_text(self):
        """Phase 3 drops reasoning_content and assistant text content."""
        msgs = [
            _system("sys"),
            _user("usr"),
            _assistant_reasoning("deep thoughts" * 50, "long text" * 50),
            _assistant_reasoning("more thinking" * 50, "more text" * 50),
            _assistant_reasoning("even more" * 50, "final" * 50),
        ]
        strat = TieredCompact()
        # Extremely tight budget: only system + user survive.
        budget = _est_tokens([_system("sys"), _user("usr")])
        out, _ = strat.compact(msgs, budget_tokens=budget)
        # No reasoning_content should remain.
        assert all("reasoning_content" not in m for m in out)
        # Assistant text content should be empty or dropped.
        asst = [m for m in out if m["role"] == "assistant"]
        for a in asst:
            assert not a.get("content")


# ---------------------------------------------------------------------------
# TieredCompact — Protection rules
# ---------------------------------------------------------------------------

class TestTieredCompactProtection:
    def test_system_prompt_never_dropped(self):
        msgs = [
            _system("critical system instructions"),
            _assistant("x" * 500),
            _assistant("y" * 500),
        ]
        strat = TieredCompact()
        out, _ = strat.compact(msgs, budget_tokens=10)
        assert any(m["role"] == "system" and m["content"] == "critical system instructions" for m in out)

    def test_user_input_never_dropped(self):
        msgs = [
            _system(),
            _user("the original question"),
            _assistant("x" * 500),
        ]
        strat = TieredCompact()
        out, _ = strat.compact(msgs, budget_tokens=10)
        assert any(m["role"] == "user" and m["content"] == "the original question" for m in out)

    def test_recent_iterations_protected(self):
        """The most recent assistant + tool exchange should survive aggressive compaction."""
        msgs = [
            _system(),
            _user(),
            _assistant("old1"),
            _assistant("old2"),
            _assistant("old3"),
            _assistant("recent"),
        ]
        strat = TieredCompact()
        out, _ = strat.compact(msgs, budget_tokens=50)
        assert out[-1]["content"] == "recent"


# ---------------------------------------------------------------------------
# TieredCompact — Return value contract
# ---------------------------------------------------------------------------

class TestTieredCompactReturnContract:
    def test_returned_tokens_within_budget_or_best_effort(self):
        """If compaction succeeds, tokens <= budget. If it can't
        (protected messages exceed budget), tokens is the minimum achievable."""
        msgs = [_system(), _user(), _assistant("answer")]
        strat = TieredCompact()
        out, tokens = strat.compact(msgs, budget_tokens=99999)
        assert tokens == _est_tokens(out)

    def test_empty_messages(self):
        strat = TieredCompact()
        out, tokens = strat.compact([], budget_tokens=100)
        assert out == []
        assert tokens == 0


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_imports_from_package_init(self):
        from omlx.context import (
            CompactStrategy,
            NoCompact,
            SlidingWindowCompact,
            TieredCompact,
        )
        assert CompactStrategy is not None
        assert NoCompact is not None
        assert SlidingWindowCompact is not None
        assert TieredCompact is not None

    def test_get_strategy_factory(self):
        """Optional convenience: strategy name -> instance."""
        from omlx.context import get_strategy
        assert isinstance(get_strategy("none"), NoCompact)
        assert isinstance(get_strategy("sliding_window"), SlidingWindowCompact)
        assert isinstance(get_strategy("tiered"), TieredCompact)

    def test_get_strategy_invalid_raises(self):
        from omlx.context import get_strategy
        with pytest.raises(ValueError):
            get_strategy("bogus")
