# SPDX-License-Identifier: Apache-2.0
"""Context compaction strategies for long conversations.

Deterministic, sub-millisecond message-level compaction (no LLM calls).

Three strategies via the :class:`CompactStrategy` ABC:

- :class:`NoCompact` — passthrough (default for API requests)
- :class:`SlidingWindowCompact` — keep last N messages (system prompt always kept)
- :class:`TieredCompact` — 3-phase priority compaction:
    Phase 1: drop nudge/retry messages + truncate tool results
    Phase 2: drop tool results entirely
    Phase 3: drop reasoning_content + assistant text

Protected (never cut): system prompt, user input, most recent assistant turn.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

TRUNCATED_MARKER = "[omlx:tool-result-truncated]"
_TOOL_RESULT_MAX_CHARS = 200
_NUDGE_PREFIXES = (
    "You provided a text response instead of making a tool call.",
    "Tool '",
)
_PROTECTED_RECENT_COUNT = 1


def _estimate_tokens(messages: list[dict]) -> int:
    if not messages:
        return 0
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
    return max(total, 1) if total > 0 else 0


def _is_nudge(message: dict) -> bool:
    content = message.get("content")
    if not isinstance(content, str):
        return False
    return any(content.startswith(p) for p in _NUDGE_PREFIXES)


def _is_tool_result(message: dict) -> bool:
    return message.get("role") == "tool"


def _truncate_tool_content(message: dict) -> dict:
    content = message.get("content")
    if not isinstance(content, str) or len(content) <= _TOOL_RESULT_MAX_CHARS:
        return message
    truncated = message.copy()
    head = content[:_TOOL_RESULT_MAX_CHARS]
    truncated["content"] = f"{head}... {TRUNCATED_MARKER}"
    return truncated


def _strip_reasoning_and_text(message: dict) -> dict:
    stripped = message.copy()
    stripped.pop("reasoning_content", None)
    if stripped.get("role") == "assistant":
        stripped["content"] = ""
    return stripped


def _within_budget(messages: list[dict], budget_tokens: int) -> bool:
    return _estimate_tokens(messages) <= budget_tokens


class CompactStrategy(ABC):
    """Abstract base for all compaction strategies."""

    @abstractmethod
    def compact(self, messages: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
        """Compact *messages* to fit within *budget_tokens*.

        Returns ``(compacted_messages, estimated_tokens)``.
        """


class NoCompact(CompactStrategy):
    """Passthrough — no compaction applied."""

    def compact(self, messages: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
        return list(messages), _estimate_tokens(messages)


class SlidingWindowCompact(CompactStrategy):
    """Keep the last *max_messages* messages; system prompt always retained.

    *max_messages* is the total count including any system messages.
    If the windowed result still exceeds *budget_tokens*, messages are
    dropped from the front (oldest first) until within budget.
    """

    def __init__(self, max_messages: int = 20):
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages

    def compact(self, messages: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
        if not messages:
            return [], 0

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        non_sys_budget = max(0, self.max_messages - len(system_msgs))
        window = non_system[-non_sys_budget:] if non_sys_budget > 0 else []
        result = system_msgs + window

        while not _within_budget(result, budget_tokens) and len(result) > 1:
            protected_head = result[0] if result[0].get("role") == "system" else None
            if protected_head is not None and len(result) == 1:
                break
            drop_start = 1 if protected_head is not None else 0
            if drop_start >= len(result):
                break
            result = result[:1] + result[2:] if protected_head is not None else result[1:]

        return result, _estimate_tokens(result)


class TieredCompact(CompactStrategy):
    """3-phase priority-based deterministic compaction.

    Phase 1: drop nudge/retry messages + truncate long tool results
    Phase 2: drop tool results entirely
    Phase 3: drop reasoning_content + assistant text content

    Protected (never cut): system prompt, user input, most recent assistant turn.
    """

    def __init__(self, protected_recent: int = _PROTECTED_RECENT_COUNT):
        self.protected_recent = protected_recent

    def _protect_indices(self, messages: list[dict]) -> set[int]:
        protected: set[int] = set()
        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "system":
                protected.add(i)
            elif role == "user":
                protected.add(i)
        if messages:
            for i in range(max(0, len(messages) - self.protected_recent), len(messages)):
                protected.add(i)
        return protected

    def _phase1(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        protected = self._protect_indices(messages)
        result: list[dict] = []
        for i, m in enumerate(messages):
            if i in protected:
                result.append(m)
                continue
            if _is_nudge(m):
                continue
            if _is_tool_result(m):
                result.append(_truncate_tool_content(m))
            else:
                result.append(m)
        return result

    def _phase2(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        protected = self._protect_indices(messages)
        result: list[dict] = []
        for i, m in enumerate(messages):
            if i in protected:
                result.append(m)
                continue
            if _is_tool_result(m):
                continue
            result.append(m)
        return result

    def _phase3(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        protected = self._protect_indices(messages)
        result: list[dict] = []
        for i, m in enumerate(messages):
            if i in protected:
                result.append(m)
                continue
            if m.get("role") == "assistant":
                result.append(_strip_reasoning_and_text(m))
            else:
                result.append(m)
        return result

    def _trim_front(self, messages: list[dict], budget_tokens: int) -> list[dict]:
        protected = self._protect_indices(messages)
        result = list(messages)
        while len(result) > 1 and not _within_budget(result, budget_tokens):
            cut = False
            for idx in range(len(result)):
                if idx not in protected:
                    del result[idx]
                    cut = True
                    break
            if not cut:
                break
        return result

    def compact(self, messages: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
        if not messages:
            return [], 0

        if _within_budget(messages, budget_tokens):
            return list(messages), _estimate_tokens(messages)

        for phase_fn in (self._phase1, self._phase2, self._phase3):
            compacted = phase_fn(messages, budget_tokens)
            if _within_budget(compacted, budget_tokens):
                return compacted, _estimate_tokens(compacted)
            messages = compacted

        final = self._trim_front(messages, budget_tokens)
        return final, _estimate_tokens(final)


_STRATEGY_MAP = {
    "none": NoCompact,
    "sliding_window": SlidingWindowCompact,
    "tiered": TieredCompact,
}


def get_strategy(name: str, **kwargs) -> CompactStrategy:
    """Return a strategy instance by name (``"none"``, ``"sliding_window"``, ``"tiered"``)."""
    key = name.strip().lower()
    cls = _STRATEGY_MAP.get(key)
    if cls is None:
        valid = ", ".join(sorted(_STRATEGY_MAP))
        raise ValueError(f"Unknown compaction strategy '{name}'. Valid: {valid}")
    return cls(**kwargs)
