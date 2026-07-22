---
comet_change: add-forge-retry-support
role: technical-design
canonical_spec: openspec
archived-with: 2026-06-22-add-forge-retry-support
status: final
---

# Design Doc: Client-Driven Retry Support

**Change**: `add-forge-retry-support`
**Date**: 2026-06-22
**Status**: Confirmed
**Phase**: comet-design

## Overview

Extends Change A's guardrail infrastructure with error budget metadata (for client-side retry loops) and tiered context compaction (for conversation management). Both features are additive — no breaking changes to existing `x_omlx_validation` consumers.

## Architecture Decisions

### AD1: Advisory budgets (server provides defaults, client tracks counts)

Server does NOT track per-session retry counts. It provides `max_retries` and `max_tool_errors` as configurable defaults serialized in `x_omlx_validation.budget`. The client tracks actual counts and enforces the budget.

### AD2: Three compaction strategies via pluggable ABC

`CompactStrategy` ABC with `NoCompact` (default), `SlidingWindowCompact`, and `TieredCompact` (3-phase priority). All pure Python, sub-millisecond, no LLM calls.

### AD3: Nudge tier for forward-looking escalation

Add `tier: int = 0` to `Nudge`. Only step/prerequisite nudges (Change C) use tiers 1-3. Current nudges stay tier 0. Including the field now is harmless and avoids a breaking change later.

## Core Data Structures

### `omlx/api/guardrails/budget.py`

```python
@dataclass(frozen=True)
class ErrorBudget:
    max_retries: int = 3
    max_tool_errors: int = 2

    def should_retry(self, retry_count: int, tool_error_count: int) -> bool:
        return retry_count <= self.max_retries and tool_error_count <= self.max_tool_errors

    def recommended_action(self, retry_count: int, tool_error_count: int) -> str:
        return "retry" if self.should_retry(retry_count, tool_error_count) else "give_up"

    def to_dict(self) -> dict:
        return {"max_retries": self.max_retries, "max_tool_errors": self.max_tool_errors}
```

### `omlx/api/guardrails/types.py` — Nudge extension

```python
@dataclass(frozen=True)
class Nudge:
    role: Literal["user", "tool"]
    content: str
    kind: Literal["retry", "unknown_tool", "tool_arg_validation"]
    tier: int = 0  # NEW: 0=N/A, 1=polite, 2=direct, 3=aggressive
```

### `omlx/api/guardrails/types.py` — ValidationResult extension

```python
@dataclass(frozen=True)
class ValidationResult:
    checks: list[CheckResult]
    nudge: Nudge | None = None
    passed: bool = False
    budget: ErrorBudget | None = None  # NEW

    def to_dict(self) -> dict:
        result = {"passed": self.passed, "checks": [...]}
        if self.nudge:
            result["nudge"] = {..., "tier": self.nudge.tier}  # NEW: tier in serialization
        if self.budget:  # NEW
            result["budget"] = self.budget.to_dict()
        return result
```

### `omlx/context/compaction.py`

```python
class CompactStrategy(ABC):
    @abstractmethod
    def compact(self, messages: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
        """Returns (compacted_messages, phase_reached). phase 0=no compaction."""

class NoCompact(CompactStrategy):
    def compact(self, messages, budget_tokens):
        return messages, 0

class SlidingWindowCompact(CompactStrategy):
    def __init__(self, keep_recent: int = 10):
        self.keep_recent = keep_recent
    def compact(self, messages, budget_tokens):
        if len(messages) <= self.keep_recent + 2:
            return messages, 0
        return messages[:2] + messages[-self.keep_recent:], 1

class TieredCompact(CompactStrategy):
    def __init__(self, keep_recent=2, thresholds=(0.75, 0.85, 0.95)):
        self.keep_recent = keep_recent
        self.phase_thresholds = thresholds
    # 3-phase implementation: Phase 1 drops nudges+truncates tool results,
    # Phase 2 drops tool results, Phase 3 drops reasoning+text.
    # Protected: messages[0] (system) and messages[1] (user input).
```

## Response Extension Format (extended)

```json
{
  "x_omlx_validation": {
    "passed": false,
    "checks": [...],
    "nudge": {"role": "tool", "content": "...", "kind": "unknown_tool", "tier": 0},
    "budget": {
      "max_retries": 3,
      "max_tool_errors": 2
    }
  }
}
```

The `budget` object tells the client the server's recommended limits. The client tracks `retry_count` and `tool_error_count` locally, using `budget.max_retries` and `budget.max_tool_errors` as bounds.

## Settings Extension

```python
# ForgeGuardrailsSettings additions
max_retries: int = 3
max_tool_errors: int = 2
compaction_strategy: str = "none"  # "none", "sliding_window", "tiered"
```

## Implementation Sequence

**Critical path**: Budget module → Types extension → Validator wiring → Settings → Compaction → Docs

Groups 1-4 (budget + types + wiring + settings) are tightly coupled and sequential.
Group 5 (compaction) is independent — can develop in parallel.
Group 6 (docs) is last.

## Testing Strategy

Same pattern as Change A: unit tests per module, integration tests for wiring. ~30 new tests expected.

## References

- Forge source: `../forge/src/forge/guardrails/error_tracker.py`, `../forge/src/forge/context/strategies.py`
- Change A: `openspec/changes/archive/2026-06-22-add-forge-guardrails/`
- Integration plan: `docs/forge-integration-plan.md` (Phase 3+4)
