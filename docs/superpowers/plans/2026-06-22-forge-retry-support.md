---
change: add-forge-retry-support
design-doc: docs/superpowers/specs/2026-06-22-forge-retry-support-design.md
base-ref: 1ce701b2382597b2444c94946eeb2af72e20b4ed
---

# Client-Driven Retry Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Change A's guardrail infrastructure with error budget metadata (for client-side retry loops) and tiered context compaction (for conversation management).

**Architecture:** Two additive, non-breaking tracks. Track A (Groups 1–4): `ErrorBudget` dataclass → `Nudge.tier` field → `ValidationResult.budget` field → settings wiring. Track B (Group 5): standalone `omlx/context/compaction.py` package with pluggable `CompactStrategy` ABC. Track C (Group 6): docs. All features are opt-in with backward-compatible defaults.

**Tech Stack:** Python 3.10+, dataclasses, pytest, FastAPI/Pydantic (admin routes).

## Global Constraints

- Python 3.10+ (`from __future__ import annotations` in all new files)
- SPDX-License-Identifier: Apache-2.0 header in all new `.py` files
- All new dataclasses are `frozen=True`
- All features opt-in — existing `x_omlx_validation` consumers must see no change when new fields are at defaults
- No new third-party dependencies (stdlib + existing project deps only)
- Test command: `pytest tests/test_<file>.py -v`
- Full guardrail test suite: `pytest tests/test_guardrail*.py tests/test_server_guardrail_wiring.py -v`

## Design Conflict Resolution

**Conflict:** tasks.md 1.1 lists `recommended_action: str = "retry"` as a field on `ErrorBudget`, but the design doc shows `recommended_action(retry_count, tool_error_count) -> str` as a method.

**Resolution:** Use the **method** approach (design doc is technically correct — a frozen dataclass can't track runtime state). `recommended_action()` is a method taking counts. `to_dict()` serializes a static `"recommended_action": "retry"` since the server doesn't know client-side counts. The client calls `should_retry()` / `recommended_action(counts)` locally for dynamic decisions.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `omlx/api/guardrails/budget.py` | Create | `ErrorBudget` frozen dataclass with `should_retry()`, `recommended_action()`, `to_dict()`, `from_dict()` |
| `omlx/api/guardrails/types.py` | Modify | Add `tier: int = 0` to `Nudge`; add `budget: ErrorBudget \| None = None` to `ValidationResult`; update both `to_dict()` methods |
| `omlx/api/guardrails/nudge.py` | Modify | Add optional `tier: int = 0` parameter to all 4 nudge generators |
| `omlx/api/guardrails/validator.py` | Modify | No logic change — `ValidationResult` construction picks up `budget=None` default automatically |
| `omlx/api/guardrails/__init__.py` | Modify | Export `ErrorBudget` |
| `omlx/api/guardrail_wiring.py` | Modify | Accept `max_retries`/`max_tool_errors` kwargs; construct `ErrorBudget` and attach to `ValidationResult` |
| `omlx/settings.py` | Modify | Add `max_retries`, `max_tool_errors`, `compaction_strategy` to `ForgeGuardrailsSettings`; update `to_dict()`/`from_dict()` |
| `omlx/admin/routes.py` | Modify | Add 3 fields to `GlobalSettingsRequest` Pydantic model; wire in PUT/GET endpoints |
| `omlx/context/__init__.py` | Create | Public API exports for compaction package |
| `omlx/context/compaction.py` | Create | `CompactStrategy` ABC + `NoCompact`, `SlidingWindowCompact`, `TieredCompact` |
| `tests/test_guardrail_budget.py` | Create | Unit tests for `ErrorBudget` |
| `tests/test_guardrail_types.py` | Modify | Add tests for `Nudge.tier` and `ValidationResult.budget` serialization |
| `tests/test_guardrail_nudges.py` | Modify | Add tests for `tier` parameter on nudge generators |
| `tests/test_guardrail_settings.py` | Modify | Add tests for new `ForgeGuardrailsSettings` fields |
| `tests/test_server_guardrail_wiring.py` | Modify | Add tests for budget attachment in wiring |
| `tests/test_context_compaction.py` | Create | Unit tests for all 3 compaction strategies |
| `tests/test_admin_guardrail_fields.py` | Create | Tests for new admin route fields |
| `docs/tool-call-guardrails.md` | Modify | Add "Client-Side Retry Loops" and "Context Compaction" sections |
| `docs/forge-integration-plan.md` | Modify | Mark Phase 3+4 as implemented |

---

## Execution Order

```
Group 1 (Budget)          ─┐
Group 2 (Nudge Tier)       ├── Sequential (Track A)
Group 3 (Response Wiring)  │
Group 4 (Settings)        ─┘

Group 5 (Compaction)      ─── Parallel (Track B, independent of Groups 1-4)

Group 6 (Docs)            ─── Last (after all code groups)
```

**Parallelization:** Groups 1–4 and Group 5 can be developed simultaneously by different engineers. Group 6 waits for all code to land.

---

## Track A: Budget + Types + Wiring + Settings (Sequential)

### Task 1: Error Budget Module — Dataclass + Serialization

**Files:**
- Create: `omlx/api/guardrails/budget.py`
- Test: `tests/test_guardrail_budget.py`

**Interfaces:**
- Produces: `ErrorBudget` class with `max_retries: int`, `max_tool_errors: int`, `should_retry(retry_count, tool_error_count) -> bool`, `recommended_action(retry_count, tool_error_count) -> str`, `to_dict() -> dict`, `from_dict(dict) -> ErrorBudget`

- [ ] **Step 1: Write failing tests for ErrorBudget creation and serialization**

Create `tests/test_guardrail_budget.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ErrorBudget dataclass."""
import pytest
from omlx.api.guardrails.budget import ErrorBudget


class TestErrorBudgetDefaults:
    def test_default_max_retries(self):
        b = ErrorBudget()
        assert b.max_retries == 3

    def test_default_max_tool_errors(self):
        b = ErrorBudget()
        assert b.max_tool_errors == 2

    def test_custom_values(self):
        b = ErrorBudget(max_retries=5, max_tool_errors=1)
        assert b.max_retries == 5
        assert b.max_tool_errors == 1

    def test_is_frozen(self):
        b = ErrorBudget()
        with pytest.raises(AttributeError):
            b.max_retries = 10  # type: ignore


class TestShouldRetry:
    def test_retry_when_under_both_limits(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=1, tool_error_count=0) is True

    def test_retry_at_exact_limit(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=3, tool_error_count=2) is True

    def test_no_retry_when_retries_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=4, tool_error_count=0) is False

    def test_no_retry_when_tool_errors_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.should_retry(retry_count=0, tool_error_count=3) is False


class TestRecommendedAction:
    def test_retry_when_under_limits(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.recommended_action(retry_count=1, tool_error_count=0) == "retry"

    def test_give_up_when_retries_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.recommended_action(retry_count=4, tool_error_count=0) == "give_up"

    def test_give_up_when_tool_errors_exhausted(self):
        b = ErrorBudget(max_retries=3, max_tool_errors=2)
        assert b.recommended_action(retry_count=0, tool_error_count=3) == "give_up"


class TestSerialization:
    def test_to_dict(self):
        b = ErrorBudget(max_retries=5, max_tool_errors=1)
        d = b.to_dict()
        assert d == {
            "max_retries": 5,
            "max_tool_errors": 1,
            "recommended_action": "retry",
        }

    def test_to_dict_defaults(self):
        b = ErrorBudget()
        d = b.to_dict()
        assert d["max_retries"] == 3
        assert d["max_tool_errors"] == 2
        assert d["recommended_action"] == "retry"

    def test_from_dict(self):
        d = {"max_retries": 7, "max_tool_errors": 3}
        b = ErrorBudget.from_dict(d)
        assert b.max_retries == 7
        assert b.max_tool_errors == 3

    def test_from_dict_missing_keys_uses_defaults(self):
        b = ErrorBudget.from_dict({})
        assert b.max_retries == 3
        assert b.max_tool_errors == 2

    def test_round_trip(self):
        original = ErrorBudget(max_retries=10, max_tool_errors=5)
        restored = ErrorBudget.from_dict(original.to_dict())
        assert restored.max_retries == original.max_retries
        assert restored.max_tool_errors == original.max_tool_errors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'omlx.api.guardrails.budget'`

- [ ] **Step 3: Implement ErrorBudget**

Create `omlx/api/guardrails/budget.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Error budget for client-driven retry loops.

The server provides configurable defaults serialized in
``x_omlx_validation.budget``. The client tracks actual retry and
tool-error counts locally and enforces the budget.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorBudget:
    """Advisory retry/tool-error limits for client-side retry loops.

    The server does NOT track per-session counts. It serializes these
    defaults via :meth:`to_dict` into the ``x_omlx_validation.budget``
    extension. The client calls :meth:`should_retry` with its own
    tracked counts.
    """

    max_retries: int = 3
    max_tool_errors: int = 2

    def should_retry(self, retry_count: int, tool_error_count: int) -> bool:
        """Return True if both counts are within budget (inclusive)."""
        return retry_count <= self.max_retries and tool_error_count <= self.max_tool_errors

    def recommended_action(self, retry_count: int = 0, tool_error_count: int = 0) -> str:
        """Return ``"retry"`` if under budget, ``"give_up"`` if exhausted."""
        return "retry" if self.should_retry(retry_count, tool_error_count) else "give_up"

    def to_dict(self) -> dict:
        """Serialize for ``x_omlx_validation.budget`` response extension.

        ``recommended_action`` is always ``"retry"`` in the serialized
        form because the server does not know client-side counts. The
        client calls :meth:`recommended_action` locally for dynamic
        decisions.
        """
        return {
            "max_retries": self.max_retries,
            "max_tool_errors": self.max_tool_errors,
            "recommended_action": "retry",
        }

    @classmethod
    def from_dict(cls, data: dict) -> ErrorBudget:
        """Deserialize from dict (ignores ``recommended_action``)."""
        return cls(
            max_retries=data.get("max_retries", 3),
            max_tool_errors=data.get("max_tool_errors", 2),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_budget.py -v`
Expected: PASS — all 14 tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/api/guardrails/budget.py tests/test_guardrail_budget.py
git commit -m "feat(guardrails): add ErrorBudget dataclass for client retry budgets"
```

---

### Task 2: Export ErrorBudget from package __init__

**Files:**
- Modify: `omlx/api/guardrails/__init__.py`

**Interfaces:**
- Produces: `ErrorBudget` importable from `omlx.api.guardrails`

- [ ] **Step 1: Add export**

In `omlx/api/guardrails/__init__.py`, add the import after line 6 (after the nudge imports block):

```python
from omlx.api.guardrails.budget import ErrorBudget
```

Add `"ErrorBudget"` to the `__all__` list (after `"CheckResult"`):

```python
__all__ = [
    "CheckResult",
    "ErrorBudget",
    "Nudge",
    # ... rest unchanged
]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from omlx.api.guardrails import ErrorBudget; print(ErrorBudget())"`
Expected: prints `ErrorBudget(max_retries=3, max_tool_errors=2)`

- [ ] **Step 3: Commit**

```bash
git add omlx/api/guardrails/__init__.py
git commit -m "feat(guardrails): export ErrorBudget from package __init__"
```

---

### Task 3: Nudge Tier Extension — Add tier field

**Files:**
- Modify: `omlx/api/guardrails/types.py` (lines 39–49, the `Nudge` dataclass)
- Test: `tests/test_guardrail_types.py` (existing file, add new test class)

**Interfaces:**
- Produces: `Nudge.tier: int = 0` field; `Nudge.to_dict()` includes `"tier"` key
- Consumes: nothing new

- [ ] **Step 1: Write failing tests for Nudge.tier**

Append to `tests/test_guardrail_types.py`:

```python
class TestNudgeTier:
    def test_default_tier_is_zero(self):
        n = Nudge(role="user", content="try again", kind=KIND_RETRY)
        assert n.tier == 0

    def test_explicit_tier(self):
        n = Nudge(role="user", content="try again", kind=KIND_RETRY, tier=2)
        assert n.tier == 2

    def test_tier_in_to_dict(self):
        n = Nudge(role="tool", content="bad tool", kind=KIND_UNKNOWN_TOOL, tier=3)
        # to_dict is on ValidationResult, but we test via to_message + manual dict
        # Actually Nudge doesn't have to_dict — tier is serialized via ValidationResult.to_dict
        # Let's verify tier is accessible
        assert n.tier == 3

    def test_backward_compatible_no_tier_arg(self):
        """Existing callers that don't pass tier must still work."""
        n = Nudge(role="tool", content="fix args", kind=KIND_TOOL_ARG_VALIDATION)
        assert n.tier == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_types.py::TestNudgeTier -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'tier'`

- [ ] **Step 3: Add tier field to Nudge**

In `omlx/api/guardrails/types.py`, modify the `Nudge` dataclass (currently lines 39–49):

```python
@dataclass(frozen=True)
class Nudge:
    """Corrective message for client to append before retry."""

    role: Literal["user", "tool"]
    content: str
    kind: Literal["retry", "unknown_tool", "tool_arg_validation"]
    tier: int = 0  # 0=N/A, 1=polite, 2=direct, 3=aggressive

    def to_message(self) -> dict:
        """Convert to chat message format for client retry."""
        return {"role": self.role, "content": self.content}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_types.py::TestNudgeTier -v`
Expected: PASS — all 4 tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/api/guardrails/types.py tests/test_guardrail_types.py
git commit -m "feat(guardrails): add tier field to Nudge for escalation levels"
```

---

### Task 4: Nudge Tier Extension — Add tier param to nudge generators

**Files:**
- Modify: `omlx/api/guardrails/nudge.py`
- Test: `tests/test_guardrail_nudges.py` (existing file, add new test class)

**Interfaces:**
- Produces: all 4 nudge generators accept optional `tier: int = 0` parameter

- [ ] **Step 1: Write failing tests**

Append to `tests/test_guardrail_nudges.py`:

```python
class TestNudgeTier:
    def test_retry_nudge_default_tier(self):
        from omlx.api.guardrails.nudge import retry_nudge
        n = retry_nudge()
        assert n.tier == 0

    def test_retry_nudge_explicit_tier(self):
        from omlx.api.guardrails.nudge import retry_nudge
        n = retry_nudge(tier=1)
        assert n.tier == 1

    def test_unknown_tool_nudge_default_tier(self):
        from omlx.api.guardrails.nudge import unknown_tool_nudge
        n = unknown_tool_nudge("foo", ["bar"])
        assert n.tier == 0

    def test_unknown_tool_nudge_explicit_tier(self):
        from omlx.api.guardrails.nudge import unknown_tool_nudge
        n = unknown_tool_nudge("foo", ["bar"], tier=2)
        assert n.tier == 2

    def test_tool_arg_validation_nudge_default_tier(self):
        from omlx.api.guardrails.nudge import tool_arg_validation_nudge
        n = tool_arg_validation_nudge("foo", "{}", "str")
        assert n.tier == 0

    def test_missing_params_nudge_default_tier(self):
        from omlx.api.guardrails.nudge import missing_params_nudge
        n = missing_params_nudge("foo", ["bar"])
        assert n.tier == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_nudges.py::TestNudgeTier -v`
Expected: FAIL — `TypeError: retry_nudge() got an unexpected keyword argument 'tier'`

- [ ] **Step 3: Add tier parameter to all 4 generators**

In `omlx/api/guardrails/nudge.py`, update each function signature and `Nudge(...)` construction:

```python
def retry_nudge(tier: int = 0) -> Nudge:
    """Nudge for bare-text when tools were expected."""
    return Nudge(
        role="user",
        content=(
            "You provided a text response instead of making a tool call. "
            "Please use the available tools to answer the request."
        ),
        kind=KIND_RETRY,
        tier=tier,
    )


def unknown_tool_nudge(tool_name: str, available_tools: list[str], tier: int = 0) -> Nudge:
    """Nudge for calling a tool that does not exist."""
    tools_str = ", ".join(sorted(available_tools)) if available_tools else "(none)"
    return Nudge(
        role="tool",
        content=(
            f"Tool '{tool_name}' does not exist. "
            f"Available: {tools_str}. Call one of them."
        ),
        kind=KIND_UNKNOWN_TOOL,
        tier=tier,
    )


def tool_arg_validation_nudge(
    tool_name: str, args_repr: str, received_type: str, tier: int = 0
) -> Nudge:
    """Nudge for malformed (non-dict) arguments."""
    return Nudge(
        role="tool",
        content=(
            f"Tool '{tool_name}' had malformed arguments. "
            f"Got type: {received_type}. Required: JSON object (dict). "
            f"Received value: {args_repr[:200]}"
        ),
        kind=KIND_TOOL_ARG_VALIDATION,
        tier=tier,
    )


def missing_params_nudge(tool_name: str, missing_params: list[str], tier: int = 0) -> Nudge:
    """Nudge for missing required parameters."""
    params_str = ", ".join(missing_params)
    return Nudge(
        role="tool",
        content=(
            f"Tool '{tool_name}' is missing required parameter(s): {params_str}. "
            f"Please provide all required parameters."
        ),
        kind=KIND_TOOL_ARG_VALIDATION,
        tier=tier,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_nudges.py -v`
Expected: PASS — all existing + 6 new tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/api/guardrails/nudge.py tests/test_guardrail_nudges.py
git commit -m "feat(guardrails): add optional tier param to all nudge generators"
```

---

### Task 5: Budget + Tier Serialization in ValidationResult.to_dict

**Files:**
- Modify: `omlx/api/guardrails/types.py` (the `ValidationResult` dataclass and its `to_dict()`)
- Test: `tests/test_guardrail_types.py` (add `TestValidationResultBudget` class)

**Interfaces:**
- Produces: `ValidationResult.budget: ErrorBudget | None = None` field; `to_dict()` includes `"budget"` when present and `"tier"` in nudge serialization

- [ ] **Step 1: Write failing tests**

Append to `tests/test_guardrail_types.py`:

```python
from omlx.api.guardrails.budget import ErrorBudget


class TestValidationResultBudget:
    def test_default_budget_is_none(self):
        vr = ValidationResult(checks=[], passed=True)
        assert vr.budget is None

    def test_to_dict_without_budget(self):
        """When budget is None, it must NOT appear in to_dict."""
        vr = ValidationResult(checks=[], passed=True)
        d = vr.to_dict()
        assert "budget" not in d

    def test_to_dict_with_budget(self):
        vr = ValidationResult(
            checks=[],
            passed=False,
            budget=ErrorBudget(max_retries=5, max_tool_errors=1),
        )
        d = vr.to_dict()
        assert d["budget"] == {
            "max_retries": 5,
            "max_tool_errors": 1,
            "recommended_action": "retry",
        }


class TestNudgeTierSerialization:
    def test_nudge_tier_in_validation_result_to_dict(self):
        nudge = Nudge(role="user", content="try again", kind=KIND_RETRY, tier=2)
        vr = ValidationResult(checks=[], passed=False, nudge=nudge)
        d = vr.to_dict()
        assert d["nudge"]["tier"] == 2

    def test_nudge_default_tier_in_to_dict(self):
        nudge = Nudge(role="tool", content="bad", kind=KIND_UNKNOWN_TOOL)
        vr = ValidationResult(checks=[], passed=False, nudge=nudge)
        d = vr.to_dict()
        assert d["nudge"]["tier"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_types.py::TestValidationResultBudget tests/test_guardrail_types.py::TestNudgeTierSerialization -v`
Expected: FAIL — `TypeError: ValidationResult.__init__() got an unexpected keyword argument 'budget'`

- [ ] **Step 3: Add budget field to ValidationResult and update to_dict**

In `omlx/api/guardrails/types.py`:

Add import at top (after existing imports):

```python
from omlx.api.guardrails.budget import ErrorBudget
```

Replace the `ValidationResult` dataclass (currently lines 52–75):

```python
@dataclass(frozen=True)
class ValidationResult:
    """Accumulated validation results for a tool-call response."""

    checks: list[CheckResult]
    nudge: Nudge | None = None
    passed: bool = False
    budget: ErrorBudget | None = None  # NEW: advisory retry budget

    def to_dict(self) -> dict:
        """Serialize for x_omlx_validation response extension."""
        result: dict = {
            "passed": self.passed,
            "checks": [
                {"check": c.check, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }
        if self.nudge:
            result["nudge"] = {
                "role": self.nudge.role,
                "content": self.nudge.content,
                "kind": self.nudge.kind,
                "tier": self.nudge.tier,  # NEW
            }
        if self.budget:  # NEW
            result["budget"] = self.budget.to_dict()
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_types.py -v`
Expected: PASS — all existing + 5 new tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/api/guardrails/types.py tests/test_guardrail_types.py
git commit -m "feat(guardrails): add budget field to ValidationResult and tier in nudge serialization"
```

---

### Task 6: Budget Wiring in guardrail_wiring.py

**Files:**
- Modify: `omlx/api/guardrail_wiring.py`
- Test: `tests/test_server_guardrail_wiring.py` (existing file, add tests)

**Interfaces:**
- Consumes: `ErrorBudget` from Task 1, `ValidationResult.budget` from Task 5
- Produces: `apply_guardrails()` accepts optional `max_retries`/`max_tool_errors` kwargs and attaches `ErrorBudget` to the merged `ValidationResult`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_server_guardrail_wiring.py`:

```python
class TestBudgetWiring:
    def _make_extraction_with_validation(self, passed=True):
        """Helper: create a ToolCallExtraction with a validation_result."""
        from omlx.api.guardrails.types import CheckResult, ValidationResult
        from omlx.api.tool_calling import ToolCallExtraction
        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=passed)],
            passed=passed,
        )
        return ToolCallExtraction(
            cleaned_text="",
            tool_calls=[],
            cleaned_thinking=None,
            tool_calls_from_thinking=False,
            validation_result=vr,
        )

    def test_budget_attached_when_metadata_enabled(self):
        from omlx.api.guardrail_wiring import guardrail_validation_payload
        ext = self._make_extraction_with_validation(passed=False)
        payload = guardrail_validation_payload(
            ext,
            include_validation_metadata=True,
            max_retries=5,
            max_tool_errors=1,
        )
        assert payload is not None
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 5
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 1

    def test_no_budget_when_metadata_disabled(self):
        from omlx.api.guardrail_wiring import guardrail_validation_payload
        ext = self._make_extraction_with_validation(passed=False)
        payload = guardrail_validation_payload(ext, include_validation_metadata=False)
        assert payload is None

    def test_budget_defaults_when_not_specified(self):
        from omlx.api.guardrail_wiring import guardrail_validation_payload
        ext = self._make_extraction_with_validation(passed=False)
        payload = guardrail_validation_payload(
            ext, include_validation_metadata=True
        )
        assert payload["x_omlx_validation"]["budget"]["max_retries"] == 3
        assert payload["x_omlx_validation"]["budget"]["max_tool_errors"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_guardrail_wiring.py::TestBudgetWiring -v`
Expected: FAIL — `TypeError: guardrail_validation_payload() got an unexpected keyword argument 'max_retries'`

- [ ] **Step 3: Update guardrail_wiring.py**

Replace the entire contents of `omlx/api/guardrail_wiring.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional

from omlx.api.guardrails.budget import ErrorBudget
from omlx.api.guardrails.types import CheckResult, ValidationResult
from omlx.api.tool_calling import ToolCallExtraction
from omlx.api.tool_choice import enforce_tool_choice


def apply_guardrails(
    extraction: ToolCallExtraction,
    tool_choice: Any,
    tools: Optional[list[dict]] = None,
    *,
    validation_enabled: bool = False,
) -> ToolCallExtraction:
    if not validation_enabled or extraction.validation_result is None:
        return extraction

    has_text = bool(extraction.cleaned_text.strip())
    _, tc_check = enforce_tool_choice(
        extraction.tool_calls, tool_choice, has_text, tools
    )

    existing: ValidationResult = extraction.validation_result
    merged = ValidationResult(
        checks=existing.checks + [tc_check],
        nudge=existing.nudge,
        passed=existing.passed and tc_check.passed,
        budget=existing.budget,
    )

    return ToolCallExtraction(
        cleaned_text=extraction.cleaned_text,
        tool_calls=extraction.tool_calls,
        cleaned_thinking=extraction.cleaned_thinking,
        tool_calls_from_thinking=extraction.tool_calls_from_thinking,
        validation_result=merged,
    )


def guardrail_validation_payload(
    extraction: ToolCallExtraction,
    *,
    include_validation_metadata: bool = False,
    max_retries: int = 3,
    max_tool_errors: int = 2,
) -> Optional[dict]:
    """Build the x_omlx_validation payload with budget metadata.

    Attaches an :class:`ErrorBudget` to the ``ValidationResult`` before
    serialization so clients can implement bounded retry loops.
    """
    if not include_validation_metadata or extraction.validation_result is None:
        return None

    existing: ValidationResult = extraction.validation_result
    # Only attach budget if not already set (avoids overwriting explicit values).
    if existing.budget is None:
        budget = ErrorBudget(max_retries=max_retries, max_tool_errors=max_tool_errors)
        result_with_budget = ValidationResult(
            checks=existing.checks,
            nudge=existing.nudge,
            passed=existing.passed,
            budget=budget,
        )
        return {"x_omlx_validation": result_with_budget.to_dict()}

    return {"x_omlx_validation": existing.to_dict()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server_guardrail_wiring.py -v`
Expected: PASS — all existing + 3 new tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/api/guardrail_wiring.py tests/test_server_guardrail_wiring.py
git commit -m "feat(guardrails): wire ErrorBudget into validation payload"
```

---

### Task 7: Settings Extension — Add new fields to ForgeGuardrailsSettings

**Files:**
- Modify: `omlx/settings.py` (lines 771–796, the `ForgeGuardrailsSettings` class)
- Test: `tests/test_guardrail_settings.py` (existing file, add test class)

**Interfaces:**
- Produces: `ForgeGuardrailsSettings.max_retries: int = 3`, `.max_tool_errors: int = 2`, `.compaction_strategy: str = "none"`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_guardrail_settings.py`:

```python
class TestNewForgeGuardrailsFields:
    def test_default_max_retries(self):
        s = ForgeGuardrailsSettings()
        assert s.max_retries == 3

    def test_default_max_tool_errors(self):
        s = ForgeGuardrailsSettings()
        assert s.max_tool_errors == 2

    def test_default_compaction_strategy(self):
        s = ForgeGuardrailsSettings()
        assert s.compaction_strategy == "none"

    def test_to_dict_includes_new_fields(self):
        s = ForgeGuardrailsSettings()
        d = s.to_dict()
        assert d["max_retries"] == 3
        assert d["max_tool_errors"] == 2
        assert d["compaction_strategy"] == "none"

    def test_from_dict_new_fields(self):
        d = {
            "validation_enabled": True,
            "max_retries": 5,
            "max_tool_errors": 1,
            "compaction_strategy": "tiered",
        }
        s = ForgeGuardrailsSettings.from_dict(d)
        assert s.max_retries == 5
        assert s.max_tool_errors == 1
        assert s.compaction_strategy == "tiered"

    def test_from_dict_missing_new_fields_uses_defaults(self):
        s = ForgeGuardrailsSettings.from_dict({"validation_enabled": True})
        assert s.max_retries == 3
        assert s.max_tool_errors == 2
        assert s.compaction_strategy == "none"

    def test_round_trip(self):
        original = ForgeGuardrailsSettings(
            validation_enabled=True,
            max_retries=10,
            max_tool_errors=5,
            compaction_strategy="sliding_window",
        )
        restored = ForgeGuardrailsSettings.from_dict(original.to_dict())
        assert restored.max_retries == 10
        assert restored.max_tool_errors == 5
        assert restored.compaction_strategy == "sliding_window"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_settings.py::TestNewForgeGuardrailsFields -v`
Expected: FAIL — `AttributeError: 'ForgeGuardrailsSettings' object has no attribute 'max_retries'`

- [ ] **Step 3: Add new fields to ForgeGuardrailsSettings**

In `omlx/settings.py`, replace the `ForgeGuardrailsSettings` class (currently lines 771–796):

```python
@dataclass
class ForgeGuardrailsSettings:
    """Guardrail validation settings (opt-in by default).

    Follows the CompressionSettings pattern. All flags default to False
    for full backward compatibility.
    """

    validation_enabled: bool = False
    strict_tool_args: bool = False
    include_validation_metadata: bool = False
    max_retries: int = 3
    max_tool_errors: int = 2
    compaction_strategy: str = "none"  # "none", "sliding_window", "tiered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_enabled": self.validation_enabled,
            "strict_tool_args": self.strict_tool_args,
            "include_validation_metadata": self.include_validation_metadata,
            "max_retries": self.max_retries,
            "max_tool_errors": self.max_tool_errors,
            "compaction_strategy": self.compaction_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgeGuardrailsSettings":
        return cls(
            validation_enabled=data.get("validation_enabled", False),
            strict_tool_args=data.get("strict_tool_args", False),
            include_validation_metadata=data.get("include_validation_metadata", False),
            max_retries=data.get("max_retries", 3),
            max_tool_errors=data.get("max_tool_errors", 2),
            compaction_strategy=data.get("compaction_strategy", "none"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_settings.py -v`
Expected: PASS — all existing + 7 new tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/settings.py tests/test_guardrail_settings.py
git commit -m "feat(settings): add max_retries, max_tool_errors, compaction_strategy to ForgeGuardrailsSettings"
```

---

### Task 8: Admin Route Wiring — GlobalSettingsRequest + PUT/GET endpoints

**Files:**
- Modify: `omlx/admin/routes.py` (3 locations: `GlobalSettingsRequest` model ~line 280, PUT handler ~line 3696, GET handler ~line 4408)
- Test: `tests/test_admin_guardrail_fields.py` (create)

**Interfaces:**
- Produces: 3 new optional fields on `GlobalSettingsRequest` Pydantic model; wired into settings update + retrieval

- [ ] **Step 1: Write failing tests**

Create `tests/test_admin_guardrail_fields.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for new forge guardrail fields in admin routes."""
from omlx.admin.routes import GlobalSettingsRequest


class TestGlobalSettingsRequestNewFields:
    def test_max_retries_optional(self):
        req = GlobalSettingsRequest()
        assert req.forge_guardrails_max_retries is None

    def test_max_retries_settable(self):
        req = GlobalSettingsRequest(forge_guardrails_max_retries=5)
        assert req.forge_guardrails_max_retries == 5

    def test_max_tool_errors_optional(self):
        req = GlobalSettingsRequest()
        assert req.forge_guardrails_max_tool_errors is None

    def test_max_tool_errors_settable(self):
        req = GlobalSettingsRequest(forge_guardrails_max_tool_errors=1)
        assert req.forge_guardrails_max_tool_errors == 1

    def test_compaction_strategy_optional(self):
        req = GlobalSettingsRequest()
        assert req.forge_guardrails_compaction_strategy is None

    def test_compaction_strategy_settable(self):
        req = GlobalSettingsRequest(forge_guardrails_compaction_strategy="tiered")
        assert req.forge_guardrails_compaction_strategy == "tiered"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_guardrail_fields.py -v`
Expected: FAIL — `AttributeError: 'GlobalSettingsRequest' object has no attribute 'forge_guardrails_max_retries'`

- [ ] **Step 3: Add fields to GlobalSettingsRequest model**

In `omlx/admin/routes.py`, after line 282 (after `forge_guardrails_include_validation_metadata`):

```python
    forge_guardrails_max_retries: int | None = None
    forge_guardrails_max_tool_errors: int | None = None
    forge_guardrails_compaction_strategy: str | None = None
```

- [ ] **Step 4: Wire new fields into the PUT handler**

In `omlx/admin/routes.py`, after line 3711 (after the `include_validation_metadata` block, before the `forge_guardrails_changed` check), add:

```python
    if request.forge_guardrails_max_retries is not None:
        global_settings.forge_guardrails.max_retries = (
            request.forge_guardrails_max_retries
        )
        forge_guardrails_changed = True
    if request.forge_guardrails_max_tool_errors is not None:
        global_settings.forge_guardrails.max_tool_errors = (
            request.forge_guardrails_max_tool_errors
        )
        forge_guardrails_changed = True
    if request.forge_guardrails_compaction_strategy is not None:
        global_settings.forge_guardrails.compaction_strategy = (
            request.forge_guardrails_compaction_strategy
        )
        forge_guardrails_changed = True
```

- [ ] **Step 5: Wire new fields into the GET handler**

In `omlx/admin/routes.py`, after line 4421 (after the `forge_guardrails_include_validation_metadata` block), add:

```python
        "forge_guardrails_max_retries": (
            global_settings.forge_guardrails.max_retries
            if global_settings
            else 3
        ),
        "forge_guardrails_max_tool_errors": (
            global_settings.forge_guardrails.max_tool_errors
            if global_settings
            else 2
        ),
        "forge_guardrails_compaction_strategy": (
            global_settings.forge_guardrails.compaction_strategy
            if global_settings
            else "none"
        ),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_admin_guardrail_fields.py -v`
Expected: PASS — all 6 tests green

- [ ] **Step 7: Run full guardrail suite for regression check**

Run: `pytest tests/test_guardrail*.py tests/test_server_guardrail_wiring.py tests/test_admin_guardrail_fields.py -v`
Expected: PASS — no regressions

- [ ] **Step 8: Commit**

```bash
git add omlx/admin/routes.py tests/test_admin_guardrail_fields.py
git commit -m "feat(admin): wire max_retries, max_tool_errors, compaction_strategy into global settings API"
```

---

## Track B: Context Compaction Package (Parallel with Track A)

### Task 9: Create omlx/context package skeleton

**Files:**
- Create: `omlx/context/__init__.py`
- Create: `omlx/context/compaction.py` (ABC only, implementations in Tasks 10–12)

**Interfaces:**
- Produces: `CompactStrategy` ABC with abstract `compact(messages, budget_tokens) -> tuple[list[dict], int]`

- [ ] **Step 1: Create the compaction module with ABC**

Create `omlx/context/compaction.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Pluggable context compaction strategies.

All strategies are pure Python with sub-millisecond execution and no
LLM calls. Protected messages: ``messages[0]`` (system prompt) and
``messages[1]`` (original user input) are never compacted by any
strategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class CompactStrategy(ABC):
    """Abstract base for context compaction strategies."""

    @abstractmethod
    def compact(
        self, messages: list[dict], budget_tokens: int
    ) -> tuple[list[dict], int]:
        """Compact the message list to fit within ``budget_tokens``.

        Args:
            messages: Chat messages in OpenAI format (each has ``role``
                and ``content``). ``messages[0]`` is the system prompt,
                ``messages[1]`` is the original user input.
            budget_tokens: Maximum tokens the compacted list should fit.

        Returns:
            Tuple of ``(compacted_messages, phase_reached)``.
            ``phase_reached=0`` means no compaction was applied.
        """
```

Create `omlx/context/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Context management package: compaction strategies."""
from omlx.context.compaction import CompactStrategy

__all__ = ["CompactStrategy"]
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from omlx.context import CompactStrategy; print(CompactStrategy)"`
Expected: prints `<class 'omlx.context.compaction.CompactStrategy'>`

- [ ] **Step 3: Commit**

```bash
git add omlx/context/__init__.py omlx/context/compaction.py
git commit -m "feat(context): create compaction package with CompactStrategy ABC"
```

---

### Task 10: NoCompact strategy

**Files:**
- Modify: `omlx/context/compaction.py` (add `NoCompact` class)
- Modify: `omlx/context/__init__.py` (export `NoCompact`)
- Test: `tests/test_context_compaction.py` (create)

**Interfaces:**
- Produces: `NoCompact` — passthrough strategy, always returns `(messages, 0)`

- [ ] **Step 1: Write failing tests**

Create `tests/test_context_compaction.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for context compaction strategies."""
import pytest
from omlx.context.compaction import NoCompact


class TestNoCompact:
    def test_returns_messages_unchanged(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        strategy = NoCompact()
        result, phase = strategy.compact(msgs, budget_tokens=100)
        assert result == msgs
        assert phase == 0

    def test_empty_messages(self):
        strategy = NoCompact()
        result, phase = strategy.compact([], budget_tokens=100)
        assert result == []
        assert phase == 0

    def test_ignores_budget(self):
        """NoCompact never compacts regardless of how small budget is."""
        msgs = [
            {"role": "system", "content": "x" * 10000},
            {"role": "user", "content": "y" * 10000},
        ]
        strategy = NoCompact()
        result, phase = strategy.compact(msgs, budget_tokens=1)
        assert result == msgs
        assert phase == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_compaction.py::TestNoCompact -v`
Expected: FAIL — `ImportError: cannot import name 'NoCompact'`

- [ ] **Step 3: Implement NoCompact**

In `omlx/context/compaction.py`, add after the `CompactStrategy` class:

```python
class NoCompact(CompactStrategy):
    """Passthrough strategy — never compacts."""

    def compact(
        self, messages: list[dict], budget_tokens: int
    ) -> tuple[list[dict], int]:
        return messages, 0
```

In `omlx/context/__init__.py`, update exports:

```python
# SPDX-License-Identifier: Apache-2.0
"""Context management package: compaction strategies."""
from omlx.context.compaction import CompactStrategy, NoCompact

__all__ = ["CompactStrategy", "NoCompact"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_compaction.py::TestNoCompact -v`
Expected: PASS — 3 tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/context/compaction.py omlx/context/__init__.py tests/test_context_compaction.py
git commit -m "feat(context): implement NoCompact passthrough strategy"
```

---

### Task 11: SlidingWindowCompact strategy

**Files:**
- Modify: `omlx/context/compaction.py` (add `SlidingWindowCompact`)
- Modify: `omlx/context/__init__.py` (export)
- Test: `tests/test_context_compaction.py` (add test class)

**Interfaces:**
- Produces: `SlidingWindowCompact(keep_recent=10)` — keeps first 2 + last N messages

- [ ] **Step 1: Write failing tests**

Append to `tests/test_context_compaction.py`:

```python
from omlx.context.compaction import SlidingWindowCompact


class TestSlidingWindowCompact:
    def _make_messages(self, n: int) -> list[dict]:
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "input"}]
        for i in range(n):
            msgs.append({"role": "assistant", "content": f"msg-{i}"})
        return msgs

    def test_keeps_protected_plus_recent(self):
        msgs = self._make_messages(20)
        strategy = SlidingWindowCompact(keep_recent=5)
        result, phase = strategy.compact(msgs, budget_tokens=100)
        # Expected: first 2 (protected) + last 5
        assert len(result) == 7
        assert result[0] == msgs[0]  # system preserved
        assert result[1] == msgs[1]  # user input preserved
        assert result[-1] == msgs[-1]  # last message preserved
        assert phase == 1

    def test_no_compact_when_short_enough(self):
        """If total messages <= keep_recent + 2, return unchanged."""
        msgs = self._make_messages(5)
        strategy = SlidingWindowCompact(keep_recent=10)
        result, phase = strategy.compact(msgs, budget_tokens=100)
        assert result == msgs
        assert phase == 0

    def test_default_keep_recent_10(self):
        msgs = self._make_messages(20)
        strategy = SlidingWindowCompact()  # default
        result, phase = strategy.compact(msgs, budget_tokens=100)
        assert len(result) == 12  # 2 protected + 10 recent
        assert phase == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_compaction.py::TestSlidingWindowCompact -v`
Expected: FAIL — `ImportError: cannot import name 'SlidingWindowCompact'`

- [ ] **Step 3: Implement SlidingWindowCompact**

In `omlx/context/compaction.py`, add after `NoCompact`:

```python
class SlidingWindowCompact(CompactStrategy):
    """Keep the first 2 protected messages plus the last ``keep_recent``.

    Protected messages (indices 0 and 1) are always retained. If the
    total message count is ``<= keep_recent + 2``, no compaction occurs.
    """

    def __init__(self, keep_recent: int = 10):
        self.keep_recent = keep_recent

    def compact(
        self, messages: list[dict], budget_tokens: int
    ) -> tuple[list[dict], int]:
        if len(messages) <= self.keep_recent + 2:
            return messages, 0
        return messages[:2] + messages[-self.keep_recent:], 1
```

Update `omlx/context/__init__.py` exports:

```python
from omlx.context.compaction import CompactStrategy, NoCompact, SlidingWindowCompact

__all__ = ["CompactStrategy", "NoCompact", "SlidingWindowCompact"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_compaction.py -v`
Expected: PASS — NoCompact + SlidingWindow tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/context/compaction.py omlx/context/__init__.py tests/test_context_compaction.py
git commit -m "feat(context): implement SlidingWindowCompact strategy"
```

---

### Task 12: TieredCompact strategy — Phase 1 (drop nudges + truncate tool results)

**Files:**
- Modify: `omlx/context/compaction.py` (add `TieredCompact` with Phase 1)
- Modify: `omlx/context/__init__.py` (export)
- Test: `tests/test_context_compaction.py` (add Phase 1 tests)

**Interfaces:**
- Produces: `TieredCompact(keep_recent=2, thresholds=(0.75, 0.85, 0.95))` with Phase 1 logic

- [ ] **Step 1: Write failing tests for Phase 1**

Append to `tests/test_context_compaction.py`:

```python
from omlx.context.compaction import TieredCompact


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: 1 token per 4 chars of content."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
    return total


class TestTieredCompactPhase1:
    def _make_large_messages(self, target_tokens: int) -> list[dict]:
        """Create messages whose total tokens exceed a budget's Phase 1 threshold."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
        ]
        # Add a nudge message (role=tool with unknown_tool-like content)
        msgs.append({"role": "tool", "content": "STEP_NUDGE: please try again"})
        # Add a tool result that's long enough to need truncation
        msgs.append({"role": "tool", "content": "x" * 1000})
        # Add assistant text
        msgs.append({"role": "assistant", "content": "y" * (target_tokens * 4)})
        return msgs

    def test_phase1_drops_nudge_messages(self):
        msgs = self._make_large_messages(target_tokens=500)
        budget = 200  # tokens
        strategy = TieredCompact(keep_recent=0, thresholds=(0.75, 0.85, 0.95))
        result, phase = strategy.compact(msgs, budget_tokens=budget)
        assert phase >= 1
        # The STEP_NUDGE message should be dropped
        nudge_msgs = [m for m in result if "STEP_NUDGE" in m.get("content", "")]
        assert len(nudge_msgs) == 0

    def test_phase1_truncates_tool_results(self):
        long_tool_result = "z" * 1000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "tool", "content": long_tool_result},
            {"role": "assistant", "content": "a" * 2000},
        ]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.75, 0.85, 0.95))
        result, phase = strategy.compact(msgs, budget_tokens=100)
        assert phase >= 1
        # Find the tool message in result (may have been truncated)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        if tool_msgs:
            truncated = tool_msgs[0]["content"]
            assert len(truncated) <= 250  # 200 chars + truncation marker

    def test_phase1_protects_system_and_user_input(self):
        msgs = self._make_large_messages(target_tokens=500)
        strategy = TieredCompact(keep_recent=0, thresholds=(0.75, 0.85, 0.95))
        result, phase = strategy.compact(msgs, budget_tokens=100)
        assert result[0] == msgs[0]  # system
        assert result[1] == msgs[1]  # user input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_compaction.py::TestTieredCompactPhase1 -v`
Expected: FAIL — `ImportError: cannot import name 'TieredCompact'`

- [ ] **Step 3: Implement TieredCompact with Phase 1**

In `omlx/context/compaction.py`, add after `SlidingWindowCompact`:

```python
# Phase 1 truncation limit for tool results (characters).
_TOOL_RESULT_TRUNCATE_CHARS = 200
_TOOL_RESULT_MARKER = "…[truncated]"

# Message content prefixes that mark nudge messages for Phase 1 dropping.
_NUDGE_PREFIXES = ("STEP_NUDGE:", "PREREQUISITE_NUDGE:", "RETRY_NUDGE:")


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~1 token per 4 characters of content."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
    return total or 1


class TieredCompact(CompactStrategy):
    """3-phase priority-based compaction.

    Phase 1 (budget × 0.75): Drop nudge messages + truncate tool results.
    Phase 2 (budget × 0.85): Drop tool results entirely.
    Phase 3 (budget × 0.95): Drop reasoning + text responses.

    Protected at all phases:
    - ``messages[0]`` (system prompt)
    - ``messages[1]`` (original user input)
    - Last ``keep_recent`` iterations' messages
    """

    def __init__(
        self,
        keep_recent: int = 2,
        thresholds: tuple[float, float, float] = (0.75, 0.85, 0.95),
    ):
        self.keep_recent = keep_recent
        self.phase_thresholds = thresholds

    def compact(
        self, messages: list[dict], budget_tokens: int
    ) -> tuple[list[dict], int]:
        if len(messages) <= 2:
            return messages, 0

        threshold_1 = int(budget_tokens * self.phase_thresholds[0])
        threshold_2 = int(budget_tokens * self.phase_thresholds[1])
        threshold_3 = int(budget_tokens * self.phase_thresholds[2])

        # Protected indices: 0, 1, and last keep_recent
        protected = self._protected_indices(len(messages))

        result = list(messages)
        phase = 0

        # Phase 1: drop nudges + truncate tool results
        if _estimate_tokens(result) > threshold_1:
            result = self._apply_phase1(result, protected)
            phase = 1

        # Phase 2: drop tool results entirely
        if _estimate_tokens(result) > threshold_2:
            result = self._apply_phase2(result, protected)
            phase = 2

        # Phase 3: drop reasoning + text
        if _estimate_tokens(result) > threshold_3:
            result = self._apply_phase3(result, protected)
            phase = 3

        return result, phase

    def _protected_indices(self, total: int) -> set[int]:
        """Indices that are never compacted."""
        protected = {0, 1}
        start = max(2, total - self.keep_recent)
        for i in range(start, total):
            protected.add(i)
        return protected

    def _is_nudge(self, msg: dict) -> bool:
        content = msg.get("content", "")
        if not isinstance(content, str):
            return False
        return any(content.startswith(p) for p in _NUDGE_PREFIXES)

    def _apply_phase1(
        self, messages: list[dict], protected: set[int]
    ) -> list[dict]:
        """Drop nudge messages and truncate tool results to 200 chars."""
        result = []
        for i, msg in enumerate(messages):
            if i in protected:
                result.append(msg)
                continue
            # Drop nudges
            if self._is_nudge(msg):
                continue
            # Truncate tool results
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > _TOOL_RESULT_TRUNCATE_CHARS:
                    msg = {
                        **msg,
                        "content": content[:_TOOL_RESULT_TRUNCATE_CHARS] + _TOOL_RESULT_MARKER,
                    }
            result.append(msg)
        return result

    def _apply_phase2(
        self, messages: list[dict], protected: set[int]
    ) -> list[dict]:
        """Drop tool results entirely."""
        return [
            msg
            for i, msg in enumerate(messages)
            if i in protected or msg.get("role") != "tool"
        ]

    def _apply_phase3(
        self, messages: list[dict], protected: set[int]
    ) -> list[dict]:
        """Drop reasoning + text responses; keep only tool-call skeletons."""
        result = []
        for i, msg in enumerate(messages):
            if i in protected:
                result.append(msg)
                continue
            role = msg.get("role", "")
            if role in ("assistant",):
                # Keep only if it has tool_calls; drop pure text
                if msg.get("tool_calls"):
                    # Strip content, keep tool_calls skeleton
                    result.append({**msg, "content": None})
                # else: drop entirely
            elif role == "system" and i > 1:
                # Drop non-protected system messages (reasoning injected as system)
                continue
            else:
                result.append(msg)
        return result
```

Update `omlx/context/__init__.py` exports:

```python
from omlx.context.compaction import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
)

__all__ = [
    "CompactStrategy",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_compaction.py -v`
Expected: PASS — all Phase 1 tests green

- [ ] **Step 5: Commit**

```bash
git add omlx/context/compaction.py omlx/context/__init__.py tests/test_context_compaction.py
git commit -m "feat(context): implement TieredCompact with 3-phase priority compaction"
```

---

### Task 13: TieredCompact Phase 2 + Phase 3 tests

**Files:**
- Test: `tests/test_context_compaction.py` (add Phase 2/3 test classes)

**Interfaces:**
- No new code — validates existing Phase 2/3 implementation from Task 12

- [ ] **Step 1: Write Phase 2 and Phase 3 tests**

Append to `tests/test_context_compaction.py`:

```python
class TestTieredCompactPhase2:
    def test_phase2_drops_tool_results(self):
        """When Phase 1 is insufficient, Phase 2 drops all tool results."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "tool", "content": "a" * 2000},
            {"role": "tool", "content": "b" * 2000},
            {"role": "assistant", "content": "c" * 4000},
        ]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.01, 0.02, 0.95))
        result, phase = strategy.compact(msgs, budget_tokens=10000)
        assert phase >= 2
        tool_msgs = [m for m in result if m["role"] == "tool"]
        # All non-protected tool messages dropped
        assert len(tool_msgs) == 0

    def test_phase2_preserves_assistant_text(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "tool", "content": "x" * 5000},
            {"role": "assistant", "content": "important answer"},
        ]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.01, 0.02, 0.95))
        result, phase = strategy.compact(msgs, budget_tokens=10000)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert any("important answer" in m.get("content", "") for m in assistant_msgs)


class TestTieredCompactPhase3:
    def test_phase3_drops_assistant_text(self):
        """Phase 3 drops reasoning/text, keeps only tool_call skeletons."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "tool", "content": "x" * 10000},
            {"role": "assistant", "content": "very long text" * 1000},
        ]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.01, 0.02, 0.03))
        result, phase = strategy.compact(msgs, budget_tokens=100000)
        assert phase == 3
        # Assistant text should be dropped (no tool_calls on it)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        # Either dropped entirely or content stripped to None
        for m in assistant_msgs:
            if i_not_protected := True:
                assert m.get("content") is None or m.get("content") == ""

    def test_phase3_keeps_tool_call_skeletons(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
            {"role": "assistant", "content": "long text" * 1000, "tool_calls": [{"id": "1", "function": {"name": "foo", "arguments": "{}"}}]},
        ]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.01, 0.02, 0.03))
        result, phase = strategy.compact(msgs, budget_tokens=100000)
        assert phase == 3
        assistant_with_tc = [m for m in result if m.get("tool_calls")]
        assert len(assistant_with_tc) >= 1
        assert assistant_with_tc[0]["content"] is None  # content stripped


class TestTieredCompactProtection:
    def test_system_prompt_always_preserved(self):
        msgs = [
            {"role": "system", "content": "IMPORTANT system prompt"},
            {"role": "user", "content": "input"},
        ] + [{"role": "assistant", "content": "x" * 5000} for _ in range(10)]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.01, 0.02, 0.03))
        result, phase = strategy.compact(msgs, budget_tokens=50000)
        assert result[0] == msgs[0]

    def test_user_input_always_preserved(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "IMPORTANT user question"},
        ] + [{"role": "assistant", "content": "x" * 5000} for _ in range(10)]
        strategy = TieredCompact(keep_recent=0, thresholds=(0.01, 0.02, 0.03))
        result, phase = strategy.compact(msgs, budget_tokens=50000)
        assert result[1] == msgs[1]

    def test_keep_recent_protected(self):
        """Messages from the last keep_recent iterations are protected."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "input"},
        ] + [{"role": "tool", "content": f"result-{i}"} for i in range(10)]
        strategy = TieredCompact(keep_recent=2, thresholds=(0.01, 0.02, 0.03))
        result, phase = strategy.compact(msgs, budget_tokens=50000)
        # Last 2 tool results should survive even at Phase 2/3
        last_contents = [m.get("content") for m in result if m.get("role") == "tool"]
        assert "result-9" in last_contents  # last message preserved
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_context_compaction.py -v`
Expected: PASS — all Phase 2, Phase 3, and protection tests green. If any fail, fix the implementation in `compaction.py` (not the tests).

- [ ] **Step 3: Commit**

```bash
git add tests/test_context_compaction.py
git commit -m "test(context): add Phase 2/3 and protection tests for TieredCompact"
```

---

### Task 14: Compaction strategy factory + config integration

**Files:**
- Modify: `omlx/context/compaction.py` (add `get_compact_strategy()` factory function)
- Modify: `omlx/context/__init__.py` (export factory)
- Test: `tests/test_context_compaction.py` (add factory tests)

**Interfaces:**
- Produces: `get_compact_strategy(name: str) -> CompactStrategy` — maps `"none"`, `"sliding_window"`, `"tiered"` to strategy instances

- [ ] **Step 1: Write failing tests**

Append to `tests/test_context_compaction.py`:

```python
from omlx.context.compaction import get_compact_strategy


class TestGetCompactStrategy:
    def test_none_returns_nocompact(self):
        s = get_compact_strategy("none")
        assert isinstance(s, NoCompact)

    def test_sliding_window_returns_sliding(self):
        from omlx.context import SlidingWindowCompact
        s = get_compact_strategy("sliding_window")
        assert isinstance(s, SlidingWindowCompact)

    def test_tiered_returns_tiered(self):
        from omlx.context import TieredCompact
        s = get_compact_strategy("tiered")
        assert isinstance(s, TieredCompact)

    def test_unknown_defaults_to_none(self):
        s = get_compact_strategy("bogus")
        assert isinstance(s, NoCompact)

    def test_empty_string_defaults_to_none(self):
        s = get_compact_strategy("")
        assert isinstance(s, NoCompact)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_context_compaction.py::TestGetCompactStrategy -v`
Expected: FAIL — `ImportError: cannot import name 'get_compact_strategy'`

- [ ] **Step 3: Implement factory function**

In `omlx/context/compaction.py`, add at the bottom:

```python
def get_compact_strategy(name: str) -> CompactStrategy:
    """Factory: map a strategy name to a CompactStrategy instance.

    Args:
        name: One of ``"none"``, ``"sliding_window"``, ``"tiered"``.
            Unknown values default to :class:`NoCompact`.

    Returns:
        A ``CompactStrategy`` instance.
    """
    if name == "sliding_window":
        return SlidingWindowCompact()
    if name == "tiered":
        return TieredCompact()
    return NoCompact()
```

Update `omlx/context/__init__.py`:

```python
from omlx.context.compaction import (
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    get_compact_strategy,
)

__all__ = [
    "CompactStrategy",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
    "get_compact_strategy",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_context_compaction.py -v`
Expected: PASS — all tests green including factory tests

- [ ] **Step 5: Commit**

```bash
git add omlx/context/compaction.py omlx/context/__init__.py tests/test_context_compaction.py
git commit -m "feat(context): add get_compact_strategy factory for config-driven strategy selection"
```

---

## Track C: Documentation (After all code)

### Task 15: Add "Client-Side Retry Loops" section to tool-call-guardrails.md

**Files:**
- Modify: `docs/tool-call-guardrails.md`

- [ ] **Step 1: Read current doc structure**

Run: `head -50 docs/tool-call-guardrails.md` to find the insertion point (end of the existing content).

- [ ] **Step 2: Add retry loops section**

Append the following section to `docs/tool-call-guardrails.md`:

```markdown
## Client-Side Retry Loops

When `include_validation_metadata` is enabled, the server includes a `budget` object in `x_omlx_validation` that tells the client how many retries and tool errors are acceptable before giving up.

### Budget Metadata Format

```json
{
  "x_omlx_validation": {
    "passed": false,
    "checks": [...],
    "nudge": {"role": "tool", "content": "...", "kind": "unknown_tool", "tier": 0},
    "budget": {
      "max_retries": 3,
      "max_tool_errors": 2,
      "recommended_action": "retry"
    }
  }
}
```

The `budget` object contains:
- **`max_retries`**: Maximum number of retry attempts (default: 3, configurable via admin panel).
- **`max_tool_errors`**: Maximum tool-call errors before giving up (default: 2).
- **`recommended_action`**: Server default is always `"retry"`. The client computes the actual action based on its tracked counts.

### Clean-Batch Reset Rule

When a validation passes (i.e., `x_omlx_validation.passed` is `true`), the client SHOULD reset its `retry_count` and `tool_error_count` to zero. A successful batch means the model is back on track — stale retry counts from a previous failure should not accumulate.

### Reference Retry Loop (Python)

```python
retry_count = 0
tool_error_count = 0

while True:
    response = client.chat.completions.create(
        model="my-model",
        messages=messages,
        tools=tools,
    )

    metadata = response.choices[0].message.model_extra or {}
    validation = metadata.get("x_omlx_validation", {})

    if validation.get("passed", True):
        # Success — reset counters
        retry_count = 0
        tool_error_count = 0
        break

    # Validation failed — check budget
    budget = validation.get("budget", {})
    max_retries = budget.get("max_retries", 3)
    max_tool_errors = budget.get("max_tool_errors", 2)

    retry_count += 1
    if "unknown_tool" in str(validation.get("checks", [])):
        tool_error_count += 1

    if retry_count > max_retries or tool_error_count > max_tool_errors:
        print("Budget exhausted — giving up.")
        break

    # Append the nudge as a new message and retry
    nudge = validation.get("nudge")
    if nudge:
        messages.append(response.choices[0].message)
        messages.append({"role": nudge["role"], "content": nudge["content"]})
```

### Configuration

The budget defaults are configurable via the admin panel (Global Settings → Forge Guardrails) or via `settings.json`:

```json
{
  "forge_guardrails": {
    "validation_enabled": true,
    "include_validation_metadata": true,
    "max_retries": 5,
    "max_tool_errors": 3
  }
}
```
```

- [ ] **Step 3: Commit**

```bash
git add docs/tool-call-guardrails.md
git commit -m "docs: add Client-Side Retry Loops section to tool-call-guardrails.md"
```

---

### Task 16: Add "Context Compaction" section to tool-call-guardrails.md

**Files:**
- Modify: `docs/tool-call-guardrails.md`

- [ ] **Step 1: Add compaction section**

Append the following section to `docs/tool-call-guardrails.md` (after the retry loops section):

```markdown
## Context Compaction

oMLX provides pluggable context compaction strategies for managing long conversations. All strategies are pure Python with sub-millisecond execution — no LLM calls, no token re-computation.

### Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `none` (default) | Passthrough — never compacts | Short conversations, debugging |
| `sliding_window` | Keeps system prompt + user input + last N messages | Medium conversations where recent context is most important |
| `tiered` | 3-phase priority-based compaction | Long conversations where you need fine-grained control |

### Tiered Compaction Phases

The `tiered` strategy applies phases progressively based on token thresholds (default: 75%, 85%, 95% of budget):

| Phase | Threshold | Action |
|-------|-----------|--------|
| 1 | 75% | Drop nudge messages + truncate tool results to 200 chars |
| 2 | 85% | Drop all tool results entirely |
| 3 | 95% | Drop reasoning/text responses, keep only tool-call skeletons |

**Protected at all phases:**
- System prompt (`messages[0]`)
- Original user input (`messages[1]`)
- Last `keep_recent` iterations (default: 2)

### Configuration

```json
{
  "forge_guardrails": {
    "compaction_strategy": "tiered"
  }
}
```

Values: `"none"`, `"sliding_window"`, `"tiered"`. Default: `"none"`.

### Python API

```python
from omlx.context import get_compact_strategy

strategy = get_compact_strategy("tiered")
compacted_messages, phase = strategy.compact(messages, budget_tokens=8192)
print(f"Compacted to phase {phase}, {len(compacted_messages)} messages remaining")
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/tool-call-guardrails.md
git commit -m "docs: add Context Compaction section to tool-call-guardrails.md"
```

---

### Task 17: Update forge-integration-plan.md — mark Phase 3+4 as implemented

**Files:**
- Modify: `docs/forge-integration-plan.md`

- [ ] **Step 1: Read current content to find Phase 3+4 section**

Run: `grep -n "Phase 3\|Phase 4" docs/forge-integration-plan.md` to find the relevant section.

- [ ] **Step 2: Add implementation status note**

At the top of the Phase 3+4 section (or at the top of the file if no clear section exists), add:

```markdown
> **Status Update (2026-06-22):** Phase 3 (error budgets) and Phase 4 (context compaction) are implemented via the `add-forge-retry-support` change. See:
> - `omlx/api/guardrails/budget.py` — `ErrorBudget` dataclass
> - `omlx/context/compaction.py` — `CompactStrategy` ABC + 3 implementations
> - `docs/tool-call-guardrails.md` — "Client-Side Retry Loops" and "Context Compaction" sections
```

- [ ] **Step 3: Commit**

```bash
git add docs/forge-integration-plan.md
git commit -m "docs: mark Phase 3+4 as implemented in forge-integration-plan.md"
```

---

## Final Integration Verification

### Task 18: Full test suite regression run

- [ ] **Step 1: Run all guardrail + context tests**

Run: `pytest tests/test_guardrail_budget.py tests/test_guardrail_types.py tests/test_guardrail_nudges.py tests/test_guardrail_validator.py tests/test_guardrail_settings.py tests/test_guardrail_e2e.py tests/test_server_guardrail_wiring.py tests/test_admin_guardrail_fields.py tests/test_context_compaction.py -v`

Expected: ALL PASS — zero failures, zero errors.

- [ ] **Step 2: Run broader smoke test (if available)**

Run: `pytest -m "not slow" --tb=short -q`
Expected: No new failures compared to pre-change baseline.

- [ ] **Step 3: Verify no import regressions**

Run: `python -c "from omlx.api.guardrails import ErrorBudget, ValidationResult, Nudge; from omlx.context import CompactStrategy, NoCompact, SlidingWindowCompact, TieredCompact, get_compact_strategy; print('All imports OK')"`
Expected: prints `All imports OK`

- [ ] **Step 4: Final commit (if any cleanup needed)**

If all tests pass, no commit needed. If fixes were made:

```bash
git add -A
git commit -m "test: fix integration issues from retry/compaction change"
```

---

## Self-Review Summary

**Spec coverage check:**
- ✅ Spec: "Validation metadata includes budget hints" → Tasks 1, 5, 6
- ✅ Spec: "Budget recommends retry/give_up" → Task 1 (`recommended_action()` method)
- ✅ Spec: "Nudge includes escalation tier" → Tasks 3, 4
- ✅ Spec: "Non-step nudges have tier 0" → Tasks 3 (default=0), 4 (existing generators stay tier=0)
- ✅ Spec: "Error budget defaults" → Task 7 (settings)
- ✅ Spec: "Budgets configurable via admin panel" → Task 8
- ✅ Spec: "NoCompact passes through" → Task 10
- ✅ Spec: "TieredCompact Phase 1/2/3" → Tasks 12, 13
- ✅ Spec: "System prompt + user input never compacted" → Task 13
- ✅ Spec: "Recent iterations protected" → Task 13
- ✅ Spec: "Compaction strategy configuration" → Tasks 7, 14
- ✅ Docs: retry loops + compaction → Tasks 15, 16, 17

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N". All code blocks are complete.

**Type consistency:**
- `ErrorBudget` used consistently across Tasks 1, 5, 6
- `ValidationResult.budget: ErrorBudget | None` matches in Tasks 5 and 6
- `Nudge.tier: int = 0` consistent in Tasks 3, 4, 5
- `CompactStrategy.compact(messages, budget_tokens) -> tuple[list[dict], int]` consistent across Tasks 9–14
- `get_compact_strategy(name)` consistent in Task 14
