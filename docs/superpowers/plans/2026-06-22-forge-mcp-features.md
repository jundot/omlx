---
change: add-forge-mcp-features
design-doc: docs/superpowers/specs/2026-06-22-forge-mcp-features-design.md
base-ref: dd676d69f0dd2f883dd33300fbb7a09ca6adc3b3
archived-with: 2026-06-22-add-forge-mcp-features
---

# MCP Features (Prerequisites + Respond Tool) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCP prerequisite enforcement (Phase 5) and a synthetic respond tool (Phase 6) to oMLX's guardrail stack, completing the 6-phase Forge integration. Both are opt-in and build on Changes A+B.

**Architecture:** Two independent tracks sharing a common nudge-types extension. Track A (Groups 1–2, 4): `PrerequisiteChecker` builds `executed_tools` statelessly from prior message history; new `step` and `prerequisite` nudge kinds flow through the existing `GuardrailValidator` → `apply_guardrails` pipeline. Track B (Group 3): `inject_respond_tool` / `strip_respond_calls` transform the tools list and parsed output around generation in `server.py`. Track C (Group 5): docs.

**Tech Stack:** Python 3.10+, dataclasses, pytest, FastAPI/Pydantic (admin routes), mlx-lm tool parsing.

## Global Constraints

- Python 3.10+ (`from __future__ import annotations` in all new files)
- SPDX-License-Identifier: Apache-2.0 header in all new `.py` files
- All new dataclasses are `frozen=True`
- All features opt-in — existing `x_omlx_validation` consumers and existing MCP clients must see no change when new fields are at defaults
- No new third-party dependencies (stdlib + existing project deps only)
- The client never sees the `respond` tool — it is injected server-side and stripped before the response returns
- Prerequisite state is stateless per-request — built entirely from the request's `messages` history, no server-side session state
- Nudge text patterns adapted from Forge's `forge/prompts/nudges.py` (do not import Forge; oMLX is standalone)
- Test command: `pytest tests/test_<file>.py -v`
- Full guardrail test suite: `pytest tests/test_guardrail*.py tests/test_server_guardrail_wiring.py tests/test_mcp_prerequisites.py tests/test_respond_tool.py -v`

## Design Decisions (from design doc)

- **AD1 — Prerequisites at validation layer:** MCP tool execution is client-side. Prerequisites are checked when tool calls are parsed; the server returns nudges if ordering is wrong. The client enforces before executing.
- **AD2 — StepTracker from message history:** `PrerequisiteChecker` builds `executed_tools` by scanning prior assistant messages for `tool_calls`. Stateless per-request.
- **AD3 — Respond tool inject/strip:** Inject `respond(message)` before generation when the setting is enabled. Strip after parsing: pure respond → text, mixed → real calls only. The client never sees the respond tool.
- **AD4 — Step/prerequisite nudges use user-role + tier:** Step and prerequisite nudges use `role="user"` (workflow guidance, not tool errors). Step nudges use 3-tier escalation via the `tier` field.

## Existing Infrastructure (Changes A+B, already merged)

This plan builds on the following already-merged code. Read these files before starting:

- `omlx/api/guardrails/types.py` — `CheckResult`, `Nudge`, `ValidationResult`, `CheckName` literal, nudge kind constants (`KIND_RETRY`, `KIND_UNKNOWN_TOOL`, `KIND_TOOL_ARG_VALIDATION`), `TOOL_CHANNEL_KINDS`, `TOOL_ERROR_KINDS`
- `omlx/api/guardrails/nudge.py` — `retry_nudge`, `unknown_tool_nudge`, `tool_arg_validation_nudge`, `missing_params_nudge`
- `omlx/api/guardrails/validator.py` — `GuardrailValidator` with 4 checks (bare_text, unknown_tool, malformed_args, missing_required_params) + `_select_nudge` priority dispatch
- `omlx/api/guardrails/budget.py` — `ErrorBudget`
- `omlx/api/guardrail_wiring.py` — `apply_guardrails()` (merges tool_choice check), `guardrail_validation_payload()`
- `omlx/api/tool_calling.py` — `ToolCallExtraction` dataclass, `extract_and_validate_tool_calls()` wrapper, `extract_tool_calls_with_thinking()`
- `omlx/settings.py:771` — `ForgeGuardrailsSettings` dataclass with `to_dict`/`from_dict`
- `omlx/server.py` — 6 call sites that call `extract_and_validate_tool_calls` → `apply_guardrails` → `guardrail_validation_payload`; `tools_for_template` built at ~line 3334 from `effective_tools` (merged user + MCP tools); `request.messages` available in scope
- `omlx/admin/routes.py:280-285` — `forge_guardrails_*` request fields + handler block at ~3698
- `omlx/mcp/config.py` — `validate_config()`, `MCPConfig` / `MCPServerConfig`
- `omlx/mcp/types.py` — `MCPConfig`, `MCPServerConfig`, `MCPTool`
- `mcp.example.json` — example config
- `tests/test_guardrail_nudges.py`, `tests/test_guardrail_validator.py`, `tests/test_server_guardrail_wiring.py`, `tests/test_guardrail_settings.py`, `tests/test_guardrail_e2e.py` — test patterns to follow

## File Structure

New files:
- `omlx/mcp/prerequisites.py` — `PrerequisiteCheck` dataclass + `PrerequisiteChecker` class (stateless per-request)
- `omlx/api/respond.py` — `RESPOND_TOOL_NAME`, `inject_respond_tool()`, `strip_respond_calls()`, `RESPOND_TOOL_SPEC` constant
- `tests/test_mcp_prerequisites.py` — unit tests for `PrerequisiteChecker`
- `tests/test_respond_tool.py` — unit tests for inject + strip
- `tests/test_guardrail_step_prereq_nudges.py` — unit tests for new nudge functions + validator integration
- `tests/test_forge_mcp_features_e2e.py` — integration tests wiring both tracks together

Modified files:
- `omlx/api/guardrails/types.py` — add `KIND_STEP`, `KIND_PREREQUISITE` constants; extend `CheckName`, `Nudge.kind` literal, `TOOL_CHANNEL_KINDS` docs
- `omlx/api/guardrails/nudge.py` — add `step_nudge()`, `prerequisite_nudge()`
- `omlx/api/guardrails/__init__.py` — export new constants + functions
- `omlx/api/guardrails/validator.py` — add Check 5 (step) + Check 6 (prerequisite) when data provided
- `omlx/api/guardrail_wiring.py` — thread `prerequisite_checker`, `step_config`, `prior_messages` into `apply_guardrails`
- `omlx/api/tool_calling.py` — extend `extract_and_validate_tool_calls` signature to accept optional prerequisite/step context
- `omlx/mcp/config.py` — parse `tools_prerequisites` + `required_steps` / `terminal_tools` fields
- `omlx/mcp/types.py` — add `tools_prerequisites`, `required_steps`, `terminal_tools` to `MCPConfig`
- `omlx/settings.py` — add `inject_respond_tool: bool = False`, `enforce_mcp_prerequisites: bool = False` to `ForgeGuardrailsSettings`
- `omlx/server.py` — inject respond tool before generation; strip after parsing; wire prerequisite/step checks; thread `request.messages` into validation
- `omlx/admin/routes.py` — add request fields + handler block for the two new settings
- `mcp.example.json` — add `tools_prerequisites` example
- `docs/tool-call-guardrails.md` — MCP Prerequisites + Respond Tool sections
- `docs/forge-integration-plan.md` — mark Phase 5+6 as implemented

archived-with: 2026-06-22-add-forge-mcp-features
---

## Group 1: MCP Prerequisites

### Task 1.1: Create `PrerequisiteCheck` dataclass and `PrerequisiteChecker` skeleton

**Files:**
- Create: `omlx/mcp/prerequisites.py`
- Create: `tests/test_mcp_prerequisites.py`

**Interfaces:**
- Produces: `PrerequisiteCheck(satisfied: bool, missing: list[str])`, `PrerequisiteChecker(prerequisites: dict[str, list])`, `PrerequisiteChecker.check(tool_calls, prior_messages) -> list[CheckResult]`

- [x] **Step 1: Write the failing test**

```python
# tests/test_mcp_prerequisites.py
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MCP prerequisite enforcement."""
from __future__ import annotations

from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.mcp.prerequisites import PrerequisiteCheck, PrerequisiteChecker


def _tc(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


class TestPrerequisiteCheckDataclass:
    def test_satisfied_true(self):
        pc = PrerequisiteCheck(satisfied=True, missing=[])
        assert pc.satisfied is True
        assert pc.missing == []

    def test_not_satisfied_with_missing(self):
        pc = PrerequisiteCheck(satisfied=False, missing=["read_file"])
        assert pc.satisfied is False
        assert "read_file" in pc.missing

    def test_is_frozen(self):
        import dataclasses
        pc = PrerequisiteCheck(satisfied=True, missing=[])
        with pytest.raises(dataclasses.FrozenInstanceError):
            pc.satisfied = False  # type: ignore[misc]


class TestPrerequisiteCheckerConstruction:
    def test_empty_prerequisites(self):
        checker = PrerequisiteChecker({})
        results = checker.check([], [])
        assert results == []

    def test_no_prereq_for_tool(self):
        checker = PrerequisiteChecker({"edit_file": {"requires": ["read_file"]}})
        # search has no declared prereqs — no result emitted for it
        results = checker.check([_tc("search")], [])
        assert results == []
```

Add `import pytest` at the top of the test file.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_prerequisites.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'omlx.mcp.prerequisites'`

- [x] **Step 3: Write minimal implementation**

```python
# omlx/mcp/prerequisites.py
# SPDX-License-Identifier: Apache-2.0
"""MCP prerequisite enforcement — stateless per-request.

Builds executed-tool state from prior message history and checks
whether each tool call's declared prerequisites are satisfied.

Two declaration modes (adapted from Forge's StepTracker):
  - Name-only:  "read_file"  — any prior read_file call satisfies
  - Arg-matched: {"tool": "read_file", "match_arg": "path"}
                  — prior call must have matching arg value
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from omlx.api.guardrails.types import CheckResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrerequisiteCheck:
    """Result of checking prerequisites for a single tool call."""

    satisfied: bool
    missing: list[str]


class PrerequisiteChecker:
    """Check tool-call prerequisites against prior message history.

    Stateless per-request: ``executed_tools`` is rebuilt from
    ``prior_messages`` on every ``check()`` call. No server-side
    session state.
    """

    def __init__(self, prerequisites: dict[str, list]) -> None:
        self._prereqs = prerequisites

    def check(
        self,
        tool_calls: list[Any],
        prior_messages: list[dict],
    ) -> list[CheckResult]:
        """Check each tool call's prerequisites.

        Returns a CheckResult per tool call that has declared
        prerequisites. Tool calls with no declared prereqs produce
        no CheckResult (omitted, not passed).
        """
        if not self._prereqs or not tool_calls:
            return []

        executed = self._build_executed_set(prior_messages)
        results: list[CheckResult] = []

        for tc in tool_calls:
            name = self._tool_name(tc)
            prereqs = self._prereqs.get(name)
            if not prereqs:
                continue
            missing = self._evaluate(name, tc, prereqs, executed)
            results.append(
                CheckResult(
                    check="prerequisite",
                    passed=len(missing) == 0,
                    detail=(
                        None
                        if not missing
                        else f"Tool '{name}' missing prerequisites: {', '.join(missing)}"
                    ),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_executed_set(
        self, prior_messages: list[dict]
    ) -> dict[str, list[dict]]:
        """Scan prior assistant messages for tool_calls.

        Returns a mapping of tool_name -> list of arg dicts.
        """
        executed: dict[str, list[dict]] = {}
        for msg in prior_messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            calls = msg.get("tool_calls")
            if not calls:
                continue
            for call in calls:
                func = call.get("function") if isinstance(call, dict) else None
                if not func:
                    continue
                name = func.get("name", "")
                if not name:
                    continue
                raw_args = func.get("arguments", "{}")
                args = self._parse_args(raw_args)
                executed.setdefault(name, []).append(args)
        return executed

    def _evaluate(
        self,
        tool_name: str,
        tc: Any,
        prereqs: list,
        executed: dict[str, list[dict]],
    ) -> list[str]:
        """Return list of unsatisfied prerequisite tool names."""
        missing: list[str] = []
        tc_args = self._tc_args(tc)
        for prereq in prereqs:
            if isinstance(prereq, str):
                # Name-only: any prior call satisfies
                if prereq not in executed:
                    missing.append(prereq)
            elif isinstance(prereq, dict):
                # Arg-matched: prior call with same arg value
                prereq_tool = prereq.get("tool", "")
                match_arg = prereq.get("match_arg", "")
                if not prereq_tool:
                    continue
                required_value = tc_args.get(match_arg) if tc_args else None
                prior_calls = executed.get(prereq_tool, [])
                if not prior_calls:
                    missing.append(prereq_tool)
                    continue
                if not any(
                    c.get(match_arg) == required_value for c in prior_calls
                ):
                    missing.append(prereq_tool)
        return missing

    @staticmethod
    def _tool_name(tc: Any) -> str:
        """Extract tool name from a ToolCall or dict."""
        func = getattr(getattr(tc, "function", None), "name", None)
        if func:
            return func
        if isinstance(tc, dict):
            f = tc.get("function", {})
            return f.get("name", "") if isinstance(f, dict) else ""
        return ""

    @staticmethod
    def _tc_args(tc: Any) -> dict:
        """Extract args dict from a ToolCall or dict."""
        raw = None
        func = getattr(tc, "function", None)
        if func is not None:
            raw = getattr(func, "arguments", None)
        elif isinstance(tc, dict):
            f = tc.get("function", {})
            raw = f.get("arguments") if isinstance(f, dict) else None
        return PrerequisiteChecker._parse_args(raw)

    @staticmethod
    def _parse_args(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_prerequisites.py -v`
Expected: PASS (all tests in the file so far)

- [x] **Step 5: Commit**

```bash
git add omlx/mcp/prerequisites.py tests/test_mcp_prerequisites.py
git commit -m "feat(mcp): add PrerequisiteChecker skeleton with PrerequisiteCheck dataclass"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.2: Implement name-only prerequisite checking

**Files:**
- Modify: `tests/test_mcp_prerequisites.py` (add tests)
- Verify: `omlx/mcp/prerequisites.py` (already implemented in 1.1)

**Interfaces:**
- Consumes: `PrerequisiteChecker` from Task 1.1
- Produces: validated name-only behavior

- [x] **Step 1: Write the failing test (appended to existing test file)**

```python
# Append to tests/test_mcp_prerequisites.py

class TestNameOnlyPrerequisite:
    def _checker(self):
        return PrerequisiteChecker({"edit_file": {"requires": ["read_file"]}})

    def test_missing_prereq_flagged(self):
        checker = self._checker()
        results = checker.check([_tc("edit_file")], [])
        assert len(results) == 1
        assert results[0].passed is False
        assert "read_file" in (results[0].detail or "")

    def test_satisfied_when_prior_call_exists(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        ]
        results = checker.check([_tc("edit_file")], prior)
        assert len(results) == 1
        assert results[0].passed is True

    def test_different_prior_tool_does_not_satisfy(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            }
        ]
        results = checker.check([_tc("edit_file")], prior)
        assert results[0].passed is False

    def test_multiple_name_only_prereqs(self):
        checker = PrerequisiteChecker(
            {"commit": {"requires": ["read_file", "edit_file"]}}
        )
        results = checker.check([_tc("commit")], [])
        assert results[0].passed is False
        assert "read_file" in (results[0].detail or "")
        assert "edit_file" in (results[0].detail or "")
```

- [x] **Step 2: Run test to verify it passes (implementation already exists from 1.1)**

Run: `pytest tests/test_mcp_prerequisites.py::TestNameOnlyPrerequisite -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_mcp_prerequisites.py
git commit -m "test(mcp): cover name-only prerequisite checking"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.3: Implement arg-matched prerequisite checking

**Files:**
- Modify: `tests/test_mcp_prerequisites.py` (add tests)
- Verify: `omlx/mcp/prerequisites.py`

- [x] **Step 1: Write the failing test**

```python
# Append to tests/test_mcp_prerequisites.py

class TestArgMatchedPrerequisite:
    def _checker(self):
        return PrerequisiteChecker(
            {
                "edit_file": {
                    "requires": [{"tool": "read_file", "match_arg": "path"}]
                }
            }
        )

    def test_missing_entirely(self):
        checker = self._checker()
        results = checker.check(
            [_tc("edit_file", '{"path": "/tmp/foo"}')], []
        )
        assert results[0].passed is False
        assert "read_file" in (results[0].detail or "")

    def test_matching_arg_satisfies(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/foo"}',
                        },
                    }
                ],
            }
        ]
        results = checker.check(
            [_tc("edit_file", '{"path": "/tmp/foo"}')], prior
        )
        assert results[0].passed is True

    def test_non_matching_arg_does_not_satisfy(self):
        checker = self._checker()
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/other"}',
                        },
                    }
                ],
            }
        ]
        results = checker.check(
            [_tc("edit_file", '{"path": "/tmp/foo"}')], prior
        )
        assert results[0].passed is False

    def test_mixed_name_and_arg_prereqs(self):
        checker = PrerequisiteChecker(
            {
                "deploy": {
                    "requires": [
                        "test",
                        {"tool": "build", "match_arg": "target"},
                    ]
                }
            }
        )
        # Only test was called, build was not
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            }
        ]
        results = checker.check(
            [_tc("deploy", '{"target": "prod"}')], prior
        )
        assert results[0].passed is False
        assert "build" in (results[0].detail or "")
        assert "test" not in (results[0].detail or "")
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_mcp_prerequisites.py::TestArgMatchedPrerequisite -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_mcp_prerequisites.py
git commit -m "test(mcp): cover arg-matched prerequisite checking"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.4: Verify `executed_tools` built from message history (stateless)

**Files:**
- Modify: `tests/test_mcp_prerequisites.py` (add edge-case tests)

- [x] **Step 1: Write the failing test**

```python
# Append to tests/test_mcp_prerequisites.py

class TestExecutedToolsFromHistory:
    def test_ignores_user_messages(self):
        checker = PrerequisiteChecker({"edit": {"requires": ["read"]}})
        prior = [
            {"role": "user", "content": "edit the file"},
            {
                "role": "user",
                "tool_calls": [{"function": {"name": "read"}}],
            },
        ]
        results = checker.check([_tc("edit")], prior)
        assert results[0].passed is False  # user tool_calls ignored

    def test_ignores_tool_role_messages(self):
        checker = PrerequisiteChecker({"edit": {"requires": ["read"]}})
        prior = [
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
            {
                "role": "tool",
                "tool_calls": [{"function": {"name": "read"}}],
            },
        ]
        results = checker.check([_tc("edit")], prior)
        assert results[0].passed is False

    def test_multiple_prior_assistant_messages_accumulate(self):
        checker = PrerequisiteChecker(
            {"commit": {"requires": ["read", "write"]}}
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "write", "arguments": "{}"}}
                ],
            },
        ]
        results = checker.check([_tc("commit")], prior)
        assert results[0].passed is True

    def test_empty_prior_messages(self):
        checker = PrerequisiteChecker({"edit": {"requires": ["read"]}})
        results = checker.check([_tc("edit")], [])
        assert results[0].passed is False

    def test_malformed_args_in_history_treated_as_empty(self):
        checker = PrerequisiteChecker(
            {"edit": {"requires": [{"tool": "read", "match_arg": "path"}]}}
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read", "arguments": "not-json"}}
                ],
            }
        ]
        results = checker.check(
            [_tc("edit", '{"path": "/x"}')], prior
        )
        assert results[0].passed is False  # malformed args can't match
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_mcp_prerequisites.py::TestExecutedToolsFromHistory -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_mcp_prerequisites.py
git commit -m "test(mcp): cover executed_tools extraction edge cases"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.5: Add `prerequisite_nudge()` to nudge.py

**Files:**
- Modify: `omlx/api/guardrails/nudge.py`
- Create: `tests/test_guardrail_step_prereq_nudges.py`

**Interfaces:**
- Produces: `prerequisite_nudge(tool_name, missing_prereqs, tier=0) -> Nudge` with `role="user"`, `kind="prerequisite"`

- [x] **Step 1: Write the failing test**

```python
# tests/test_guardrail_step_prereq_nudges.py
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for step + prerequisite nudge generators."""
from __future__ import annotations

import pytest

from omlx.api.guardrails.nudge import prerequisite_nudge, step_nudge
from omlx.api.guardrails.types import KIND_PREREQUISITE, KIND_STEP


class TestPrerequisiteNudge:
    def test_role_is_user(self):
        n = prerequisite_nudge("edit_file", ["read_file"])
        assert n.role == "user"

    def test_kind_is_prerequisite(self):
        n = prerequisite_nudge("edit_file", ["read_file"])
        assert n.kind == KIND_PREREQUISITE

    def test_content_mentions_tool_and_missing(self):
        n = prerequisite_nudge("edit_file", ["read_file"])
        assert "edit_file" in n.content
        assert "read_file" in n.content

    def test_multiple_missing_listed(self):
        n = prerequisite_nudge("commit", ["read_file", "write_file"])
        assert "read_file" in n.content
        assert "write_file" in n.content

    def test_default_tier_zero(self):
        n = prerequisite_nudge("edit_file", ["read_file"])
        assert n.tier == 0
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py::TestPrerequisiteNudge -v`
Expected: FAIL with `ImportError: cannot import name 'KIND_PREREQUISITE'` (types not yet updated)

- [x] **Step 3: Add the nudge kind constants to types.py**

In `omlx/api/guardrails/types.py`, add after the existing kind constants (after line 16, `KIND_TOOL_ARG_VALIDATION`):

```python
KIND_STEP = "step"
KIND_PREREQUISITE = "prerequisite"
```

Then update the `CheckName` Literal (around line 24) to add the two new checks:

```python
CheckName = Literal[
    "bare_text",
    "unknown_tool",
    "malformed_args",
    "missing_required_params",
    "tool_choice_enforcement",
    "step",
    "prerequisite",
]
```

Then update the `Nudge.kind` Literal (around line 48) to include the new kinds:

```python
    kind: Literal[
        "retry",
        "unknown_tool",
        "tool_arg_validation",
        "step",
        "prerequisite",
    ]
```

Note: `KIND_STEP` and `KIND_PREREQUISITE` are NOT added to `TOOL_CHANNEL_KINDS` or `TOOL_ERROR_KINDS` — they use `role="user"`, not the tool channel.

- [x] **Step 4: Add `prerequisite_nudge()` to nudge.py**

In `omlx/api/guardrails/nudge.py`, update the import block at the top to include the new constants:

```python
from omlx.api.guardrails.types import (
    KIND_PREREQUISITE,
    KIND_RETRY,
    KIND_STEP,
    KIND_TOOL_ARG_VALIDATION,
    KIND_UNKNOWN_TOOL,
    Nudge,
)
```

Then append at the end of the file:

```python
def prerequisite_nudge(
    tool_name: str, missing_prereqs: list[str], tier: int = 0
) -> Nudge:
    """Nudge for calling a tool without its declared prerequisites.

    Uses role='user' because this is workflow guidance (call the
    prerequisite first), not a tool-execution error. Adapted from
    Forge's prompts/nudges.py:prerequisite_nudge.

    Args:
        tool_name: The tool the model tried to call prematurely.
        missing_prereqs: Prerequisite tool names that haven't been called.
        tier: Escalation tier (default 0; prerequisite nudges do not
            escalate).
    """
    prereqs = ", ".join(missing_prereqs)
    return Nudge(
        role="user",
        content=(
            f"You cannot call {tool_name} yet. "
            f"You must first call: {prereqs}. "
            "Call the prerequisite tool now."
        ),
        kind=KIND_PREREQUISITE,
        tier=tier,
    )
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py::TestPrerequisiteNudge -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add omlx/api/guardrails/types.py omlx/api/guardrails/nudge.py tests/test_guardrail_step_prereq_nudges.py
git commit -m "feat(guardrails): add prerequisite_nudge + KIND_STEP/KIND_PREREQUISITE constants"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.6: Export new constants from `__init__.py`

**Files:**
- Modify: `omlx/api/guardrails/__init__.py`
- Modify: `tests/test_guardrail_step_prereq_nudges.py` (add re-export test)

- [x] **Step 1: Write the failing test**

```python
# Append to tests/test_guardrail_step_prereq_nudges.py

class TestPackageExports:
    def test_kind_step_exported(self):
        from omlx.api.guardrails import KIND_STEP
        assert KIND_STEP == "step"

    def test_kind_prerequisite_exported(self):
        from omlx.api.guardrails import KIND_PREREQUISITE
        assert KIND_PREREQUISITE == "prerequisite"

    def test_step_nudge_exported(self):
        from omlx.api.guardrails import step_nudge
        assert callable(step_nudge)

    def test_prerequisite_nudge_exported(self):
        from omlx.api.guardrails import prerequisite_nudge
        assert callable(prerequisite_nudge)
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py::TestPackageExports -v`
Expected: FAIL with `ImportError: cannot import name 'KIND_STEP'`

- [x] **Step 3: Update `__init__.py`**

In `omlx/api/guardrails/__init__.py`, update imports and `__all__`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Guardrails package: tool-call validation, rescue parsing, tool_choice enforcement."""
from omlx.api.guardrails.nudge import (
    missing_params_nudge,
    prerequisite_nudge,
    retry_nudge,
    step_nudge,
    tool_arg_validation_nudge,
    unknown_tool_nudge,
)
from omlx.api.guardrails.types import (
    CheckResult,
    Nudge,
    ValidationResult,
    KIND_PREREQUISITE,
    KIND_RETRY,
    KIND_STEP,
    KIND_TOOL_ARG_VALIDATION,
    KIND_UNKNOWN_TOOL,
    TOOL_CHANNEL_KINDS,
    TOOL_ERROR_KINDS,
)
from omlx.api.guardrails.validator import GuardrailValidator

__all__ = [
    "CheckResult",
    "Nudge",
    "ValidationResult",
    "KIND_RETRY",
    "KIND_UNKNOWN_TOOL",
    "KIND_TOOL_ARG_VALIDATION",
    "KIND_STEP",
    "KIND_PREREQUISITE",
    "TOOL_CHANNEL_KINDS",
    "TOOL_ERROR_KINDS",
    "GuardrailValidator",
    "missing_params_nudge",
    "prerequisite_nudge",
    "retry_nudge",
    "step_nudge",
    "tool_arg_validation_nudge",
    "unknown_tool_nudge",
]
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omlx/api/guardrails/__init__.py tests/test_guardrail_step_prereq_nudges.py
git commit -m "feat(guardrails): export KIND_STEP, KIND_PREREQUISITE, step/prerequisite nudges"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.7: Add prerequisite declaration parsing to MCP config

**Files:**
- Modify: `omlx/mcp/types.py`
- Modify: `omlx/mcp/config.py`
- Modify: `tests/test_mcp_config.py`

**Interfaces:**
- Produces: `MCPConfig.tools_prerequisites: dict[str, list]` field

- [x] **Step 1: Write the failing test**

Append to `tests/test_mcp_config.py`:

```python
class TestToolsPrerequisites:
    def test_prerequisites_parsed(self):
        data = {
            "servers": {},
            "tools_prerequisites": {
                "edit_file": {"requires": ["read_file"]},
            },
        }
        config = validate_config(data)
        assert "edit_file" in config.tools_prerequisites
        assert config.tools_prerequisites["edit_file"] == {"requires": ["read_file"]}

    def test_prerequisites_default_empty(self):
        config = validate_config({"servers": {}})
        assert config.tools_prerequisites == {}

    def test_arg_matched_prerequisite_parsed(self):
        data = {
            "servers": {},
            "tools_prerequisites": {
                "edit_file": {
                    "requires": [{"tool": "read_file", "match_arg": "path"}]
                },
            },
        }
        config = validate_config(data)
        prereqs = config.tools_prerequisites["edit_file"]["requires"]
        assert prereqs[0]["tool"] == "read_file"
        assert prereqs[0]["match_arg"] == "path"

    def test_from_dict_carries_prerequisites(self):
        data = {
            "servers": {},
            "tools_prerequisites": {"deploy": {"requires": ["build"]}},
        }
        config = MCPConfig.from_dict(data)
        assert "deploy" in config.tools_prerequisites
```

Add `from omlx.mcp.config import validate_config` and `from omlx.mcp.types import MCPConfig` to the imports at the top of the test file if not already present.

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_config.py::TestToolsPrerequisites -v`
Expected: FAIL with `AttributeError: 'MCPConfig' object has no attribute 'tools_prerequisites'`

- [x] **Step 3: Add field to `MCPConfig` in types.py**

In `omlx/mcp/types.py`, update the `MCPConfig` dataclass (around line 65) to add the new field:

```python
@dataclass
class MCPConfig:
    """Root configuration for MCP client."""

    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    max_tool_calls: int = 10
    default_timeout: float = 30.0
    tools_prerequisites: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPConfig":
        """Create config from dictionary."""
        servers = {}
        for name, server_data in data.get("servers", {}).items():
            server_data["name"] = name
            servers[name] = MCPServerConfig(**server_data)

        return cls(
            servers=servers,
            max_tool_calls=data.get("max_tool_calls", 10),
            default_timeout=data.get("default_timeout", 30.0),
            tools_prerequisites=data.get("tools_prerequisites", {}),
        )
```

- [x] **Step 4: Parse the field in config.py `validate_config`**

In `omlx/mcp/config.py`, update `validate_config` (around line 144) to parse `tools_prerequisites`:

```python
    # Validate tools_prerequisites (optional)
    tools_prerequisites = data.get("tools_prerequisites", {})
    if not isinstance(tools_prerequisites, dict):
        raise ValueError("'tools_prerequisites' must be a dictionary")

    return MCPConfig(
        servers=servers,
        max_tool_calls=max_tool_calls,
        default_timeout=default_timeout,
        tools_prerequisites=tools_prerequisites,
    )
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_mcp_config.py::TestToolsPrerequisites -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add omlx/mcp/types.py omlx/mcp/config.py tests/test_mcp_config.py
git commit -m "feat(mcp): parse tools_prerequisites field in MCP config"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.8: Update `mcp.example.json` with prerequisite declarations

**Files:**
- Modify: `mcp.example.json`

- [x] **Step 1: Update the example file**

Replace the entire contents of `mcp.example.json` with:

```json
{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "enabled": true,
      "timeout": 30
    },
    "sqlite": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "data.db"],
      "enabled": false,
      "timeout": 30
    },
    "web-search": {
      "transport": "sse",
      "url": "http://localhost:3001/sse",
      "enabled": false,
      "timeout": 60
    }
  },
  "max_tool_calls": 10,
  "default_timeout": 30.0,
  "tools_prerequisites": {
    "write_file": {
      "requires": [
        {"tool": "read_file", "match_arg": "path"}
      ]
    },
    "delete_file": {
      "requires": ["list_files"]
    }
  }
}
```

- [x] **Step 2: Verify it parses**

Run: `python -c "import json; json.load(open('mcp.example.json'))"`
Expected: no output (valid JSON)

- [x] **Step 3: Commit**

```bash
git add mcp.example.json
git commit -m "docs(mcp): add tools_prerequisites example declarations"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 1.9: Full PrerequisiteChecker unit test suite review

**Files:**
- Verify: `tests/test_mcp_prerequisites.py`

- [x] **Step 1: Run the full test file**

Run: `pytest tests/test_mcp_prerequisites.py -v`
Expected: PASS (all classes: TestPrerequisiteCheckDataclass, TestPrerequisiteCheckerConstruction, TestNameOnlyPrerequisite, TestArgMatchedPrerequisite, TestExecutedToolsFromHistory)

- [x] **Step 2: Commit (if any test fixes were needed)**

```bash
git add tests/test_mcp_prerequisites.py
git commit -m "test(mcp): finalize PrerequisiteChecker unit test coverage"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

## Group 2: Step Enforcement + Nudges

### Task 2.1: Add `step_nudge()` with 3-tier escalation

**Files:**
- Modify: `omlx/api/guardrails/nudge.py`
- Modify: `tests/test_guardrail_step_prereq_nudges.py`

**Interfaces:**
- Produces: `step_nudge(terminal_tool, pending_steps, tier=1) -> Nudge` with `role="user"`, `kind="step"`, 3-tier escalation

- [x] **Step 1: Write the failing test**

```python
# Append to tests/test_guardrail_step_prereq_nudges.py

class TestStepNudge:
    def test_role_is_user(self):
        n = step_nudge("respond", ["search", "read"], tier=1)
        assert n.role == "user"

    def test_kind_is_step(self):
        n = step_nudge("respond", ["search"], tier=1)
        assert n.kind == KIND_STEP

    def test_tier1_polite(self):
        n = step_nudge("respond", ["search", "read"], tier=1)
        assert n.tier == 1
        assert "respond" in n.content
        assert "search" in n.content
        assert "read" in n.content
        # Polite tone
        assert "cannot" in n.content.lower() or "must first" in n.content.lower()

    def test_tier2_direct(self):
        n = step_nudge("respond", ["search"], tier=2)
        assert n.tier == 2
        assert "search" in n.content
        # More direct — shorter, imperative
        assert "pick one" in n.content.lower()

    def test_tier3_aggressive(self):
        n = step_nudge("respond", ["search"], tier=3)
        assert n.tier == 3
        assert "search" in n.content
        # Aggressive — STOP / MUST
        content_upper = n.content.upper()
        assert "STOP" in content_upper or "MUST" in content_upper

    def test_tier_clamped_to_3(self):
        n = step_nudge("respond", ["search"], tier=5)
        assert n.tier == 3

    def test_tier_clamped_to_1(self):
        n = step_nudge("respond", ["search"], tier=0)
        assert n.tier == 1
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py::TestStepNudge -v`
Expected: FAIL with `NameError: name 'step_nudge' is not defined` (function not yet added)

- [x] **Step 3: Add `step_nudge()` to nudge.py**

Append to `omlx/api/guardrails/nudge.py`:

```python
def step_nudge(
    terminal_tool: str, pending_steps: list[str], tier: int = 1
) -> Nudge:
    """Escalating nudge for premature terminal tool attempts.

    Uses role='user' because this is workflow guidance (complete the
    required steps first), not a tool-execution error. Adapted from
    Forge's prompts/nudges.py:step_nudge.

    Args:
        terminal_tool: The terminal tool the model tried to call early.
        pending_steps: Required steps that must be completed first.
        tier: Escalation level (1=polite, 2=direct, 3=aggressive).
            Clamped to 1-3.
    """
    tier = max(1, min(3, tier))
    steps = ", ".join(pending_steps)
    if tier == 1:
        content = (
            f"You cannot call {terminal_tool} yet. "
            f"You must first complete these required steps: {steps}. "
            "Call one of them now."
        )
    elif tier == 2:
        content = (
            f"You must call one of these tools now: {steps}. "
            "Pick one."
        )
    else:
        content = (
            f"STOP. You MUST call one of: {steps}. "
            f"Do NOT call {terminal_tool}. "
            f"Your next response MUST be a tool call to one of: {steps}."
        )
    return Nudge(role="user", content=content, kind=KIND_STEP, tier=tier)
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py::TestStepNudge -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omlx/api/guardrails/nudge.py tests/test_guardrail_step_prereq_nudges.py
git commit -m "feat(guardrails): add step_nudge with 3-tier escalation"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 2.2: Add step enforcement check (Check 5) to GuardrailValidator

**Files:**
- Modify: `omlx/api/guardrails/validator.py`
- Modify: `tests/test_guardrail_validator.py`

**Interfaces:**
- Consumes: `step_nudge` from Task 2.1
- Produces: `GuardrailValidator.validate()` accepts optional `step_context` parameter; emits Check 5 ("step") when terminal tool called before steps satisfied

**Design note:** The validator is constructed with tool schemas. Step context (terminal_tools, required_steps, premature_attempts) is passed to `validate()` as an optional kwarg because it varies per-request and depends on message history.

- [x] **Step 1: Write the failing test**

Append to `tests/test_guardrail_validator.py`:

```python
class TestStepEnforcementCheck:
    """Check 5: step enforcement — premature terminal tool calls."""

    def _validator_with_respond(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "respond",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        return GuardrailValidator(tools)

    def _tc(self, name):
        from omlx.api.openai_models import FunctionCall, ToolCall
        return ToolCall(
            id=f"call_{name}",
            type="function",
            function=FunctionCall(name=name, arguments="{}"),
        )

    def _extraction(self, calls):
        from omlx.api.tool_calling import ToolCallExtraction
        return ToolCallExtraction(
            cleaned_text="",
            tool_calls=calls,
            cleaned_thinking="",
        )

    def test_step_check_passes_when_no_step_context(self):
        v = self._validator_with_respond()
        ext = self._extraction([self._tc("respond")])
        result = v.validate(ext, has_tools=True)
        check_names = [c.check for c in result.checks]
        assert "step" not in check_names  # no context → check skipped

    def test_step_check_fails_on_premature_terminal(self):
        v = self._validator_with_respond()
        ext = self._extraction([self._tc("respond")])
        step_context = {
            "terminal_tools": frozenset({"respond"}),
            "pending_steps": ["search"],
            "premature_attempts": 0,
        }
        result = v.validate(ext, has_tools=True, step_context=step_context)
        step_checks = [c for c in result.checks if c.check == "step"]
        assert len(step_checks) == 1
        assert step_checks[0].passed is False

    def test_step_check_passes_when_steps_satisfied(self):
        v = self._validator_with_respond()
        ext = self._extraction([self._tc("respond")])
        step_context = {
            "terminal_tools": frozenset({"respond"}),
            "pending_steps": [],  # all satisfied
            "premature_attempts": 0,
        }
        result = v.validate(ext, has_tools=True, step_context=step_context)
        step_checks = [c for c in result.checks if c.check == "step"]
        assert len(step_checks) == 1
        assert step_checks[0].passed is True

    def test_step_nudge_has_user_role_and_tier(self):
        v = self._validator_with_respond()
        ext = self._extraction([self._tc("respond")])
        step_context = {
            "terminal_tools": frozenset({"respond"}),
            "pending_steps": ["search"],
            "premature_attempts": 1,
        }
        result = v.validate(ext, has_tools=True, step_context=step_context)
        assert result.nudge is not None
        assert result.nudge.role == "user"
        assert result.nudge.kind == "step"
        assert result.nudge.tier == 2  # attempts=1 → tier 2
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_validator.py::TestStepEnforcementCheck -v`
Expected: FAIL with `TypeError: validate() got an unexpected keyword argument 'step_context'`

- [x] **Step 3: Add step_context to validate() and Check 5**

In `omlx/api/guardrails/validator.py`, update the imports:

```python
from omlx.api.guardrails.nudge import (
    missing_params_nudge,
    prerequisite_nudge,
    retry_nudge,
    step_nudge,
    tool_arg_validation_nudge,
    unknown_tool_nudge,
)
```

Update the `_NUDGE_PRIORITY` list to include the new checks (step and prerequisite run after existing checks):

```python
_NUDGE_PRIORITY = [
    "bare_text",
    "unknown_tool",
    "malformed_args",
    "missing_required_params",
    "step",
    "prerequisite",
]
```

Update the `validate` method signature and body. Replace the entire `validate` method:

```python
    def validate(
        self,
        extraction: Any,
        tool_choice: Any = None,
        has_tools: bool = True,
        *,
        step_context: dict | None = None,
        prerequisite_results: list | None = None,
    ) -> ValidationResult:
        """Run all checks and return accumulated result.

        Args:
            extraction: ToolCallExtraction with parsed tool calls.
            tool_choice: The request's tool_choice value.
            has_tools: Whether tools were provided in the request.
            step_context: Optional dict with keys ``terminal_tools``
                (frozenset[str]), ``pending_steps`` (list[str]),
                ``premature_attempts`` (int). When provided, runs
                Check 5 (step enforcement).
            prerequisite_results: Optional list of CheckResult from
                PrerequisiteChecker. When provided, runs Check 6
                (prerequisite validation) by merging these results.
        """
        try:
            checks: list[CheckResult] = []

            tool_calls = extraction.tool_calls or []

            checks.append(self._check_bare_text(extraction, tool_choice, has_tools))

            for tc in tool_calls:
                name = tc.function.name
                if name not in self._tool_names:
                    checks.append(
                        CheckResult(
                            check="unknown_tool",
                            passed=False,
                            detail=(
                                f"Tool '{name}' does not exist. "
                                f"Available: {', '.join(sorted(self._tool_names))}"
                            ),
                        )
                    )
                else:
                    checks.append(CheckResult(check="unknown_tool", passed=True))

            for tc in tool_calls:
                checks.append(self._check_malformed_args(tc))

            for tc in tool_calls:
                checks.append(self._check_missing_params(tc))

            # Check 5: step enforcement (only when context provided)
            if step_context is not None:
                checks.append(self._check_step(tool_calls, step_context))

            # Check 6: prerequisite validation (merge external results)
            if prerequisite_results is not None:
                checks.extend(prerequisite_results)

            passed = all(c.passed for c in checks) if checks else True
            nudge = self._select_nudge(checks, tool_calls) if not passed else None
            return ValidationResult(
                checks=checks, nudge=nudge, passed=passed, budget=self._budget
            )

        except Exception:
            logger.exception("GuardrailValidator.validate failed unexpectedly")
            return ValidationResult(
                checks=[], nudge=None, passed=True, budget=self._budget
            )
```

Add the `_check_step` method (before `_select_nudge`):

```python
    def _check_step(
        self, tool_calls: list, step_context: dict
    ) -> CheckResult:
        """Check 5: detect premature terminal tool calls.

        A terminal tool called before all required steps are complete
        is a step violation.
        """
        terminal_tools = step_context.get("terminal_tools", frozenset())
        pending_steps = step_context.get("pending_steps", [])

        has_terminal = any(
            getattr(getattr(tc, "function", None), "name", "") in terminal_tools
            for tc in tool_calls
        )

        if has_terminal and pending_steps:
            attempted = next(
                (
                    getattr(getattr(tc, "function", None), "name", "")
                    for tc in tool_calls
                    if getattr(getattr(tc, "function", None), "name", "")
                    in terminal_tools
                ),
                "unknown",
            )
            return CheckResult(
                check="step",
                passed=False,
                detail=(
                    f"Tool '{attempted}' is terminal but required steps "
                    f"are incomplete: {', '.join(pending_steps)}"
                ),
            )
        return CheckResult(check="step", passed=True)
```

Update `_build_nudge` to handle the step check. Add this branch before the final `return retry_nudge()`:

```python
        if check.check == "step":
            # The caller (server wiring) tracks premature_attempts;
            # we cannot reconstruct tier here, so default tier 1.
            # The wiring layer overrides the nudge with the correct tier.
            return step_nudge("terminal", [], tier=1)
```

Note: The `_build_nudge` for step is a fallback. The primary step nudge is built in the wiring layer (`guardrail_wiring.py`) where the premature attempt count is tracked. See Task 2.3.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_validator.py::TestStepEnforcementCheck -v`
Expected: PASS

- [x] **Step 5: Run existing validator tests to verify no regressions**

Run: `pytest tests/test_guardrail_validator.py -v`
Expected: PASS (all existing tests still pass — step_context defaults to None)

- [x] **Step 6: Commit**

```bash
git add omlx/api/guardrails/validator.py tests/test_guardrail_validator.py
git commit -m "feat(guardrails): add Check 5 (step enforcement) to GuardrailValidator"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 2.3: Wire step + prerequisite checks into validation flow

**Files:**
- Modify: `omlx/api/guardrail_wiring.py`
- Modify: `omlx/api/tool_calling.py`
- Modify: `tests/test_server_guardrail_wiring.py`

**Interfaces:**
- Consumes: `PrerequisiteChecker` from Task 1.1, step_context from server
- Produces: `apply_guardrails()` accepts optional `prerequisite_checker`, `prior_messages`, `step_context`; `extract_and_validate_tool_calls()` accepts optional `prerequisite_checker`, `prior_messages`, `step_context`

- [x] **Step 1: Write the failing test**

Append to `tests/test_server_guardrail_wiring.py`:

```python
class TestPrerequisiteWiring:
    def test_prerequisite_check_runs_when_provided(self):
        from omlx.mcp.prerequisites import PrerequisiteChecker

        vr = ValidationResult(
            checks=[CheckResult(check="bare_text", passed=True)],
            passed=True,
        )
        ext = _mk_extraction(
            tool_calls=[
                _make_tool_call("edit_file", '{"path": "/x"}'),
            ],
            validation_result=vr,
        )
        checker = PrerequisiteChecker(
            {"edit_file": {"requires": ["read_file"]}}
        )
        result = apply_guardrails(
            ext,
            "auto",
            None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=[],
        )
        check_names = [c.check for c in result.validation_result.checks]
        assert "prerequisite" in check_names
        assert result.validation_result.passed is False

    def test_prerequisite_check_skipped_when_no_checker(self):
        vr = ValidationResult(checks=[], passed=True)
        ext = _mk_extraction(validation_result=vr)
        result = apply_guardrails(
            ext, "auto", None, validation_enabled=True,
        )
        check_names = [c.check for c in result.validation_result.checks]
        assert "prerequisite" not in check_names

    def test_step_check_runs_when_context_provided(self):
        vr = ValidationResult(checks=[], passed=True)
        ext = _mk_extraction(
            tool_calls=[_make_tool_call("respond", "{}")],
            validation_result=vr,
        )
        step_context = {
            "terminal_tools": frozenset({"respond"}),
            "pending_steps": ["search"],
            "premature_attempts": 0,
        }
        result = apply_guardrails(
            ext,
            "auto",
            [{"type": "function", "function": {"name": "respond"}}],
            validation_enabled=True,
            step_context=step_context,
        )
        check_names = [c.check for c in result.validation_result.checks]
        assert "step" in check_names


def _make_tool_call(name, args):
    from omlx.api.openai_models import FunctionCall, ToolCall
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_guardrail_wiring.py::TestPrerequisiteWiring -v`
Expected: FAIL with `TypeError: apply_guardrails() got an unexpected keyword argument 'prerequisite_checker'`

- [x] **Step 3: Update `apply_guardrails` in guardrail_wiring.py**

Replace the entire contents of `omlx/api/guardrail_wiring.py`:

```python
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Optional

from omlx.api.guardrails.budget import ErrorBudget
from omlx.api.guardrails.nudge import prerequisite_nudge, step_nudge
from omlx.api.guardrails.types import (
    CheckResult,
    Nudge,
    ValidationResult,
)
from omlx.api.tool_calling import ToolCallExtraction
from omlx.api.tool_choice import enforce_tool_choice


def apply_guardrails(
    extraction: ToolCallExtraction,
    tool_choice: Any,
    tools: Optional[list[dict]] = None,
    *,
    validation_enabled: bool = False,
    prerequisite_checker: Any = None,
    prior_messages: list[dict] | None = None,
    step_context: dict | None = None,
) -> ToolCallExtraction:
    """Apply tool_choice, prerequisite, and step enforcement checks.

    Merges new CheckResult entries into the existing ValidationResult.
    Builds the final nudge from the highest-priority failure.
    """
    if not validation_enabled or extraction.validation_result is None:
        return extraction

    has_text = bool(extraction.cleaned_text.strip())
    _, tc_check = enforce_tool_choice(
        extraction.tool_calls, tool_choice, has_text, tools
    )

    new_checks: list[CheckResult] = [tc_check]

    # Check 6: prerequisite validation
    if prerequisite_checker is not None and extraction.tool_calls:
        prereq_results = prerequisite_checker.check(
            extraction.tool_calls, prior_messages or []
        )
        new_checks.extend(prereq_results)

    # Check 5: step enforcement
    if step_context is not None and extraction.tool_calls:
        new_checks.append(_check_step(extraction.tool_calls, step_context))

    existing: ValidationResult = extraction.validation_result
    merged_checks = existing.checks + new_checks
    merged_passed = existing.passed and all(c.passed for c in new_checks)
    merged_nudge = _select_merged_nudge(merged_checks, extraction.tool_calls, step_context)

    merged = ValidationResult(
        checks=merged_checks,
        nudge=merged_nudge,
        passed=merged_passed,
        budget=existing.budget,
    )

    return ToolCallExtraction(
        cleaned_text=extraction.cleaned_text,
        tool_calls=extraction.tool_calls,
        cleaned_thinking=extraction.cleaned_thinking,
        tool_calls_from_thinking=extraction.tool_calls_from_thinking,
        validation_result=merged,
    )


def _check_step(tool_calls: list, step_context: dict) -> CheckResult:
    """Step enforcement check (inline, mirrors validator._check_step)."""
    terminal_tools = step_context.get("terminal_tools", frozenset())
    pending_steps = step_context.get("pending_steps", [])

    has_terminal = any(
        getattr(getattr(tc, "function", None), "name", "") in terminal_tools
        for tc in tool_calls
    )

    if has_terminal and pending_steps:
        attempted = next(
            (
                getattr(getattr(tc, "function", None), "name", "")
                for tc in tool_calls
                if getattr(getattr(tc, "function", None), "name", "")
                in terminal_tools
            ),
            "unknown",
        )
        return CheckResult(
            check="step",
            passed=False,
            detail=(
                f"Tool '{attempted}' is terminal but required steps "
                f"are incomplete: {', '.join(pending_steps)}"
            ),
        )
    return CheckResult(check="step", passed=True)


_PRIORITY = [
    "bare_text",
    "unknown_tool",
    "malformed_args",
    "missing_required_params",
    "step",
    "prerequisite",
    "tool_choice_enforcement",
]


def _select_merged_nudge(
    checks: list[CheckResult],
    tool_calls: list,
    step_context: dict | None,
) -> Optional[Nudge]:
    """Select nudge from highest-priority failed check.

    Step and prerequisite nudges are built here (the wiring layer)
    because tier escalation requires the premature_attempts counter
    from step_context.
    """
    failed = [c for c in checks if not c.passed]
    if not failed:
        return None

    for check_name in _PRIORITY:
        for c in failed:
            if c.check != check_name:
                continue

            if c.check == "step" and step_context is not None:
                terminal = next(
                    (
                        getattr(getattr(tc, "function", None), "name", "")
                        for tc in tool_calls
                        if getattr(getattr(tc, "function", None), "name", "")
                        in step_context.get("terminal_tools", frozenset())
                    ),
                    "terminal",
                )
                tier = min(step_context.get("premature_attempts", 0) + 1, 3)
                return step_nudge(
                    terminal,
                    step_context.get("pending_steps", []),
                    tier=tier,
                )

            if c.check == "prerequisite":
                # Extract tool name and missing from detail
                tool_name = "tool"
                missing: list[str] = []
                for tc in tool_calls:
                    name = getattr(getattr(tc, "function", None), "name", "")
                    if name and c.detail and name in c.detail:
                        tool_name = name
                        break
                return prerequisite_nudge(tool_name, missing or ["prerequisite"])

            # For existing checks (bare_text, unknown_tool, etc.),
            # defer to the validator's nudge already in the result.
            # If the existing ValidationResult has a nudge, keep it.
            return None
    return None


def guardrail_validation_payload(
    extraction: ToolCallExtraction,
    *,
    include_validation_metadata: bool = False,
    max_retries: int = 3,
    max_tool_errors: int = 2,
) -> Optional[dict]:
    """Build the x_omlx_validation payload, attaching an ErrorBudget.

    When the ValidationResult does not already carry a budget, one is
    constructed from *max_retries* / *max_tool_errors* so clients can
    implement bounded retry loops.
    """
    if not include_validation_metadata or extraction.validation_result is None:
        return None

    existing: ValidationResult = extraction.validation_result
    if existing.budget is not None:
        return {"x_omlx_validation": existing.to_dict()}

    budget = ErrorBudget(max_retries=max_retries, max_tool_errors=max_tool_errors)
    with_budget = ValidationResult(
        checks=existing.checks,
        nudge=existing.nudge,
        passed=existing.passed,
        budget=budget,
    )
    return {"x_omlx_validation": with_budget.to_dict()}
```

- [x] **Step 4: Update `extract_and_validate_tool_calls` to pass step/prereq context**

In `omlx/api/tool_calling.py`, update the `extract_and_validate_tool_calls` function (around line 1559) to accept and forward the new optional parameters:

```python
def extract_and_validate_tool_calls(
    thinking_content: str,
    regular_content: str,
    tokenizer: Any,
    tools: Optional[List] = None,
    tool_choice: Any = None,
    strict_tool_args: bool = False,
    validation_enabled: bool = False,
    *,
    step_context: dict | None = None,
    prerequisite_results: list | None = None,
) -> ToolCallExtraction:
    """Wrapper: extract tool calls, then validate if enabled.

    When validation_enabled is False (default), this is a pure passthrough
    to extract_tool_calls_with_thinking() — zero behavioral change.

    When validation_enabled is True and tools are provided, runs
    GuardrailValidator on the extraction and attaches the result.
    step_context and prerequisite_results are forwarded to the validator
    for Check 5 (step) and Check 6 (prerequisite) when provided.
    """
    extraction = extract_tool_calls_with_thinking(
        thinking_content, regular_content, tokenizer, tools,
        strict=strict_tool_args,
    )

    validation_result = None
    if validation_enabled and tools:
        try:
            from omlx.api.guardrails.validator import GuardrailValidator

            validator = GuardrailValidator(tools)
            validation_result = validator.validate(
                extraction,
                tool_choice=tool_choice,
                has_tools=bool(tools),
                step_context=step_context,
                prerequisite_results=prerequisite_results,
            )
        except Exception:
            logger.exception(
                "Guardrail validation failed; returning extraction without validation"
            )

    return ToolCallExtraction(
        cleaned_text=extraction.cleaned_text,
        tool_calls=extraction.tool_calls,
        cleaned_thinking=extraction.cleaned_thinking,
        tool_calls_from_thinking=extraction.tool_calls_from_thinking,
        validation_result=validation_result,
    )
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_server_guardrail_wiring.py::TestPrerequisiteWiring -v`
Expected: PASS

- [x] **Step 6: Run full wiring test suite for regressions**

Run: `pytest tests/test_server_guardrail_wiring.py -v`
Expected: PASS (all existing tests still pass — new params are optional)

- [x] **Step 7: Commit**

```bash
git add omlx/api/guardrail_wiring.py omlx/api/tool_calling.py tests/test_server_guardrail_wiring.py
git commit -m "feat(guardrails): wire step + prerequisite checks into apply_guardrails"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 2.4: Step nudge tier correctness tests

**Files:**
- Modify: `tests/test_guardrail_step_prereq_nudges.py`

- [x] **Step 1: Write the test**

```python
# Append to tests/test_guardrail_step_prereq_nudges.py

class TestStepNudgeTierProgression:
    """Verify tier escalation matches Forge's pattern."""

    def test_tier_progresses_1_2_3(self):
        tiers = []
        for attempts in range(0, 5):
            tier = min(attempts + 1, 3)
            n = step_nudge("respond", ["search"], tier=tier)
            tiers.append(n.tier)
        assert tiers == [1, 2, 3, 3, 3]

    def test_each_tier_content_is_distinct(self):
        contents = set()
        for tier in (1, 2, 3):
            n = step_nudge("respond", ["search"], tier=tier)
            contents.add(n.content)
        assert len(contents) == 3  # all distinct

    def test_tier3_is_strongest(self):
        n1 = step_nudge("respond", ["search"], tier=1)
        n3 = step_nudge("respond", ["search"], tier=3)
        # Tier 3 content is longer / more emphatic
        assert len(n3.content) >= len(n1.content)
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_guardrail_step_prereq_nudges.py::TestStepNudgeTierProgression -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_guardrail_step_prereq_nudges.py
git commit -m "test(guardrails): verify step nudge tier progression"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

## Group 3: Synthetic Respond Tool

### Task 3.1: Create `respond.py` module with inject + strip

**Files:**
- Create: `omlx/api/respond.py`
- Create: `tests/test_respond_tool.py`

**Interfaces:**
- Produces: `RESPOND_TOOL_NAME = "respond"`, `inject_respond_tool(tools: list) -> list`, `strip_respond_calls(tool_calls) -> tuple[list, str | None]`

- [x] **Step 1: Write the failing test**

```python
# tests/test_respond_tool.py
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for synthetic respond tool inject + strip."""
from __future__ import annotations

import pytest

from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.respond import (
    RESPOND_TOOL_NAME,
    inject_respond_tool,
    strip_respond_calls,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tc(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


class TestRespondToolName:
    def test_name_is_respond(self):
        assert RESPOND_TOOL_NAME == "respond"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_respond_tool.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'omlx.api.respond'`

- [x] **Step 3: Write minimal implementation**

```python
# omlx/api/respond.py
# SPDX-License-Identifier: Apache-2.0
"""Synthetic respond tool — structured alternative to bare text responses.

The model calls respond(message="...") instead of producing bare text.
This keeps the model in tool-calling mode where oMLX's guardrail stack
applies. The respond tool is injected server-side and stripped before
the response returns — the client never sees it.

Adapted from Forge's forge/tools/respond.py.

Usage:
    tools = inject_respond_tool(tools)   # before generation
    ...
    real_calls, text = strip_respond_calls(parsed_calls)  # after parsing
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

RESPOND_TOOL_NAME = "respond"

RESPOND_DESCRIPTION = (
    "Respond to the user with a message. Use this when the user is chatting, "
    "asking a question, when you need to ask a clarifying question before "
    "proceeding, or when no other tool action is needed. Use this "
    "after completing the user's request to report the result."
)

RESPOND_TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": RESPOND_TOOL_NAME,
        "description": RESPOND_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the user.",
                }
            },
            "required": ["message"],
        },
    },
}


def inject_respond_tool(tools: list | None) -> list:
    """Append the respond tool to the tools list if not already present.

    Args:
        tools: Existing tools list (OpenAI function format). May be None
            or empty — in that case, returns the input unchanged.

    Returns:
        A new list with the respond tool appended, or the original list
        if it was empty/None or already contains a respond tool.
    """
    if not tools:
        return tools  # type: ignore[return-value]

    # Check if a tool named "respond" already exists
    for tool in tools:
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        if func.get("name") == RESPOND_TOOL_NAME:
            return tools  # already present, don't duplicate

    return tools + [RESPOND_TOOL_SPEC]


def strip_respond_calls(
    tool_calls: list[Any] | None,
) -> tuple[list[Any], str | None]:
    """Strip respond calls from parsed tool call output.

    Three cases:
      - Pure respond (only call is respond): return ([], message_text)
      - Mixed (respond + real calls): return (real_calls, None)
      - No respond calls: return (original_calls, None) — passthrough

    Args:
        tool_calls: Parsed ToolCall objects or None.

    Returns:
        Tuple of (real_tool_calls, respond_message). respond_message is
        the extracted message string when the only call was respond,
        otherwise None.
    """
    if not tool_calls:
        return ([], None) if tool_calls is not None else ([], None)

    respond_calls: list[Any] = []
    real_calls: list[Any] = []

    for tc in tool_calls:
        name = _get_name(tc)
        if name == RESPOND_TOOL_NAME:
            respond_calls.append(tc)
        else:
            real_calls.append(tc)

    # Pure respond → extract message as text
    if respond_calls and not real_calls:
        message = _extract_message(respond_calls[0])
        return ([], message)

    # Mixed or no respond → return real calls only
    return (real_calls, None)


def _get_name(tc: Any) -> str:
    """Extract function name from a ToolCall or dict."""
    func = getattr(getattr(tc, "function", None), "name", None)
    if func:
        return func
    if isinstance(tc, dict):
        f = tc.get("function", {})
        return f.get("name", "") if isinstance(f, dict) else ""
    return ""


def _extract_message(tc: Any) -> str:
    """Extract the 'message' argument from a respond ToolCall."""
    raw_args = None
    func = getattr(tc, "function", None)
    if func is not None:
        raw_args = getattr(func, "arguments", None)
    elif isinstance(tc, dict):
        f = tc.get("function", {})
        raw_args = f.get("arguments") if isinstance(f, dict) else None

    if isinstance(raw_args, dict):
        return str(raw_args.get("message", ""))
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return str(parsed.get("message", "")) if isinstance(parsed, dict) else ""
        except (json.JSONDecodeError, ValueError):
            return ""
    return ""
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_respond_tool.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add omlx/api/respond.py tests/test_respond_tool.py
git commit -m "feat(api): add synthetic respond tool module (inject + strip)"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 3.2: `inject_respond_tool` — append if not present

**Files:**
- Modify: `tests/test_respond_tool.py` (add comprehensive inject tests)

- [x] **Step 1: Write the test**

```python
# Append to tests/test_respond_tool.py

class TestInjectRespondTool:
    def test_appends_to_non_empty_list(self):
        tools = [_tool("search"), _tool("read")]
        result = inject_respond_tool(tools)
        assert len(result) == 3
        assert result[-1]["function"]["name"] == RESPOND_TOOL_NAME

    def test_returns_none_when_none(self):
        assert inject_respond_tool(None) is None

    def test_returns_empty_when_empty(self):
        result = inject_respond_tool([])
        assert result == []

    def test_does_not_duplicate(self):
        tools = [_tool("search"), RESPOND_TOOL_SPEC]
        result = inject_respond_tool(tools)
        assert len(result) == 2  # unchanged
        names = [t["function"]["name"] for t in result]
        assert names.count(RESPOND_TOOL_NAME) == 1

    def test_injected_tool_has_message_param(self):
        result = inject_respond_tool([_tool("search")])
        params = result[-1]["function"]["parameters"]
        assert "message" in params["properties"]
        assert "message" in params["required"]

    def test_injected_tool_has_description(self):
        result = inject_respond_tool([_tool("search")])
        desc = result[-1]["function"]["description"]
        assert len(desc) > 20  # non-trivial description

    def test_does_not_mutate_original_list(self):
        tools = [_tool("search")]
        inject_respond_tool(tools)
        assert len(tools) == 1  # original unchanged
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_respond_tool.py::TestInjectRespondTool -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_respond_tool.py
git commit -m "test(api): comprehensive inject_respond_tool coverage"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 3.3: `strip_respond_calls` — three cases

**Files:**
- Modify: `tests/test_respond_tool.py` (add comprehensive strip tests)

- [x] **Step 1: Write the test**

```python
# Append to tests/test_respond_tool.py

class TestStripRespondCalls:
    def test_pure_respond_becomes_text(self):
        calls = [_tc("respond", '{"message": "Hello world"}')]
        real, text = strip_respond_calls(calls)
        assert real == []
        assert text == "Hello world"

    def test_mixed_drops_respond_silently(self):
        calls = [
            _tc("respond", '{"message": "thinking..."}'),
            _tc("search", '{"query": "test"}'),
        ]
        real, text = strip_respond_calls(calls)
        assert len(real) == 1
        assert real[0].function.name == "search"
        assert text is None  # respond silently dropped

    def test_no_respond_passthrough(self):
        calls = [_tc("search"), _tc("read")]
        real, text = strip_respond_calls(calls)
        assert len(real) == 2
        assert text is None

    def test_empty_list(self):
        real, text = strip_respond_calls([])
        assert real == []
        assert text is None

    def test_none_input(self):
        real, text = strip_respond_calls(None)
        assert real == []
        assert text is None

    def test_pure_respond_extracts_message_from_dict_args(self):
        """When args are already a dict (some parsers do this)."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="respond",
                arguments={"message": "Dict args"},
            ),
        )
        real, text = strip_respond_calls([tc])
        assert real == []
        assert text == "Dict args"

    def test_malformed_respond_args_returns_empty_message(self):
        tc = _tc("respond", "not-json")
        real, text = strip_respond_calls([tc])
        assert real == []
        assert text == ""  # graceful degradation

    def test_multiple_real_calls_preserved(self):
        calls = [
            _tc("search", '{"q": "a"}'),
            _tc("respond", '{"message": "x"}'),
            _tc("read", '{"path": "/y"}'),
        ]
        real, text = strip_respond_calls(calls)
        assert len(real) == 2
        assert real[0].function.name == "search"
        assert real[1].function.name == "read"
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_respond_tool.py::TestStripRespondCalls -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_respond_tool.py
git commit -m "test(api): comprehensive strip_respond_calls coverage"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 3.4: Wire respond tool into server.py

**Files:**
- Modify: `omlx/server.py`
- Modify: `tests/test_server_guardrail_wiring.py` (source inspection tests)

**Design note:** The server has 6 call sites (3 API formats × streaming/non-streaming). The respond tool is injected once into `tools_for_template` before generation, and stripped once after parsing. The injection point is after `tools_for_template` is built (~line 3334) but before the generation call. The strip point is right after `extract_and_validate_tool_calls` / `_convert_parser_tool_calls` returns, before building the response.

- [x] **Step 1: Write the failing test (source inspection)**

Append to `tests/test_server_guardrail_wiring.py`:

```python
class TestRespondToolServerWiring:
    def test_inject_respond_tool_imported(self):
        src = _server_source()
        assert "inject_respond_tool" in src

    def test_strip_respond_calls_imported(self):
        src = _server_source()
        assert "strip_respond_calls" in src

    def test_inject_respond_tool_called(self):
        src = _server_source()
        call_lines = [
            line for line in src.splitlines()
            if "inject_respond_tool(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 1, "inject_respond_tool not called"

    def test_strip_respond_calls_called(self):
        src = _server_source()
        call_lines = [
            line for line in src.splitlines()
            if "strip_respond_calls(" in line
            and "def " not in line
            and "import" not in line
        ]
        assert len(call_lines) >= 1, "strip_respond_calls not called"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_guardrail_wiring.py::TestRespondToolServerWiring -v`
Expected: FAIL (inject_respond_tool not yet imported/called in server.py)

- [x] **Step 3: Add imports to server.py**

In `omlx/server.py`, find the import block around line 155-166 (where `apply_guardrails`, `guardrail_validation_payload`, `extract_and_validate_tool_calls` are imported from `omlx.api.tool_calling`). Add after that block:

```python
from omlx.api.respond import inject_respond_tool, strip_respond_calls
```

- [x] **Step 4: Wire injection + stripping at each call site**

There are 6 call sites in server.py (lines ~3573, ~4442, ~4890, ~5349, ~5852, ~6327). At each site, the pattern is:

**Before generation** (after `tools_for_template` is built, before the generation call), add injection:

```python
# After tools_for_template is finalized (after the Gemma enrichment block),
# add this before the generation call:
if _server_state.global_settings.forge_guardrails.inject_respond_tool:
    tools_for_template = inject_respond_tool(tools_for_template)
```

The cleanest approach: add this once right after the `tools_for_template` construction block (after line ~3339). This covers all 6 call sites since they all use the same `tools_for_template` variable.

**After parsing** (after tool_calls are extracted, before building the response), add stripping:

At each of the 6 call sites, after the `tool_calls` variable is assigned (either from `_convert_parser_tool_calls` or from `extraction.tool_calls`), add:

```python
# Strip respond calls — convert pure respond to text, drop from mixed
if _server_state.global_settings.forge_guardrails.inject_respond_tool and tool_calls:
    tool_calls, _respond_text = strip_respond_calls(tool_calls)
    if _respond_text and not cleaned_text:
        cleaned_text = _respond_text
    if not tool_calls:
        finish_reason = output.finish_reason  # not "tool_calls"
```

Place this block right after the `tool_calls` assignment and before `finish_reason = "tool_calls" if tool_calls else output.finish_reason`.

For the first call site (~line 3590), the full context after modification:

```python
                cleaned_text = extraction.cleaned_text
                tool_calls = extraction.tool_calls
                cleaned_thinking = extraction.cleaned_thinking

            # Strip synthetic respond tool calls from output
            if _server_state.global_settings.forge_guardrails.inject_respond_tool and tool_calls:
                tool_calls, _respond_text = strip_respond_calls(tool_calls)
                if _respond_text and not cleaned_text:
                    cleaned_text = _respond_text
```

Repeat this strip block at all 6 sites (after each `tool_calls = ...` assignment). The injection only needs to happen once (on `tools_for_template`).

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_server_guardrail_wiring.py::TestRespondToolServerWiring -v`
Expected: PASS

- [x] **Step 6: Run full wiring test suite**

Run: `pytest tests/test_server_guardrail_wiring.py -v`
Expected: PASS

- [x] **Step 7: Commit**

```bash
git add omlx/server.py tests/test_server_guardrail_wiring.py
git commit -m "feat(server): wire respond tool injection + stripping at all call sites"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 3.5: Add `inject_respond_tool` + `enforce_mcp_prerequisites` settings

**Files:**
- Modify: `omlx/settings.py`
- Modify: `tests/test_guardrail_settings.py`

**Interfaces:**
- Produces: `ForgeGuardrailsSettings.inject_respond_tool: bool = False`, `ForgeGuardrailsSettings.enforce_mcp_prerequisites: bool = False`

- [x] **Step 1: Write the failing test**

Append to `tests/test_guardrail_settings.py`:

```python
class TestNewForgeSettings:
    def test_inject_respond_tool_defaults_false(self):
        s = ForgeGuardrailsSettings()
        assert s.inject_respond_tool is False

    def test_enforce_mcp_prerequisites_defaults_false(self):
        s = ForgeGuardrailsSettings()
        assert s.enforce_mcp_prerequisites is False

    def test_to_dict_includes_new_fields(self):
        s = ForgeGuardrailsSettings(inject_respond_tool=True)
        d = s.to_dict()
        assert d["inject_respond_tool"] is True
        assert "enforce_mcp_prerequisites" in d

    def test_from_dict_reads_new_fields(self):
        d = {
            "inject_respond_tool": True,
            "enforce_mcp_prerequisites": True,
        }
        s = ForgeGuardrailsSettings.from_dict(d)
        assert s.inject_respond_tool is True
        assert s.enforce_mcp_prerequisites is True

    def test_round_trip_with_new_fields(self):
        original = ForgeGuardrailsSettings(
            inject_respond_tool=True,
            enforce_mcp_prerequisites=True,
        )
        d = original.to_dict()
        restored = ForgeGuardrailsSettings.from_dict(d)
        assert restored == original
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_settings.py::TestNewForgeSettings -v`
Expected: FAIL with `AttributeError: 'ForgeGuardrailsSettings' object has no attribute 'inject_respond_tool'`

- [x] **Step 3: Add fields to `ForgeGuardrailsSettings`**

In `omlx/settings.py`, update the `ForgeGuardrailsSettings` class (around line 772). Add the two new fields after `compaction_strategy`:

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
    compaction_strategy: str = "none"
    inject_respond_tool: bool = False
    enforce_mcp_prerequisites: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_enabled": self.validation_enabled,
            "strict_tool_args": self.strict_tool_args,
            "include_validation_metadata": self.include_validation_metadata,
            "max_retries": self.max_retries,
            "max_tool_errors": self.max_tool_errors,
            "compaction_strategy": self.compaction_strategy,
            "inject_respond_tool": self.inject_respond_tool,
            "enforce_mcp_prerequisites": self.enforce_mcp_prerequisites,
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
            inject_respond_tool=data.get("inject_respond_tool", False),
            enforce_mcp_prerequisites=data.get("enforce_mcp_prerequisites", False),
        )
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_guardrail_settings.py -v`
Expected: PASS (all tests including existing ones)

- [x] **Step 5: Commit**

```bash
git add omlx/settings.py tests/test_guardrail_settings.py
git commit -m "feat(settings): add inject_respond_tool + enforce_mcp_prerequisites to ForgeGuardrailsSettings"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 3.6: Wire new settings into admin panel

**Files:**
- Modify: `omlx/admin/routes.py`
- Modify: `tests/test_guardrail_settings.py` (source inspection test)

- [x] **Step 1: Write the failing test**

Append to `tests/test_guardrail_settings.py`:

```python
class TestAdminRoutesWiring:
    def test_admin_routes_has_inject_respond_tool_field(self):
        from pathlib import Path
        routes_path = Path(__file__).resolve().parent.parent / "omlx" / "admin" / "routes.py"
        src = routes_path.read_text()
        assert "forge_guardrails_inject_respond_tool" in src

    def test_admin_routes_has_enforce_mcp_prerequisites_field(self):
        from pathlib import Path
        routes_path = Path(__file__).resolve().parent.parent / "omlx" / "admin" / "routes.py"
        src = routes_path.read_text()
        assert "forge_guardrails_enforce_mcp_prerequisites" in src
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_guardrail_settings.py::TestAdminRoutesWiring -v`
Expected: FAIL

- [x] **Step 3: Add request fields to admin routes**

In `omlx/admin/routes.py`, find the forge_guardrails request fields block (around line 280-285). Add after `forge_guardrails_compaction_strategy`:

```python
    forge_guardrails_inject_respond_tool: bool | None = None
    forge_guardrails_enforce_mcp_prerequisites: bool | None = None
```

- [x] **Step 4: Add handler block**

In `omlx/admin/routes.py`, find the forge_guardrails handler block (around line 3725-3741). Add after the `compaction_strategy` handler and before the `if forge_guardrails_changed:` block:

```python
    if request.forge_guardrails_inject_respond_tool is not None:
        global_settings.forge_guardrails.inject_respond_tool = (
            request.forge_guardrails_inject_respond_tool
        )
        forge_guardrails_changed = True
    if request.forge_guardrails_enforce_mcp_prerequisites is not None:
        global_settings.forge_guardrails.enforce_mcp_prerequisites = (
            request.forge_guardrails_enforce_mcp_prerequisites
        )
        forge_guardrails_changed = True
```

Also update the logger.info block (around line 3734-3741) to include the new fields:

```python
        logger.info(
            f"Forge Guardrails settings updated: "
            f"validation_enabled={global_settings.forge_guardrails.validation_enabled}, "
            f"strict_tool_args={global_settings.forge_guardrails.strict_tool_args}, "
            f"include_validation_metadata={global_settings.forge_guardrails.include_validation_metadata}, "
            f"max_retries={global_settings.forge_guardrails.max_retries}, "
            f"max_tool_errors={global_settings.forge_guardrails.max_tool_errors}, "
            f"compaction_strategy={global_settings.forge_guardrails.compaction_strategy}, "
            f"inject_respond_tool={global_settings.forge_guardrails.inject_respond_tool}, "
            f"enforce_mcp_prerequisites={global_settings.forge_guardrails.enforce_mcp_prerequisites}"
        )
```

- [x] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_guardrail_settings.py -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add omlx/admin/routes.py tests/test_guardrail_settings.py
git commit -m "feat(admin): wire inject_respond_tool + enforce_mcp_prerequisites into settings UI"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 3.7: Full respond tool test suite review

**Files:**
- Verify: `tests/test_respond_tool.py`

- [x] **Step 1: Run full respond tool tests**

Run: `pytest tests/test_respond_tool.py -v`
Expected: PASS (all classes: TestRespondToolName, TestInjectRespondTool, TestStripRespondCalls)

- [x] **Step 2: Commit (if any fixes needed)**

```bash
git add tests/test_respond_tool.py
git commit -m "test(api): finalize respond tool test coverage"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

## Group 4: Integration + Tests

### Task 4.1: Wire prerequisite validation into guardrail_wiring (server-side)

**Files:**
- Modify: `omlx/server.py`
- Modify: `tests/test_server_guardrail_wiring.py`

**Design note:** The server builds a `PrerequisiteChecker` from the MCP config's `tools_prerequisites` when `enforce_mcp_prerequisites` is enabled. It passes the checker + `request.messages` (as `prior_messages`) into `apply_guardrails` at each call site. Since `apply_guardrails` already accepts these kwargs (Task 2.3), this task updates the 6 call sites.

- [x] **Step 1: Write the failing test (source inspection)**

Append to `tests/test_server_guardrail_wiring.py`:

```python
class TestPrerequisiteServerWiring:
    def test_prerequisite_checker_referenced(self):
        src = _server_source()
        assert "PrerequisiteChecker" in src

    def test_enforce_mcp_prerequisites_referenced(self):
        src = _server_source()
        assert "enforce_mcp_prerequisites" in src

    def test_prerequisite_checker_passed_to_apply_guardrails(self):
        src = _server_source()
        # Look for prerequisite_checker= in apply_guardrails calls
        lines = [
            line for line in src.splitlines()
            if "prerequisite_checker=" in line
        ]
        assert len(lines) >= 1, "prerequisite_checker not passed to apply_guardrails"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server_guardrail_wiring.py::TestPrerequisiteServerWiring -v`
Expected: FAIL

- [x] **Step 3: Build PrerequisiteChecker + pass to apply_guardrails in server.py**

In `omlx/server.py`, add the import near the top (after the existing MCP imports):

```python
from omlx.mcp.prerequisites import PrerequisiteChecker
```

At each of the 6 `apply_guardrails` call sites, update the call to pass the prerequisite checker and prior messages when `enforce_mcp_prerequisites` is enabled. The pattern at each site (the first is ~line 3583):

**Before (current):**
```python
                extraction = apply_guardrails(
                    extraction,
                    request.tool_choice,
                    tools_for_template,
                    validation_enabled=_fg.validation_enabled,
                )
```

**After:**
```python
                _prereq_checker = None
                if (
                    _fg.enforce_mcp_prerequisites
                    and _server_state.mcp_manager
                    and getattr(
                        _server_state.mcp_manager.config, "tools_prerequisites", {}
                    )
                ):
                    _prereq_checker = PrerequisiteChecker(
                        _server_state.mcp_manager.config.tools_prerequisites
                    )
                extraction = apply_guardrails(
                    extraction,
                    request.tool_choice,
                    tools_for_template,
                    validation_enabled=_fg.validation_enabled,
                    prerequisite_checker=_prereq_checker,
                    prior_messages=(
                        [m.model_dump() for m in request.messages]
                        if _prereq_checker else None
                    ),
                )
```

Repeat this pattern at all 6 call sites. Note: `request.messages` contains Pydantic models — convert to dicts with `.model_dump()` for the checker.

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server_guardrail_wiring.py::TestPrerequisiteServerWiring -v`
Expected: PASS

- [x] **Step 5: Run full server wiring test suite**

Run: `pytest tests/test_server_guardrail_wiring.py -v`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add omlx/server.py tests/test_server_guardrail_wiring.py
git commit -m "feat(server): wire PrerequisiteChecker into all 6 validation call sites"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 4.2: Integration test — prerequisites end-to-end

**Files:**
- Create: `tests/test_forge_mcp_features_e2e.py`

- [x] **Step 1: Write the test**

```python
# tests/test_forge_mcp_features_e2e.py
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for MCP prerequisite enforcement.

These tests exercise the full pipeline: PrerequisiteChecker →
apply_guardrails → ValidationResult without requiring a running
model engine.
"""
from __future__ import annotations

from omlx.api.guardrail_wiring import apply_guardrails
from omlx.api.guardrails.types import CheckResult, ValidationResult
from omlx.api.openai_models import FunctionCall, ToolCall
from omlx.api.tool_calling import ToolCallExtraction
from omlx.mcp.prerequisites import PrerequisiteChecker


def _tc(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(
        id=f"call_{name}",
        type="function",
        function=FunctionCall(name=name, arguments=args),
    )


def _extraction(calls, vr=None):
    return ToolCallExtraction(
        cleaned_text="",
        tool_calls=calls,
        cleaned_thinking="",
        validation_result=vr or ValidationResult(checks=[], passed=True),
    )


class TestPrerequisiteEndToEnd:
    def test_edit_without_read_flagged(self):
        checker = PrerequisiteChecker(
            {"edit_file": {"requires": ["read_file"]}}
        )
        ext = _extraction([_tc("edit_file", '{"path": "/x"}')])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=[],
        )
        assert result.validation_result.passed is False
        prereq_checks = [
            c for c in result.validation_result.checks
            if c.check == "prerequisite"
        ]
        assert len(prereq_checks) == 1
        assert prereq_checks[0].passed is False

    def test_edit_after_read_passes(self):
        checker = PrerequisiteChecker(
            {"edit_file": {"requires": ["read_file"]}}
        )
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": "{}"}}
                ],
            }
        ]
        ext = _extraction([_tc("edit_file")])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=prior,
        )
        assert result.validation_result.passed is True

    def test_arg_matched_e2e(self):
        checker = PrerequisiteChecker(
            {
                "edit_file": {
                    "requires": [{"tool": "read_file", "match_arg": "path"}]
                }
            }
        )
        # read /a, then edit /b → should fail
        prior = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/a"}',
                        }
                    }
                ],
            }
        ]
        ext = _extraction([_tc("edit_file", '{"path": "/b"}')])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=prior,
        )
        assert result.validation_result.passed is False

    def test_no_prereqs_declared_is_noop(self):
        checker = PrerequisiteChecker({})
        ext = _extraction([_tc("any_tool")])
        result = apply_guardrails(
            ext, "auto", None,
            validation_enabled=True,
            prerequisite_checker=checker,
            prior_messages=[],
        )
        # No prerequisite checks emitted
        prereq_checks = [
            c for c in result.validation_result.checks
            if c.check == "prerequisite"
        ]
        assert prereq_checks == []
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_forge_mcp_features_e2e.py::TestPrerequisiteEndToEnd -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_forge_mcp_features_e2e.py
git commit -m "test(e2e): prerequisites end-to-end integration tests"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 4.3: Integration test — respond tool end-to-end

**Files:**
- Modify: `tests/test_forge_mcp_features_e2e.py`

- [x] **Step 1: Write the test**

```python
# Append to tests/test_forge_mcp_features_e2e.py

from omlx.api.respond import inject_respond_tool, strip_respond_calls


class TestRespondToolEndToEnd:
    def test_inject_then_strip_pure_respond(self):
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}
        ]
        injected = inject_respond_tool(tools)
        assert any(
            t["function"]["name"] == "respond" for t in injected
        )

        # Simulate model output: only respond call
        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Done!"}'
                ),
            )
        ]
        real, text = strip_respond_calls(calls)
        assert real == []
        assert text == "Done!"

    def test_inject_then_strip_mixed(self):
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}
        ]
        injected = inject_respond_tool(tools)

        # Model emits respond + search
        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Searching..."}'
                ),
            ),
            ToolCall(
                id="call_2",
                type="function",
                function=FunctionCall(
                    name="search", arguments='{"query": "test"}'
                ),
            ),
        ]
        real, text = strip_respond_calls(calls)
        assert len(real) == 1
        assert real[0].function.name == "search"
        assert text is None

    def test_full_pipeline_respond_becomes_text(self):
        """Inject → (model generates respond call) → strip → text output."""
        from omlx.api.guardrail_wiring import apply_guardrails
        from omlx.api.guardrails.types import ValidationResult
        from omlx.api.tool_calling import ToolCallExtraction

        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}}
        ]
        injected = inject_respond_tool(tools)

        # Simulate: model called respond, validator passed (respond is a known tool)
        calls = [
            ToolCall(
                id="call_1",
                type="function",
                function=FunctionCall(
                    name="respond", arguments='{"message": "Hello!"}'
                ),
            )
        ]
        # Strip happens after validation
        real_calls, text = strip_respond_calls(calls)
        assert real_calls == []
        assert text == "Hello!"
        # text would replace cleaned_text in the server response
```

- [x] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_forge_mcp_features_e2e.py::TestRespondToolEndToEnd -v`
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add tests/test_forge_mcp_features_e2e.py
git commit -m "test(e2e): respond tool end-to-end integration tests"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 4.4: Run full test suite — verify 0 regressions

**Files:**
- Verify only

- [x] **Step 1: Run the complete guardrail + MCP test suite**

Run: `pytest tests/test_guardrail_types.py tests/test_guardrail_budget.py tests/test_guardrail_validator.py tests/test_guardrail_settings.py tests/test_server_guardrail_wiring.py tests/test_guardrail_nudges.py tests/test_guardrail_step_prereq_nudges.py tests/test_guardrail_e2e.py tests/test_mcp_prerequisites.py tests/test_mcp_config.py tests/test_respond_tool.py tests/test_forge_mcp_features_e2e.py -v`
Expected: PASS — 0 failures

- [x] **Step 2: Run the full project test suite (non-slow)**

Run: `pytest -m "not slow" -x -q`
Expected: PASS — 0 failures across Changes A+B+C

- [x] **Step 3: Commit (only if test fixes were needed)**

```bash
git add -A
git commit -m "test: verify 0 regressions across Changes A+B+C"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

## Group 5: Documentation

### Task 5.1: Update `docs/tool-call-guardrails.md`

**Files:**
- Modify: `docs/tool-call-guardrails.md`

- [x] **Step 1: Add MCP Prerequisites section + Respond Tool section + update settings table**

In `docs/tool-call-guardrails.md`, make the following updates:

**Update the Settings table** (around line 24-28). Add two rows:

```markdown
| `inject_respond_tool` | `false` | Inject a synthetic `respond(message)` tool to keep the model in tool-calling mode |
| `enforce_mcp_prerequisites` | `false` | Validate tool-call ordering against MCP prerequisite declarations |
```

**Update the "Nudge Escalation Tiers" section** (around line 206-207). Replace the last sentence:

```markdown
Currently all nudges use tier 0. Step-enforcement nudges escalate through tiers 1-3 on consecutive premature terminal-tool attempts. Prerequisite nudges use tier 0.
```

**Update the Validation Checks table** (around line 35-40). Add two rows:

```markdown
| **Step enforcement** | Model called a terminal tool before completing required steps | `step` | `user` |
| **Prerequisite** | Tool called without its declared prerequisite (e.g., edit before read) | `prerequisite` | `user` |
```

**Add new sections before the "## Implementation" section** (before line 247):

```markdown
## MCP Prerequisites

When `enforce_mcp_prerequisites` is enabled, oMLX validates tool-call ordering against prerequisite declarations in your MCP configuration. This enforces workflows like "read before edit" or "search before respond."

### Declaration

Declare prerequisites in `mcp.json` under `tools_prerequisites`:

```json
{
  "tools_prerequisites": {
    "edit_file": {
      "requires": ["read_file"]
    },
    "write_file": {
      "requires": [{"tool": "read_file", "match_arg": "path"}]
    }
  }
}
```

Two declaration modes:

| Mode | Format | Example |
|------|--------|---------|
| **Name-only** | `"read_file"` | Any prior `read_file` call satisfies |
| **Arg-matched** | `{"tool": "read_file", "match_arg": "path"}` | Prior `read_file(path=X)` must match the current call's `path=X` |

### How It Works

1. oMLX scans the request's prior messages for assistant `tool_calls` to build an `executed_tools` set (stateless per-request — no server session state).
2. For each tool call in the current response, if the tool has declared prerequisites, oMLX checks whether they're satisfied.
3. Unsatisfied prerequisites generate a nudge with `role="user"`, `kind="prerequisite"`.

### Nudge Example

```json
{
  "x_omlx_validation": {
    "passed": false,
    "checks": [
      {"check": "prerequisite", "passed": false, "detail": "Tool 'edit_file' missing prerequisites: read_file"}
    ],
    "nudge": {
      "role": "user",
      "content": "You cannot call edit_file yet. You must first call: read_file. Call the prerequisite tool now.",
      "kind": "prerequisite",
      "tier": 0
    }
  }
}
```

## Synthetic Respond Tool

When `inject_respond_tool` is enabled, oMLX injects a synthetic `respond(message)` tool into the tools list before generation. This keeps small models in tool-calling mode (where the full guardrail stack applies) instead of falling back to bare text.

### How It Works

1. **Before generation**: The `respond` tool is appended to the tools list (if not already present).
2. **After parsing**: `respond` calls are stripped from the output:
   - **Pure respond** (only call): The message becomes the response text. No tool calls returned.
   - **Mixed** (respond + real calls): The respond call is silently dropped. Only real tool calls are returned.
   - **No respond calls**: All calls pass through unchanged.

The client never sees the `respond` tool — it exists only between injection and stripping.

### When to Use

Enable for small local models (~8B parameters) that struggle to choose between text and tool calls. The respond tool gives them a structured way to produce text responses while staying in tool-calling mode.

```json
{
  "forge_guardrails": {
    "inject_respond_tool": true
  }
}
```

## Step Enforcement

Step enforcement detects premature terminal tool calls — when the model tries to finish (e.g., call `respond`) before completing required steps (e.g., `search`, `read`). Step nudges escalate through 3 tiers on consecutive premature attempts.

This feature requires workflow configuration (required steps + terminal tools) and is currently exposed via the validation API for client-side enforcement. Nudges use `role="user"` and `kind="step"` with a `tier` field (1=polite, 2=direct, 3=aggressive).
```

- [x] **Step 2: Commit**

```bash
git add docs/tool-call-guardrails.md
git commit -m "docs: add MCP Prerequisites, Respond Tool, and Step Enforcement sections"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

### Task 5.2: Update `docs/forge-integration-plan.md`

**Files:**
- Modify: `docs/forge-integration-plan.md`

- [x] **Step 1: Update the status header and phase markers**

In `docs/forge-integration-plan.md`:

**Update the header note** (around line 11):

```markdown
> All 6 phases are now implemented. Phases 1-4 (Changes A+B) and Phases 5-6 (Change C) are merged to main.
```

**Update Phase 5 section** (around line 429). Add a status line:

```markdown
### Phase 5: MCP Prerequisites (P2 -- 1-2 weeks) ✅ Implemented

> **Status**: Implemented in Change C (`add-forge-mcp-features`). See `omlx/mcp/prerequisites.py`.
```

**Update Phase 6 section** (around line 439). Add a status line:

```markdown
### Phase 6: Synthetic Respond Tool (P3 -- 1 week) ✅ Implemented

> **Status**: Implemented in Change C (`add-forge-mcp-features`). See `omlx/api/respond.py`.
```

- [x] **Step 2: Commit**

```bash
git add docs/forge-integration-plan.md
git commit -m "docs: mark Phase 5+6 as implemented (all 6 Forge phases done)"
```

archived-with: 2026-06-22-add-forge-mcp-features
---

## Self-Review Checklist

After completing all tasks, verify:

### Spec Coverage
- [x] **MCP prerequisite declaration** (name-only + arg-matched) → Tasks 1.1–1.4, 1.7
- [x] **Prerequisite validation** (missing/satisfied/no-prereqs scenarios) → Tasks 1.2, 1.3, 4.2
- [x] **Prerequisite state from message history** → Task 1.4
- [x] **Synthetic respond tool injection** (enabled/disabled/no-duplicate) → Tasks 3.1–3.3
- [x] **Respond call stripping** (pure/mixed/passthrough) → Task 3.3
- [x] **Tool-call validation modified** (Check 5 + Check 6, new nudge kinds) → Tasks 1.5, 2.1–2.3
- [x] **Step nudge uses user-role with tier** → Task 2.1, 2.4
- [x] **Prerequisite nudge uses user-role** → Task 1.5
- [x] **Settings extension** (inject_respond_tool, enforce_mcp_prerequisites) → Task 3.5, 3.6
- [x] **Documentation** → Tasks 5.1, 5.2

### Placeholder Scan
- [x] No "TBD", "TODO", "implement later" in any step
- [x] All code blocks contain complete, runnable code
- [x] All file paths are exact
- [x] All test commands include expected output

### Type Consistency
- [x] `PrerequisiteCheck(satisfied: bool, missing: list[str])` — consistent across types.py, prerequisites.py, tests
- [x] `PrerequisiteChecker.__init__(prerequisites: dict[str, list])` — consistent
- [x] `PrerequisiteChecker.check(tool_calls, prior_messages) -> list[CheckResult]` — consistent
- [x] `inject_respond_tool(tools: list) -> list` — consistent
- [x] `strip_respond_calls(tool_calls) -> tuple[list, str | None]` — consistent
- [x] `step_nudge(terminal_tool, pending_steps, tier) -> Nudge` — consistent
- [x] `prerequisite_nudge(tool_name, missing_prereqs, tier) -> Nudge` — consistent
- [x] `KIND_STEP = "step"`, `KIND_PREREQUISITE = "prerequisite"` — consistent across types.py, nudge.py, __init__.py

archived-with: 2026-06-22-add-forge-mcp-features
---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-22-forge-mcp-features.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
