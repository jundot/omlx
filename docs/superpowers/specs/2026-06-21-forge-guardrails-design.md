---
comet_change: add-forge-guardrails
role: technical-design
canonical_spec: openspec
---

# Design Doc: Forge Guardrails Foundation

**Change**: `add-forge-guardrails`
**Date**: 2026-06-21
**Status**: Confirmed
**Phase**: comet-design

## Overview

This document specifies the implementation approach for adding tool-call validation, rescue parsing, and `tool_choice` enforcement to oMLX. The design adapts patterns from [Forge](https://github.com/jundot/forge) (IEEE DOI 10.1145/3786335.3813193) to oMLX's stateless, per-request architecture.

**Core principle**: All new behavior is opt-in via settings. Existing clients see identical responses unless they enable the new flags. Validation runs inline (not in a retry loop); corrective nudges are returned as response metadata for clients to act on.

## Architecture Decisions

### AD1: Wrapper function — `extract_and_validate_tool_calls()`

The existing `extract_tool_calls_with_thinking()` (`tool_calling.py:1339`) stays untouched. A new wrapper function handles validation:

```
extract_and_validate_tool_calls()
  ├── extract_tool_calls_with_thinking()    [EXISTING — unchanged]
  │     └── ToolCallExtraction(text, tool_calls)
  └── GuardrailValidator.validate()         [NEW — if enabled]
        └── ValidationResult(checks, nudge, passed)
```

**Rationale**: Zero changes to the existing function means full backward compatibility. The 6 server.py call sites (lines 3567, 4418, 4850, 5292, 5776, 6232) switch from `extract_tool_calls_with_thinking()` to the wrapper when `validation_enabled=True`. When disabled, the wrapper is a passthrough.

### AD2: Accumulate all failures, emit one nudge

All 5 checks (4 validation + `tool_choice` enforcement) run unconditionally. Results accumulate into a `checks` list. The `nudge` is pre-computed from the highest-priority failure.

**Priority order** (highest to lowest):
1. `bare_text` — model emitted text instead of tool calls when tools were expected
2. `unknown_tool` — model called a tool not in the request
3. `malformed_args` — arguments are not a JSON object (dict)
4. `missing_required_params` — required params absent from arguments
5. `tool_choice_enforcement` — `tool_choice` mode violated

**Rationale**: Clients get the complete validation picture (all checks) but only need to act on one nudge per retry. This matches Forge's proven single-nudge pattern and keeps client retry logic simple.

### AD3: Per-endpoint response extension

`x_omlx_validation` as a non-standard top-level JSON field on all 3 endpoints. Streaming: metadata piggybacks on the final SSE chunk before `data: [DONE]`.

### AD4: Stateless validation, client-driven retry

The server validates but does NOT retry. Corrective nudges are returned via `x_omlx_validation` for clients to append before retrying. No server-side error budgets or retry loops in this change (deferred to Change B).

## Core Data Structures

### `omlx/api/guardrails/types.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class CheckResult:
    """Result of a single validation check."""
    check: Literal[
        "bare_text",              # text when tools expected
        "unknown_tool",           # tool name not in tools list
        "malformed_args",         # args not a dict
        "missing_required_params", # required param missing
        "tool_choice_enforcement" # tool_choice mode violated
    ]
    passed: bool
    detail: str | None = None

@dataclass(frozen=True)
class Nudge:
    """Corrective message for client to append before retry."""
    role: Literal["user", "tool"]
    content: str
    kind: Literal["retry", "unknown_tool", "tool_arg_validation"]

    def to_message(self) -> dict:
        """Convert to chat message format for client retry."""
        return {"role": self.role, "content": self.content}

@dataclass(frozen=True)
class ValidationResult:
    """Accumulated validation results for a tool-call response."""
    checks: list[CheckResult]
    nudge: Nudge | None = None
    passed: bool = False

    def to_dict(self) -> dict:
        """Serialize for x_omlx_validation response extension."""
        result = {
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
            }
        return result
```

### Nudge kind → role mapping

| Kind | Role | Channel | Rationale |
|------|------|---------|-----------|
| `retry` | `user` | User | Bare-text correction; user-role instruction |
| `unknown_tool` | `tool` | Tool-result | Models expect "tool call failed → try again" wire shape |
| `tool_arg_validation` | `tool` | Tool-result | Same — tool-channel correction |

This mapping follows Forge's design (nudge.py:36-47) where tool-channel corrections use `role="tool"` because models are pretrained on the "tool call failed" pattern and attend to it better than trailing user messages.

## Validation Flow

### `omlx/api/guardrails/validator.py`

```python
class GuardrailValidator:
    """Stateless validator for parsed tool-call responses."""

    CHECK_PRIORITY = [
        "bare_text", "unknown_tool", "malformed_args",
        "missing_required_params", "tool_choice_enforcement"
    ]

    def __init__(self, tools: list[dict] | None):
        self._tool_schemas: dict[str, dict] = {}
        self._tool_names: set[str] = set()
        if tools:
            for tool in tools:
                func = tool.get("function", tool)
                name = func.get("name", "")
                self._tool_names.add(name)
                self._tool_schemas[name] = func.get("parameters", {})

    def validate(
        self,
        extraction: ToolCallExtraction,
        tool_choice: Any = None,
        has_tools: bool = True,
    ) -> ValidationResult:
        checks: list[CheckResult] = []

        # Check 1: bare-text-when-tools-expected
        checks.append(self._check_bare_text(extraction, tool_choice, has_tools))

        # Check 2: unknown tool names
        for tc in (extraction.tool_calls or []):
            if tc.function.name not in self._tool_names:
                checks.append(CheckResult(
                    check="unknown_tool", passed=False,
                    detail=f"Tool '{tc.function.name}' does not exist. "
                           f"Available: {', '.join(sorted(self._tool_names))}"
                ))
            else:
                checks.append(CheckResult(check="unknown_tool", passed=True))

        # Check 3: malformed args (non-dict)
        for tc in (extraction.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
            except (json.JSONDecodeError, TypeError):
                args = None
            if not isinstance(args, dict):
                checks.append(CheckResult(
                    check="malformed_args", passed=False,
                    detail=f"Tool '{tc.function.name}' had malformed arguments. "
                           f"Got type: {type(args).__name__}. Required: JSON object (dict)."
                ))
            else:
                checks.append(CheckResult(check="malformed_args", passed=True))

        # Check 4: missing required params (oMLX-original, uses validate_json_schema)
        for tc in (extraction.tool_calls or []):
            checks.append(self._check_missing_params(tc))

        # Check 5: tool_choice enforcement
        checks.append(self._check_tool_choice(extraction, tool_choice))

        # Compute result
        passed = all(c.passed for c in checks)
        nudge = self._select_nudge(checks, extraction) if not passed else None
        return ValidationResult(checks=checks, nudge=nudge, passed=passed)
```

### Wrapper function

```python
# omlx/api/tool_calling.py (additions)

@dataclass
class ToolCallExtraction:
    """Extended with optional validation result."""
    text: str
    tool_calls: list[ToolCall] | None
    thinking: str | None = None
    validation_result: ValidationResult | None = None  # NEW field

def extract_and_validate_tool_calls(
    thinking_content: str,
    regular_content: str,
    tokenizer,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    settings: "ForgeGuardrailsSettings | None" = None,
) -> ToolCallExtraction:
    """Wrapper: extract tool calls, then validate if enabled."""
    extraction = extract_tool_calls_with_thinking(
        thinking_content, regular_content, tokenizer, tools
    )
    if settings and settings.validation_enabled and tools:
        validator = GuardrailValidator(tools)
        extraction.validation_result = validator.validate(
            extraction, tool_choice, has_tools=bool(tools)
        )
    return extraction
```

## Request/Response Flow

### Non-streaming chat completion

```
POST /v1/chat/completions
  │
  ├── server.py:create_chat_completion()
  │     ├── Parse ChatCompletionRequest (existing)
  │     ├── Resolve effective_tools (existing, line 3318)
  │     ├── engine.generate() → raw model output (existing)
  │     ├── extract_and_validate_tool_calls()  ← NEW
  │     │     ├── extract_tool_calls_with_thinking()  [existing]
  │     │     └── GuardrailValidator.validate()  [new, if enabled]
  │     ├── Build ChatCompletionResponse (existing)
  │     ├── Attach x_omlx_validation if include_validation_metadata  ← NEW
  │     └── Return response
  │
  └── Response:
        {
          "choices": [...],
          "x_omlx_validation": {          ← NEW (if enabled)
            "passed": false,
            "checks": [...],
            "nudge": {"role": "tool", "content": "...", "kind": "unknown_tool"}
          }
        }
```

### Streaming chat completion

```
POST /v1/chat/completions (stream=true)
  │
  ├── Stream content/tool_calls as parsed (existing behavior)
  │     data: {"choices": [{"delta": {"content": "..."}}]}
  │     data: {"choices": [{"delta": {"tool_calls": [...]}}]}
  │
  ├── After stream completes (all tool calls parsed):
  │     Run validation on accumulated extraction
  │
  └── Final chunk with validation metadata:
        data: {"choices": [...], "x_omlx_validation": {...}}
        data: [DONE]
```

## Response Extension Format

### OpenAI `/v1/chat/completions`

Non-streaming: top-level `x_omlx_validation` field on the JSON response body.
Streaming: `x_omlx_validation` on the final SSE data event before `data: [DONE]`.

### Anthropic `/v1/messages`

Top-level `x_omlx_validation` field. Anthropic SDKs ignore unknown top-level fields. For streaming, a final SSE event carries the metadata.

### Responses API `/v1/responses`

Same pattern — top-level `x_omlx_validation`. The Responses API supports arbitrary metadata fields.

### Serialization

```json
{
  "x_omlx_validation": {
    "passed": false,
    "checks": [
      {"check": "bare_text", "passed": true},
      {"check": "unknown_tool", "passed": false, "detail": "Tool 'nonexistent' does not exist. Available: search, read, write"},
      {"check": "malformed_args", "passed": false, "detail": "Tool 'nonexistent' had malformed arguments. Got type: str. Required: JSON object (dict)."},
      {"check": "missing_required_params", "passed": true},
      {"check": "tool_choice_enforcement", "passed": true}
    ],
    "nudge": {
      "role": "tool",
      "content": "Tool 'nonexistent' does not exist. Available: search, read, write. Call one of them.",
      "kind": "unknown_tool"
    }
  }
}
```

## Settings Integration

### `omlx/settings.py` — `ForgeGuardrailsSettings`

```python
@dataclass
class ForgeGuardrailsSettings:
    """Guardrail validation settings (opt-in by default)."""
    validation_enabled: bool = False           # Enable post-parse validation
    strict_tool_args: bool = False             # Preserve malformed args (don't coerce to "{}")
    include_validation_metadata: bool = False  # Include x_omlx_validation in responses

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_enabled": self.validation_enabled,
            "strict_tool_args": self.strict_tool_args,
            "include_validation_metadata": self.include_validation_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ForgeGuardrailsSettings":
        return cls(
            validation_enabled=data.get("validation_enabled", False),
            strict_tool_args=data.get("strict_tool_args", False),
            include_validation_metadata=data.get("include_validation_metadata", False),
        )
```

Composed into `GlobalSettings` as `forge_guardrails: ForgeGuardrailsSettings`, following the `CompressionSettings` pattern (settings.py:746). Exposed via admin panel `GlobalSettingsRequest` (routes.py:206) and `update_global_settings` handler (routes.py:3234).

## Testing Strategy

### Test pyramid

| Layer | Count | Approach | Mocking |
|-------|-------|----------|---------|
| Unit | 45+ | Pure functions, construct `ToolCallExtraction` inline | None needed |
| Integration | 7 | FastAPI TestClient + enabled settings | `monkeypatch` model output |
| E2E | 3 | Full request → mocked model → verify response | `monkeypatch` engine.generate |

### Unit test structure

```python
# tests/test_guardrail_validator.py
class TestBareTextCheck:
    def test_text_when_tools_expected_fails(self): ...
    def test_text_when_tool_choice_none_passes(self): ...
    def test_tool_calls_present_passes(self): ...

class TestUnknownToolCheck:
    def test_known_tool_passes(self): ...
    def test_unknown_tool_fails_with_nudge(self): ...
    def test_mixed_known_unknown_reports_unknown(self): ...

class TestMalformedArgsCheck:
    def test_dict_args_passes(self): ...
    def test_string_args_fails(self): ...
    def test_array_args_fails(self): ...

class TestMissingRequiredParams:
    def test_all_required_present_passes(self): ...
    def test_missing_required_fails(self): ...
    def test_no_required_field_passes(self): ...
    def test_empty_parameters_schema_passes(self): ...

class TestCheckOrdering:
    def test_bare_text_takes_priority(self): ...
    def test_unknown_tool_before_malformed_args(self): ...

class TestToolChoiceEnforcement:
    def test_required_rejects_bare_text(self): ...
    def test_named_tool_filters_others(self): ...
    def test_none_suppresses_tool_calls(self): ...
```

### Integration test structure

```python
# tests/test_guardrail_e2e.py
def test_validation_metadata_in_response(client, monkeypatch, enable_guardrails):
    """Enable validation, mock model to emit unknown tool, verify response."""
    monkeypatch_model_output(monkeypatch, '<tool_call>{"name":"bad","arguments":{}}</tool_call>')
    response = client.post("/v1/chat/completions", json={
        "model": "test", "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {...}}}]
    })
    body = response.json()
    assert "x_omlx_validation" in body
    assert not body["x_omlx_validation"]["passed"]
```

## Implementation Sequence

**Critical path** (sequential dependencies):

```
Task Group 1 (Types) → Task Group 2 (Nudges) → Task Group 3 (Validator)
                                                 ↓
Task Group 4 (Rescue parsers) ──────────────→ Task Group 5 (Strict args)
                                                 ↓
Task Group 6 (tool_choice)                       │
                                                 ↓
Task Group 7 (Settings) ─────────────────────→ Task Group 8 (Server wiring)
                                                 ↓
                                          Task Group 9 (Tests)
                                                 ↓
                                          Task Group 10 (Docs)
```

**Parallelizable**: Groups 1-2 (types + nudges) can start in parallel. Group 4 (rescue parsers) is independent of groups 2-3 and can run in parallel. Group 6 (tool_choice) depends only on group 1 (types).

**Critical path length**: 1 → 3 → 5 → 8 → 9 (5 sequential groups).

## Error Handling Matrix

| Scenario | Behavior |
|----------|----------|
| Tool definition missing `parameters` field | Treat as empty schema `{}` — no required params, check passes |
| Tool definition has `parameters: {}` | Same — no required params |
| `json.loads(arguments)` fails (malformed JSON string) | Treat as malformed args (non-dict) — check fails |
| `arguments` is already a dict (not string) | Validate directly — no json.loads needed |
| `arguments` is `None` | Malformed args — check fails |
| Tool schema `required` field is not a list | Skip missing-params check (defensive) — log warning |
| Validator construction fails (bad tools list) | Skip validation entirely — log warning, return extraction without validation result |
| `x_omlx_validation` serialization fails | Omit from response — log warning (shouldn't happen with frozen dataclasses) |

## Open Questions Resolved

1. **Q: Should `x_omlx_validation` be namespaced per-endpoint?**
   A: No — same `x_omlx_validation` top-level field across all 3 endpoints. Simpler for clients; SDKs tolerate unknown fields.

2. **Q: Should validation be permissive or strict for unknown `tool_choice` values?**
   A: Reject with HTTP 400 at request validation layer (`openai_models.py`). Separate from runtime enforcement.

3. **Q: Should rescue parsers be individually toggleable?**
   A: No — single `rescue_parsing_enabled` flag for Change A. Per-parser toggles deferred unless false positives appear in practice.

## References

- Forge source: `../forge/src/forge/guardrails/` (response_validator.py, nudge.py, guardrails.py)
- Forge rescue parsers: `../forge/src/forge/prompts/templates.py`
- Integration plan: `docs/forge-integration-plan.md` (Phase 1 + Phase 2)
- OpenSpec proposal: `openspec/changes/add-forge-guardrails/proposal.md`
- OpenSpec specs: `openspec/changes/add-forge-guardrails/specs/*/spec.md`
- oMLX integration points: `extract_tool_calls_with_thinking()` at `tool_calling.py:1339`, `_serialize_tool_call_arguments()` at `tool_calling.py:94`, `validate_json_schema()` at `tool_calling.py:1948`
