# Comet Design Handoff

- Change: add-forge-guardrails
- Phase: design
- Mode: compact
- Context hash: f63064b9df13f10eabcdada375b3d2b6ceae595f9d694d656cbd5c3d9402a9d3

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/add-forge-guardrails/proposal.md

- Source: openspec/changes/add-forge-guardrails/proposal.md
- Lines: 1-50
- SHA256: 06e5df3df147122333de2316b10be4fdcca48a230c66853c7edbb7f09c3625f3

```md
## Why

oMLX currently provides tool-call parsing but no validation or self-correction. When a model emits malformed JSON arguments, hallucinates a tool name, or produces bare text instead of a tool call, oMLX either silently corrupts the response (coercing bad args to `"{}"` at `tool_calling.py:94`) or drops it with only a `logger.warning`. The `tool_choice` parameter is accepted but never enforced (only `"none"` suppresses tools at `server.py:3307`; `"required"` and named-tool filtering are unhandled). This makes tool-calling unreliable for the 8B–14B local models oMLX primarily serves, where malformations are common. Forge (IEEE DOI 10.1145/3786335.3813193) demonstrates that validation + corrective feedback improves small-model tool-calling success from single-digit to 84%+. This change brings the foundational layer of that reliability stack to oMLX.

## What Changes

- **NEW: Guardrail validation middleware** (`omlx/api/guardrails/`) — validates every parsed tool-call response before it reaches the client. Four checks: unknown tool name, malformed arguments (non-dict), missing required parameters (validated against tool JSON schema using the existing `validate_json_schema()`), and bare-text-when-tools-expected.
- **NEW: Corrective nudge generation** — when validation fails, generates a structured nudge message (role, content, kind) that clients can append before retrying. Two channels: user-role (bare text corrections) and tool-result-role (tool call failures — models are pretrained on this wire shape).
- **NEW: Rescue parser additions** — rehearsal syntax (`tool_name[ARGS]{...json...}` for reasoning models) and improved Mistral bracket-tag parsing (brace-balance scan with string/escape awareness for nested JSON).
- **NEW: `tool_choice` enforcement** — `"none"` (already works), `"required"` (reject bare text, require ≥1 tool call), `{"type":"function","function":{"name":"X"}}` (filter to named tool only), `"auto"` (pass through).
- **CHANGED: `_serialize_tool_call_arguments()`** — stops silently coercing non-dict args to `"{}"`. Behind `strict_tool_args` feature flag (default off for backward compat). When flag is on, malformed args are preserved and surfaced via validation results.
- **NEW: `x_omlx_validation` response extension** — optional (opt-in via `include_validation_metadata` setting) non-standard extension carrying validation check results + suggested nudge. Zero breaking changes when disabled.
- **NEW: Server settings** — `guardrail_validation` (bool, default off), `strict_tool_args` (bool, default off), `include_validation_metadata` (bool, default off). All live-mutable via admin panel (modeled on `CompressionSettings` at `settings.py:746`).

**Non-breaking by default**: all new behavior is opt-in via settings. Existing clients see identical responses unless they enable the new flags. This is Change A of 3 — error budgets/retry loops (Change B) and MCP prerequisites/respond tool (Change C) are deferred.

## Capabilities

### New Capabilities
- `tool-call-validation`: Post-parse validation of tool-call responses (unknown tool, malformed args, missing required params, bare text detection) with corrective nudge generation. Stateless per-request.
- `tool-choice-enforcement`: Enforcement of `tool_choice` semantics (`"none"`, `"auto"`, `"required"`, named-tool filtering) after parsing and validation.
- `tool-call-rescue-parsing`: Additional rescue parser strategies (rehearsal syntax, improved Mistral bracket-tag) to extract tool calls from text that fails native parsing.

### Modified Capabilities
<!-- No existing specs to modify — openspec/specs/ is empty. All capabilities above are new. -->

## Impact

**Code affected:**
- `omlx/api/tool_calling.py` — add rescue parsers, change `_serialize_tool_call_arguments()` coercion (flagged), add validation hook in `extract_tool_calls_with_thinking()` (single chokepoint covering all 6 server.py call sites)
- `omlx/api/guardrails/` — NEW package: `__init__.py`, `validator.py` (ResponseValidator), `nudge.py` (Nudge dataclass + generators), `types.py` (CheckResult, ValidationResult)
- `omlx/api/tool_choice.py` — NEW module: `enforce_tool_choice()` function
- `omlx/server.py` — wire validation + tool_choice enforcement at response construction (non-streaming at line 3600, streaming at line 4418+)
- `omlx/settings.py` — add `ForgeGuardrailsSettings` dataclass to `GlobalSettings`
- `omlx/admin/routes.py` — add guardrail toggle fields to `GlobalSettingsRequest` and `update_global_settings` handler
- `omlx/api/openai_models.py` — add `tool_choice` field validation (reject malformed values)
- `tests/` — unit tests for all validators, nudges, rescue parsers, tool_choice modes

**APIs affected:**
- `POST /v1/chat/completions`, `POST /v1/messages`, `POST /v1/responses` — optional `x_omlx_validation` extension in responses (non-breaking, opt-in)
- Admin panel — new guardrail settings section

**Dependencies:**
- Uses existing `jsonschema` library (already imported at `tool_calling.py:29` for `validate_json_schema`)
- No new external dependencies

**Reference:**
- Forge source: `../forge/src/forge/guardrails/` (response_validator.py, nudge.py, guardrails.py)
- Forge rescue parsers: `../forge/src/forge/prompts/templates.py`
- Integration plan: `docs/forge-integration-plan.md` (Phase 1 + Phase 2)
```

## openspec/changes/add-forge-guardrails/design.md

- Source: openspec/changes/add-forge-guardrails/design.md
- Lines: 1-131
- SHA256: e5117d89c79f65dd7f629eea01759a212715458b768cc9d371529d7ee6b673b0

[TRUNCATED]

```md
## Context

oMLX serves 8B–14B local models on Apple Silicon via OpenAI/Anthropic-compatible APIs. Its current tool-call pipeline (`parse_tool_calls()` → `_serialize_tool_call_arguments()`) parses model output but performs no post-parse validation. Malformed arguments are silently coerced to `"{}"` (`tool_calling.py:94-120`), unknown tool names pass through, and `tool_choice` is accepted but only `"none"` is enforced (`server.py:3307`). The existing `validate_json_schema()` (`tool_calling.py:1948`) validates `response_format` JSON outputs but is never applied to tool-call arguments.

Forge (`../forge/`) is an IEEE-published reliability layer whose `ResponseValidator` + `Nudge` architecture improves small-model tool-calling success from <10% to 84%+. However, forge runs a stateful agentic loop (`WorkflowRunner`); oMLX is a stateless per-request API server. This design adapts forge's validation/nudge patterns to oMLX's stateless architecture — validation happens inline, corrective nudges are returned as response metadata for clients to act on.

Three chat endpoints (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) each call `extract_tool_calls_with_thinking()` (`tool_calling.py:1339`) at 6 sites in `server.py` (lines 3567, 4418, 4850, 5292, 5776, 6232). All 6 funnel through this single function, making it the natural validation insertion point.

**Settings architecture note**: oMLX has two config systems. `config.py` is static startup config (CLI/env, read once at boot). `settings.py` contains `GlobalSettings` — live-mutable, admin-panel-editable, persisted to `~/.omlx/settings.json`. Guardrail toggles belong in `settings.py`, modeled on `CompressionSettings` (`settings.py:746`).

## Goals / Non-Goals

**Goals:**
- Validate every parsed tool-call response before it reaches the client (unknown tool, malformed args, missing required params, bare text)
- Generate corrective nudge messages clients can use to implement retry loops
- Enforce all `tool_choice` modes (`"none"`, `"auto"`, `"required"`, named-tool)
- Add 2 rescue parsers (rehearsal syntax, improved Mistral bracket-tag)
- Surface validation metadata via optional `x_omlx_validation` response extension
- All new behavior opt-in, zero breaking changes by default

**Non-Goals:**
- Server-side retry loops / error budgets (deferred to Change B)
- Context compaction (deferred to Change B)
- MCP prerequisites / step enforcement (deferred to Change C)
- Synthetic respond tool (deferred to Change C)
- Modifying `ToolExecutor` or `/v1/mcp/execute` (not in chat flow — tool execution is client-side)
- Building a stateful admin chat API (staying stateless)

## Decisions

### D1: Single-chokepoint validation at `extract_tool_calls_with_thinking()`

**Decision**: Insert validation inside `extract_tool_calls_with_thinking()` (`tool_calling.py:1339`) rather than at each of the 6 server.py call sites.

**Rationale**: All 3 endpoints (chat/messages/responses) × 2 modes (streaming/non-streaming) funnel through this one function. A validation hook here covers all 6 sites with a single code change. The alternative — 6 separate insertions in the 6652-line `server.py` — is error-prone and fragile.

**Alternatives considered**:
- FastAPI middleware: rejected — tool-call validation needs parsed tool calls + tool definitions, which are only available deep in the handler, not at the HTTP layer.
- Per-endpoint insertion: rejected — 6× duplication, high maintenance burden.

**Implementation**: `extract_tool_calls_with_thinking()` gains an optional `tools` parameter (already passed at most call sites) and an optional `validator` callback. When validation is enabled (via settings), the function returns a `ToolCallExtraction` that includes a `validation_result` field.

### D2: Stateless validation, client-driven retry

**Decision**: Server validates but does NOT retry. Validation results + suggested nudge are returned via `x_omlx_validation` response extension. Clients (Claude Code, OpenCode, custom) implement their own retry loops.

**Rationale**: oMLX is stateless per-request. A server-side retry loop would require holding the generation pipeline open across multiple inference passes, breaking the request/response contract and complicating streaming. Forge's retry loop works because it owns the agentic loop; oMLX doesn't.

**Alternatives considered**:
- Server-side retry with error budget: rejected for Change A — requires stateful session management (Change B territory).
- No retry support at all: rejected — the nudge metadata is the building block that makes client-side retry practical.

**Nudge shape** (adapted from forge `Nudge` dataclass):
```
Nudge:
  role: "user" | "tool"           # user-role for bare-text; tool-role for tool errors
  content: str                     # corrective message
  kind: "retry" | "unknown_tool" | "tool_arg_validation" | "missing_params"
```

**Channel mapping** (critical invariant from forge): `role="tool"` for `unknown_tool` and `tool_arg_validation` kinds (models expect "tool call failed → try again" wire shape); `role="user"` for `retry` kind (bare-text correction).

### D3: Four validation checks (forge does 3, oMLX adds 1)

**Decision**: Implement 4 checks in order: (1) bare-text-when-tools-expected, (2) unknown tool name, (3) malformed args (non-dict), (4) missing required params. Checks 1–3 adapt forge's `ResponseValidator` (`../forge/src/forge/guardrails/response_validator.py`). Check 4 is oMLX-original, using the existing `validate_json_schema()`.

**Rationale**: The integration plan (`docs/forge-integration-plan.md` §3.1) claims forge does 4 checks, but codebase exploration confirmed forge does only 3 — no JSON-schema required-params validation. oMLX already has `validate_json_schema()` (`tool_calling.py:1948`, using `jsonschema.validate`) that validates `response_format` outputs. Extending it to tool-call arguments is low-effort and high-value: missing required params currently cause runtime failures with unclear errors.

**Alternatives considered**:
- Port forge's 3 checks only: rejected — misses the highest-impact validation for structured tool use.
- Full Pydantic model validation per tool: rejected — tools provide JSON Schema, not Pydantic models. `jsonschema.validate` is the right tool and already imported.

**Check ordering rationale**: Bare-text check first because it's cheapest (no tool schema lookup). Unknown-tool check second because it determines whether subsequent per-tool checks can run (if the tool name is unknown, we can't look up its schema). Malformed-args (isinstance dict) before missing-required-params (schema check) because the schema check requires a dict to validate against.

### D4: Feature-flagged `_serialize_tool_call_arguments()` change

**Decision**: The current coercion of non-dict args to `"{}"` (`tool_calling.py:94-120`) is preserved as default behavior. When `strict_tool_args=True` (settings flag, default off), malformed args are preserved as-is and surfaced via validation results.

**Rationale**: Existing clients depend on always-string arguments. Changing the default would break clients that don't expect non-string argument values. The flag allows opt-in to the stricter behavior.

```

Full source: openspec/changes/add-forge-guardrails/design.md

## openspec/changes/add-forge-guardrails/tasks.md

- Source: openspec/changes/add-forge-guardrails/tasks.md
- Lines: 1-81
- SHA256: bed5ce2a688bd989f246b0d98175551e050a3b22b9ef9341b3ba1419691e1a1b

[TRUNCATED]

```md
## 1. Guardrails Package Foundation

- [ ] 1.1 Create `omlx/api/guardrails/__init__.py` with public API exports (`GuardrailValidator`, `ValidationResult`, `Nudge`, `CheckResult`)
- [ ] 1.2 Create `omlx/api/guardrails/types.py` with frozen dataclasses: `ValidationResult` (checks list, nudge, passed bool), `CheckResult` (check name, passed, detail), `Nudge` (role, content, kind)
- [ ] 1.3 Define nudge kind constants: `KIND_RETRY="retry"`, `KIND_UNKNOWN_TOOL="unknown_tool"`, `KIND_TOOL_ARG_VALIDATION="tool_arg_validation"` and role/channel mapping (`TOOL_CHANNEL_KINDS`, `TOOL_ERROR_KINDS`)

## 2. Nudge Generation

- [ ] 2.1 Create `omlx/api/guardrails/nudge.py` with 4 nudge generator functions: `retry_nudge()`, `unknown_tool_nudge(tool_name, available_tools)`, `tool_arg_validation_nudge(tool_name, args)`, `missing_params_nudge(tool_name, missing_params)`
- [ ] 2.2 Each nudge function returns a `Nudge` with appropriate `role` ("user" for retry, "tool" for tool errors) and `kind` — reference forge's `prompts/nudges.py` for message text patterns
- [ ] 2.3 Unit tests for all 4 nudge generators (verify role, kind, and message content for representative inputs)

## 3. Response Validator (4 Checks)

- [ ] 3.1 Create `omlx/api/guardrails/validator.py` with `GuardrailValidator` class, constructor takes `tools: list[ToolDef]` and builds a name→schema lookup
- [ ] 3.2 Implement Check 1 (bare-text-when-tools-expected): if no tool calls parsed AND tools were provided AND `tool_choice != "none"`, return `ValidationResult` with `retry_nudge()`
- [ ] 3.3 Implement Check 2 (unknown tool name): for each parsed tool call, if name not in tools lookup, return `ValidationResult` with `unknown_tool_nudge()`
- [ ] 3.4 Implement Check 3 (malformed args / non-dict): for each tool call with non-dict arguments, return `ValidationResult` with `tool_arg_validation_nudge()` (adapted from forge `response_validator.py:100-110`)
- [ ] 3.5 Implement Check 4 (missing required params / oMLX-original): for each tool call, load `json.loads(tc.function.arguments)`, look up the tool's JSON Schema, extract `required` array, check each required param exists; on failure return `ValidationResult` with `missing_params_nudge()` — uses existing `validate_json_schema()` at `tool_calling.py:1948`
- [ ] 3.6 Check ordering: bare-text → unknown-tool → malformed-args → missing-params (first failure wins; remaining checks skipped)
- [ ] 3.7 Handle edge case: tool with `parameters={}` or no `required` field → Check 4 passes trivially (no false positives)
- [ ] 3.8 Unit tests for each check independently + integration test for check ordering

## 4. Rescue Parsers

- [ ] 4.1 Add rehearsal syntax parser `_parse_rehearsal_tool_calls(text)` to `omlx/api/tool_calling.py` — regex `r"(\w+)\[ARGS\](\{.*?\})"` with `re.DOTALL`, validate brace content as JSON, return list of parsed tool calls or empty list
- [ ] 4.2 Add improved Mistral bracket-tag parser using brace-balance scanner (track `in_string` and `escape` flags for nested JSON) — reference `../forge/src/forge/prompts/templates.py:206-234`
- [ ] 4.3 Integrate both parsers into `_parse_tool_calls_impl` chain at `tool_calling.py:1108` as last-resort fallbacks (after existing parsers, before final marker-strip at line 1275)
- [ ] 4.4 Unit tests: rehearsal syntax with single/multiple/invalid-JSON cases; Mistral parser with nested objects, string-escaped braces, escaped quotes

## 5. Strict Args Mode

- [ ] 5.1 Modify `_serialize_tool_call_arguments()` at `tool_calling.py:94` — accept `strict: bool = False` parameter; when `strict=True`, preserve original args (don't coerce to `"{}"`) and return a tuple `(serialized, is_malformed)` or attach validation flag
- [ ] 5.2 Thread the `strict` flag from `ForgeGuardrailsSettings.strict_tool_args` through `extract_tool_calls_with_thinking()` and the parser chain
- [ ] 5.3 Ensure backward compatibility: when `strict=False` (default), behavior is identical to current (coerce to `"{}"`)
- [ ] 5.4 Unit tests: strict mode preserves malformed args; default mode coerces as before

## 6. tool_choice Enforcement Module

- [ ] 6.1 Create `omlx/api/tool_choice.py` with `enforce_tool_choice(tool_calls, tool_choice, has_text, tools) -> tuple[list[ToolCall], ValidationResult | None]`
- [ ] 6.2 Implement `"none"` mode: suppress all tool calls (also enforced upstream at `server.py:3307`, but duplicate here for layering consistency)
- [ ] 6.3 Implement `"auto"` mode: pass through (no-op)
- [ ] 6.4 Implement `"required"` mode: if `not tool_calls and has_text`, return nudge `kind="retry"`
- [ ] 6.5 Implement named-tool mode `{"type":"function","function":{"name":"X"}}`: filter `tool_calls` to only calls to `X`, flag rejected calls with `unknown_tool` nudge
- [ ] 6.6 Add request-time validation in `omlx/api/openai_models.py:292` — reject malformed `tool_choice` values (strings other than the 4 valid modes, dicts missing required structure) with HTTP 400
- [ ] 6.7 Unit tests for all 5 modes + invalid-value rejection

## 7. Settings Integration

- [ ] 7.1 Add `ForgeGuardrailsSettings` dataclass to `omlx/settings.py` (near `CompressionSettings` at line 746) with fields: `validation_enabled: bool = False`, `strict_tool_args: bool = False`, `include_validation_metadata: bool = False`, plus `to_dict()`/`from_dict()` methods
- [ ] 7.2 Compose `forge_guardrails: ForgeGuardrailsSettings` field into `GlobalSettings` (near line 799), wire into `GlobalSettings.to_dict()` (line 1453), `from_dict()`/load (near line 888), and `validate()` (line 1237)
- [ ] 7.3 Add fields to `GlobalSettingsRequest` in `omlx/admin/routes.py:206`: `forge_guardrails_validation_enabled`, `forge_guardrails_strict_tool_args`, `forge_guardrails_include_validation_metadata`
- [ ] 7.4 Wire application in `update_global_settings` handler at `routes.py:3234` (mirror the Claude Code block at lines 3650-3679)
- [ ] 7.5 Expose in GET response (around line 4366 where `claude_code_context_scaling_enabled` is returned)
- [ ] 7.6 Unit tests for settings round-trip (set → get → verify)

## 8. Server Wiring (Single Chokepoint)

- [ ] 8.1 Modify `extract_tool_calls_with_thinking()` at `tool_calling.py:1339` — accept `tools` and `validator` optional params; when validator is provided, run validation on extracted tool calls; return extended `ToolCallExtraction` with optional `validation_result` field
- [ ] 8.2 Verify all 6 call sites in `server.py` (lines 3567, 4418, 4850, 5292, 5776, 6232) pass through the `tools` param (most already do) and consume the new validation result
- [ ] 8.3 Wire `enforce_tool_choice()` after validation, before response construction — non-streaming chat at line 3600, streaming at line 4418+
- [ ] 8.4 Build `x_omlx_validation` response extension for non-streaming responses (when `include_validation_metadata=True`) — include `checks` array and `nudge` object
- [ ] 8.5 Build streaming SSE event for validation metadata — emit as final event before `data: [DONE]`
- [ ] 8.6 Repeat wiring for `/v1/messages` (Anthropic, lines 4850, 5292) and `/v1/responses` (Responses API, lines 5776, 6232) — adapt extension location to endpoint convention (Anthropic may use `metadata` field)
- [ ] 8.7 Integration tests: send real tool-call requests with deliberately malformed model output, verify validation metadata + enforcement + nudges

## 9. Test Suite

- [ ] 9.1 `tests/test_guardrail_validator.py` — 4 checks independently + ordering + edge cases (loose schemas, empty args)
- [ ] 9.2 `tests/test_guardrail_nudges.py` — all 4 nudge generators (role, kind, content)
- [ ] 9.3 `tests/test_rescue_parsers.py` — rehearsal + Mistral + integration with existing parser chain
- [ ] 9.4 `tests/test_tool_choice_enforcement.py` — all 5 modes + invalid value rejection
- [ ] 9.5 `tests/test_guardrail_settings.py` — settings round-trip + admin handler
- [ ] 9.6 `tests/test_guardrail_e2e.py` — end-to-end through the API: enable flags via settings, send request with malformed tool output (mocked model), verify response extension
- [ ] 9.7 Run existing test suite (`pytest -m "not slow"`) and verify no regressions

## 10. Documentation

- [ ] 10.1 Add `docs/tool-call-guardrails.md` — user-facing docs: what the flags do, how to enable, how to use `x_omlx_validation` for client-side retry, example nudge messages
- [ ] 10.2 Add server setting descriptions to admin panel help text (if applicable)
```

Full source: openspec/changes/add-forge-guardrails/tasks.md

## openspec/changes/add-forge-guardrails/specs/tool-call-rescue-parsing/spec.md

- Source: openspec/changes/add-forge-guardrails/specs/tool-call-rescue-parsing/spec.md
- Lines: 1-50
- SHA256: 85f2f65d00d583fcca4d0a79793b2dff798ee96cb6f66cf88d84ff2c50622d55

```md
## ADDED Requirements

### Requirement: Rehearsal syntax rescue parser
The system SHALL parse tool calls expressed in rehearsal syntax (`tool_name[ARGS]{...json...}`) as a fallback strategy when all existing parsers return no matches. This format is produced by reasoning models that "rehearse" tool calls inside thinking tokens.

#### Scenario: Rehearsal syntax parsed successfully
- **WHEN** the model output (post thinking-tag extraction) contains text matching `tool_name[ARGS]{...}` where the brace-enclosed content is valid JSON
- **THEN** the system extracts a tool call with the parsed tool name and JSON arguments

#### Scenario: Multiple rehearsal calls extracted
- **WHEN** the model output contains multiple rehearsal-syntax expressions
- **THEN** the system extracts all valid expressions as separate tool calls

#### Scenario: Invalid JSON in rehearsal syntax ignored
- **WHEN** the brace-enclosed content after `[ARGS]` is not valid JSON
- **THEN** the system skips that expression and continues scanning for other matches (does not crash)

#### Scenario: Rehearsal parser only runs after existing parsers fail
- **WHEN** an existing parser (native, XML, namespaced, Hermes, bracket) successfully extracts tool calls
- **THEN** the rehearsal parser is NOT invoked (existing parser output takes precedence)

### Requirement: Improved Mistral bracket-tag rescue parser
The system SHALL parse Mistral `[TOOL_CALLS]` bracket-tag format using a brace-balance scanner with string and escape awareness, replacing or augmenting the current regex-based extraction. This handles nested JSON objects and string-escaped braces more robustly than regex.

#### Scenario: Nested JSON in Mistral bracket-tag
- **WHEN** the model output contains `[TOOL_CALLS]tool_name{...}` where the JSON object contains nested objects or arrays
- **THEN** the system correctly extracts the complete JSON object, respecting brace nesting depth

#### Scenario: Literal braces in string values
- **WHEN** the JSON arguments contain string values with literal `{` or `}` characters (e.g., `{"pattern": "use {placeholder}"}`)
- **THEN** the brace-balance scanner correctly distinguishes structural braces from string content and extracts the complete object

#### Scenario: Escaped quotes in string values
- **WHEN** the JSON arguments contain escaped quote characters (e.g., `{"text": "say \"hello\""}`)
- **THEN** the scanner correctly tracks escape state and extracts the complete object without premature termination

### Requirement: Rescue parser integration
The system SHALL integrate rescue parsers into the existing parser chain in `_parse_tool_calls_impl` (`tool_calling.py:1108`) as additional fallback strategies, tried after all existing parsers return no matches.

#### Scenario: Rescue parsers are last in the chain
- **WHEN** the parser chain runs, existing parsers (native, XML, namespaced, Hermes, bracket) are tried first
- **THEN** rehearsal and improved Mistral parsers are tried only if all existing parsers return no matches

#### Scenario: Rescue parsing toggleable
- **WHEN** rescue parsing is disabled (future setting, not in Change A scope)
- **THEN** only existing parsers run (current behavior preserved)

#### Scenario: Multiple rescue strategies tried in order
- **WHEN** the rehearsal parser returns no matches and the Mistral bracket-tag marker is present
- **THEN** the Mistral parser is tried next; if it also returns nothing, parsing falls through to the final marker-strip pass
```

## openspec/changes/add-forge-guardrails/specs/tool-call-validation/spec.md

- Source: openspec/changes/add-forge-guardrails/specs/tool-call-validation/spec.md
- Lines: 1-65
- SHA256: 4a60998964b3aba341a4b9fc4f9132a624567ad7a04ce361867d172e0322f0bb

```md
## ADDED Requirements

### Requirement: Tool-call response validation
The system SHALL validate every parsed tool-call response before returning it to the client, when `guardrail_validation` setting is enabled. Validation SHALL perform four checks in order: bare-text-when-tools-expected, unknown tool name, malformed arguments (non-dict), and missing required parameters.

#### Scenario: Valid tool call passes all checks
- **WHEN** the model emits a well-formed tool call to a known tool with valid dict arguments satisfying all required parameters
- **THEN** the system returns the tool call normally with validation result `passed=True` and no nudge

#### Scenario: Unknown tool name is detected
- **WHEN** the model emits a tool call to a tool name not present in the request's `tools` list
- **THEN** the system flags the validation check `unknown_tool` as failed and generates a nudge with `kind="unknown_tool"`, `role="tool"`, listing the available tool names

#### Scenario: Malformed arguments (non-dict) are detected
- **WHEN** the model emits a tool call whose arguments cannot be parsed as a JSON object (e.g., a bare string, number, or array)
- **THEN** the system flags the validation check `malformed_args` as failed and generates a nudge with `kind="tool_arg_validation"`, `role="tool"`, indicating the received type and the required dict shape

#### Scenario: Missing required parameters are detected
- **WHEN** the model emits a tool call to a known tool with dict arguments, but one or more parameters listed in the tool's JSON Schema `required` array are absent
- **THEN** the system flags the validation check `missing_required_params` as failed and generates a nudge with `kind="tool_arg_validation"`, `role="tool"`, listing the missing parameter names

#### Scenario: Bare text when tools are expected is detected
- **WHEN** the model emits only text content (no tool calls) and `tools` were provided in the request and `tool_choice` is not `"none"`
- **THEN** the system flags the validation check `bare_text` as failed and generates a nudge with `kind="retry"`, `role="user"`, instructing the model to emit a tool call

#### Scenario: Validation disabled by default
- **WHEN** `guardrail_validation` setting is `False` (default)
- **THEN** the system performs no validation and returns responses identical to current behavior

### Requirement: Corrective nudge generation
The system SHALL generate a structured nudge message for each failed validation check. Each nudge SHALL specify `role` (either `"user"` for bare-text corrections or `"tool"` for tool-call corrections), `content` (the corrective message text), and `kind` (one of `"retry"`, `"unknown_tool"`, `"tool_arg_validation"`).

#### Scenario: Nudge uses tool-result role for tool-call errors
- **WHEN** a validation check fails for `unknown_tool` or `malformed_args` or `missing_required_params`
- **THEN** the generated nudge SHALL have `role="tool"` to match the wire shape models are pretrained on ("tool call failed → try again")

#### Scenario: Nudge uses user role for bare-text correction
- **WHEN** the `bare_text` check fails
- **THEN** the generated nudge SHALL have `role="user"` with corrective instructions

### Requirement: Validation metadata in response extension
The system SHALL include validation results and suggested nudge in the API response when `include_validation_metadata` setting is enabled. The metadata SHALL be carried in a non-standard `x_omlx_validation` extension field that does not interfere with standard OpenAI/Anthropic response fields.

#### Scenario: Validation metadata included when enabled
- **WHEN** `include_validation_metadata` is `True` and validation runs
- **THEN** the response includes an `x_omlx_validation` object containing `checks` (array of check results with `passed` boolean and optional `detail`) and `nudge` (the suggested corrective message or null if validation passed)

#### Scenario: No metadata when disabled
- **WHEN** `include_validation_metadata` is `False` (default)
- **THEN** the response contains no `x_omlx_validation` field and is indistinguishable from current behavior

#### Scenario: Streaming validation metadata
- **WHEN** validation runs on a streaming response and `include_validation_metadata` is `True`
- **THEN** the validation metadata SHALL be emitted as a final SSE event before `data: [DONE]`, since earlier events cannot be retroactively modified

### Requirement: Strict tool arguments mode
The system SHALL preserve malformed tool-call arguments as-is (rather than coercing to `"{}"`) when `strict_tool_args` setting is enabled. Validation results SHALL surface the original malformed value regardless of this setting.

#### Scenario: Strict mode preserves malformed args
- **WHEN** `strict_tool_args` is `True` and the model emits a tool call with non-dict arguments
- **THEN** the system preserves the original argument value in the response and flags it via validation

#### Scenario: Default mode coerces for backward compatibility
- **WHEN** `strict_tool_args` is `False` (default) and the model emits a tool call with non-dict arguments
- **THEN** the system coerces arguments to `"{}"` as per current behavior, but still reports the validation failure in `x_omlx_validation` (if metadata is enabled)
```

## openspec/changes/add-forge-guardrails/specs/tool-choice-enforcement/spec.md

- Source: openspec/changes/add-forge-guardrails/specs/tool-choice-enforcement/spec.md
- Lines: 1-35
- SHA256: 58a9e0d877e910f2f361ada4734a4aa56b1c1e22938b2af8c859a471d1e45852

```md
## ADDED Requirements

### Requirement: tool_choice enforcement
The system SHALL enforce `tool_choice` semantics after parsing and validation, before response construction. All four modes SHALL be supported: `"none"`, `"auto"`, `"required"`, and `{"type":"function","function":{"name":"X"}}`.

#### Scenario: tool_choice "none" suppresses tool calls
- **WHEN** `tool_choice` is `"none"` and the model produces tool calls in its response
- **THEN** the system suppresses all tool calls and returns only the text content

#### Scenario: tool_choice "auto" passes through
- **WHEN** `tool_choice` is `"auto"` (or omitted)
- **THEN** the system passes through whatever the model produces (tool calls and/or text)

#### Scenario: tool_choice "required" rejects bare text
- **WHEN** `tool_choice` is `"required"` and the model produces only text content with no tool calls
- **THEN** the system flags a validation failure with nudge `kind="retry"`, informing that a tool call is required

#### Scenario: tool_choice "required" accepts tool calls
- **WHEN** `tool_choice` is `"required"` and the model produces at least one tool call
- **THEN** the system accepts the tool calls normally

#### Scenario: Named-tool filtering
- **WHEN** `tool_choice` is `{"type":"function","function":{"name":"search"}}` and the model produces tool calls including some to tools other than `search`
- **THEN** the system filters the response to include only tool calls matching the named tool `search`, and flags rejected calls with nudge `kind="unknown_tool"`

#### Scenario: Invalid tool_choice value rejected at request time
- **WHEN** the client sends a `tool_choice` value that is not one of the supported formats (e.g., a random string like `"weird"`)
- **THEN** the system rejects the request with HTTP 400 before inference begins

### Requirement: tool_choice enforcement ordering
The system SHALL apply `tool_choice` enforcement AFTER validation checks and BEFORE response construction, so that enforcement failures benefit from the same nudge infrastructure as validation failures.

#### Scenario: Enforcement after validation
- **WHEN** a response has both a malformed-args validation failure and a `tool_choice="required"` enforcement failure
- **THEN** both failures appear in the validation metadata, with validation checks listed before enforcement checks
```

