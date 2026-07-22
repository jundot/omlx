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
