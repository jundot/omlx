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

**Migration path**: After Change A ships, monitor adoption. Consider flipping the default to `True` in a future major version if validation metadata shows clients handle malformed args correctly.

### D5: Rescue parsers as additional fallback strategies

**Decision**: Add rehearsal syntax and improved Mistral bracket-tag parsers as new fallback strategies in the existing parser chain (`_parse_tool_calls_impl`, `tool_calling.py:1108`), tried after existing parsers fail.

**Rehearsal syntax**: `re.search(r"(\w+)\[ARGS\](\{.*\})", text, re.DOTALL)` — matches reasoning models that "rehearse" tool calls in thinking tokens.

**Mistral bracket-tag improvement**: Replace the current regex-based extraction with forge's brace-balance scanner (`../forge/src/forge/prompts/templates.py:206-234`) that tracks `in_string` and `escape` flags for robust nested-JSON extraction.

**Rationale**: These are drop-in additions to the existing parser chain with no architectural change. The brace-balance scanner is more robust than regex for nested objects containing literal `{`/`}` in string values.

### D6: `tool_choice` enforcement as a post-validation filter

**Decision**: New `omlx/api/tool_choice.py` module with `enforce_tool_choice(tool_calls, tool_choice, has_text)`. Applied AFTER validation, BEFORE response construction. Modes:
- `"none"`: suppress all tool calls (already works at `server.py:3307`, but now also suppresses in the validation layer for consistency)
- `"auto"`: pass through (no-op)
- `"required"`: if no tool calls and `has_text`, generate a nudge (`kind="retry"`) telling the model it must emit a tool call
- `{"type":"function","function":{"name":"X"}}`: filter `tool_calls` to only calls matching `X`; reject others with `unknown_tool` nudge

**Rationale**: Applying after validation means `tool_choice` enforcement benefits from the same nudge infrastructure. A `"required"` failure produces the same retry nudge as a bare-text validation failure.

### D7: Settings in `GlobalSettings`, not `config.py`

**Decision**: New `ForgeGuardrailsSettings` dataclass in `settings.py`, composed into `GlobalSettings`. Fields: `validation_enabled: bool = False`, `strict_tool_args: bool = False`, `include_validation_metadata: bool = False`. Exposed via admin panel `GlobalSettingsRequest` (`admin/routes.py:206`) and `update_global_settings` handler (`routes.py:3234`).

**Rationale**: `config.py` is read once at boot and not admin-mutable. Guardrail toggles need to be live-adjustable (enable validation for testing, disable if it causes issues). The `CompressionSettings` pattern (`settings.py:746`) is the canonical example.

## Risks / Trade-offs

**[Validation adds latency]** → Validation is pure Python with no I/O (in-memory schema lookup + `jsonschema.validate`). Sub-millisecond per check. Negligible vs. inference time (seconds). No mitigation needed.

**[Strict args mode breaks existing clients]** → Feature-flagged off by default (`strict_tool_args=False`). Validation results are always reported in `x_omlx_validation` (when `include_validation_metadata=True`), even when coercion is active. Clients can monitor validation failures without opting into stricter behavior.

**[`x_omlx_validation` confuses strict OpenAI clients]** → Extension is only included when `include_validation_metadata=True` (default off). Non-standard fields are in a namespaced extension (`x_omlx_*`), which OpenAI clients ignore. Tested OpenAI/Anthropic SDKs tolerate unknown response fields.

**[Streaming validation timing]** → Validation runs at stream end (after all tool-call markup is parsed). The `x_omlx_validation` data is sent as a final SSE event before `data: [DONE]`. Clients cannot retroactively fix earlier-streamed events. Acceptable: validation metadata is informational (tells client whether to retry), not corrective (doesn't modify the current response).

**[Missing-required-params false positives]** → Some tools have loose schemas (`parameters: {}` or no `required` field). The check passes trivially for these (no required fields = no missing params). Only flags tools that explicitly declare required fields. No false positives possible.

**[Rescue parser false positives]** → Rehearsal syntax (`tool_name[ARGS]{...}`) could match non-tool-call text. Mitigation: only tried when all existing parsers return nothing, and the extracted JSON must parse successfully. The brace-balance scanner for Mistral is anchored on `[TOOL_CALLS]` marker, reducing false matches.

**[Nudge channel/budget mapping confusion]** → The forge invariant (tool channel vs retry budget are orthogonal) is subtle. If `unknown_tool` drains the wrong budget, retry limits misfire. Mitigation: document the mapping in code comments and unit tests; for Change A (stateless), budgets aren't used yet, so this risk is deferred to Change B.

## Open Questions

1. **Should `x_omlx_validation` be namespaced per-endpoint?** OpenAI uses `x_omlx_*`, Anthropic uses a different extension pattern. For `/v1/messages`, should the metadata go in a `metadata` field instead? → Resolve in implementation: use endpoint-appropriate location.

2. **Should validation be permissive or strict for unknown `tool_choice` values?** If a client sends `tool_choice="weird"`, do we reject the request or pass through? → Recommendation: reject with 400 at the request validation layer (`openai_models.py`), separate from runtime enforcement.

3. **Should rescue parsers be individually toggleable?** Some models might produce false-positive rehearsal matches. → Recommendation: global `rescue_parsing_enabled` flag for Change A; per-parser toggles deferred unless needed.
