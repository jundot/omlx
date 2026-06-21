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
- [ ] 10.3 Update `docs/forge-integration-plan.md` — mark Phase 1 + Phase 2 as "in progress via change `add-forge-guardrails`" and note Change B/C as follow-ups
