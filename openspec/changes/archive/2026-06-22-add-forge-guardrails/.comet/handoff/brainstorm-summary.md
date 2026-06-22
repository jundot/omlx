# Brainstorm Summary

- Change: add-forge-guardrails
- Date: 2026-06-21

## Confirmed Technical Approach

**Wrapper function pattern**: New `extract_and_validate_tool_calls()` wraps the existing `extract_tool_calls_with_thinking()` (which stays untouched). When `guardrail_validation` setting is enabled, the wrapper constructs a `GuardrailValidator` and runs validation on the extraction output. All 6 server.py call sites switch to the wrapper when validation is enabled.

**Accumulate failures, one nudge**: All 5 checks (4 validation + tool_choice enforcement) run unconditionally. Results accumulate into a `checks` list. The `nudge` field is pre-computed from the highest-priority failure (priority: bare_text > unknown_tool > malformed_args > missing_required_params > tool_choice_enforcement). Client gets the complete picture but only needs to act on one nudge per retry.

**Per-endpoint response extension**: `x_omlx_validation` as a non-standard top-level field on all 3 endpoints (OpenAI chat, Anthropic messages, Responses API). Streaming: validation metadata piggybacks on the final SSE chunk before `[DONE]`. All SDKs tolerate unknown top-level fields.

**Opt-in by default**: `include_validation_metadata=False` means no `x_omlx_validation` field anywhere — responses are byte-identical to current behavior.

## Key Trade-offs and Risks

1. **Wrapper indirection**: One extra function call when validation is enabled. Negligible — Python function call overhead is ~100ns vs. seconds of inference time.

2. **All-checks-run overhead**: Even when the first check fails, all subsequent checks still run. Justified because clients need the complete picture for informed retry decisions. Each check is sub-millisecond pure Python.

3. **Streaming timing**: Validation metadata arrives at stream end — the client already received (possibly malformed) tool calls via earlier events. Acceptable: metadata is informational (tells client whether to retry), not corrective (doesn't modify streamed events).

4. **Nudge priority ambiguity**: The priority order (bare_text > unknown_tool > malformed_args > missing_params > tool_choice) is a design choice. Forge uses a similar ordering. If a response has both an unknown tool AND malformed args, the unknown-tool nudge is emitted first because it's higher priority and more actionable (client knows immediately which tool to correct).

5. **Non-standard extension field**: `x_omlx_validation` is not part of the OpenAI/Anthropic spec. Future API changes could theoretically conflict. Mitigation: `x_omlx_` namespace prefix is the established convention for non-standard extensions.

## Testing Strategy

**Test pyramid**: 45+ unit tests (pure functions, mock at `ToolCallExtraction` boundary) → 7 integration tests (FastAPI TestClient + enabled settings) → 3 E2E tests (mock model output → verify response).

- **Unit tests**: Construct `ToolCallExtraction` inline (no fixture files). Test each check independently, then check ordering, then edge cases (loose schemas, empty args, all-pass).
- **Integration tests**: `monkeypatch` the model to return canned output. Enable settings via `GlobalSettings`. Verify `x_omlx_validation` appears in response body.
- **E2E tests**: Full request → mocked model → verify response shape + validation metadata + nudge content.
- **Regression**: Run existing `pytest -m "not slow"` and verify zero new failures.

One test file per module: `test_guardrail_validator.py`, `test_guardrail_nudges.py`, `test_rescue_parsers.py`, `test_tool_choice_enforcement.py`, `test_guardrail_settings.py`, `test_guardrail_e2e.py`.

## Spec Patches

None needed. The 3 capability specs (`tool-call-validation`, `tool-choice-enforcement`, `tool-call-rescue-parsing`) already have 31 scenarios covering the confirmed design. The brainstorming deepened implementation details (data structures, flow, testing) but did not change requirements or add new acceptance scenarios.
