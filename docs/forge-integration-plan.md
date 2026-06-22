# Forge Integration Plan for oMLX

> Analysis of [forge-guardrails](https://github.com/jundot/forge) v0.7.4 features applicable to oMLX,
> with concrete integration strategies, code impact, and phased rollout.

> **Status (2026-06-21):** Phase 1 (Validation & Rescue) + Phase 2 (tool_choice Enforcement) are
> **implemented** via OpenSpec change `add-forge-guardrails` (branch `feature/20260621/add-forge-guardrails`).
> See [Tool Call Guardrails](tool-call-guardrails.md) for user-facing docs.
> Phase 3 (Error Budgets) + Phase 4 (Context Compaction) are planned as Change B (client-driven retry
> via response extensions). Phase 5 (MCP Prerequisites) + Phase 6 (Synthetic Respond Tool) are planned
> as Change C.

---

## 1. Executive Summary

[Forge](https://github.com/jundot/forge) is a reliability layer for self-hosted LLM tool-calling.
Published as an IEEE paper (DOI: 10.1145/3786335.3813193), it improves small model (8B-14B)
tool-calling success rates from single-digit to 84%+, and frontier models from 85% to 98%.

oMLX currently provides **parsing but no validation or self-correction** for tool calls.
When a model emits malformed JSON, hallucinates a tool name, or produces text instead of a
tool call, oMLX either silently corrupts the response (coercing bad args to `{}`) or drops
it with a `logger.warning`. There is no retry loop, no corrective feedback to the model,
and no enforcement of `tool_choice` semantics.

This document identifies 8 forge features applicable to oMLX, ranks them by impact and
integration effort, and provides a phased implementation plan.

---

## 2. Current State: oMLX Tool-Calling Gaps

### 2.1 The Tool-Call Pipeline Today

```
Client Request (tools, tool_choice)
  |
  v
server.py -- ChatCompletionRequest parsing
  |  tool_choice accepted but NEVER enforced
  |  MCP tools merged (user tools take precedence)
  v
engine/batched.py -- _apply_chat_template()
  |  tools injected into prompt via tokenizer.apply_chat_template
  v
Model Inference
  |  Raw text output (may contain tool-call markup)
  v
tool_calling.py -- parse_tool_calls()
  |  5 fallback parsers tried in sequence
  |  Unparseable matches: DROPPED with logger.warning
  |  Malformed args: COERCED to "{}" with logger.warning
  |  Unknown tool names: PASSED THROUGH
  v
API Response
   tool_calls returned as-is, no validation
```

### 2.2 Specific Gaps

| Gap | Current Behavior | Impact |
|-----|-----------------|--------|
| **Malformed args** | `_serialize_tool_call_arguments()` coerces non-dict to `"{}"` | Silent data loss; model never learns it made an error |
| **Unknown tool names** | Passed through; MCP executor logs warning, sets `all_valid=False` but returns invalid calls | Client receives calls to non-existent tools |
| **Bare text instead of tool call** | Returned as regular `content`, no re-prompt | Model's intent (tool call) is lost; client must detect and retry manually |
| **Missing required params** | Not checked against tool schema | Tool execution fails at runtime with unclear error |
| **`tool_choice` not enforced** | `"none"` -> still returns parsed tool calls; forced tool -> no filtering | API contract violated |
| **No retry/nudge loop** | Single-shot; model never gets a chance to self-correct | One malformed response = permanent failure |
| **No argument schema validation** | `validate_json_schema()` exists but only used for `response_format`, never for tool-call args | Invalid types/missing fields propagate silently |
| **No context compaction** | KV cache has token-level eviction but no message-level semantic compaction | Long conversations fail; no prioritized truncation |

---

## 3. Forge Features Applicable to oMLX

### Feature 1: Guardrail Middleware (`check()` / `record()`)

**Forge implementation**: `src/forge/guardrails/guardrails.py`

Two-method middleware API that validates every model response before tool execution:

- `check(response)` -> returns one of: `"execute"`, `"retry"`, `"tool_error"`, `"step_blocked"`, `"fatal"`
- `record(executed_tools)` -> tracks step completion, resets error counters on clean batch

The check pipeline flows through `ResponseValidator` -> `StepEnforcer` -> `ErrorTracker`,
producing a `Nudge` (corrective message) on validation failure. Nudges use two injection
channels:
- **User-role** (`"retry"`, `"step"` kinds): for bare-text corrections, step nudges
- **Tool-result-role** (`"unknown_tool"`, `"tool_arg_validation"` kinds): for tool-channel
  corrections -- models are pretrained on "tool call failed -> try again" patterns, so this
  channel survives heavy context and attention drop-off better than a trailing user nudge.

**Why this matters for oMLX**: oMLX's `parse_tool_calls()` currently has no post-parse
validation. Every malformed response either gets corrupted (args -> `{}`) or silently
dropped. The guardrail middleware would insert a validation step between parsing and
response emission, catching errors before they reach the client.

**Integration approach**: Add a `GuardrailMiddleware` class to `omlx/api/` that wraps the
existing `parse_tool_calls()` output. The middleware does NOT own the inference loop --
oMLX's server is stateless per-request. Instead, it provides:
1. Validation result (valid/invalid + reason)
2. Corrective nudge message (for the client to send back in a retry)
3. Error classification (for logging/metrics)

This is the **extracted middleware** pattern from forge (ADR-011), not the full
`WorkflowRunner` loop.

---

### Feature 2: Rescue Parsing Enhancements

**Forge implementation**: `src/forge/prompts/templates.py` -- `rescue_tool_call()`

Four ordered strategies for extracting tool calls from text that failed native parsing:

| Strategy | Pattern | Models |
|----------|---------|--------|
| Prompt-injected JSON | `{"tool": "name", "args": {...}}` including code-fenced | Generic small models in prompt mode |
| Rehearsal syntax | `tool_name[ARGS]{...json...}` | Reasoning models (thinking tokens) |
| Qwen Coder XML | `<function=name><parameter=key>value</parameter></function>` | Qwen Coder family |
| Mistral bracket-tag | `[TOOL_CALLS]name{json_args}` | Devstral-Small-2, Mistral-Small-3.x |

**oMLX's existing parsers** cover XML, namespaced, Hermes, bracket, and Gemma4 formats.
Forge adds two patterns not currently handled:

1. **Rehearsal syntax** -- reasoning models sometimes "rehearse" tool calls inside thinking
   tokens using `tool_name[ARGS]{...}`. oMLX's thinking-tag extraction (`extract_thinking`)
   already separates thinking from content, but the content parser doesn't recognize this
   rehearsal format.

2. **Mistral bracket-tag** -- oMLX handles one-sided `[TOOL_CALLS]` markers via its native
   tokenizer path, but the regex-based brace-balance scan in forge is more robust for
   nested JSON objects and string-escaped braces.

Additionally, forge's approach to **malformed args** differs fundamentally:
- **oMLX**: `_serialize_tool_call_arguments()` coerces non-dict to `"{}"` (silent data loss)
- **Forge**: Keeps raw malformed args as-is (non-dict), routes them through the
  `tool_arg_validation` nudge channel so the model is told exactly what it did wrong

**Integration approach**: Add the two missing parsing strategies to `parse_tool_calls()`
and change the malformed-args handling from silent coercion to explicit error signaling.

---

### Feature 3: Error Budgets with Nudge Loop

**Forge implementation**: `src/forge/guardrails/error_tracker.py`, `src/forge/prompts/nudges.py`

Two independent error budgets:
- **Retry budget** (`max_retries=3`): consecutive formatting failures (bare text, unknown tool)
- **Tool-error budget** (`max_tool_errors=2`): consecutive tool execution errors

Key design decisions:
- Only a **clean batch** (all tools succeed) resets the tool-error counter, preventing
  success/failure oscillation
- **Soft errors** (`ToolResolutionError`, `is_soft_error=True`) don't count -- "try again
  with different args" (HTTP 4xx equivalent)
- Five nudge templates with escalation: `retry`, `unknown_tool`, `tool_arg_validation`,
  `step`, `prerequisite`

**Why this matters for oMLX**: oMLX is a **stateless API server** -- each request is
independent. The full forge retry loop (re-prompting the model within a single request)
doesn't fit this architecture. However, oMLX can provide the *building blocks* for clients
to implement their own retry loops:

1. **Validation metadata in responses**: Include validation results (which checks passed/failed)
   as non-standard extensions in the API response, so clients know whether to retry.
2. **Nudge message generation**: Provide the corrective message the client should append
   before retrying, so they don't have to craft it themselves.
3. **Error budget tracking**: For the admin dashboard's built-in chat (which IS stateful),
   implement the full retry loop with budget tracking.

**Integration approach**: Two-tier --
- **Tier A (API responses)**: Add `x-omlx-validation` header or response extension with
  validation results + suggested nudge. Zero breaking changes.
- **Tier B (Admin chat)**: Full retry loop with error budgets in the admin dashboard's
  conversation handler, which maintains state across turns.

---

### Feature 4: `tool_choice` Enforcement

**Forge implementation**: Implicit -- validated tool calls are checked against the tool
registry; unknown tools are rejected.

oMLX accepts `tool_choice` in the request but never enforces it:

| `tool_choice` value | oMLX behavior | Correct behavior |
|---------------------|--------------|-----------------|
| `"none"` | Tool calls still parsed and returned | Suppress all tool calls; return only text |
| `"auto"` | Whatever the model produces | Pass through (correct) |
| `{"type":"function","function":{"name":"X"}}` | All calls returned | Only calls to tool X; reject others |
| `"required"` | Not supported | Force at least one tool call |

**Integration approach**: Add a `enforce_tool_choice()` function that filters/validates
parsed tool calls against the requested `tool_choice` mode. Apply after `parse_tool_calls()`
and before response construction.

---

### Feature 5: Tiered Context Compaction

**Forge implementation**: `src/forge/context/strategies.py` -- `TieredCompact`

Three-phase deterministic compaction (no LLM calls, sub-millisecond execution):

| Priority | Phase 1 | Phase 2 | Phase 3 |
|----------|---------|---------|---------|
| Drop first | Nudges/retries | + Drop tool results | + Drop reasoning & text |
| Preserve | Reasoning, text, tool_call skeletons | Reasoning, text | Tool_call skeletons only |
| Never cut | System prompt, user input, recent iterations | System prompt, user input | System prompt, user input |

**Key insight**: Reasoning survives through Phase 2 because the model's *interpretation*
of tool results is more valuable than the raw results themselves.

**oMLX's current approach**: KV cache management is at the **token level** (paged blocks,
hot RAM -> cold SSD tiering, prefix sharing). There is no **message-level** compaction.
When context exceeds the model's window, the request fails.

**Integration approach**: oMLX's architecture makes this applicable in two places:
1. **Admin chat**: The built-in conversation handler can implement tiered compaction
   when conversation history approaches the model's context limit.
2. **Context scaling for Claude Code**: oMLX already has `claude_code_context_scaling`
   for small-context models. Tiered compaction would improve this by making intelligent
   decisions about what to truncate instead of simple truncation.

The `CompactStrategy` ABC pattern from forge (pluggable strategies) maps well to oMLX's
existing model-specific handling patterns.

---

### Feature 6: Step Enforcement & Prerequisites (MCP)

**Forge implementation**: `src/forge/guardrails/step_enforcer.py`, `src/forge/core/steps.py`

Two enforcement modes:
1. **Required steps**: Terminal tool can't be called until prerequisite steps complete
   (with 3-tier escalating nudges: polite -> direct -> aggressive)
2. **Arg-matched prerequisites**: `read_file(path=X)` must precede `edit_file(path=X)`

Critical design: **Step state lives outside the message history** in `StepTracker`.
Context compaction can't corrupt step completion -- once a step is recorded, it stays
recorded even if the message that produced it is compacted away.

**oMLX relevance**: MCP tool execution often has implicit ordering dependencies
(search -> read -> edit). Currently, oMLX's `ToolExecutor` fires all tool calls
in parallel with no ordering guarantees.

**Integration approach**: Add prerequisite declarations to the MCP tool configuration
schema (`mcp.example.json`), and enforce ordering in `ToolExecutor.execute_tool_calls()`.
This is a configuration-driven feature -- MCP server authors declare prerequisites in their
tool metadata, and oMLX enforces them.

---

### Feature 7: Synthetic Respond Tool

**Forge implementation**: `src/forge/tools/respond.py`

A synthetic `respond(message="...")` tool that gives small models a structured alternative
to bare text. Instead of choosing between "emit text" and "call a tool" (which small models
do poorly), the model always stays in tool-calling mode. The respond call is stripped from
the outbound response -- the client sees normal text.

**oMLX relevance**: oMLX primarily serves 8B-14B local models on Apple Silicon. These
models struggle with the text/tool-call decision. Injecting a respond tool keeps them in
the grammar where forge's full guardrail stack applies.

**Integration approach**: When `tool_choice != "none"` and the model is detected as a
small local model (< 14B parameters), automatically inject a `respond` tool into the
tools list. Strip `respond` calls from the response before returning to the client.
Make this configurable via a server setting (default: off).

---

### Feature 8: Per-Model Sampling Defaults

**Forge implementation**: `src/forge/clients/sampling_defaults.py`

50+ model entries keyed by Ollama tag, GGUF stem, and llamafile stem -- sourced from
HuggingFace model cards with inline citation URLs. Opt-in via `recommended_sampling=True`.

**oMLX relevance**: oMLX already has model-specific handling (Gemma4 param renaming,
Qwen thinking detection, etc.) but no centralized sampling registry. A structured
registry would reduce ad-hoc model detection code and provide sensible defaults for
new models.

**Integration approach**: Lower priority -- oMLX's existing per-model handling works.
Could be adopted incrementally as a `ModelRegistry` that centralizes model-specific
configuration (sampling params, tool-calling format, thinking tag style, etc.).

---

## 4. Integration Architecture

### 4.1 Design Principles (from Forge, Adapted for oMLX)

1. **Fail Fast, Fail Loud** -- No silent coercion of malformed args to `{}`. Surface
   errors as validation results so clients can act on them.

2. **Validation After Parsing** -- Insert a validation layer between `parse_tool_calls()`
   and response construction. This is the `GuardrailMiddleware` pattern.

3. **Control Flow != Memory** -- For stateful features (step tracking, error budgets),
   maintain state outside the message history so it survives compaction.

4. **Stateless API, Stateful Chat** -- The OpenAI/Anthropic API endpoints remain stateless
   per-request. Stateful features (retry loops, step enforcement) are available in the
   admin chat and as building blocks for clients.

5. **Opt-In, Not Opt-Out** -- New guardrail features are off by default and enabled via
   server configuration or per-request headers. Existing behavior is preserved.

### 4.2 Proposed Module Structure

```
omlx/
  api/
    tool_calling.py          # EXISTING -- add rescue parsers, change args coercion
    guardrails/              # NEW -- validation middleware
      __init__.py            #   Public API: GuardrailMiddleware, Nudge, CheckResult
      validator.py           #   ResponseValidator (unknown tool, malformed args, bare text)
      nudge.py               #   Nudge generation (retry, unknown_tool, tool_arg_validation)
      budget.py              #   ErrorBudget (retry + tool-error counters, clean-batch reset)
    tool_choice.py           # NEW -- tool_choice enforcement logic
  mcp/
    executor.py              # MODIFY -- add prerequisite enforcement
    prerequisites.py         # NEW -- prerequisite declaration and checking
  context/
    __init__.py              # NEW
    compaction.py            # NEW -- CompactStrategy ABC + TieredCompact + SlidingWindow
  admin/
    routes.py                # MODIFY -- full retry loop in chat handler
```

### 4.3 Integration Points in the Request Flow

```
Client Request (tools, tool_choice)
  |
  v
server.py -- ChatCompletionRequest parsing
  |
  +-- NEW: tool_choice.py -- validate tool_choice value
  +-- NEW: guardrails/validator.py -- validate tool definitions (name, parameters schema)
  |
  v
engine/batched.py -- _apply_chat_template()
  |  NEW: optionally inject synthetic respond tool
  v
Model Inference
  |
  v
tool_calling.py -- parse_tool_calls()
  |  NEW: rehearsal syntax parser
  |  NEW: improved Mistral bracket-tag parser
  |  CHANGED: malformed args preserved (not coerced to "{}")
  v
NEW: guardrails/validator.py -- ResponseValidator
  |  Check 1: unknown tool names -> Nudge(kind="unknown_tool")
  |  Check 2: malformed args (non-dict) -> Nudge(kind="tool_arg_validation")
  |  Check 3: missing required params -> Nudge(kind="tool_arg_validation")
  |  Check 4: bare text when tools expected -> Nudge(kind="retry")
  v
NEW: tool_choice.py -- enforce_tool_choice()
  |  "none" -> suppress all tool calls
  |  forced tool -> filter to specified tool only
  |  "required" -> reject bare text, require at least one tool call
  v
API Response
  |  NEW: x-omlx-validation header (validation results for client retry)
  |  NEW: nudge messages (for client to append before retry)
  v
Client -- may retry with nudge (stateless) or
Admin Chat -- automatic retry loop with error budgets (stateful)
```

---

## 5. Phased Implementation Plan

### Phase 1: Validation & Rescue (P0 -- 2-3 weeks)

**Goal**: Stop silently corrupting tool-call responses. Surface errors clearly.

| Task | Files | Description |
|------|-------|-------------|
| 1.1 | `omlx/api/guardrails/__init__.py` | Create guardrails package with public API |
| 1.2 | `omlx/api/guardrails/validator.py` | Implement `ResponseValidator` with 4 checks: unknown tool, malformed args, missing required params, bare text detection |
| 1.3 | `omlx/api/guardrails/nudge.py` | Implement `Nudge` dataclass and 4 nudge generators: `retry_nudge`, `unknown_tool_nudge`, `tool_arg_validation_nudge`, `missing_params_nudge` |
| 1.4 | `omlx/api/tool_calling.py` | Change `_serialize_tool_call_arguments()`: preserve malformed args instead of coercing to `"{}"`. Add a `ValidationResult` field to `ToolCall` or a parallel validation result. |
| 1.5 | `omlx/api/tool_calling.py` | Add rehearsal syntax parser: `_parse_rehearsal_tool_calls(text)` matching `tool_name[ARGS]{...json...}` |
| 1.6 | `omlx/server.py` | Wire `ResponseValidator` into chat completion handlers (non-streaming first). Add `x-omlx-validation` extension to responses. |
| 1.7 | `tests/` | Unit tests for all new validators, nudges, and rescue parsers |

**Breaking change mitigation**: `_serialize_tool_call_arguments()` coercion change is
behind a feature flag (`OMLX_STRICT_TOOL_ARGS=1`). Default preserves current `"{}"`
behavior. Validation results are always reported regardless of flag.

### Phase 2: tool_choice Enforcement (P1 -- 1 week)

| Task | Files | Description |
|------|-------|-------------|
| 2.1 | `omlx/api/tool_choice.py` | New module: `enforce_tool_choice(tool_calls, tool_choice, tools)` |
| 2.2 | `omlx/api/openai_models.py` | Add `tool_choice` field validator (reject invalid values) |
| 2.3 | `omlx/server.py` | Apply `enforce_tool_choice()` after parsing + validation |
| 2.4 | `tests/` | Unit tests for all tool_choice modes |

### Phase 3: Error Budgets & Admin Chat Retry Loop (P1 -- 2 weeks)

| Task | Files | Description |
|------|-------|-------------|
| 3.1 | `omlx/api/guardrails/budget.py` | Implement `ErrorBudget` with retry budget, tool-error budget, clean-batch reset |
| 3.2 | `omlx/admin/routes.py` | Implement full retry loop in chat handler: validate -> nudge -> re-prompt, bounded by error budgets |
| 3.3 | `omlx/admin/routes.py` | Display retry/validator state in chat UI (current budget, last validation result) |
| 3.4 | `tests/` | Integration tests for retry loop |

### Phase 4: Context Compaction (P2 -- 2-3 weeks)

| Task | Files | Description |
|------|-------|-------------|
| 4.1 | `omlx/context/__init__.py` | Create context package |
| 4.2 | `omlx/context/compaction.py` | Implement `CompactStrategy` ABC, `NoCompact`, `SlidingWindowCompact`, `TieredCompact` |
| 4.3 | `omlx/admin/routes.py` | Apply tiered compaction in chat handler when context approaches limit |
| 4.4 | `omlx/server.py` | Improve `claude_code_context_scaling` with tiered compaction instead of simple truncation |
| 4.5 | `tests/` | Unit tests for all compaction strategies |

### Phase 5: MCP Prerequisites (P2 -- 1-2 weeks)

| Task | Files | Description |
|------|-------|-------------|
| 5.1 | `omlx/mcp/prerequisites.py` | Implement `PrerequisiteChecker` with name-only and arg-matched prerequisites |
| 5.2 | `omlx/mcp/config.py` | Add `prerequisites` field to `MCPServerConfig` / tool metadata |
| 5.3 | `omlx/mcp/executor.py` | Add prerequisite validation before execution; reorder parallel calls to satisfy dependencies |
| 5.4 | `mcp.example.json` | Add example prerequisite declarations |
| 5.5 | `tests/` | Unit tests for prerequisite enforcement |

### Phase 6: Synthetic Respond Tool (P3 -- 1 week)

| Task | Files | Description |
|------|-------|-------------|
| 6.1 | `omlx/api/tool_calling.py` | Add `inject_respond_tool(tools)` and `strip_respond_calls(tool_calls)` |
| 6.2 | `omlx/server.py` | Conditional injection based on model size + `tool_choice` + server setting |
| 6.3 | `omlx/config.py` | Add `inject_respond_tool` server setting (default: off) |
| 6.4 | `tests/` | Unit tests for respond tool injection/stripping |

---

## 6. API Surface Changes

### 6.1 New Response Extensions (Non-Breaking)

For chat completion responses, add optional validation metadata:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "...",
      "tool_calls": [...]
    }
  }],
  "x_omlx_validation": {
    "checks": [
      {"check": "unknown_tool", "passed": true},
      {"check": "malformed_args", "passed": false, "detail": "Tool 'search' had malformed arguments. Got type: str. Required: JSON object (dict)."},
      {"check": "missing_required_params", "passed": false, "detail": "Tool 'search' missing required parameter 'query'"},
      {"check": "tool_choice_enforcement", "passed": true}
    ],
    "nudge": {
      "role": "tool",
      "content": "Tool call to 'search' had malformed arguments...",
      "tool_call_id": "call_abc123"
    }
  }
}
```

This allows clients to:
1. Detect that the model produced a malformed response
2. Append the nudge message and retry
3. Implement their own error budget logic

### 6.2 New Server Settings

```python
# omlx/config.py additions

# Enable guardrail validation on tool-call responses
guardrail_validation: bool = True  # Phase 1

# Strict args mode: reject malformed args instead of coercing to {}
strict_tool_args: bool = False  # Phase 1 (off by default for backward compat)

# Include validation metadata in API responses
include_validation_metadata: bool = False  # Phase 1

# Inject synthetic respond tool for small models
inject_respond_tool: bool = False  # Phase 6

# Context compaction strategy for admin chat
compaction_strategy: str = "tiered"  # Phase 4: "none", "sliding_window", "tiered"
```

### 6.3 New MCP Configuration Fields

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "...",
      "args": [],
      "tools_prerequisites": {
        "write_file": {
          "requires": [
            {"tool": "read_file", "match_arg": "path"}
          ]
        }
      }
    }
  }
}
```

---

## 7. Feature Impact Matrix

| Feature | Priority | Impact on Tool-Calling Reliability | Integration Effort | Breaking Changes | Phase |
|---------|----------|-----------------------------------|-------------------|-----------------|-------|
| Guardrail Middleware | P0 | HIGH -- catches malformed args, unknown tools, bare text | Medium (new module, wire into server.py) | None (opt-in) | 1 |
| Rescue Parsing | P0 | HIGH -- more tool calls successfully extracted | Low (add parsers to existing file) | None | 1 |
| Error Budgets | P1 | MEDIUM-HIGH -- enables self-correcting loops | Medium (admin chat integration) | None | 3 |
| tool_choice Enforcement | P1 | MEDIUM -- API contract compliance | Low (new module) | Minimal (previously invalid responses now filtered) | 2 |
| Tiered Compaction | P2 | MEDIUM -- longer conversations stay coherent | Medium (new module + admin chat) | None | 4 |
| MCP Prerequisites | P2 | MEDIUM -- prevents out-of-order tool execution | Low (config-driven) | None | 5 |
| Synthetic Respond Tool | P3 | LOW-MEDIUM -- helps small models stay in tool mode | Low (inject/strip helpers) | None (opt-in) | 6 |
| Sampling Defaults | P3 | LOW -- sensible defaults | Low (registry) | None | -- |

---

## 8. Key Forge Design Patterns to Adopt

### 8.1 Tool-Error Channel for Corrections

Forge routes corrections via `role="tool"` messages with error prefixes because models are
pretrained on the "tool call failed -> try again" wire shape. This is counterintuitive but
effective:

```
# Instead of a user-role correction (easily ignored):
{"role": "user", "content": "Your tool call was malformed."}

# Use a tool-result correction (model expects to retry):
{"role": "tool", "tool_call_id": "call_abc", "content": "[ToolArgValidationError] Tool 'search' had malformed arguments. Got type: str. Required: JSON object (dict)."}
```

This pattern should be adopted for oMLX's admin chat retry loop and recommended in
documentation for clients implementing their own retry loops.

### 8.2 Clean-Batch Reset

Only a fully successful tool-call batch resets the error counter. A single success among
failures does NOT reset. This prevents the model from oscillating between success and
failure indefinitely. Critical for the admin chat retry loop.

### 8.3 Control Flow != Memory

Step completion tracked outside the message history. Context compaction can drop a tool
result without corrupting workflow state. The model may redundantly re-call a compacted
tool (wastes one iteration but doesn't corrupt). This principle applies to oMLX's MCP
prerequisite tracking -- prerequisites should be tracked in a `StepTracker`, not inferred
from conversation history.

### 8.4 Deterministic Compaction

All three compaction phases are pure text manipulation with sub-millisecond execution. No
LLM calls for summarization. This is critical for oMLX's latency-sensitive API -- compaction
must not add perceptible delay.

---

## 9. Forge Features NOT Applicable to oMLX

| Feature | Why Not Applicable |
|---------|-------------------|
| `WorkflowRunner` (full agentic loop) | oMLX is a stateless API server; the agentic loop belongs to the client (e.g., Claude Code, OpenCode) |
| `SlotWorker` (priority-queued GPU access) | oMLX already has its own scheduler (`scheduler.py`) with FCFS and concurrency control |
| `HardwareProfile` / `detect_hardware()` | oMLX runs exclusively on Apple Silicon; hardware detection is already handled |
| `ServerManager` / budget resolution modes | oMLX has its own context budget management via KV cache configuration |
| `LLMClient` protocol / backend adapters | oMLX is itself the inference backend; it doesn't call external LLM APIs |
| Per-model sampling defaults (full 50+ registry) | Lower priority; oMLX's existing model-specific handling covers key cases |
| Proxy server (dual OpenAI/Anthropic protocol) | oMLX already implements both protocols natively |

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Validation adds latency to every response | Low (sub-ms per check) | Low | Validation is pure Python with no I/O; negligible overhead |
| Strict args mode breaks existing clients | Medium | Medium | Feature-flagged off by default; gradual opt-in |
| Nudge messages confuse clients that don't expect them | Low | Low | Nudges only in `x_omlx_validation` extension; not in standard response fields |
| Admin chat retry loop increases token usage | High | Medium | Error budgets cap retries (default: 3 retry + 2 tool-error) |
| MCP prerequisites are too restrictive | Medium | Low | Prerequisites are opt-in per-tool; no prerequisites by default |
| Tiered compaction drops important context | Low | Medium | Phase-based approach preserves reasoning through Phase 2; configurable thresholds |

---

## 11. Success Metrics

| Metric | Baseline (Current) | Target (Post Phase 1-3) |
|--------|-------------------|------------------------|
| Tool calls with silently corrupted args | Non-zero (coerced to `{}`) | Zero (all flagged via validation) |
| Unknown tool names reaching clients | Non-zero | Zero (filtered or flagged) |
| `tool_choice="none"` violations | Non-zero | Zero |
| Admin chat: successful tool-call completion rate | Unknown | >90% for supported models |
| Rescue parser coverage | 5 strategies | 7 strategies (rehearsal + Mistral bracket-tag) |
| Clients able to implement retry loops | Must craft nudges manually | Use `x_omlx_validation` nudge directly |

---

## 12. References

- Forge source: `../forge/` (local), GitHub: forge-guardrails
- Forge IEEE paper: DOI 10.1145/3786335.3813193
- Forge ADRs (relevant): ADR-011 (Guardrail middleware), ADR-013 (Respond tool), ADR-015 (Cache control), ADR-016 (Malformed args tool error channel)
- oMLX tool calling: `omlx/api/tool_calling.py`
- oMLX MCP: `omlx/mcp/`
- oMLX server: `omlx/server.py`
