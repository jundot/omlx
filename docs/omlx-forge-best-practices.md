# oMLX Forge Guardrails — Best Practices

> Practical guide for configuring and using the Forge-inspired tool-call guardrail system in oMLX.
> Covers all 6 phases of the Forge integration (Changes A, B, C).

---

## Quick Reference: Settings

All settings live under `forge_guardrails` in `~/.omlx/settings.json` or the admin panel (Settings → Forge Guardrails). **Everything defaults to off** — zero behavior change until you opt in.

| Setting | Default | What it does |
|---------|---------|-------------|
| `validation_enabled` | `false` | Run 6 validation checks on every tool-call response |
| `strict_tool_args` | `false` | Preserve malformed args instead of coercing to `"{}"` |
| `include_validation_metadata` | `false` | Include `x_omlx_validation` extension in responses |
| `max_retries` | `3` | Advisory retry budget for clients |
| `max_tool_errors` | `2` | Advisory tool-error budget for clients |
| `compaction_strategy` | `"none"` | Context compaction: `"none"`, `"sliding_window"`, `"tiered"` |
| `inject_respond_tool` | `false` | Inject synthetic respond tool for small models |
| `enforce_mcp_prerequisites` | `false` | Validate MCP tool call ordering |

---

## 1. Gradual Rollout Strategy

Don't enable everything at once. Roll out in phases to monitor impact:

### Phase 1: Observe (No Breaking Changes)

```json
{
  "validation_enabled": true,
  "include_validation_metadata": true
}
```

Validation runs but doesn't modify responses. You see what's failing via `x_omlx_validation` without changing client behavior. **Monitor**: what percentage of responses have validation failures? Which checks fail most?

### Phase 2: Enable Strict Mode

```json
{
  "validation_enabled": true,
  "include_validation_metadata": true,
  "strict_tool_args": true
}
```

Malformed args are now preserved instead of coerced to `"{}"`. Clients that were silently receiving `"{}"` will now see the actual malformed value. **Monitor**: do any clients break when they receive non-dict arguments?

### Phase 3: Client Retry Loops

```json
{
  "validation_enabled": true,
  "include_validation_metadata": true,
  "strict_tool_args": true,
  "max_retries": 3,
  "max_tool_errors": 2
}
```

Update your client to read `x_omlx_validation.budget` and implement a bounded retry loop (see [Client Retry Pattern](#3-client-retry-pattern) below). **Monitor**: retry success rate, token usage increase.

### Phase 4: Advanced Features

```json
{
  "validation_enabled": true,
  "include_validation_metadata": true,
  "compaction_strategy": "tiered",
  "enforce_mcp_prerequisites": true
}
```

Enable compaction for long conversations and prerequisite enforcement for MCP tools. **Monitor**: compaction phase frequency, prerequisite violation rate.

### Phase 5: Respond Tool (Small Models Only)

```json
{
  "inject_respond_tool": true
}
```

Only for models <14B parameters that struggle with text/tool-call decisions. **Monitor**: reduction in bare-text validation failures.

---

## 2. Model-Size Recommendations

| Model Size | Recommended Settings | Rationale |
|------------|---------------------|-----------|
| **Frontier (70B+)** | `validation_enabled: true`, `include_validation_metadata: true` | High accuracy, rare failures. Validation catches the occasional hallucination. |
| **Mid-range (32B-70B)** | Above + `strict_tool_args: true`, retry budgets | Occasional malformed args. Strict mode surfaces issues for debugging. |
| **Small (8B-14B)** | Above + `inject_respond_tool: true`, `compaction_strategy: "tiered"` | Frequent text/tool-call confusion. Respond tool keeps them in grammar. Compaction handles retry-heavy conversations. |
| **Tiny (<8B)** | All of the above + `max_retries: 5` | Very high failure rate. More retries needed. Consider whether tool-calling is viable at this size. |

---

## 3. Client Retry Pattern

The server is stateless — it doesn't track retry counts. Your client must. Here's a production-ready retry loop:

```python
import json

def chat_with_retry(client, model, messages, tools, max_iterations=10):
    """Chat with automatic retry on validation failures."""
    retry_count = 0
    tool_error_count = 0

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            extra_headers={"X-Request-ID": f"retry-{retry_count}"},
        )

        # Check for validation metadata
        raw = response.model_dump()
        validation = raw.get("x_omlx_validation")

        if not validation or validation["passed"]:
            return response  # Success or no validation

        # Validation failed — extract budget and nudge
        budget = validation.get("budget", {})
        max_retries = budget.get("max_retries", 3)
        max_tool_errors = budget.get("max_tool_errors", 2)

        # Check budgets
        if retry_count >= max_retries and tool_error_count >= max_tool_errors:
            return response  # Budgets exhausted — return last response

        # Extract nudge
        nudge = validation.get("nudge")
        if not nudge:
            return response  # No nudge to retry with

        # Track which budget to increment
        if nudge["kind"] in ("unknown_tool", "tool_arg_validation"):
            tool_error_count += 1
        else:
            retry_count += 1

        # Append assistant response + nudge to conversation
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content or "",
            "tool_calls": response.choices[0].message.tool_calls or [],
        })
        messages.append({
            "role": nudge["role"],
            "content": nudge["content"],
        })

    return response  # Max iterations reached
```

### Clean-Batch Reset Rule

Only a **fully successful tool-call batch** resets the tool-error counter. If 3 out of 4 tools succeed but 1 fails, the counter does NOT reset. Implement this in your client:

```python
# After executing tools:
all_succeeded = all(result.success for result in tool_results)
if all_succeeded:
    tool_error_count = 0  # Clean batch — reset
```

---

## 4. MCP Prerequisite Configuration

### Declaring Prerequisites

Add `tools_prerequisites` to your MCP server config:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "tools_prerequisites": {
        "edit_file": {
          "requires": [
            {"tool": "read_file", "match_arg": "path"}
          ]
        },
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

### Best Practices for Prerequisites

1. **Use arg-matched prerequisites for path-based tools**. Name-only is too loose — `read_file("/a")` shouldn't satisfy `edit_file("/b")`.

2. **Don't over-constrain**. Only declare prerequisites where ordering truly matters. Unnecessary prerequisites increase false positives.

3. **Test with real workflows**. Enable `include_validation_metadata` first to observe violation patterns before enforcing.

4. **Prerequisites are per-request**. The server scans the current request's message history. If your client doesn't include prior tool results in the messages array, prerequisites won't see them.

---

## 5. Respond Tool — When to Use

### Do Use When:
- Model is <14B parameters
- You see frequent bare-text validation failures (`check: "bare_text"`)
- Model produces text when `tool_choice="required"` or tools are provided
- You're using a reasoning model that "thinks out loud" instead of calling tools

### Don't Use When:
- Model is 32B+ and rarely produces bare text
- Your client can't handle the injected tool transparently (most can — it's stripped before return)
- You're debugging tool-calling issues and need to see the model's raw behavior

### How to Verify It's Working

Enable `include_validation_metadata` and check:
- `bare_text` check failures should decrease significantly
- `x_omlx_validation.passed` should be `true` more often
- No `respond` tool should appear in the client-facing response

---

## 6. Context Compaction — When to Use

### Decision Matrix

| Scenario | Strategy | Why |
|----------|----------|-----|
| Short conversations (<10 messages) | `"none"` | No compaction needed |
| Medium conversations (10-50 messages) | `"sliding_window"` | Simple, predictable |
| Long conversations with retries (50+ messages) | `"tiered"` | Preserves reasoning, drops low-value messages |
| Stateless API requests | `"none"` | Each request is independent — no history to compact |

### Tiered Compaction Phase Guide

Phase 1 (triggers at 75% of context budget):
- Drops: nudge messages, retry messages, truncated tool results
- Keeps: everything else
- Impact: Minimal — only removes correction-related chatter

Phase 2 (triggers at 85% if Phase 1 wasn't enough):
- Drops: tool results entirely
- Keeps: reasoning, text, tool-call skeletons
- Impact: Model may re-call tools it already called (wastes one iteration)

Phase 3 (triggers at 95% if Phase 2 wasn't enough):
- Drops: reasoning + text
- Keeps: tool-call skeletons only
- Impact: Significant — model loses context about what it was doing

**Never compacted**: System prompt, original user input, last 2 iterations.

---

## 7. Streaming Considerations

When using streaming responses (`stream: true`):

- Validation runs **after** all tool-call markup is parsed (at stream end)
- The `x_omlx_validation` metadata arrives as the **final SSE event** before `[DONE]`
- You cannot retroactively fix earlier-streamed events — the metadata is informational only
- For retry loops: buffer the full stream, validate at end, then decide whether to retry

```python
# Streaming retry pattern
events = []
for chunk in client.chat.completions.create(..., stream=True):
    events.append(chunk)
    if hasattr(chunk, "x_omlx_validation"):
        validation = chunk.x_omlx_validation
        # Decide: retry or accept
```

---

## 8. API Compatibility

### Non-Breaking by Design

- `x_omlx_validation` is a non-standard extension field — OpenAI/Anthropic SDKs ignore unknown fields
- All features are opt-in — disabled means byte-identical responses
- `strict_tool_args` only affects how malformed args are serialized when enabled

### Client SDK Notes

| SDK | `x_omlx_validation` access | Notes |
|-----|---------------------------|-------|
| OpenAI Python | `response.model_extra.get("x_omlx_validation")` | Pydantic stores unknown fields in `model_extra` |
| Anthropic Python | `response.model_extra.get("x_omlx_validation")` | Same pattern |
| Raw HTTP | `response.json()["x_omlx_validation"]` | Direct dict access |
| Streaming | Final SSE data event | Parse the last `data:` line before `[DONE]` |

---

## 9. Monitoring and Metrics

Track these metrics to measure guardrail effectiveness:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Validation pass rate | `% of responses where x_omlx_validation.passed == true` | >90% for supported models |
| Bare-text failure rate | `% where check "bare_text" failed` | <5% (use respond tool if higher) |
| Unknown-tool rate | `% where check "unknown_tool" failed` | <2% |
| Malformed-args rate | `% where check "malformed_args" failed` | <1% |
| Retry success rate | `% of retries that eventually pass` | >80% |
| Average retries per request | Count from client | <1.5 |

### Logging

oMLX logs validation failures at `WARNING` level. Enable structured logging to capture:

```
{
  "event": "guardrail_validation_failed",
  "checks_failed": ["unknown_tool", "malformed_args"],
  "nudge_kind": "unknown_tool",
  "model": "Qwen3-8B",
  "retry_count": 1
}
```

---

## 10. Common Pitfalls

### Pitfall: Enabling strict_tool_args breaks existing client

**Symptom**: Client crashes on non-dict arguments after enabling `strict_tool_args`.

**Fix**: Keep `strict_tool_args: false` until your client handles malformed args. The validation metadata still reports what's wrong — you just don't change the serialization.

### Pitfall: Retry loop increases token costs unexpectedly

**Symptom**: Token usage doubles or triples after enabling retries.

**Fix**: Set conservative budgets (`max_retries: 2`, `max_tool_errors: 1`). Monitor token usage per request. The nudge messages add to context length — use `compaction_strategy: "tiered"` for long retry chains.

### Pitfall: Prerequisites not detecting prior calls

**Symptom**: `enforce_mcp_prerequisites` enabled but prior tool calls aren't detected.

**Fix**: The checker scans the `messages` array in the request. Ensure your client includes prior assistant messages with `tool_calls` and tool result messages in the conversation history. If your client only sends the latest message, prerequisites can't see prior calls.

### Pitfall: Respond tool not being stripped

**Symptom**: Client sees a `respond` tool call in the response.

**Fix**: This shouldn't happen — stripping is automatic. If it does, verify you're on the latest version. The stripping happens in `server.py` after parsing, before response construction. File a bug if `respond` calls leak.

### Pitfall: Streaming validation arrives too late

**Symptom**: Client already processed tool calls before seeing validation failure.

**Fix**: This is by design — streaming validation is informational. For corrective action, buffer the stream and validate at end. Or use non-streaming requests when validation is critical.

---

## 11. Testing Your Integration

### Unit Test Pattern

```python
def test_client_handles_validation_failure():
    """Verify client correctly handles x_omlx_validation."""
    mock_response = {
        "choices": [{"message": {"role": "assistant", "tool_calls": [...]}}],
        "x_omlx_validation": {
            "passed": False,
            "checks": [{"check": "unknown_tool", "passed": False}],
            "nudge": {"role": "tool", "content": "Tool 'bad' doesn't exist", "kind": "unknown_tool", "tier": 0},
            "budget": {"max_retries": 3, "max_tool_errors": 2}
        }
    }
    # Your client should: detect failure, extract nudge, append and retry
```

### Integration Test Checklist

- [ ] Client detects `x_omlx_validation.passed == false`
- [ ] Client extracts nudge and appends to messages
- [ ] Client respects `budget.max_retries` and stops when exhausted
- [ ] Client handles clean-batch reset correctly
- [ ] Client handles streaming validation metadata (final SSE event)
- [ ] Client works normally when validation is disabled (backward compat)
- [ ] Client handles `strict_tool_args` malformed values gracefully

---

## 12. Feature Interaction Matrix

| Feature A | Feature B | Interaction |
|-----------|-----------|-------------|
| Validation | Respond tool | Respond tool reduces bare-text failures → fewer validation nudges |
| Validation | Prerequisites | Prerequisites add Checks 5-6 to the validation pipeline |
| Validation | Compaction | Compaction drops nudge/retry messages (Phase 1) to free context |
| Respond tool | tool_choice="required" | Respond tool satisfies the "at least one tool call" requirement |
| Strict args | Validation | Strict mode preserves malformed args → validation flags them |
| Prerequisites | Compaction | StepTracker state survives compaction (lives outside message history) |
| Retry budgets | Compaction | Retries add messages → triggers compaction sooner |

---

## References

- [Tool Call Guardrails](tool-call-guardrails.md) — User-facing feature documentation
- [Forge Integration Plan](forge-integration-plan.md) — Original 6-phase plan
- [Forge IEEE Paper](https://doi.org/10.1145/3786335.3813193) — Academic reference
- Forge source: `../forge/src/forge/` — Reference implementation
- OpenSpec specs: `openspec/specs/` — 6 capability specifications
- Design docs: `docs/superpowers/specs/2026-06-2*-forge-*.md` — Technical design documents
