# Tool Call Guardrails

> Optional validation layer for tool-calling responses, adapted from [Forge](https://github.com/jundot/forge) (IEEE DOI: 10.1145/3786335.3813193).

oMLX provides an opt-in guardrail system that validates tool-call responses before they reach the client. When enabled, it catches malformed arguments, unknown tool names, missing required parameters, and bare-text-when-tools-expected — and generates corrective nudge messages clients can use to implement retry loops.

## Quick Start

All guardrail features are **off by default**. Enable them via the admin panel (Settings → Forge Guardrails) or by editing `~/.omlx/settings.json`:

```json
{
  "forge_guardrails": {
    "validation_enabled": true,
    "strict_tool_args": false,
    "include_validation_metadata": true
  }
}
```

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `validation_enabled` | `false` | Enable post-parse validation of tool-call responses |
| `strict_tool_args` | `false` | Preserve malformed args as-is instead of coercing to `"{}"` |
| `include_validation_metadata` | `false` | Include `x_omlx_validation` extension in API responses |

All settings are live-mutable — no server restart required. Changes take effect on the next request.

## Validation Checks

When `validation_enabled` is `true`, every tool-call response runs through 4 checks:

| Check | What it catches | Nudge kind | Nudge role |
|-------|----------------|------------|------------|
| **Bare text** | Model emitted text instead of a tool call when tools were expected | `retry` | `user` |
| **Unknown tool** | Model called a tool not in the request's `tools` list | `unknown_tool` | `tool` |
| **Malformed args** | Arguments are not a JSON object (dict) | `tool_arg_validation` | `tool` |
| **Missing required params** | Required parameters (from tool JSON Schema) are absent | `tool_arg_validation` | `tool` |

Checks run in priority order. All results are accumulated — the client sees every check that failed, but only one nudge (from the highest-priority failure) per response.

## Response Extension

When `include_validation_metadata` is `true`, responses include an `x_omlx_validation` field:

### Non-streaming

```json
{
  "choices": [{"message": {"role": "assistant", "content": "...", "tool_calls": [...]}}],
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

### Streaming

For streaming responses, the validation metadata is emitted as a final SSE event before the terminal signal:

- **OpenAI** (`/v1/chat/completions`): `data: {"x_omlx_validation": {...}}\n\n` before `data: [DONE]`
- **Anthropic** (`/v1/messages`): `event: x_omlx_validation\ndata: {...}\n\n` before `message_stop`
- **Responses API** (`/v1/responses`): `event: response.x_omlx_validation\ndata: {...}\n\n` before `response.completed`

## Client-Side Retry

The nudge message is designed for direct use in a retry loop. Append it to the conversation and re-send:

```python
# Python client example
response = client.chat.completions.create(
    model="local-model",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

validation = response.model_extra.get("x_omlx_validation") if hasattr(response, "model_extra") else None
# Or access via the raw response dict

if validation and not validation["passed"]:
    nudge = validation["nudge"]
    # Append the nudge as a new message
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    messages.append({"role": nudge["role"], "content": nudge["content"]})
    # Retry
    response = client.chat.completions.create(
        model="local-model",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
```

### Why tool-role nudges?

For `unknown_tool` and `tool_arg_validation` failures, the nudge uses `role: "tool"` (not `role: "user"`). Models are pretrained on the "tool call failed → try again" wire shape and attend to tool-result messages better than trailing user messages. This pattern comes from Forge's IEEE-published research.

## tool_choice Enforcement

When validation is enabled, `tool_choice` is also enforced:

| `tool_choice` | Behavior |
|---------------|----------|
| `"none"` | Suppresses all tool calls (returns only text) |
| `"auto"` | Pass through (no-op) |
| `"required"` | Rejects bare text — model must emit at least one tool call |
| `{"type":"function","function":{"name":"X"}}` | Filters to named tool only; other calls rejected |

Invalid `tool_choice` values are rejected at request time with HTTP 422.

## Rescue Parsers

Two additional rescue parsers run as last-resort fallbacks when all existing parsers fail:

1. **Rehearsal syntax**: Matches `tool_name[ARGS]{...json...}` — produced by reasoning models that "rehearse" tool calls in thinking tokens.

2. **Mistral bracket-tag**: Improved brace-balance scanner with string/escape awareness for `[TOOL_CALLS]` format. Handles nested JSON objects and literal braces in string values more robustly than regex.

These parsers only run when existing parsers (native, XML, namespaced, Hermes, bracket) return nothing.

## Strict Args Mode

By default (`strict_tool_args: false`), malformed arguments are coerced to `"{}"` for backward compatibility — the client always receives a valid JSON string. The validation metadata still reports the original malformed value.

When `strict_tool_args: true`, malformed arguments are preserved as-is in the response. This is useful for debugging and for clients that want to handle malformed args themselves.

## API Compatibility

- Works with all 3 chat endpoints: `/v1/chat/completions`, `/v1/messages`, `/v1/responses`
- Both streaming and non-streaming supported
- `x_omlx_validation` is a non-standard extension field — existing OpenAI/Anthropic clients ignore it
- All features are opt-in — zero behavior change when disabled

## Implementation

This feature is Change A of the Forge integration plan (Phases 1+2). See:
- [Forge Integration Plan](forge-integration-plan.md) — full plan with 6 phases
- [Design Doc](../docs/superpowers/specs/2026-06-21-forge-guardrails-design.md) — technical design
- Forge source: `../forge/src/forge/guardrails/` — reference implementation
