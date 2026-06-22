---
comet_change: add-forge-mcp-features
role: technical-design
canonical_spec: openspec
---

# Design Doc: MCP Features (Prerequisites + Respond Tool)

**Change**: `add-forge-mcp-features`
**Date**: 2026-06-22
**Phase**: comet-design

## Overview

Final change in the Forge integration. Adds MCP prerequisite enforcement (Phase 5) and synthetic respond tool (Phase 6). Both are opt-in, building on Changes A+B infrastructure.

## Architecture Decisions

### AD1: Prerequisites at validation layer
MCP tool execution is client-side. Prerequisites checked when tool calls are parsed — server returns nudges if ordering wrong. Client enforces before executing.

### AD2: StepTracker from message history
`PrerequisiteChecker` builds `executed_tools` by scanning prior assistant messages for tool_calls. Stateless per-request — no server session state.

### AD3: Respond tool inject/strip
Inject `respond(message)` before generation when setting enabled. Strip after parsing: pure respond → text, mixed → real calls only. Client never sees respond tool.

### AD4: Step nudges use user-role + tier
Step/prerequisite nudges use `role="user"` (workflow guidance, not tool errors). 3-tier escalation via `tier` field.

## Data Structures

```python
# omlx/mcp/prerequisites.py
@dataclass(frozen=True)
class PrerequisiteCheck:
    satisfied: bool
    missing: list[str]

class PrerequisiteChecker:
    def __init__(self, prerequisites: dict[str, list]):
        self._prereqs = prerequisites
    def check(self, tool_calls, prior_messages) -> list[CheckResult]:
        executed = self._build_executed_set(prior_messages)
        # For each tool call, check if its prerequisites are satisfied

# omlx/api/respond.py
RESPOND_TOOL_NAME = "respond"
def inject_respond_tool(tools: list) -> list:
    if not tools or any(t.get("function",{}).get("name")==RESPOND_TOOL_NAME for t in tools):
        return tools
    return tools + [{"type":"function","function":{"name":"respond","description":"...","parameters":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}}}]
def strip_respond_calls(tool_calls) -> tuple[list, str|None]:
    respond = [tc for tc in tool_calls if tc.function.name == RESPOND_TOOL_NAME]
    real = [tc for tc in tool_calls if tc.function.name != RESPOND_TOOL_NAME]
    if respond and not real: return [], respond[0].function.arguments_msg
    return real, None
```

## Settings Extension
```python
inject_respond_tool: bool = False
enforce_mcp_prerequisites: bool = False
```

## Implementation Sequence
1. Prerequisites module (independent) — can start immediately
2. Step/prerequisite nudges — depends on types change
3. Respond tool (independent) — can start immediately
4. Validator wiring — depends on 1+2
5. Server wiring — depends on 3+4
6. Settings — needed before server wiring
7. Tests + docs — last

## References
- Forge: `../forge/src/forge/guardrails/step_enforcer.py`, `../forge/src/forge/core/steps.py`, `../forge/src/forge/tools/respond.py`
