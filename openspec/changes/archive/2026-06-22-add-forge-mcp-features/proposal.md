## Why

Changes A and B delivered tool-call validation, rescue parsers, error budgets, and context compaction. Two forge features remain: MCP prerequisite enforcement (prevents out-of-order tool execution) and the synthetic respond tool (helps small models stay in tool-calling mode). MCP tool calls often have implicit ordering dependencies (search → read → edit) that oMLX currently doesn't enforce. Small models (8B-14B) struggle with the text/tool-call decision — a synthetic respond tool keeps them in the grammar where guardrails apply.

## What Changes

- **NEW: MCP Prerequisite Checker** (`omlx/mcp/prerequisites.py`) — `PrerequisiteChecker` class that validates tool call ordering against declared prerequisites. Two modes: name-only (`read_file` must precede `edit_file`) and arg-matched (`read_file(path=X)` must precede `edit_file(path=X)`). Uses `StepTracker` pattern from forge — state lives outside message history so compaction can't corrupt it.
- **NEW: Step + Prerequisite Nudges** — add `step_nudge(terminal, pending, tier)` and `prerequisite_nudge(tool, missing)` generators to `omlx/api/guardrails/nudge.py`. Step nudges use 3-tier escalation (polite → direct → aggressive) via the `tier` field added in Change B.
- **NEW: Synthetic Respond Tool** (`omlx/api/respond.py`) — `inject_respond_tool(tools)` adds a `respond(message)` tool to the tools list. `strip_respond_calls(tool_calls)` removes respond calls from parsed output before returning to client. Pure respond → text response; mixed → real calls only.
- **NEW: Settings** — `inject_respond_tool: bool = False`, `enforce_mcp_prerequisites: bool = False` added to `ForgeGuardrailsSettings`.
- **NEW: MCP Config** — `tools_prerequisites` field in `mcp.example.json` showing prerequisite declarations.
- **CHANGED: GuardrailValidator** — add step enforcement check (Check 5) and prerequisite check (Check 6) when prerequisite data is available.

## Capabilities

### New Capabilities
- `mcp-prerequisite-enforcement`: Validate MCP tool call ordering against declared prerequisites. Name-only and arg-matched modes. StepTracker pattern for compaction-safe state.
- `synthetic-respond-tool`: Inject a synthetic respond(message) tool for small models. Strip respond calls from responses before returning to client.

### Modified Capabilities
- `tool-call-validation`: Add step enforcement and prerequisite validation checks (Checks 5 and 6) when MCP prerequisite data is available. Add step/prerequisite nudge kinds.

## Impact

**Code affected:**
- `omlx/mcp/prerequisites.py` — NEW: PrerequisiteChecker + StepTracker
- `omlx/api/respond.py` — NEW: inject_respond_tool + strip_respond_calls
- `omlx/api/guardrails/nudge.py` — MODIFY: add step_nudge + prerequisite_nudge generators
- `omlx/api/guardrails/types.py` — MODIFY: add "step" and "prerequisite" to nudge kind literals
- `omlx/api/guardrails/validator.py` — MODIFY: add step + prerequisite checks (Checks 5-6)
- `omlx/settings.py` — MODIFY: add inject_respond_tool + enforce_mcp_prerequisites fields
- `omlx/server.py` — MODIFY: wire respond tool injection + prerequisite validation
- `mcp.example.json` — MODIFY: add example prerequisite declarations
- `tests/` — NEW: prerequisite tests, respond tool tests, step nudge tests

**Reference:**
- Forge source: `../forge/src/forge/guardrails/step_enforcer.py`, `../forge/src/forge/core/steps.py`, `../forge/src/forge/tools/respond.py`
- Integration plan: `docs/forge-integration-plan.md` (Phase 5 + Phase 6)
