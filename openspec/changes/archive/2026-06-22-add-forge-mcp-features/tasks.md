## 1. MCP Prerequisites

- [x] 1.1 Create `omlx/mcp/prerequisites.py` with `PrerequisiteChecker` class: `__init__(prerequisites: dict)`, `check(tool_calls, prior_messages) -> list[CheckResult]`
- [x] 1.2 Implement name-only prerequisite checking: `"edit_file": {"requires": ["read_file"]}` — any prior `read_file` call satisfies
- [x] 1.3 Implement arg-matched prerequisite checking: `{"tool": "read_file", "match_arg": "path"}` — prior call must have matching arg value
- [x] 1.4 Build `executed_tools` set by scanning prior messages for tool calls (stateless per-request)
- [x] 1.5 Add `prerequisite_nudge(tool_name, missing_prereqs)` to `omlx/api/guardrails/nudge.py` — role="user", kind="prerequisite"
- [x] 1.6 Add `"step"` and `"prerequisite"` to nudge kind literals in `omlx/api/guardrails/types.py`
- [x] 1.7 Add prerequisite declaration parsing to `omlx/mcp/config.py` — `tools_prerequisites` field
- [x] 1.8 Update `mcp.example.json` with example prerequisite declarations
- [x] 1.9 Unit tests for PrerequisiteChecker (name-only, arg-matched, satisfied, missing, no-prereqs)

## 2. Step Enforcement + Nudges

- [x] 2.1 Add `step_nudge(terminal_tool, pending_steps, tier=1)` to `omlx/api/guardrails/nudge.py` — 3-tier escalation text, role="user", kind="step"
- [x] 2.2 Add step enforcement check (Check 5) to `GuardrailValidator` — detects premature terminal tool calls
- [x] 2.3 Wire step + prerequisite checks into validation flow (Checks 5-6)
- [x] 2.4 Unit tests for step nudges (3 tiers, content correctness)

## 3. Synthetic Respond Tool

- [x] 3.1 Create `omlx/api/respond.py` with `inject_respond_tool(tools)` and `strip_respond_calls(tool_calls)`
- [x] 3.2 `inject_respond_tool`: append respond tool spec if not already present
- [x] 3.3 `strip_respond_calls`: pure respond → text, mixed → real calls only, none → passthrough
- [x] 3.4 Wire injection into `server.py` — inject before generation, strip after parsing
- [x] 3.5 Add `inject_respond_tool: bool = False` and `enforce_mcp_prerequisites: bool = False` to `ForgeGuardrailsSettings`
- [x] 3.6 Wire new settings into admin panel
- [x] 3.7 Unit tests for inject + strip

## 4. Integration + Tests

- [x] 4.1 Wire prerequisite validation into `guardrail_wiring.py`
- [x] 4.2 Integration test: prerequisites end-to-end
- [x] 4.3 Integration test: respond tool end-to-end
- [x] 4.4 Run full test suite — 0 regressions across Changes A+B+C

## 5. Documentation

- [x] 5.1 Update `docs/tool-call-guardrails.md` — MCP Prerequisites + Respond Tool sections
- [x] 5.2 Update `docs/forge-integration-plan.md` — mark Phase 5+6 as implemented (all 6 phases done)
