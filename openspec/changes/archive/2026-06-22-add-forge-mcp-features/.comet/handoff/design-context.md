# Comet Design Handoff

- Change: add-forge-mcp-features
- Phase: design
- Mode: compact
- Context hash: 07e9d83826984d722bcdc6e3cc3a782de49396e27337b37630d6915c7037a3eb

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/add-forge-mcp-features/proposal.md

- Source: openspec/changes/add-forge-mcp-features/proposal.md
- Lines: 1-38
- SHA256: 8da1ac401518daf6b19822217895cd39084cecf82f86720ca9474cace1fbfaad

```md
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
```

## openspec/changes/add-forge-mcp-features/design.md

- Source: openspec/changes/add-forge-mcp-features/design.md
- Lines: 1-38
- SHA256: 109585ccb53e324bad2a28f0a7b30df543305908e1ff1a429b09108f566de26a

```md
## Context

Changes A+B delivered validation, rescue parsing, tool_choice enforcement, error budgets, and compaction. Change C adds the final two forge features: MCP prerequisite enforcement (Phase 5) and synthetic respond tool (Phase 6). MCP tool execution is client-side — prerequisites are checked at the validation layer (server returns nudges, client enforces ordering). The respond tool helps small models avoid bare-text responses.

## Goals / Non-Goals

**Goals:**
- Validate MCP tool call ordering against declared prerequisites (name-only + arg-matched)
- Provide escalating step nudges for prerequisite violations (3 tiers)
- Inject synthetic respond tool for small models, strip from responses
- All features opt-in, zero breaking changes

**Non-Goals:**
- Server-side tool execution (client-side only)
- Full agentic loop / WorkflowRunner
- SlotWorker / hardware detection

## Decisions

### D1: Prerequisites checked at validation layer, not execution layer
MCP tool execution is client-side in oMLX. Prerequisites are checked when tool calls are parsed/validated — the server returns nudges if ordering is wrong. The client is responsible for enforcing the ordering before executing.

### D2: StepTracker pattern — state outside message history
`StepTracker` holds `executed_tools` dict. It's constructed per-request from the request's prior message history (scanning for tool calls). This is stateless per-request but provides prerequisite checking within a single conversation turn.

### D3: Respond tool — inject for small models, strip before return
When `inject_respond_tool=True`, the server adds a `respond(message)` tool to the tools list before generation. After parsing, `respond` calls are stripped: pure respond → text response; mixed → real tool calls only. The client never sees the respond tool.

### D4: Step + prerequisite nudges use user-role channel
Step nudges use `role="user"` (not tool-role) because they're workflow guidance, not tool errors. The 3-tier escalation (polite → direct → aggressive) uses the `tier` field from Change B.

## Risks / Trade-offs

**[Prerequisite false positives]** → Only fires when prerequisites are explicitly declared in config. No prerequisites = no checking. Non-blocking.

**[Respond tool confuses models]** → Opt-in (default off). Only recommended for small models (<14B) that struggle with text/tool-call decisions.

**[StepTracker stateless]** → Cannot track across requests (each request is independent). Within a single request's message history, prior tool calls are scanned to build the executed_tools set. This is a limitation but matches oMLX's stateless architecture.
```

## openspec/changes/add-forge-mcp-features/tasks.md

- Source: openspec/changes/add-forge-mcp-features/tasks.md
- Lines: 1-40
- SHA256: ac16598fc64237de023b1b14ca101e7ff1d30dae52b59fd34ae24d040543e328

```md
## 1. MCP Prerequisites

- [ ] 1.1 Create `omlx/mcp/prerequisites.py` with `PrerequisiteChecker` class: `__init__(prerequisites: dict)`, `check(tool_calls, prior_messages) -> list[CheckResult]`
- [ ] 1.2 Implement name-only prerequisite checking: `"edit_file": {"requires": ["read_file"]}` — any prior `read_file` call satisfies
- [ ] 1.3 Implement arg-matched prerequisite checking: `{"tool": "read_file", "match_arg": "path"}` — prior call must have matching arg value
- [ ] 1.4 Build `executed_tools` set by scanning prior messages for tool calls (stateless per-request)
- [ ] 1.5 Add `prerequisite_nudge(tool_name, missing_prereqs)` to `omlx/api/guardrails/nudge.py` — role="user", kind="prerequisite"
- [ ] 1.6 Add `"step"` and `"prerequisite"` to nudge kind literals in `omlx/api/guardrails/types.py`
- [ ] 1.7 Add prerequisite declaration parsing to `omlx/mcp/config.py` — `tools_prerequisites` field
- [ ] 1.8 Update `mcp.example.json` with example prerequisite declarations
- [ ] 1.9 Unit tests for PrerequisiteChecker (name-only, arg-matched, satisfied, missing, no-prereqs)

## 2. Step Enforcement + Nudges

- [ ] 2.1 Add `step_nudge(terminal_tool, pending_steps, tier=1)` to `omlx/api/guardrails/nudge.py` — 3-tier escalation text, role="user", kind="step"
- [ ] 2.2 Add step enforcement check (Check 5) to `GuardrailValidator` — detects premature terminal tool calls
- [ ] 2.3 Wire step + prerequisite checks into validation flow (Checks 5-6)
- [ ] 2.4 Unit tests for step nudges (3 tiers, content correctness)

## 3. Synthetic Respond Tool

- [ ] 3.1 Create `omlx/api/respond.py` with `inject_respond_tool(tools)` and `strip_respond_calls(tool_calls)`
- [ ] 3.2 `inject_respond_tool`: append respond tool spec if not already present
- [ ] 3.3 `strip_respond_calls`: pure respond → text, mixed → real calls only, none → passthrough
- [ ] 3.4 Wire injection into `server.py` — inject before generation, strip after parsing
- [ ] 3.5 Add `inject_respond_tool: bool = False` and `enforce_mcp_prerequisites: bool = False` to `ForgeGuardrailsSettings`
- [ ] 3.6 Wire new settings into admin panel
- [ ] 3.7 Unit tests for inject + strip

## 4. Integration + Tests

- [ ] 4.1 Wire prerequisite validation into `guardrail_wiring.py`
- [ ] 4.2 Integration test: prerequisites end-to-end
- [ ] 4.3 Integration test: respond tool end-to-end
- [ ] 4.4 Run full test suite — 0 regressions across Changes A+B+C

## 5. Documentation

- [ ] 5.1 Update `docs/tool-call-guardrails.md` — MCP Prerequisites + Respond Tool sections
- [ ] 5.2 Update `docs/forge-integration-plan.md` — mark Phase 5+6 as implemented (all 6 phases done)
```

## openspec/changes/add-forge-mcp-features/specs/mcp-prerequisite-enforcement/spec.md

- Source: openspec/changes/add-forge-mcp-features/specs/mcp-prerequisite-enforcement/spec.md
- Lines: 1-34
- SHA256: e73a7f04734f4678bbebb8f7ca604b13d52fea71f5e343b0c9a2862b75b7e498

```md
## ADDED Requirements

### Requirement: MCP prerequisite declaration
The system SHALL support declaring tool prerequisites in MCP configuration. Two declaration modes: name-only (`"read_file"`) and arg-matched (`{"tool": "read_file", "match_arg": "path"}`).

#### Scenario: Name-only prerequisite declared
- **WHEN** MCP config declares `"edit_file": {"requires": ["read_file"]}`
- **THEN** any call to `edit_file` requires a prior call to `read_file` (any args)

#### Scenario: Arg-matched prerequisite declared
- **WHEN** MCP config declares `"edit_file": {"requires": [{"tool": "read_file", "match_arg": "path"}]}`
- **THEN** `edit_file(path=X)` requires a prior `read_file(path=X)` with matching path arg

### Requirement: Prerequisite validation
The system SHALL validate tool call ordering against declared prerequisites when `enforce_mcp_prerequisites` is enabled. Violations generate a prerequisite nudge with `role="user"`, `kind="prerequisite"`.

#### Scenario: Missing prerequisite detected
- **WHEN** `edit_file(path="/tmp/foo")` is called without prior `read_file(path="/tmp/foo")`
- **THEN** validation flags the call and generates a nudge listing the missing prerequisite

#### Scenario: Satisfied prerequisite passes
- **WHEN** `edit_file(path="/tmp/foo")` is called after `read_file(path="/tmp/foo")` appeared in prior messages
- **THEN** the prerequisite check passes

#### Scenario: No prerequisites declared passes
- **WHEN** no prerequisites are declared for any tool
- **THEN** prerequisite validation is a no-op (all calls pass)

### Requirement: Prerequisite state from message history
The system SHALL build prerequisite state (executed tools) by scanning the request's prior message history for tool calls. This is stateless per-request — no server-side session state.

#### Scenario: Prior tool calls scanned from history
- **WHEN** a request contains prior assistant messages with tool_calls
- **THEN** the system extracts tool names and args from those messages to populate the executed_tools set
```

## openspec/changes/add-forge-mcp-features/specs/synthetic-respond-tool/spec.md

- Source: openspec/changes/add-forge-mcp-features/specs/synthetic-respond-tool/spec.md
- Lines: 1-31
- SHA256: 629fda6d031cd92143e78084e0ef9145b3b6f55d7989f8dd8f9d9b373f522f4c

```md
## ADDED Requirements

### Requirement: Synthetic respond tool injection
The system SHALL inject a synthetic `respond(message)` tool into the tools list when `inject_respond_tool` setting is enabled. The tool has a single `message: str` parameter and a description instructing the model to use it for text responses.

#### Scenario: Respond tool injected when enabled
- **WHEN** `inject_respond_tool` is `True` and tools are provided
- **THEN** a `respond` tool is appended to the tools list before generation

#### Scenario: Respond tool not injected when disabled
- **WHEN** `inject_respond_tool` is `False` (default)
- **THEN** no respond tool is added to the tools list

#### Scenario: Respond tool not duplicated
- **WHEN** a tool named `respond` already exists in the tools list
- **THEN** no additional respond tool is injected

### Requirement: Respond call stripping
The system SHALL strip `respond` calls from parsed tool call output before returning to the client. Three cases: pure respond (becomes text), mixed (real calls only), empty (empty response).

#### Scenario: Pure respond becomes text
- **WHEN** the only tool call is `respond(message="Hello")`
- **THEN** the response contains the message as text content with no tool calls

#### Scenario: Mixed calls — respond silently dropped
- **WHEN** tool calls include both `respond(message="...")` and `search(query="...")`
- **THEN** only the `search` call is returned; the respond call is silently dropped

#### Scenario: No respond calls — passthrough
- **WHEN** no respond calls are present in the parsed output
- **THEN** all tool calls pass through unchanged
```

## openspec/changes/add-forge-mcp-features/specs/tool-call-validation/spec.md

- Source: openspec/changes/add-forge-mcp-features/specs/tool-call-validation/spec.md
- Lines: 1-12
- SHA256: d56dc89c5e720456aa20c13b5482fcb3bbc866ac19c457081175dc7289c6471f

```md
## MODIFIED Requirements

### Requirement: Tool-call response validation
The system SHALL validate every parsed tool-call response. **Modified**: adds Check 5 (step enforcement) and Check 6 (prerequisite validation) when MCP prerequisite data is available. Adds "step" and "prerequisite" nudge kinds.

#### Scenario: Step nudge uses user-role with tier
- **WHEN** step enforcement detects a premature terminal tool call
- **THEN** the nudge has `role="user"`, `kind="step"`, and `tier` (1-3) based on consecutive failure count

#### Scenario: Prerequisite nudge uses user-role
- **WHEN** prerequisite validation detects a missing prerequisite
- **THEN** the nudge has `role="user"`, `kind="prerequisite"`, listing the missing prerequisite tool
```

