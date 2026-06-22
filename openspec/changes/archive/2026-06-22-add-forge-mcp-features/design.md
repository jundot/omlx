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
