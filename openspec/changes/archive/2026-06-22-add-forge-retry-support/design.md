## Context

Change A delivered tool-call validation with `x_omlx_validation` response extension containing check results and nudges. Clients can detect validation failures and have the corrective nudge text, but they lack:
1. **Budget guidance** — how many retries are reasonable before giving up?
2. **Compaction** — what to do when the conversation (including retry messages) exceeds the model's context window?

Forge solves both via dual error budgets (retry=3, tool_error=2, clean-batch reset) and 3-phase deterministic compaction. This change adapts both to oMLX's stateless architecture.

## Goals / Non-Goals

**Goals:**
- Provide budget hints in `x_omlx_validation` so clients can implement bounded retry loops
- Add nudge escalation tiers for step-by-step correction
- Implement tiered context compaction as a reusable library (admin chat + client-side)
- Document reference retry loop patterns

**Non-Goals:**
- Server-side retry loop (staying stateless)
- Stateful admin chat API
- LLM-based summarization (compaction is deterministic only)

## Decisions

### D1: Advisory budgets (server provides defaults, client tracks counts)

**Decision**: The server does NOT track retry counts per session. It provides `max_retries` and `max_tool_errors` as configurable defaults in `x_omlx_validation.budget`. The client tracks actual counts across retry turns.

**Rationale**: oMLX is stateless per-request. Tracking retry counts requires session state, which we explicitly don't build (Change B non-goal). The server's role is to provide sensible budget defaults; the client's role is to enforce them.

### D2: Clean-batch reset as documentation, not enforcement

**Decision**: The clean-batch reset rule (only a fully successful tool-call batch resets the tool-error counter) is documented in the reference client code but not enforced server-side. The server provides the rule as metadata; clients implement it.

**Rationale**: Same as D1 — no server-side state. The rule is important for client correctness, so it's documented prominently.

### D3: Three compaction strategies, pluggable ABC

**Decision**: `CompactStrategy` ABC with 3 implementations:
- `NoCompact` — passthrough (default for API requests)
- `SlidingWindowCompact` — keep last N messages
- `TieredCompact` — 3-phase priority-based (nudges → tool results → reasoning+text)

All are pure Python, sub-millisecond, no LLM calls.

### D4: Nudge tier for escalation only

**Decision**: Add `tier: int = 0` to `Nudge`. Only step/prerequisite nudges use tiers 1-3. Other nudges (retry, unknown_tool, tool_arg_validation) stay at tier 0 (N/A). The tier is informational — clients can use it to customize message display or escalate wording.

## Risks / Trade-offs

**[Budget defaults may not fit all use cases]** → Configurable via settings (`max_retries`, `max_tool_errors`). Clients can override. Non-blocking.

**[Compaction drops important context]** → Tiered approach preserves reasoning through Phase 2. Phase 3 (most aggressive) only triggers when Phase 2 result still exceeds budget. Protected: system prompt + user input are never cut.

**[Nudge tier unused by current validation checks]** → Only meaningful for step enforcement (Change C). Including the field now is forward-looking but harmless (defaults to 0).
