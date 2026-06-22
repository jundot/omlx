## Why

Change A (`add-forge-guardrails`) added tool-call validation and corrective nudges via `x_omlx_validation` response extension. However, clients receive no guidance on **how many times to retry** or **when to stop**. The validation metadata tells clients WHAT went wrong but not the retry budget (max attempts, current attempt, tool-error count). Additionally, oMLX has no message-level context compaction — when conversations exceed the model's window, the request fails.

Forge solves both problems: dual error budgets (retry + tool-error) with clean-batch reset, and 3-phase deterministic context compaction (sub-millisecond, no LLM calls). This change adapts both to oMLX's stateless architecture — budgets are tracked client-side using server-provided metadata, and compaction is available as a library for the admin chat and client-side use.

## What Changes

- **NEW: Error budget metadata** (`omlx/api/guardrails/budget.py`) — `ErrorBudget` dataclass with two independent counters (retry + tool-error), clean-batch reset logic, and exhaustion semantics. Serialized into `x_omlx_validation` as budget hints for clients.
- **CHANGED: `x_omlx_validation` extension** — adds optional `budget` object with `max_retries`, `max_tool_errors`, and `recommended_action` (`retry` | `give_up`) fields. Clients use this to bound their retry loops.
- **CHANGED: Nudge escalation** — adds `tier: int` field to `Nudge` dataclass (0 = N/A, 1 = polite, 2 = direct, 3 = aggressive). Step nudges escalate with consecutive failures.
- **NEW: Tiered context compaction** (`omlx/context/`) — `CompactStrategy` ABC with `NoCompact`, `SlidingWindowCompact`, `TieredCompact` implementations. 3-phase deterministic compaction: (1) drop nudges + truncate tool results, (2) drop tool results entirely, (3) drop reasoning + text. Sub-millisecond, pure Python.
- **NEW: Reference client retry loop** — documentation in `docs/tool-call-guardrails.md` showing how to use budget metadata + nudges for bounded client-side retry.

**Non-breaking**: All new metadata fields are optional. Clients that don't read them see identical behavior to Change A.

## Capabilities

### New Capabilities
- `context-compaction`: Tiered context compaction strategies for managing conversation length. Deterministic, sub-millisecond, no LLM calls. Three phases of increasing aggressiveness.

### Modified Capabilities
- `tool-call-validation`: Adds error budget metadata (max_retries, max_tool_errors, recommended_action) to the `x_omlx_validation` response extension. Adds nudge escalation tier field.

## Impact

**Code affected:**
- `omlx/api/guardrails/budget.py` — NEW: `ErrorBudget` dataclass with dual counters + clean-batch reset
- `omlx/api/guardrails/types.py` — MODIFY: add `tier: int = 0` to `Nudge`, add `budget` field to `ValidationResult.to_dict()` output
- `omlx/api/guardrails/nudge.py` — MODIFY: add tier parameter to step nudge generators
- `omlx/context/__init__.py` — NEW: context package
- `omlx/context/compaction.py` — NEW: `CompactStrategy` ABC + `NoCompact` + `SlidingWindowCompact` + `TieredCompact`
- `omlx/api/guardrail_wiring.py` — MODIFY: serialize budget metadata into `x_omlx_validation`
- `omlx/server.py` — MINIMAL: wire budget defaults from settings
- `omlx/settings.py` — MODIFY: add `max_retries`, `max_tool_errors`, `compaction_strategy` to `ForgeGuardrailsSettings`
- `docs/tool-call-guardrails.md` — MODIFY: add retry loop section + compaction docs
- `tests/` — NEW: budget tests, compaction tests, nudge tier tests

**APIs affected:**
- `x_omlx_validation` extension gains optional `budget` and `tier` fields (additive, non-breaking)

**Dependencies:**
- Builds on Change A infrastructure (`GuardrailValidator`, `ValidationResult`, `Nudge`, `ForgeGuardrailsSettings`)
- No new external dependencies

**Reference:**
- Forge source: `../forge/src/forge/guardrails/error_tracker.py`, `../forge/src/forge/context/strategies.py`
- Integration plan: `docs/forge-integration-plan.md` (Phase 3 + Phase 4)
- Change A: `openspec/changes/archive/2026-06-22-add-forge-guardrails/`
