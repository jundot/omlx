# Comet Design Handoff

- Change: add-forge-retry-support
- Phase: design
- Mode: compact
- Context hash: c3cdf6f97f70c645b66d0faa09b932bd3970405d25ea04b87600e287b1c47de5

Generated-by: comet-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/add-forge-retry-support/proposal.md

- Source: openspec/changes/add-forge-retry-support/proposal.md
- Lines: 1-49
- SHA256: 5f05e83123bd0391e2e360cde9a9c59ee690c2c9d17a478f94f27646c73b9b07

```md
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
```

## openspec/changes/add-forge-retry-support/design.md

- Source: openspec/changes/add-forge-retry-support/design.md
- Lines: 1-55
- SHA256: 8215c5dec3f3aa90876f6541a3866306074d8f990d1b95aa58a34291d839f5d2

```md
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
```

## openspec/changes/add-forge-retry-support/tasks.md

- Source: openspec/changes/add-forge-retry-support/tasks.md
- Lines: 1-41
- SHA256: 5da9d3db27938e0d817fd5cb46535e9b202ce6e555e2aa96692fedd95c189cf1

```md
## 1. Error Budget Module

- [ ] 1.1 Create `omlx/api/guardrails/budget.py` with `ErrorBudget` dataclass: `max_retries: int = 3`, `max_tool_errors: int = 2`, `recommended_action: str = "retry"`, plus `to_dict()` and `from_dict()`
- [ ] 1.2 Implement `should_retry(retry_count, tool_error_count) -> bool` method on `ErrorBudget` — returns False when either budget exhausted
- [ ] 1.3 Unit tests for ErrorBudget (defaults, exhaustion, to_dict/from_dict round-trip)

## 2. Nudge Tier Extension

- [ ] 2.1 Modify `omlx/api/guardrails/types.py` — add `tier: int = 0` field to `Nudge` dataclass (frozen, backward compatible)
- [ ] 2.2 Modify `omlx/api/guardrails/nudge.py` — existing nudge generators stay tier=0; add optional `tier` param for future step nudges
- [ ] 2.3 Update `Nudge.to_dict()` serialization to include `tier` field
- [ ] 2.4 Unit tests for tier field (default 0, explicit tier, serialization)

## 3. Budget Metadata in Response Extension

- [ ] 3.1 Modify `omlx/api/guardrails/validator.py` — add `budget: ErrorBudget | None` field to `ValidationResult`
- [ ] 3.2 Modify `ValidationResult.to_dict()` — include `budget` object when present
- [ ] 3.3 Modify `omlx/api/guardrail_wiring.py` — construct `ErrorBudget` from settings and attach to `ValidationResult`
- [ ] 3.4 Unit tests for budget serialization in validation result

## 4. Settings Extension

- [ ] 4.1 Modify `omlx/settings.py` — add `max_retries: int = 3`, `max_tool_errors: int = 2`, `compaction_strategy: str = "none"` to `ForgeGuardrailsSettings`
- [ ] 4.2 Update `ForgeGuardrailsSettings.to_dict()` and `from_dict()` for new fields
- [ ] 4.3 Wire new fields into `GlobalSettingsRequest` in `admin/routes.py` (admin panel)
- [ ] 4.4 Unit tests for settings round-trip with new fields

## 5. Context Compaction Package

- [ ] 5.1 Create `omlx/context/__init__.py` with public API exports
- [ ] 5.2 Create `omlx/context/compaction.py` with `CompactStrategy` ABC (abstract `compact(messages, budget_tokens) -> tuple[list, int]`)
- [ ] 5.3 Implement `NoCompact` — passthrough strategy
- [ ] 5.4 Implement `SlidingWindowCompact` — keep last N messages
- [ ] 5.5 Implement `TieredCompact` — 3-phase priority compaction (Phase 1: drop nudges + truncate tool results; Phase 2: drop tool results; Phase 3: drop reasoning + text). Protected: system prompt, user input, recent iterations.
- [ ] 5.6 Unit tests for all 3 strategies (phase transitions, protection rules, truncation markers)

## 6. Documentation

- [ ] 6.1 Update `docs/tool-call-guardrails.md` — add "Client-Side Retry Loops" section with: budget metadata explanation, clean-batch reset rule, reference retry loop code example
- [ ] 6.2 Add "Context Compaction" section to docs — explain 3 strategies, when to use each, configuration
- [ ] 6.3 Update `docs/forge-integration-plan.md` — mark Phase 3+4 as implemented via Change B
```

## openspec/changes/add-forge-retry-support/specs/context-compaction/spec.md

- Source: openspec/changes/add-forge-retry-support/specs/context-compaction/spec.md
- Lines: 1-39
- SHA256: 207d5e6328641a5830111728468ce0c92b2a9c8ccada4fd2243babb9e36cc623

```md
## ADDED Requirements

### Requirement: Tiered context compaction strategies
The system SHALL provide pluggable context compaction strategies via a `CompactStrategy` ABC with three implementations: `NoCompact` (passthrough), `SlidingWindowCompact` (keep last N messages), and `TieredCompact` (3-phase priority-based). All strategies SHALL be pure Python with sub-millisecond execution and no LLM calls.

#### Scenario: NoCompact passes messages through unchanged
- **WHEN** `NoCompact` strategy is applied to a message list
- **THEN** the message list is returned unchanged regardless of token count

#### Scenario: TieredCompact Phase 1 drops nudges and truncates tool results
- **WHEN** `TieredCompact` is applied and messages exceed the Phase 1 threshold (budget × 0.75)
- **THEN** STEP_NUDGE, PREREQUISITE_NUDGE, and RETRY_NUDGE messages are dropped, and TOOL_RESULT messages are truncated to the first 200 characters with a truncation marker

#### Scenario: TieredCompact Phase 2 drops tool results entirely
- **WHEN** Phase 1 result still exceeds the Phase 2 threshold
- **THEN** TOOL_RESULT messages are dropped entirely; REASONING and TEXT_RESPONSE are preserved

#### Scenario: TieredCompact Phase 3 drops reasoning and text
- **WHEN** Phase 2 result still exceeds the Phase 3 threshold
- **THEN** REASONING and TEXT_RESPONSE messages are dropped; only TOOL_CALL skeletons remain

#### Scenario: System prompt and user input are never compacted
- **WHEN** any compaction strategy runs
- **THEN** the system prompt (messages[0]) and original user input (messages[1]) are always preserved

#### Scenario: Recent iterations are protected
- **WHEN** TieredCompact runs with `keep_recent=2`
- **THEN** messages from the last 2 iterations are protected from compaction regardless of phase

### Requirement: Compaction strategy configuration
The system SHALL allow configuring the compaction strategy via `ForgeGuardrailsSettings.compaction_strategy` field with values `"none"`, `"sliding_window"`, or `"tiered"`. Default: `"none"`.

#### Scenario: Default compaction is none
- **WHEN** no compaction strategy is configured
- **THEN** the system uses `NoCompact` (passthrough)

#### Scenario: Tiered compaction enabled
- **WHEN** `compaction_strategy` is set to `"tiered"`
- **THEN** `TieredCompact` is used with default thresholds (0.75, 0.85, 0.95) and keep_recent=2
```

## openspec/changes/add-forge-retry-support/specs/tool-call-validation/spec.md

- Source: openspec/changes/add-forge-retry-support/specs/tool-call-validation/spec.md
- Lines: 1-37
- SHA256: aeef04df7727dcc040bf9cb44022513d1439f23a8d12a267abe3ee06d2c09272

```md
## MODIFIED Requirements

### Requirement: Validation metadata in response extension
The system SHALL include validation results and suggested nudge in the API response when `include_validation_metadata` setting is enabled. The metadata SHALL be carried in a non-standard `x_omlx_validation` extension field. **Modified**: adds optional `budget` object and `tier` field.

#### Scenario: Validation metadata includes budget hints
- **WHEN** `include_validation_metadata` is `True` and validation fails
- **THEN** the response includes `x_omlx_validation.budget` with `max_retries` (default 3), `max_tool_errors` (default 2), and `recommended_action` (`"retry"` if budgets not exhausted, `"give_up"` if exhausted)

#### Scenario: Budget recommends retry when under limits
- **WHEN** validation fails and budget hints indicate remaining retry capacity
- **THEN** `recommended_action` is `"retry"`

#### Scenario: Budget recommends give_up when exhausted
- **WHEN** validation fails and budget hints indicate all retries would be exhausted
- **THEN** `recommended_action` is `"give_up"`

#### Scenario: Nudge includes escalation tier
- **WHEN** a nudge is generated for a step or prerequisite failure
- **THEN** the nudge includes `tier` field (1=polite, 2=direct, 3=aggressive) indicating escalation level

#### Scenario: Non-step nudges have tier 0
- **WHEN** a nudge is generated for retry, unknown_tool, or tool_arg_validation
- **THEN** the nudge `tier` is 0 (not applicable)

## ADDED Requirements

### Requirement: Error budget defaults
The system SHALL provide configurable error budget defaults via `ForgeGuardrailsSettings`: `max_retries: int = 3` and `max_tool_errors: int = 2`. These defaults are serialized into the `x_omlx_validation.budget` extension for clients to use in bounded retry loops.

#### Scenario: Default budgets provided
- **WHEN** validation metadata is included in a response
- **THEN** the `budget` object contains `max_retries` and `max_tool_errors` from settings (defaults: 3 and 2)

#### Scenario: Budgets configurable via admin panel
- **WHEN** an admin updates `max_retries` or `max_tool_errors` via the admin panel
- **THEN** subsequent responses reflect the updated budget values
```

