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
