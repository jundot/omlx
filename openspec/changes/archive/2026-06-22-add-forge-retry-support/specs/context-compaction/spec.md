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
