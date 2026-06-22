# tool-call-validation Specification

## Purpose
TBD - created by archiving change add-forge-guardrails. Update Purpose after archive.
## Requirements
### Requirement: Tool-call response validation
The system SHALL validate every parsed tool-call response. **Modified**: adds Check 5 (step enforcement) and Check 6 (prerequisite validation) when MCP prerequisite data is available. Adds "step" and "prerequisite" nudge kinds.

#### Scenario: Step nudge uses user-role with tier
- **WHEN** step enforcement detects a premature terminal tool call
- **THEN** the nudge has `role="user"`, `kind="step"`, and `tier` (1-3) based on consecutive failure count

#### Scenario: Prerequisite nudge uses user-role
- **WHEN** prerequisite validation detects a missing prerequisite
- **THEN** the nudge has `role="user"`, `kind="prerequisite"`, listing the missing prerequisite tool

### Requirement: Corrective nudge generation
The system SHALL generate a structured nudge message for each failed validation check. Each nudge SHALL specify `role` (either `"user"` for bare-text corrections or `"tool"` for tool-call corrections), `content` (the corrective message text), and `kind` (one of `"retry"`, `"unknown_tool"`, `"tool_arg_validation"`).

#### Scenario: Nudge uses tool-result role for tool-call errors
- **WHEN** a validation check fails for `unknown_tool` or `malformed_args` or `missing_required_params`
- **THEN** the generated nudge SHALL have `role="tool"` to match the wire shape models are pretrained on ("tool call failed → try again")

#### Scenario: Nudge uses user role for bare-text correction
- **WHEN** the `bare_text` check fails
- **THEN** the generated nudge SHALL have `role="user"` with corrective instructions

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

### Requirement: Strict tool arguments mode
The system SHALL preserve malformed tool-call arguments as-is (rather than coercing to `"{}"`) when `strict_tool_args` setting is enabled. Validation results SHALL surface the original malformed value regardless of this setting.

#### Scenario: Strict mode preserves malformed args
- **WHEN** `strict_tool_args` is `True` and the model emits a tool call with non-dict arguments
- **THEN** the system preserves the original argument value in the response and flags it via validation

#### Scenario: Default mode coerces for backward compatibility
- **WHEN** `strict_tool_args` is `False` (default) and the model emits a tool call with non-dict arguments
- **THEN** the system coerces arguments to `"{}"` as per current behavior, but still reports the validation failure in `x_omlx_validation` (if metadata is enabled)

### Requirement: Error budget defaults
The system SHALL provide configurable error budget defaults via `ForgeGuardrailsSettings`: `max_retries: int = 3` and `max_tool_errors: int = 2`. These defaults are serialized into the `x_omlx_validation.budget` extension for clients to use in bounded retry loops.

#### Scenario: Default budgets provided
- **WHEN** validation metadata is included in a response
- **THEN** the `budget` object contains `max_retries` and `max_tool_errors` from settings (defaults: 3 and 2)

#### Scenario: Budgets configurable via admin panel
- **WHEN** an admin updates `max_retries` or `max_tool_errors` via the admin panel
- **THEN** subsequent responses reflect the updated budget values

