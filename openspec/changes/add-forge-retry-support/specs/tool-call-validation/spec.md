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
