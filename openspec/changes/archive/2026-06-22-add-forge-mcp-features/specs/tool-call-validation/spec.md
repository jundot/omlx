## MODIFIED Requirements

### Requirement: Tool-call response validation
The system SHALL validate every parsed tool-call response. **Modified**: adds Check 5 (step enforcement) and Check 6 (prerequisite validation) when MCP prerequisite data is available. Adds "step" and "prerequisite" nudge kinds.

#### Scenario: Step nudge uses user-role with tier
- **WHEN** step enforcement detects a premature terminal tool call
- **THEN** the nudge has `role="user"`, `kind="step"`, and `tier` (1-3) based on consecutive failure count

#### Scenario: Prerequisite nudge uses user-role
- **WHEN** prerequisite validation detects a missing prerequisite
- **THEN** the nudge has `role="user"`, `kind="prerequisite"`, listing the missing prerequisite tool
