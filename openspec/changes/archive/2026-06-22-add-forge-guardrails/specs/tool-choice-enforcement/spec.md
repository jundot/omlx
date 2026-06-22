## ADDED Requirements

### Requirement: tool_choice enforcement
The system SHALL enforce `tool_choice` semantics after parsing and validation, before response construction. All four modes SHALL be supported: `"none"`, `"auto"`, `"required"`, and `{"type":"function","function":{"name":"X"}}`.

#### Scenario: tool_choice "none" suppresses tool calls
- **WHEN** `tool_choice` is `"none"` and the model produces tool calls in its response
- **THEN** the system suppresses all tool calls and returns only the text content

#### Scenario: tool_choice "auto" passes through
- **WHEN** `tool_choice` is `"auto"` (or omitted)
- **THEN** the system passes through whatever the model produces (tool calls and/or text)

#### Scenario: tool_choice "required" rejects bare text
- **WHEN** `tool_choice` is `"required"` and the model produces only text content with no tool calls
- **THEN** the system flags a validation failure with nudge `kind="retry"`, informing that a tool call is required

#### Scenario: tool_choice "required" accepts tool calls
- **WHEN** `tool_choice` is `"required"` and the model produces at least one tool call
- **THEN** the system accepts the tool calls normally

#### Scenario: Named-tool filtering
- **WHEN** `tool_choice` is `{"type":"function","function":{"name":"search"}}` and the model produces tool calls including some to tools other than `search`
- **THEN** the system filters the response to include only tool calls matching the named tool `search`, and flags rejected calls with nudge `kind="unknown_tool"`

#### Scenario: Invalid tool_choice value rejected at request time
- **WHEN** the client sends a `tool_choice` value that is not one of the supported formats (e.g., a random string like `"weird"`)
- **THEN** the system rejects the request with HTTP 400 before inference begins

### Requirement: tool_choice enforcement ordering
The system SHALL apply `tool_choice` enforcement AFTER validation checks and BEFORE response construction, so that enforcement failures benefit from the same nudge infrastructure as validation failures.

#### Scenario: Enforcement after validation
- **WHEN** a response has both a malformed-args validation failure and a `tool_choice="required"` enforcement failure
- **THEN** both failures appear in the validation metadata, with validation checks listed before enforcement checks
