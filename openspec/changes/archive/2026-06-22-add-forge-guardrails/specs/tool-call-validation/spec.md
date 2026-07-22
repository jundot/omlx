## ADDED Requirements

### Requirement: Tool-call response validation
The system SHALL validate every parsed tool-call response before returning it to the client, when `guardrail_validation` setting is enabled. Validation SHALL perform four checks in order: bare-text-when-tools-expected, unknown tool name, malformed arguments (non-dict), and missing required parameters.

#### Scenario: Valid tool call passes all checks
- **WHEN** the model emits a well-formed tool call to a known tool with valid dict arguments satisfying all required parameters
- **THEN** the system returns the tool call normally with validation result `passed=True` and no nudge

#### Scenario: Unknown tool name is detected
- **WHEN** the model emits a tool call to a tool name not present in the request's `tools` list
- **THEN** the system flags the validation check `unknown_tool` as failed and generates a nudge with `kind="unknown_tool"`, `role="tool"`, listing the available tool names

#### Scenario: Malformed arguments (non-dict) are detected
- **WHEN** the model emits a tool call whose arguments cannot be parsed as a JSON object (e.g., a bare string, number, or array)
- **THEN** the system flags the validation check `malformed_args` as failed and generates a nudge with `kind="tool_arg_validation"`, `role="tool"`, indicating the received type and the required dict shape

#### Scenario: Missing required parameters are detected
- **WHEN** the model emits a tool call to a known tool with dict arguments, but one or more parameters listed in the tool's JSON Schema `required` array are absent
- **THEN** the system flags the validation check `missing_required_params` as failed and generates a nudge with `kind="tool_arg_validation"`, `role="tool"`, listing the missing parameter names

#### Scenario: Bare text when tools are expected is detected
- **WHEN** the model emits only text content (no tool calls) and `tools` were provided in the request and `tool_choice` is not `"none"`
- **THEN** the system flags the validation check `bare_text` as failed and generates a nudge with `kind="retry"`, `role="user"`, instructing the model to emit a tool call

#### Scenario: Validation disabled by default
- **WHEN** `guardrail_validation` setting is `False` (default)
- **THEN** the system performs no validation and returns responses identical to current behavior

### Requirement: Corrective nudge generation
The system SHALL generate a structured nudge message for each failed validation check. Each nudge SHALL specify `role` (either `"user"` for bare-text corrections or `"tool"` for tool-call corrections), `content` (the corrective message text), and `kind` (one of `"retry"`, `"unknown_tool"`, `"tool_arg_validation"`).

#### Scenario: Nudge uses tool-result role for tool-call errors
- **WHEN** a validation check fails for `unknown_tool` or `malformed_args` or `missing_required_params`
- **THEN** the generated nudge SHALL have `role="tool"` to match the wire shape models are pretrained on ("tool call failed → try again")

#### Scenario: Nudge uses user role for bare-text correction
- **WHEN** the `bare_text` check fails
- **THEN** the generated nudge SHALL have `role="user"` with corrective instructions

### Requirement: Validation metadata in response extension
The system SHALL include validation results and suggested nudge in the API response when `include_validation_metadata` setting is enabled. The metadata SHALL be carried in a non-standard `x_omlx_validation` extension field that does not interfere with standard OpenAI/Anthropic response fields.

#### Scenario: Validation metadata included when enabled
- **WHEN** `include_validation_metadata` is `True` and validation runs
- **THEN** the response includes an `x_omlx_validation` object containing `checks` (array of check results with `passed` boolean and optional `detail`) and `nudge` (the suggested corrective message or null if validation passed)

#### Scenario: No metadata when disabled
- **WHEN** `include_validation_metadata` is `False` (default)
- **THEN** the response contains no `x_omlx_validation` field and is indistinguishable from current behavior

#### Scenario: Streaming validation metadata
- **WHEN** validation runs on a streaming response and `include_validation_metadata` is `True`
- **THEN** the validation metadata SHALL be emitted as a final SSE event before `data: [DONE]`, since earlier events cannot be retroactively modified

### Requirement: Strict tool arguments mode
The system SHALL preserve malformed tool-call arguments as-is (rather than coercing to `"{}"`) when `strict_tool_args` setting is enabled. Validation results SHALL surface the original malformed value regardless of this setting.

#### Scenario: Strict mode preserves malformed args
- **WHEN** `strict_tool_args` is `True` and the model emits a tool call with non-dict arguments
- **THEN** the system preserves the original argument value in the response and flags it via validation

#### Scenario: Default mode coerces for backward compatibility
- **WHEN** `strict_tool_args` is `False` (default) and the model emits a tool call with non-dict arguments
- **THEN** the system coerces arguments to `"{}"` as per current behavior, but still reports the validation failure in `x_omlx_validation` (if metadata is enabled)
