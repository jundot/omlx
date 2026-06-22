## ADDED Requirements

### Requirement: Synthetic respond tool injection
The system SHALL inject a synthetic `respond(message)` tool into the tools list when `inject_respond_tool` setting is enabled. The tool has a single `message: str` parameter and a description instructing the model to use it for text responses.

#### Scenario: Respond tool injected when enabled
- **WHEN** `inject_respond_tool` is `True` and tools are provided
- **THEN** a `respond` tool is appended to the tools list before generation

#### Scenario: Respond tool not injected when disabled
- **WHEN** `inject_respond_tool` is `False` (default)
- **THEN** no respond tool is added to the tools list

#### Scenario: Respond tool not duplicated
- **WHEN** a tool named `respond` already exists in the tools list
- **THEN** no additional respond tool is injected

### Requirement: Respond call stripping
The system SHALL strip `respond` calls from parsed tool call output before returning to the client. Three cases: pure respond (becomes text), mixed (real calls only), empty (empty response).

#### Scenario: Pure respond becomes text
- **WHEN** the only tool call is `respond(message="Hello")`
- **THEN** the response contains the message as text content with no tool calls

#### Scenario: Mixed calls — respond silently dropped
- **WHEN** tool calls include both `respond(message="...")` and `search(query="...")`
- **THEN** only the `search` call is returned; the respond call is silently dropped

#### Scenario: No respond calls — passthrough
- **WHEN** no respond calls are present in the parsed output
- **THEN** all tool calls pass through unchanged
