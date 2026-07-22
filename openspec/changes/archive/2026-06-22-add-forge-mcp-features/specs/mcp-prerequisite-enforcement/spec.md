## ADDED Requirements

### Requirement: MCP prerequisite declaration
The system SHALL support declaring tool prerequisites in MCP configuration. Two declaration modes: name-only (`"read_file"`) and arg-matched (`{"tool": "read_file", "match_arg": "path"}`).

#### Scenario: Name-only prerequisite declared
- **WHEN** MCP config declares `"edit_file": {"requires": ["read_file"]}`
- **THEN** any call to `edit_file` requires a prior call to `read_file` (any args)

#### Scenario: Arg-matched prerequisite declared
- **WHEN** MCP config declares `"edit_file": {"requires": [{"tool": "read_file", "match_arg": "path"}]}`
- **THEN** `edit_file(path=X)` requires a prior `read_file(path=X)` with matching path arg

### Requirement: Prerequisite validation
The system SHALL validate tool call ordering against declared prerequisites when `enforce_mcp_prerequisites` is enabled. Violations generate a prerequisite nudge with `role="user"`, `kind="prerequisite"`.

#### Scenario: Missing prerequisite detected
- **WHEN** `edit_file(path="/tmp/foo")` is called without prior `read_file(path="/tmp/foo")`
- **THEN** validation flags the call and generates a nudge listing the missing prerequisite

#### Scenario: Satisfied prerequisite passes
- **WHEN** `edit_file(path="/tmp/foo")` is called after `read_file(path="/tmp/foo")` appeared in prior messages
- **THEN** the prerequisite check passes

#### Scenario: No prerequisites declared passes
- **WHEN** no prerequisites are declared for any tool
- **THEN** prerequisite validation is a no-op (all calls pass)

### Requirement: Prerequisite state from message history
The system SHALL build prerequisite state (executed tools) by scanning the request's prior message history for tool calls. This is stateless per-request — no server-side session state.

#### Scenario: Prior tool calls scanned from history
- **WHEN** a request contains prior assistant messages with tool_calls
- **THEN** the system extracts tool names and args from those messages to populate the executed_tools set
