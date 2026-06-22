# tool-call-rescue-parsing Specification

## Purpose
TBD - created by archiving change add-forge-guardrails. Update Purpose after archive.
## Requirements
### Requirement: Rehearsal syntax rescue parser
The system SHALL parse tool calls expressed in rehearsal syntax (`tool_name[ARGS]{...json...}`) as a fallback strategy when all existing parsers return no matches. This format is produced by reasoning models that "rehearse" tool calls inside thinking tokens.

#### Scenario: Rehearsal syntax parsed successfully
- **WHEN** the model output (post thinking-tag extraction) contains text matching `tool_name[ARGS]{...}` where the brace-enclosed content is valid JSON
- **THEN** the system extracts a tool call with the parsed tool name and JSON arguments

#### Scenario: Multiple rehearsal calls extracted
- **WHEN** the model output contains multiple rehearsal-syntax expressions
- **THEN** the system extracts all valid expressions as separate tool calls

#### Scenario: Invalid JSON in rehearsal syntax ignored
- **WHEN** the brace-enclosed content after `[ARGS]` is not valid JSON
- **THEN** the system skips that expression and continues scanning for other matches (does not crash)

#### Scenario: Rehearsal parser only runs after existing parsers fail
- **WHEN** an existing parser (native, XML, namespaced, Hermes, bracket) successfully extracts tool calls
- **THEN** the rehearsal parser is NOT invoked (existing parser output takes precedence)

### Requirement: Improved Mistral bracket-tag rescue parser
The system SHALL parse Mistral `[TOOL_CALLS]` bracket-tag format using a brace-balance scanner with string and escape awareness, replacing or augmenting the current regex-based extraction. This handles nested JSON objects and string-escaped braces more robustly than regex.

#### Scenario: Nested JSON in Mistral bracket-tag
- **WHEN** the model output contains `[TOOL_CALLS]tool_name{...}` where the JSON object contains nested objects or arrays
- **THEN** the system correctly extracts the complete JSON object, respecting brace nesting depth

#### Scenario: Literal braces in string values
- **WHEN** the JSON arguments contain string values with literal `{` or `}` characters (e.g., `{"pattern": "use {placeholder}"}`)
- **THEN** the brace-balance scanner correctly distinguishes structural braces from string content and extracts the complete object

#### Scenario: Escaped quotes in string values
- **WHEN** the JSON arguments contain escaped quote characters (e.g., `{"text": "say \"hello\""}`)
- **THEN** the scanner correctly tracks escape state and extracts the complete object without premature termination

### Requirement: Rescue parser integration
The system SHALL integrate rescue parsers into the existing parser chain in `_parse_tool_calls_impl` (`tool_calling.py:1108`) as additional fallback strategies, tried after all existing parsers return no matches.

#### Scenario: Rescue parsers are last in the chain
- **WHEN** the parser chain runs, existing parsers (native, XML, namespaced, Hermes, bracket) are tried first
- **THEN** rehearsal and improved Mistral parsers are tried only if all existing parsers return no matches

#### Scenario: Rescue parsing toggleable
- **WHEN** rescue parsing is disabled (future setting, not in Change A scope)
- **THEN** only existing parsers run (current behavior preserved)

#### Scenario: Multiple rescue strategies tried in order
- **WHEN** the rehearsal parser returns no matches and the Mistral bracket-tag marker is present
- **THEN** the Mistral parser is tried next; if it also returns nothing, parsing falls through to the final marker-strip pass

