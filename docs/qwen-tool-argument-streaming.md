# Incremental Qwen tool arguments

Chat Completions clients can request partial JSON argument deltas with
`"stream_options": {"incremental_tool_arguments": true}`.
The default remains false, which keeps existing completed-envelope delivery.
The option requires an engine explicitly advertising raw early tool streaming
and the native mlx-lm `qwen3_coder` parser.

For a long `write` call, the first argument bytes can now arrive while the model
is still generating that call's content.
The existing envelope filter supplies fragments in order alongside ordinary
content, so coalesced text/tool/text chunks keep their order.
Thinking-channel calls and other parser families keep their existing handling.

Each call uses the usual OpenAI `index`, `id` and `function` fields.
Concatenate `function.arguments` deltas by index, and execute only complete,
validated calls, never partial JSON.
The final closing JSON bytes are withheld until the complete envelope passes
native parsing and its arguments match every byte already emitted.
Duplicate calls retain separate indexes and IDs.
Repeated XML parameters emit only one JSON key and succeed when native parsing
confirms the value already sent. If a later occurrence changes that value, the
stream ends with an error because earlier argument bytes cannot be withdrawn.

Malformed, oversized or interrupted partial calls produce an SSE error and no
successful completion; the client should discard the incomplete call.
An envelope is bounded to 2 Mi characters and an unfinished header to 4,096
characters.
Numeric and structured values stay buffered until their parameter closes;
string contents stream after resolving Qwen's special `null` handling.

This changes delivery timing without changing model inference or sampling.
Tests compare complete arguments, content and reasoning against the existing
server on synthetic fixtures and check that long arguments arrive before the
outer envelope closes.
