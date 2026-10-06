# Incremental Qwen tool arguments

Chat Completions streams Qwen XML argument fragments through the existing raw
Qwen engine capability gate. Stock clients need no extra request option.
Other engines and non-streaming responses retain their existing handling.

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

Length stops retain `finish_reason: length` and terminal usage when available.
Incomplete arguments remain open and must not execute. Malformed complete calls
and argument validation failures produce an SSE error. Cancellation closes the
producer. No closing syntax is invented, and no mutation is automatically retried.
A complete function missing only its outer envelope close retains the existing
final-parser recovery rule; its validated suffix uses the original call ID.
Early envelope delivery is bounded to 2 Mi characters. Beyond the limit,
final parsing validates any emitted prefix and delivers the remaining suffix.
Unfinished headers retain at most 4,096 characters.
Numeric and structured values stay buffered until their parameter closes;
string contents stream after resolving Qwen's special `null` handling.

This changes delivery timing without changing model inference or sampling.
Tests compare complete arguments, content and reasoning against the existing
server on synthetic fixtures and check that long arguments arrive before the
outer envelope closes.

Ported from ashhart's commits 3a1979a and 09400ec in
https://github.com/jundot/omlx/pull/3646. The port retains the existing naked-function
filter and terminal-generation accounting, then adapts activation and finalization
for the custom GA stack.

Triple-quoted Python docstring prefixes stream as literal string contents.
Split opening quotes wait until the prefix is settled: three quotes cannot
become a JSON-encoded string in the recovery parser. Single/double-quote
prefixes and meaningful leading whitespace keep their existing deferral.
Native parsing, recovery values and completion validation are unchanged.

The incremental parser retains a SHA-256 digest and character count of emitted
arguments, checked once against final parsing. Non-string values remain buffered
until their parameter closes. The existing server transcript and envelope filter
still retain raw output for final parsing; total request memory is not constant.

Strings with ambiguous leading quotes/whitespace or trailing whitespace defer
those bytes to final parsing so missing-outer-close recovery keeps its existing
semantics. Boolean schemas and untyped properties also use final parsing.

Only literal `<tool_call>` outer envelopes supply incremental fragments. Bare
functions and namespaced wrappers retain complete-envelope delivery, including
when their string arguments contain literal paired-tool opening markup.

The gate recognizes the exact Qwen parser functions shipped by both mlx-lm and
mlx-vlm. Conversion helpers come from the registered function's own module;
the pinned VLM parser has different integer and structured-value conversion
rules. The local engine must still explicitly report a raw scheduler without
a structured output parser.
