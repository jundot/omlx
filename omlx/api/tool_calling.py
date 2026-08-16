# SPDX-License-Identifier: Apache-2.0
"""
Tool calling parsing and conversion utilities.

Uses mlx-lm's modular tool parser system to support multiple model formats:
- json_tools: Pure JSON format
- minimax_m2: MiniMax M2 XML format
- function_gemma: Google Gemma function calling format
- glm47: GLM-4.7 format
- qwen3_coder: Qwen3 Coder XML format

The tool parser is automatically selected based on the model's chat template.

Also includes structured output (JSON Schema) utilities:
- parse_json_output: Extract JSON from model output
- validate_json_schema: Validate JSON against a schema
"""

import ast
import bisect
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import regex
from jsonschema import ValidationError, validate

from .openai_models import FunctionCall, ResponseFormat, ToolCall

logger = logging.getLogger(__name__)


def _template_safe_description(value: Any) -> str:
    """Return a string description safe for strict chat templates."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _copy_schema_with_template_defaults(value: Any, *, is_schema: bool) -> Any:
    """Copy JSON Schema data while filling missing schema descriptions."""
    if isinstance(value, dict):
        copied = {}
        for key, child in value.items():
            if key == "properties" and isinstance(child, dict):
                copied[key] = {
                    name: _copy_schema_with_template_defaults(
                        prop_schema, is_schema=True
                    )
                    for name, prop_schema in child.items()
                }
            elif key in {
                "items",
                "additionalProperties",
                "contains",
                "propertyNames",
                "not",
                "if",
                "then",
                "else",
            }:
                copied[key] = _copy_schema_with_template_defaults(child, is_schema=True)
            elif key in {"oneOf", "anyOf", "allOf", "prefixItems"} and isinstance(
                child, list
            ):
                copied[key] = [
                    _copy_schema_with_template_defaults(item, is_schema=True)
                    for item in child
                ]
            else:
                copied[key] = _copy_schema_with_template_defaults(
                    child, is_schema=False
                )

        if is_schema:
            copied["description"] = _template_safe_description(
                copied.get("description")
            )
        return copied

    if isinstance(value, list):
        return [
            _copy_schema_with_template_defaults(item, is_schema=False) for item in value
        ]

    return value


def _serialize_tool_call_arguments(arguments: Any) -> str:
    """Serialize parser output to a JSON-object arguments string.

    Chat templates for models with native tool calling (Qwen 3.5/3.6 XML,
    GLM, MiniMax) iterate `arguments.items()` when the call is echoed back
    in history. Anything that does not represent a JSON object must be
    coerced to "{}" here so we never hand the client a non-JSON value that
    the next turn's template would crash on.
    """
    # `json.dumps` recurses per nesting level just as the decoders do, and it
    # runs after a value has already decoded successfully, from a deeper stack
    # frame. A value nested just under the limit at decode time can therefore
    # breach it here (#2545, found by DiscoStew6082 at depth ~987 on 3.11).
    #
    # A breach is deliberately NOT coerced to "{}" like a non-object value is.
    # The two are different failures: a non-object is a parser quirk we can
    # safely normalize, whereas failing to serialize means we HAVE the
    # arguments and cannot render them. Returning "{}" there would hand back a
    # runnable tool call with its arguments silently removed, so `write_file`
    # would still fire with nothing to write. Let it raise instead, so
    # `_build_tool_call` drops that one call with a warning, which is what the
    # issue asks for (jundot's review on #2593).
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    # mlx-vlm / mlx-lm gemma4 parser returns a JSON-object string per the
    # OpenAI spec. Accept it when it parses back to a dict.
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            # Deep-nest errors are deliberately NOT caught here. Catching them
            # sends a value we failed to decode into the "{}" coercion below,
            # which is the same silent argument loss as the dict branch above:
            # the parser handed us real arguments and we would return a
            # runnable call without them. Letting it propagate reaches
            # `_build_tool_call`, which drops that one call. Malformed JSON
            # still coerces, since that is a parser quirk rather than lost
            # data. (DiscoStew6082 caught this branch on #2593.)
            parsed = None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
    logger.warning(
        "Tool parser returned non-dict arguments (type=%s, repr=%.200r); "
        "coercing to empty object to keep downstream template safe.",
        type(arguments).__name__,
        arguments,
    )
    return "{}"


def _build_tool_call(name: str, arguments: Any) -> Optional[ToolCall]:
    """Build a ToolCall, dropping this one call if validation cannot finish.

    ``FunctionCall`` re-parses the arguments string while validating it, which
    is a *third* decode of a value the parser already decoded and serialized,
    and it runs from a deeper stack frame than either (#2545).  Recursion
    limits are about remaining stack rather than input depth, so a value that
    was fine at both earlier steps can still breach the limit here.

    Validation failure has to drop this one call and leave the rest of the
    batch alone, the same way an unparseable match does, rather than escape
    the parse chain.
    """
    try:
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name=name,
                arguments=_serialize_tool_call_arguments(arguments),
            ),
        )
    except (TypeError, ValueError, *_DEEP_NEST_ERRORS) as exc:
        logger.warning(
            "Dropping tool call %.80r: arguments failed validation (%s: %s)",
            name,
            type(exc).__name__,
            exc,
        )
        return None


@dataclass(frozen=True)
class ToolCallExtraction:
    """Parsed tool-call result plus sanitized reasoning text."""

    cleaned_text: str
    tool_calls: Optional[List[ToolCall]]
    cleaned_thinking: str
    tool_calls_from_thinking: bool = False


# Declared-type buckets for schema-aware parameter coercion, mirroring
# mlx-lm's qwen3_coder parser so fallback-recovered calls keep the same
# argument types as natively parsed calls (#2332).
_SCHEMA_STRING_TYPES = {"string", "str", "text", "varchar", "char", "enum"}
_SCHEMA_BOOL_TYPES = {"boolean", "bool", "binary"}
_SCHEMA_CONTAINER_TYPES = {"object", "array", "arr"}
_SCHEMA_INT_PREFIXES = ("int", "uint", "long", "short", "unsigned")


def _tool_param_properties(func_name: str, tools: Optional[List]) -> dict:
    """Return the declared parameter properties for a tool, or {}."""
    if not tools:
        return {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict) or func.get("name") != func_name:
            continue
        params = func.get("parameters")
        if isinstance(params, dict):
            props = params.get("properties")
            if isinstance(props, dict):
                return props
        return {}
    return {}


def _repair_json_value(val: str) -> Optional[Any]:
    """Best-effort repair of near-valid JSON with unbalanced brackets.

    Rewrites closing brackets that do not match the innermost open bracket
    (a common small-model slip, e.g. closing an array with "}"), drops
    closers with no matching opener, terminates an unclosed string, and
    appends missing closers.  Returns the parsed value, or None when the
    repaired text still fails to parse.
    """
    out: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in val:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
            out.append(ch)
        elif ch in "]}":
            if stack:
                out.append(stack.pop())
        else:
            out.append(ch)
    if in_string:
        out.append('"')
    while stack:
        out.append(stack.pop())
    try:
        return json.loads("".join(out), strict=False)
    except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
        return None


def _coerce_param_value(val: str, key: str, props: dict, func_name: str) -> Any:
    """Convert an XML-extracted parameter value per its declared schema type.

    Mirrors the type conversion mlx-lm's native qwen3_coder parser applies,
    with an extra bracket-repair pass for declared container params whose
    value is near-valid JSON (#2332).  Parameters without a usable declared
    type keep the legacy best-effort JSON parse.
    """
    spec = props.get(key)
    raw_type = spec.get("type") if isinstance(spec, dict) else None
    if not isinstance(raw_type, str):
        # Undeclared param, union type list, or anyOf: legacy behavior.
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
            return val
    if val.strip().lower() == "null":
        return None
    ptype = raw_type.strip().lower()
    if ptype in _SCHEMA_STRING_TYPES:
        # A JSON-quoted string literal (e.g. MiniMax emits "SF" for a string
        # param) carries JSON encoding on the wire; decode it back to the
        # underlying string.  Plain values that merely look like JSON (42,
        # {"a": 1}) are kept verbatim so a string param never gets coerced to a
        # non-string.
        stripped = val.strip()
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            try:
                decoded = json.loads(stripped)
            except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
                decoded = None
            if isinstance(decoded, str):
                return decoded
        return val
    if ptype in _SCHEMA_BOOL_TYPES:
        lowered = val.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    if ptype.startswith(_SCHEMA_INT_PREFIXES):
        try:
            return int(val.strip())
        except ValueError:
            pass
    elif ptype.startswith(("num", "float")):
        try:
            num = float(val.strip())
            return int(num) if num == int(num) else num
        except (ValueError, OverflowError):
            pass
    try:
        return json.loads(val, strict=False)
    except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
        pass
    try:
        literal = ast.literal_eval(val)
        if isinstance(literal, (dict, list, tuple)):
            return list(literal) if isinstance(literal, tuple) else literal
    except (ValueError, SyntaxError, TypeError, MemoryError, *_DEEP_NEST_ERRORS):
        pass
    if ptype in _SCHEMA_CONTAINER_TYPES or ptype.startswith(("dict", "list")):
        repaired = _repair_json_value(val)
        if repaired is not None:
            logger.warning(
                "Repaired malformed JSON for parameter %r of tool %r "
                "(declared type %r)",
                key,
                func_name,
                ptype,
            )
            return repaired
        logger.warning(
            "Parameter %r of tool %r failed to parse as declared type %r; "
            "keeping raw string",
            key,
            func_name,
            ptype,
        )
    return val


# Shared decoder for locating payload boundaries. ``strict=False`` matches the
# tolerance already used when parsing tool-call JSON elsewhere in this module,
# so boundary detection never rejects a payload the parser would have accepted.
_TOOL_CALL_JSON_DECODER = json.JSONDecoder(strict=False)

# Deeply nested model output breaks the decoders in a version-dependent way:
# `json.loads` raises RecursionError on some Python versions and a plain
# JSONDecodeError on others, while `ast.literal_eval`/`ast.parse` raise
# SyntaxError or RecursionError depending on where the compiler gives up.
# Neither RecursionError nor SyntaxError is a ValueError, so both slip past
# excepts written for decode errors and escape the parse chain (#2545).
# Model output is untrusted and attacker-influenceable, so a breach has to be
# a clean parse failure on the existing drop path, never a raised exception.
# Added to every decode-site except in this module rather than relying on the
# bounds, because only some of these paths are bounded.
_DEEP_NEST_ERRORS = (RecursionError, SyntaxError)

# Boundary detection is a hint, not the real parse, so it is cheap to bound.
# Same rationale as _GEMMA4_MAX_ARGS_LEN below: model output is untrusted and
# attacker-influenceable, so exceeding the bound must degrade to the historical
# behaviour rather than burn time on a payload we are only measuring.
_TOOL_CALL_MAX_BOUNDARY_SCAN = 262_144


def _json_value_end(text: str, start: int) -> Optional[int]:
    """Index just past the complete JSON value beginning at/after ``start``.

    Returns ``None`` when the text does not begin a JSON object/array there, or
    when that value is still incomplete.  Only ``{``/``[`` openers are treated
    as JSON: tool-call payloads in the non-JSON dialects (GLM
    ``<arg_key>``/``<arg_value>``, Qwen ``<function=...>``) must keep their
    historical handling, and a bare scalar would let ``raw_decode`` claim a
    prefix of some other markup.

    Never raises. ``RecursionError`` is caught alongside ``ValueError`` because
    ``raw_decode`` recurses per nesting level, so deeply nested model output
    (``"[" * 100000``) blows the stack rather than returning a decode error.
    A boundary hint must never turn into an exception escaping the parse chain.
    """
    idx = start
    while idx < len(text) and text[idx] in " \t\r\n":
        idx += 1
    if idx >= len(text) or text[idx] not in "{[":
        return None
    if len(text) - idx > _TOOL_CALL_MAX_BOUNDARY_SCAN:
        return None
    try:
        _, end = _TOOL_CALL_JSON_DECODER.raw_decode(text, idx)
    except (ValueError, RecursionError, *_DEEP_NEST_ERRORS):
        return None
    return end


_XML_FUNCTION_OPEN = "<function="
_XML_FUNCTION_CLOSE = "</function>"
_XML_PARAMETER_OPEN = "<parameter="
_XML_PARAMETER_CLOSE = "</parameter>"
# Candidate envelope ends examined before giving up. Each candidate costs a
# balance count over the payload, so this keeps hostile output linear.
_XML_MAX_END_CANDIDATES = 32
# Trailing payload context the stream filter keeps so a `</function>` split
# across chunks is still visible when the close marker arrives.
_XML_TAIL_KEEP = 64


def _skip_ws(text: str, idx: int) -> int:
    """First index at/after ``idx`` that is not ASCII whitespace."""
    while idx < len(text) and text[idx] in " \t\r\n":
        idx += 1
    return idx


def _xml_function_payload_end(
    text: str, payload_start: int, end_marker: str
) -> Optional[int]:
    """Index just past the ``</function>`` closing a qwen3_coder payload.

    The ``qwen3_coder`` parser (Qwen3.5/3.6 builds) wraps XML rather than JSON
    in the envelope::

        <tool_call><function=name><parameter=k>value</parameter></function></tool_call>

    so ``_json_value_end`` cannot bound it and a literal close marker inside a
    parameter value truncates the payload the same way (#2507).

    The structural anchor is that a real envelope ends with ``</function>``
    followed by the close marker, and that its parameter elements balance.
    Requiring both skips copies embedded in values, including a value that
    contains the whole ``</function>`` + close-marker sequence, while still
    ending at the FIRST call when several are concatenated.

    Returns ``None`` when the payload is not this dialect or has no such pair,
    which leaves the caller on the historical first-match behaviour.
    """
    idx = _skip_ws(text, payload_start)
    if not text.startswith(_XML_FUNCTION_OPEN, idx):
        return None
    search = idx
    # Bounded so a value stuffed with fake terminators cannot make the balance
    # check quadratic; past the cap we simply decline to bound the payload.
    for _ in range(_XML_MAX_END_CANDIDATES):
        close = text.find(_XML_FUNCTION_CLOSE, search)
        if close < 0:
            return None
        after = close + len(_XML_FUNCTION_CLOSE)
        if text.startswith(end_marker, _skip_ws(text, after)):
            payload = text[idx:close]
            if payload.count(_XML_PARAMETER_OPEN) == payload.count(
                _XML_PARAMETER_CLOSE
            ):
                return after
        search = after
    return None


def _xml_element_value_end(text: str, start: int, close_tag: str, next_open: str) -> int:
    """End index of an XML value, tolerating ``close_tag`` inside the value.

    The value really ends at the ``close_tag`` whose next non-space token is
    either ``next_open`` (another sibling element) or the end of the enclosing
    text. Copies sitting inside the value are followed by neither, so they are
    skipped. Falls back to the first ``close_tag`` when nothing matches, which
    is the historical behaviour, and to ``len(text)`` when there is none.
    """
    search = start
    while True:
        close = text.find(close_tag, search)
        if close < 0:
            first = text.find(close_tag, start)
            return first if first >= 0 else len(text)
        after = _skip_ws(text, close + len(close_tag))
        if after >= len(text) or text.startswith(next_open, after):
            return close
        search = close + len(close_tag)


_XML_PARAMETER_OPEN_RE = re.compile(r"<parameter=(\w+)>")


def _iter_xml_parameters(params_text: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(key, value)`` for each ``<parameter=k>v</parameter>`` element.

    Scanning advances past each value rather than using ``finditer`` over the
    open tag alone, so a literal ``<parameter=`` sitting inside a value cannot
    start a spurious element, and a literal ``</parameter>`` cannot end one
    early (#2507).
    """
    pos = 0
    while True:
        match = _XML_PARAMETER_OPEN_RE.search(params_text, pos)
        if not match:
            return
        value_end = _xml_element_value_end(
            params_text, match.end(), _XML_PARAMETER_CLOSE, _XML_PARAMETER_OPEN
        )
        yield match.group(1), params_text[match.end() : value_end].strip()
        pos = value_end + len(_XML_PARAMETER_CLOSE)


def _find_marker_span_end(
    text: str, payload_start: int, end_marker: str
) -> Optional[Tuple[int, int]]:
    """Locate the close marker that actually terminates a tool-call payload.

    Returns ``(payload_end, span_end)`` or ``None`` when no close marker
    follows at all.

    Why this exists instead of a plain ``start(.*?)end`` regex: the non-greedy
    match stops at the FIRST close marker, so a tool call whose argument
    contains a literal copy of that marker is cut mid-JSON, fails to parse and
    is dropped silently (#2507).  When the payload is JSON, ``raw_decode``
    reports where the value really ends, which is authoritative and ignores
    marker text sitting inside a string.

    Two payload shapes can be bounded structurally: JSON, via ``raw_decode``,
    and the ``qwen3_coder`` XML dialect, via its ``</function>`` close.

    Deliberately conservative: a boundary is used *only* when it proves the
    payload extends past the first close marker AND a later close marker
    exists.  Every other case falls back to the historical first-match
    behaviour, so this can only change outputs that are broken today.
    """
    plain = text.find(end_marker, payload_start)
    if plain < 0:
        return None
    boundary = _json_value_end(text, payload_start)
    if boundary is None:
        boundary = _xml_function_payload_end(text, payload_start, end_marker)
    if boundary is not None and boundary > plain:
        later = text.find(end_marker, boundary)
        if later >= 0:
            return later, later + len(end_marker)
    return plain, plain + len(end_marker)


def _iter_marker_spans(
    text: str, start_marker: str, end_marker: str
) -> Iterator[Tuple[int, int, str]]:
    """Yield ``(span_start, span_end, payload)`` for each marker-delimited call.

    ``span_start`` indexes the start marker and ``span_end`` is just past the
    close marker, so callers can both parse the payload and excise the whole
    span. Unterminated trailing envelopes are not yielded, matching the regex
    behaviour this replaces.
    """
    pos = 0
    while True:
        start = text.find(start_marker, pos)
        if start < 0:
            return
        payload_start = start + len(start_marker)
        found = _find_marker_span_end(text, payload_start, end_marker)
        if found is None:
            return
        payload_end, span_end = found
        yield start, span_end, text[payload_start:payload_end]
        pos = span_end


def _marker_payloads(text: str, start_marker: str, end_marker: str) -> List[str]:
    """Payloads of every complete marker-delimited tool call, in order."""
    return [payload for _s, _e, payload in _iter_marker_spans(text, start_marker, end_marker)]


def _strip_marker_spans(text: str, start_marker: str, end_marker: str) -> str:
    """Remove every complete marker-delimited span, keeping surrounding prose.

    Span-based rather than ``re.sub`` with a non-greedy pattern so that a call
    containing a literal close marker is removed whole instead of leaving its
    tail behind as visible content (#2507).
    """
    out: List[str] = []
    last = 0
    for span_start, span_end, _payload in _iter_marker_spans(
        text, start_marker, end_marker
    ):
        out.append(text[last:span_start])
        last = span_end
    out.append(text[last:])
    return "".join(out)


def _parse_xml_tool_calls(
    text: str, tools: Optional[List] = None
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Fallback parser for XML-based tool call formats.

    Handles models that use <tool_call>...</tool_call> XML format, including:
    - GLM format: <tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    - Qwen/Llama format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    - Generic JSON: <tool_call>{"name": ..., "arguments": ...}</tool_call>

    When ``tools`` is provided, parameter values are coerced to their
    declared schema types instead of best-effort JSON parsing.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    matches = _marker_payloads(text, "<tool_call>", "</tool_call>")

    for match in matches:
        content = match.strip()
        try:
            # Try JSON format first: {"name": "func", "arguments": {...}}
            parsed = json.loads(content, strict=False)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            _built = _build_tool_call(name, arguments)
            if _built is not None:
                tool_calls.append(_built)
            continue
        except (json.JSONDecodeError, AttributeError, *_DEEP_NEST_ERRORS):
            pass

        # Qwen/Llama format: <function=name><parameter=key>value</parameter></function>
        # Bounded by rfind rather than a non-greedy match: the payload is already
        # delimited, so the element closes at the LAST </function>, and a literal
        # copy inside a parameter value must not end it early (#2507).
        func_open = re.match(r"<function=(\w+)>", content)
        func_close = content.rfind(_XML_FUNCTION_CLOSE)
        if func_open and func_close >= func_open.end():
            func_name = func_open.group(1)
            params_text = content[func_open.end() : func_close]
            props = _tool_param_properties(func_name, tools)
            arguments = {}
            for key, val in _iter_xml_parameters(params_text):
                arguments[key] = _coerce_param_value(val, key, props, func_name)
            _built = _build_tool_call(func_name, arguments)
            if _built is not None:
                tool_calls.append(_built)
            continue

        # GLM XML format: func_name<arg_key>k</arg_key><arg_value>v</arg_value>...
        arg_keys = re.findall(r"<arg_key>(.*?)</arg_key>", content)
        arg_values = re.findall(r"<arg_value>(.*?)</arg_value>", content, re.DOTALL)
        if arg_keys:
            # Function name is the text before the first <arg_key>
            name_match = re.match(r"^(.*?)<arg_key>", content, re.DOTALL)
            func_name = (
                name_match.group(1).strip()
                if name_match
                else content.split("<")[0].strip()
            )
            props = _tool_param_properties(func_name, tools)
            arguments = {}
            for k, v in zip(arg_keys, arg_values):
                arguments[k] = _coerce_param_value(v, k, props, func_name)
            _built = _build_tool_call(func_name, arguments)
            if _built is not None:
                tool_calls.append(_built)

    if not tool_calls:
        return text, None

    # Remove tool call tags from text
    cleaned = _strip_marker_spans(text, "<tool_call>", "</tool_call>").strip()
    return cleaned, tool_calls


def _parse_namespaced_tool_calls(
    text: str, namespace: str, tools: Optional[List] = None
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse namespaced tool call tags like <minimax:tool_call>...</minimax:tool_call>.

    Handles the <invoke name="func"><parameter name="key">value</parameter></invoke>
    format used by MiniMax and similar models.

    When ``tools`` is provided, parameter values are coerced to their
    declared schema types instead of best-effort JSON parsing.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    tag_start = f"<{namespace}:tool_call>"
    tag_end = f"</{namespace}:tool_call>"
    pattern = re.escape(tag_start) + r"(.*?)" + re.escape(tag_end)
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        content = match.strip()
        # Parse <invoke name="func_name">...<parameter name="key">value</parameter>...</invoke>
        for invoke_match in re.finditer(
            r'<invoke\s+name="([^"]+)">(.*?)</invoke>', content, re.DOTALL
        ):
            func_name = invoke_match.group(1)
            params_text = invoke_match.group(2)
            props = _tool_param_properties(func_name, tools)
            arguments = {}
            for pm in re.finditer(
                r'<parameter\s+name="([^"]+)">(.*?)</parameter>', params_text, re.DOTALL
            ):
                key = pm.group(1)
                val = pm.group(2).strip()
                arguments[key] = _coerce_param_value(val, key, props, func_name)
            _built = _build_tool_call(func_name, arguments)
            if _built is not None:
                tool_calls.append(_built)

    if not tool_calls:
        return text, None

    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


def _parse_hermes_tool_calls(text: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Fallback parser for Hermes-style tool call formats.

    Handles outputs that use <|tool_call_start|>...<|tool_call_end|> markers
    with bracket-style content inside:
        <|tool_call_start|>[function_name(arg1=value1, arg2=value2)]<|tool_call_end|>

    Also handles JSON variant:
        <|tool_call_start|>{"name": "func", "arguments": {...}}<|tool_call_end|>

    Some clients/agents emit tool calls using this Hermes-style wire format.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    matches = _marker_payloads(text, "<|tool_call_start|>", "<|tool_call_end|>")

    for match in matches:
        content = match.strip()

        # Try JSON format first: {"name": "func", "arguments": {...}}
        try:
            parsed = json.loads(content)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            if name:
                _built = _build_tool_call(name, arguments)
                if _built is not None:
                    tool_calls.append(_built)
                continue
        except (json.JSONDecodeError, AttributeError, *_DEEP_NEST_ERRORS):
            pass

        # Hermes bracket format: [func_name(arg1=val1), other_tool(arg2=val2)]
        # The payload is Python-expression-like; use ast so commas inside quoted
        # strings or nested lists/dicts do not split calls incorrectly.
        try:
            parsed_expr = ast.parse(content, mode="eval").body
        except (SyntaxError, *_DEEP_NEST_ERRORS):
            parsed_expr = None

        calls = parsed_expr.elts if isinstance(parsed_expr, ast.List) else [parsed_expr]
        for call in calls:
            if not isinstance(call, ast.Call):
                continue

            if isinstance(call.func, ast.Name):
                func_name = call.func.id
            elif isinstance(call.func, ast.Attribute):
                try:
                    func_name = ast.unparse(call.func)
                except _DEEP_NEST_ERRORS:
                    continue
            else:
                continue

            arguments = {}
            unrepresentable = False
            for kw in call.keywords:
                if kw.arg is None:
                    continue
                try:
                    arguments[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError, *_DEEP_NEST_ERRORS):
                    # Fall back to the source text. `ast.unparse` walks the
                    # tree recursively, so an expression `ast.parse` built
                    # successfully can still breach the limit being rendered
                    # back out, from a deeper frame (#2545). An argument we
                    # cannot represent drops the whole call rather than
                    # yielding one that is missing it.
                    try:
                        arguments[kw.arg] = ast.unparse(kw.value)
                    except _DEEP_NEST_ERRORS:
                        unrepresentable = True
                        break

            if unrepresentable:
                logger.warning(
                    "Dropping tool call %.80r: argument %.40r could not be "
                    "represented (nested too deeply)",
                    func_name,
                    kw.arg,
                )
                continue

            _built = _build_tool_call(func_name, arguments)
            if _built is not None:
                tool_calls.append(_built)

    if not tool_calls:
        return text, None

    cleaned = _strip_marker_spans(
        text, "<|tool_call_start|>", "<|tool_call_end|>"
    ).strip()
    return cleaned, tool_calls


def _parse_bracket_tool_calls(text: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Fallback parser for bracket-style tool call formats.

    Recognizes both ``[Calling tool: name(args)]`` and ``[Tool call: name(args)]``
    prefixes, with or without arguments.  Models may emit the args-less form
    ``[Tool call: name]`` when mimicking conversation history.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    # Match with args first (higher fidelity)
    pattern_with_args = (
        r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)\(({.*?})\)\]"
    )
    matched_spans: list = []
    for match in re.finditer(pattern_with_args, text, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2)
        try:
            arguments = json.loads(args_str)
        except _DEEP_NEST_ERRORS as exc:
            logger.warning(
                "Dropping bracket tool call %.80r: arguments nested too deeply "
                "to decode (%s: %s)",
                name,
                type(exc).__name__,
                exc,
            )
            matched_spans.append(match.span())
            continue
        except (json.JSONDecodeError, ValueError):
            arguments = {"raw": args_str}
        _built = _build_tool_call(name, arguments)
        if _built is not None:
            tool_calls.append(_built)
        matched_spans.append(match.span())

    # Match without args (model-generated simplified form)
    pattern_no_args = r"\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)\]"
    for match in re.finditer(pattern_no_args, text):
        # Skip if this span overlaps with an already-matched with-args span
        start, end = match.span()
        if any(s <= start < e for s, e in matched_spans):
            continue
        name = match.group(1)
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments="{}",
                ),
            )
        )
        matched_spans.append((start, end))

    if not tool_calls:
        return text, None

    # Remove all matched spans from text
    cleaned = re.sub(pattern_with_args, "", text, flags=re.DOTALL)
    cleaned = re.sub(pattern_no_args, "", cleaned).strip()
    return cleaned, tool_calls


# ---------------------------------------------------------------------------
# Gemma 4 robust fallback parser
# ---------------------------------------------------------------------------

# Gemma 4's non-standard string delimiter (mlx_lm.tool_parsers.gemma4 uses
# the same literal in its regex).
_GEMMA4_STR_DELIM = '<|"|>'

# Bounds for parsing model-emitted arguments.  Model output is untrusted and
# attacker-influenceable (prompt injection can steer emissions verbatim), so
# parsing must stay linear-time and bounded: breaching a bound is a clean
# parse failure that flows into the existing drop-with-warning path, never an
# exception that escapes the parse chain.
_GEMMA4_MAX_ARGS_LEN = 262_144
_GEMMA4_MAX_DEPTH = 64


class _Gemma4ArgsTooComplexError(ValueError):
    """A defensive bound (length/depth) was breached parsing args.

    Distinct from an ordinary parse failure so the orchestrator can reject
    hard rather than retry with the legacy parser: the bounds are DoS guards
    against attacker-influenceable model output, and the legacy parser would
    happily parse oversized/deeply-nested input and defeat them.  Subclasses
    ValueError so the public parse chain still treats it as a clean drop.
    """

# A tool-call head: the name plus its opening ``{``.  Only the head is matched
# by regex; the argument span is found by _scan_gemma4_args_span, not by a
# recursive pattern (see that function).  The name segment captures namespaced
# MCP names (colon/dot/hyphen separated, e.g.
# call:google:mcp:text_generation:create-pdf-file, #1830).  The ``call:``
# opener is made optional and tolerant — ``(?:call)?:?`` — so the diffusion
# lane's degenerate prefixes (``calldone{`` missing the colon, ``:done{``
# missing ``call``, #1837) still match; the fallback only runs on
# marker-delimited content, so a permissive prefix cannot misfire on prose.
#
# Compiled with the ``regex`` module, NOT ``re``: once the ``call:`` literal
# anchor became optional (above), ``re``'s engine restarts the greedy
# ``[\w.-]+`` match at every position of a long bare argument value and
# backtracks O(n^2) hunting an opening ``{`` that never comes, hanging on
# adversarial output (a 300 KB bare value pegs a core indefinitely).  The
# ``regex`` engine fails that same partial match fast, so finditer stays
# linear.  See test_oversized_args_fail_cleanly.
_GEMMA4_CALL_HEAD = regex.compile(r"(?:call)?:?([\w.-]+(?::[\w.-]+)*)\{")

# The parenthesized variant of the head (#1846): under instruction-dense
# agentic load (large system prompt + many tool schemas + a big tool result)
# Gemma 4 26B reproducibly degrades from ``call:name{...}`` to a Python-kwargs
# shell ``call:name(key=value, ...)`` — same name grammar and same ``<|"|>``
# string quoting, only the OUTER ``{}`` becomes ``()`` and the top-level
# ``:`` separator becomes ``=``.  The nested grammar (objects, arrays, strings)
# is unchanged, so the hardened transcoder is reused for the inner content; only
# the outer shell needs a separate head + span scan.  Compiled with ``regex``,
# not ``re``, for the SAME reason as ``_GEMMA4_CALL_HEAD``: the optional/tolerant
# prefix makes ``re`` backtrack O(n^2) on a long bare value hunting an opening
# ``(`` that never comes (the #1854 ReDoS).  See test_oversized_paren_args.
_GEMMA4_CALL_HEAD_PAREN = regex.compile(r"(?:call)?:?([\w.-]+(?::[\w.-]+)*)\(")

# Matching close character for each balanced opener the parsers track.
_GEMMA4_CLOSE_CHAR = {"{": "}", "[": "]", "(": ")"}


def _squote_close_positions(s: str) -> list:
    """Indices of single quotes that can CLOSE a single-quoted value.

    A closing quote is one whose next non-whitespace character is ``,``,
    ``}``, ``]``, ``)`` or end of input.  ``)`` is an anchor for the
    parenthesized variant (#1846): a single-quoted value can be the last
    argument right before the call's closing paren, as in
    ``call:f(msg='hi')`` where the quote is followed immediately by ``)``.
    The curly form never closes a value with ``)`` (its values end at
    ``,``/``}``/``]``), so adding it does not change curly parsing.
    Anchoring closes this way (rather than taking the first quote) keeps
    apostrophes inside values from pairing across values: in
    ``{a: 'it's ok', b: 1}`` the quote in ``it's`` is followed by ``s`` so
    it cannot close the string.

    Computed in one reverse pass so each lookup is O(log n) via bisect; a
    forward scan that peeks past whitespace at every quote would be
    quadratic on whitespace-heavy input, and this text is model-emitted.
    """
    closes: list[int] = []
    next_sig = ""  # next non-whitespace char AFTER the current index
    for idx in range(len(s) - 1, -1, -1):
        ch = s[idx]
        if ch == "'" and (next_sig == "" or next_sig in ",}])"):
            closes.append(idx)
        if not ch.isspace():
            next_sig = ch
    closes.reverse()
    return closes


def _scan_gemma4_args_span(
    text: str, open_idx: int, squote_closes: list,
    open_ch: str = "{", close_ch: str = "}",
) -> int:
    """Return the end index (exclusive) of the balanced ``open_ch...close_ch``
    span starting at ``open_idx``, or -1 if no balanced span exists within
    bounds.  Defaults to braces (the canonical ``call:name{...}`` form); the
    parenthesized variant (#1846) passes ``(``/``)`` to find the call's outer
    span — nested ``{}``/``[]`` then pass through as ordinary characters for
    depth purposes (only the tracked pair is counted), while the string-skip
    logic below is character-agnostic so a ``)`` inside any string still
    cannot close the span.

    Iterative single-pass walk that counts brace depth only OUTSIDE string
    literals (``<|"|>``-paired strings, standard JSON double-quoted strings,
    and anchored single-quoted values), so a brace inside string content
    cannot truncate or unbalance the span.
    This deliberately replaces a recursive regex:
    - linear time: recursive alternation patterns degrade quadratically on
      unbalanced model output (measured ~590ms at 80KB), an injection-driven
      CPU burn on a server,
    - iterative: RecursionError is a RuntimeError subclass that no except
      tuple in the parse chain catches, so recursion on deeply nested model
      output would escape as a 500.

    A single-quoted string OPENS only at a value position (previous
    significant char is ``:``, ``,`` or ``[``); a bare apostrophe anywhere
    else (don't, it's) is ordinary content.
    """
    n = len(text)
    depth = 0
    last_sig = ""
    i = open_idx
    limit = min(n, open_idx + _GEMMA4_MAX_ARGS_LEN)
    while i < limit:
        if text.startswith(_GEMMA4_STR_DELIM, i):
            close = text.find(_GEMMA4_STR_DELIM, i + len(_GEMMA4_STR_DELIM))
            if close == -1:
                return -1  # unterminated string: malformed, give up cleanly
            i = close + len(_GEMMA4_STR_DELIM)
            last_sig = '"'
            continue
        ch = text[i]
        if ch == '"':
            # Standard JSON double-quoted string. The <|"|> delimiter is
            # matched by the startswith branch above before we reach here, so
            # a bare ``"`` is an ordinary JSON string open: skip to its closing
            # unescaped quote so a ``}`` inside the value cannot truncate the
            # span (#1854 — without this the suffix remap turned the corrupted
            # parse into an executable call with silently mangled arguments).
            # Honors ``\"`` and ``\\`` so an escaped quote never closes early.
            j = i + 1
            while j < limit:
                if text[j] == "\\":
                    j += 2  # escaped char is literal, never closes the string
                    continue
                if text[j] == '"':
                    break
                j += 1
            else:
                return -1  # unterminated string within bounds: drop cleanly
            i = j + 1
            last_sig = '"'
            continue
        if ch == "'" and last_sig in ":=,[":
            k = bisect.bisect_right(squote_closes, i)
            if k < len(squote_closes):
                i = squote_closes[k] + 1
                last_sig = "'"
                continue
            # No valid close ahead: treat the quote as ordinary content.
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        if not ch.isspace():
            last_sig = ch
        i += 1
    return -1


def _gemma4_args_to_json_robust(args_str: str) -> dict:
    """Convert Gemma 4 tool-call args to a Python dict.

    Tries the strict single-pass transcoder first
    (``_gemma4_transcode_to_json``); on failure, falls back to the legacy
    key-anchored recovery (``_gemma4_args_to_json_legacy``) for the one
    case the transcoder deliberately rejects: bare values that themselves
    contain commas, braces, or newlines — long markdown emitted into an
    ``answer:`` argument, observed live on the diffusion lane (#1837).
    The transcoder stops bare values at the first structural separator, so
    only key-anchored capture can recover that shape.
    """
    try:
        return _gemma4_transcode_to_json(args_str)
    except _Gemma4ArgsTooComplexError:
        # Defensive bound breached: reject hard.  The legacy parser ignores
        # these bounds and would parse the input anyway, defeating the DoS
        # guard, so it must NOT see oversized/deeply-nested args.
        raise
    except _DEEP_NEST_ERRORS as exc:
        # Nesting deep enough to break a decoder is a bound breach in all but
        # name, so it takes the reject-hard path above rather than the legacy
        # retry below (#2545).  Falling through would hand the very input the
        # bounds exist to stop to the parser that ignores them.
        raise _Gemma4ArgsTooComplexError(
            "Gemma 4 args nested too deeply to decode"
        ) from exc
    except (ValueError, json.JSONDecodeError):
        # The legacy path's NUL-placeholder forge vector is reintroduced
        # ONLY for ambiguous input the strict transcoder could not parse
        # (e.g. bare multi-comma markdown values, #1837); the common path
        # keeps the transcoder's no-placeholder, injection-safe guarantee.
        try:
            return _gemma4_args_to_json_legacy(args_str)
        except _DEEP_NEST_ERRORS as exc:
            # The legacy parser is deliberately unbounded, so it is the one
            # place a deep payload can still reach a raw decoder.
            raise _Gemma4ArgsTooComplexError(
                "Gemma 4 args nested too deeply to decode"
            ) from exc


def _gemma4_transcode_to_json(args_str: str) -> dict:
    """Transcode Gemma 4 tool-call args to a dict in a single pass.

    Handles what mlx-lm's parser cannot:
    - bare keys and values (``{location: Tokyo}``)
    - single-quoted values, including commas/colons/braces/apostrophes
      inside them (``{content: 'a, b: c'}``, #1830)
    - ``<|"|>``-delimited strings, arrays, and nested objects

    Implemented as a single-pass transcode to JSON text followed by one
    ``json.loads`` after local length/depth checks.  Every piece of captured
    string content is emitted through ``json.dumps`` and structural characters
    are emitted only by the state machine, so model output cannot inject JSON
    structure.  The legacy
    implementation substituted ``\\x00N\\x00`` placeholders, which literal
    NUL bytes in model output could forge, cross-contaminating argument
    values; transcoding directly leaves nothing to forge.

    Bare values stop at the first ``,``/``}``/``]`` by design: a bare value
    that embeds those characters is ambiguous here, and the caller
    (``_gemma4_args_to_json_robust``) recovers it via the legacy
    key-anchored fallback.
    """
    if len(args_str) > _GEMMA4_MAX_ARGS_LEN:
        raise _Gemma4ArgsTooComplexError("Gemma 4 args too large to parse")

    squote_closes = _squote_close_positions(args_str)
    n = len(args_str)
    out: list[str] = []  # JSON text fragments
    stack: list[str] = []  # open containers: "{" or "["
    expect = "object"  # object | key | value | delim
    i = 0

    def _skip_ws(i: int) -> int:
        while i < n and args_str[i].isspace():
            i += 1
        return i

    def _read_marked_string(i: int):
        """Read a <|"|>- or single-quoted string at i, or return None."""
        if args_str.startswith(_GEMMA4_STR_DELIM, i):
            close = args_str.find(
                _GEMMA4_STR_DELIM, i + len(_GEMMA4_STR_DELIM)
            )
            if close == -1:
                raise ValueError("unterminated Gemma 4 string")
            return (
                args_str[i + len(_GEMMA4_STR_DELIM): close],
                close + len(_GEMMA4_STR_DELIM),
            )
        if args_str[i] == "'":
            k = bisect.bisect_right(squote_closes, i)
            if k < len(squote_closes):
                close = squote_closes[k]
                return args_str[i + 1: close], close + 1
            # No anchored close ahead: not a string, treat as bare content.
        return None

    def _read_json_string(i: int):
        """Read a standard double-quoted JSON string token verbatim."""
        j = i + 1
        while j < n:
            if args_str[j] == "\\":
                j += 2
                continue
            if args_str[j] == '"':
                return args_str[i: j + 1], j + 1
            j += 1
        raise ValueError("unterminated double-quoted string")

    while True:
        i = _skip_ws(i)
        if expect == "object":
            # The outer container is ``{`` for the canonical form and ``(`` for
            # the parenthesized variant (#1846).  Either way it is an object;
            # we always emit ``{`` to JSON and remember the real opener on the
            # stack so its matching closer (``}`` or ``)``) is accepted below.
            if i >= n or args_str[i] not in "{(":
                raise ValueError("Gemma 4 args must start with '{' or '('")
            out.append("{")
            stack.append(args_str[i])
            i += 1
            expect = "key"
        elif expect == "key":
            if i >= n:
                raise ValueError("unterminated object")
            if args_str[i] == _GEMMA4_CLOSE_CHAR[stack[-1]]:
                # Empty object, or tolerated trailing comma.  Closer is ``}``
                # for a ``{`` opener and ``)`` for the paren variant's ``(``.
                if out and out[-1] == ", ":
                    out.pop()
                out.append("}")
                stack.pop()
                i += 1
                expect = "delim"
                continue
            if args_str[i] == '"':
                tok, i = _read_json_string(i)
                key = json.loads(tok)
            else:
                marked = _read_marked_string(i)
                if marked is not None:
                    key, i = marked
                else:
                    # Bare key: everything up to the separator.  ``=`` is a
                    # separator too for the parenthesized kwargs variant
                    # (#1846), so it bounds the key just like ``:``.
                    j = i
                    while j < n and args_str[j] not in ":=,{}[]'\"":
                        j += 1
                    key = args_str[i:j].strip()
                    if not key:
                        raise ValueError("malformed object key")
                    i = j
            i = _skip_ws(i)
            # Accept ``=`` as well as ``:`` — the parenthesized variant (#1846)
            # uses ``key=value``.  This is applied universally (not only at top
            # level), so it does widen the curly grammar to also accept ``=``
            # separators; that is a strict superset, so every valid ``:``-based
            # curly parse is unchanged and a previously-rejected ``{a = 1}`` now
            # succeeds rather than corrupting anything.  ``=`` inside a value is
            # untouched: strings are read atomically and bare values stop only
            # at ``,``/``}``/``]`` (plus ``)`` when a paren container is open).
            if i >= n or args_str[i] not in ":=":
                raise ValueError("expected ':' or '=' after object key")
            out.append(json.dumps(key))
            out.append(": ")
            i += 1
            expect = "value"
        elif expect == "value":
            if i >= n:
                raise ValueError("unterminated value")
            ch = args_str[i]
            if ch == "{" or ch == "[":
                # Depth bound, not recursion: a breach must surface as a
                # clean parse failure on the existing drop path, never as a
                # RecursionError (uncaught by the parse chain's excepts).
                if len(stack) >= _GEMMA4_MAX_DEPTH:
                    raise _Gemma4ArgsTooComplexError(
                        "Gemma 4 args nested too deeply"
                    )
                out.append(ch)
                stack.append(ch)
                i += 1
                expect = "key" if ch == "{" else "value"
                continue
            if ch == "]" and stack and stack[-1] == "[":
                # Empty array, or tolerated trailing comma.
                if out and out[-1] == ", ":
                    out.pop()
                out.append("]")
                stack.pop()
                i += 1
                expect = "delim"
                continue
            if ch == '"':
                tok, i = _read_json_string(i)
                out.append(tok)
                expect = "delim"
                continue
            marked = _read_marked_string(i)
            if marked is not None:
                content, i = marked
                out.append(json.dumps(content))
                expect = "delim"
                continue
            # Bare value: runs to the next structural separator.  When the
            # call uses the parenthesized outer shell (#1846), ``)`` also
            # terminates a bare value so the closing paren of the call is not
            # swallowed (``call:f(units=metric)`` — the value is ``metric``,
            # not ``metric)``).  For the curly form ``)`` stays ordinary
            # content so a value may legitimately contain parentheses
            # (``{expr: f(x)}``).  ``(`` is only ever the outermost container,
            # so its presence on the stack is the reliable signal.
            stops = ",)}]" if "(" in stack else ",}]"
            j = i
            while j < n and args_str[j] not in stops:
                j += 1
            value = args_str[i:j].strip()
            i = j
            if not value:
                raise ValueError("empty value")
            low = value.lower()
            if low in ("true", "false", "null"):
                out.append(low)  # normalize case (models emit True/False)
            else:
                try:
                    json.loads(value)  # already a valid scalar (number, ...)
                    out.append(value)
                except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
                    out.append(json.dumps(value))
            expect = "delim"
        else:  # expect == "delim"
            if not stack:
                if i < n:
                    raise ValueError("trailing data after args object")
                break
            if i >= n:
                raise ValueError("unterminated args")
            ch = args_str[i]
            if ch == ",":
                out.append(", ")
                i += 1
                # After a comma, an object (``{`` or the paren outer ``(``)
                # expects a key; an array expects a value.
                expect = "key" if stack[-1] in "{(" else "value"
            elif stack[-1] in "{(" and ch == _GEMMA4_CLOSE_CHAR[stack[-1]]:
                # Object close: ``}`` for ``{``, ``)`` for the paren outer.
                out.append("}")
                stack.pop()
                i += 1
            elif ch == "]" and stack[-1] == "[":
                out.append("]")
                stack.pop()
                i += 1
            else:
                raise ValueError("malformed args structure")

    result = json.loads("".join(out))
    if not isinstance(result, dict):
        raise ValueError("Gemma 4 args did not parse to an object")
    return result


def _gemma4_args_to_json_legacy(args_str: str) -> dict:
    """Legacy regex-based Gemma 4 args parser (upstream #1837).

    Kept as the last-resort fallback behind ``_gemma4_transcode_to_json``.
    Its value over the transcoder is step 6: key-anchored value capture for
    bare values that themselves contain commas, braces, or newlines (long
    markdown emitted into an ``answer:`` argument, observed live on the
    diffusion lane).  The transcoder stops bare values at the first
    separator, so this is the only path that recovers that shape.

    Carries the placeholder mechanism (``\\x00N\\x00``) the transcoder was
    written to avoid; it runs only on input the transcoder already rejected.
    """
    import regex

    # 1. Extract <|"|>-delimited strings and replace with placeholders
    strings: list[str] = []

    def _capture(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = regex.sub(r'<\|"\|>(.*?)<\|"\|>', _capture, args_str, flags=regex.DOTALL)

    # 2. Quote bare keys (allow whitespace after { or ,)
    text = regex.sub(r"(?<=[{,])\s*(\w+)\s*:", r' "\1":', text)

    # 3. Restore captured strings as properly escaped JSON strings
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s))

    # 4. Try json.loads — works when all values are already valid JSON primitives
    try:
        return json.loads(text)
    except (json.JSONDecodeError, *_DEEP_NEST_ERRORS):
        pass

    # 5. Quote bare string values that are not numbers, booleans, or null
    def _quote_bare(m):
        value = m.group(2).strip()
        suffix = m.group(3)
        if value.lower() in ("true", "false", "null"):
            return f": {value}{suffix}"
        try:
            json.loads(value)
            return f": {value}{suffix}"
        except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
            return f": {json.dumps(value)}{suffix}"

    # Keep the pre-step-5 text: if step 5 fails, its partial quoting has
    # corrupted multi-line bare values and step 6 must start clean.
    pre_quote_text = text
    text = regex.sub(
        r"(:\s*)([^\",\[\]{}\s][^,}]*?)(\s*[,}])", _quote_bare, text
    )
    try:
        return json.loads(text)
    except (json.JSONDecodeError, *_DEEP_NEST_ERRORS):
        pass

    # 6. Last resort: key-anchored value capture. Bare values that
    # themselves contain commas, braces, or newlines (e.g. long markdown
    # emitted into an ``answer:`` argument — observed live on the
    # diffusion lane) defeat the per-pair regex in step 5. Anchor on the
    # quoted keys produced by step 2 and treat everything between a
    # key's colon and the next key (or the end) as that key's value.
    # Operates on the pre-step-5 text so step 5's partial quoting cannot
    # corrupt the captured values.
    inner = pre_quote_text.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    key_pat = regex.compile(r'"([A-Za-z_]\w*)"\s*:')
    key_matches = list(key_pat.finditer(inner))
    if not key_matches:
        return json.loads(text)  # re-raise original-style error
    result: dict = {}
    for i, km in enumerate(key_matches):
        value_start = km.end()
        value_end = (
            key_matches[i + 1].start() if i + 1 < len(key_matches) else len(inner)
        )
        raw_value = inner[value_start:value_end].strip()
        if i + 1 < len(key_matches):
            raw_value = raw_value.rstrip().rstrip(",").rstrip()
        try:
            result[km.group(1)] = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError, *_DEEP_NEST_ERRORS):
            result[km.group(1)] = raw_value
    return result


def _parse_gemma4_tool_call_fallback(text: str) -> Union[dict, list]:
    """Robust fallback parser for Gemma 4 ``call:name{args}`` format.

    Activated only for Gemma 4 models (guarded by the ``tool_call_start``
    check at the call site).  Extends mlx-lm's parser to handle:
    - colons / dots / hyphens in function names (namespaced MCP tools,
      e.g. ``call:google:mcp:text_generation:create-pdf-file``, #1830)
    - bare string values without ``<|"|>`` delimiters
    - single-quoted values, including commas, colons, braces and
      apostrophes inside them (#1830)
    - the parenthesized kwargs variant ``call:name(key=value, ...)`` that
      Gemma 4 26B degrades to under instruction-dense agentic load (#1846).
      The nested grammar is identical to the curly form, so only the outer
      ``()`` shell and the ``=`` separator are new; both head forms are
      collected and processed in document order (see below).
    - degenerate ``call:`` prefixes from the diffusion lane's parallel
      denoising, which can drop a token from the opening (observed live:
      ``calldone{...}`` — missing colon — and ``:done{...}`` — missing
      ``call``, #1837).  ``_GEMMA4_CALL_HEAD`` matches these; the text is
      already marker-delimited (between ``<|tool_call>`` and
      ``<tool_call|>``), so the permissive prefix cannot misfire on prose.

    Name remapping onto registered tools is deliberately NOT done here:
    that is a post-parse concern handled by ``_remap_tool_call_names`` so
    it covers every producer path (native parser, this fallback, XML
    recovery, thinking-content promotion), not just this one.
    """
    squote_closes = _squote_close_positions(text)

    # Collect heads from both the canonical curly form and the parenthesized
    # variant (#1846), then process them in document order so the consumed-span
    # dedup below holds across BOTH forms: a curly head that falls inside an
    # already-consumed paren span (the inner ``{...}`` of ``call:f(a={...})``)
    # is string/structure content, not a sibling call, and must be skipped.
    heads = [(m.start(), m, "{", "}") for m in _GEMMA4_CALL_HEAD.finditer(text)]
    heads += [
        (m.start(), m, "(", ")")
        for m in _GEMMA4_CALL_HEAD_PAREN.finditer(text)
    ]
    heads.sort(key=lambda h: h[0])

    results = []
    consumed_until = 0
    for start, m, open_ch, close_ch in heads:
        # A head inside an already-consumed args span is string content
        # (e.g. quoted prose mentioning a tool call), not a sibling call.
        if start < consumed_until:
            continue
        open_idx = m.end() - 1
        end = _scan_gemma4_args_span(
            text, open_idx, squote_closes, open_ch, close_ch
        )
        if end == -1:
            continue
        args_str = text[open_idx:end]
        try:
            arguments = _gemma4_args_to_json_robust(args_str)
        except (ValueError, json.JSONDecodeError, RecursionError):
            continue  # one malformed call must not drop its siblings
        if not isinstance(arguments, dict):
            continue
        results.append({"name": m.group(1), "arguments": arguments})
        consumed_until = end

    if not results:
        raise ValueError("No function call found in Gemma 4 format")
    return results[0] if len(results) == 1 else results


def _remap_tool_call_names(
    tool_calls: List[ToolCall], tools: Optional[List]
) -> None:
    """Remap namespace-prefixed emitted tool names onto registered tools.

    Gemma 4 emits names like ``google:mcp:text_generation:create-pdf-file``
    for a tool registered as ``create-pdf-file`` (#1830); clients match by
    exact name, so the call would be unusable.  Runs post-parse so every
    producer path is covered and so the behavior survives changes to
    mlx-lm's native parser (which currently rejects colon names and routes
    these to the fallback, but may not forever).

    Rule: remap only when the emitted name matches no registered tool AND
    exactly one registered tool is a ``:``-boundary suffix of it; on zero
    or several candidates keep the name verbatim.  The comparison is
    boundary-aligned by construction (split on ':'), never str.endswith:
    a bare endswith would let a crafted emission like 'evilcreate-pdf-file'
    coerce into a registered 'create-pdf-file' (model output is
    attacker-influenceable via prompt injection).
    """
    if not tool_calls or not tools:
        return
    valid_names = _extract_tool_names(tools)
    if not valid_names:
        return
    for tc in tool_calls:
        name = tc.function.name if tc.function else ""
        if not name or name in valid_names or ":" not in name:
            continue
        parts = name.split(":")
        suffixes = {":".join(parts[i:]) for i in range(1, len(parts))}
        candidates = suffixes & valid_names
        if len(candidates) == 1:
            target = next(iter(candidates))
            logger.info(
                "Remapped namespaced tool call name %r to registered "
                "tool %r",
                name[:200],
                target,
            )
            tc.function.name = target


def parse_tool_calls(
    text: str,
    tokenizer: Any,
    tools: Optional[List] = None,
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse tool calls from model output.

    Uses mlx-lm's TokenizerWrapper tool parser if available, otherwise
    falls back to generic XML tool call parsing for models like GLM.

    Emitted names that match no registered tool are conservatively remapped
    onto registered tools afterwards (see _remap_tool_call_names); doing it
    here, at the single post-parse chokepoint, covers every producer path
    including the thinking-content promotion in
    extract_tool_calls_with_thinking, whose exact-name validity filter would
    otherwise silently drop cleanly-parsed namespaced calls (#1830).

    Args:
        text: Raw model output text
        tokenizer: mlx-lm's TokenizerWrapper (required)
        tools: Tool definitions for type conversion (optional)

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags and thinking tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
    cleaned_text, tool_calls = _parse_tool_calls_impl(text, tokenizer, tools)
    if tool_calls:
        _remap_tool_call_names(tool_calls, tools)
    return cleaned_text, tool_calls


def _parse_tool_calls_impl(
    text: str,
    tokenizer: Any,
    tools: Optional[List] = None,
) -> Tuple[str, Optional[List[ToolCall]]]:
    """parse_tool_calls body, pre-remap. See the public wrapper's docstring."""
    cleaned_text = text

    # Remove thinking tags if present (reasoning models)
    cleaned_text = re.sub(
        r"<think>.*?</think>", "", cleaned_text, flags=re.DOTALL
    ).strip()

    # Try mlx-lm's native tool parser first
    if getattr(tokenizer, "has_tool_calling", False):
        tool_call_start = tokenizer.tool_call_start
        tool_call_end = tokenizer.tool_call_end
        tool_parser = tokenizer.tool_parser

        if tool_call_start is not None and tool_parser is not None:
            tool_calls = []
            start_escaped = re.escape(tool_call_start)

            if tool_call_end:
                # Paired markers (e.g. <tool_call>...</tool_call>).  Span-based
                # rather than a non-greedy regex so an argument containing a
                # literal close marker does not truncate the payload (#2507).
                matches = _marker_payloads(text, tool_call_start, tool_call_end)
            else:
                # One-sided marker (e.g. Mistral/Devstral "[TOOL_CALLS]"):
                # split on the start marker and parse each segment.
                # The model emits: [TOOL_CALLS]name[ARGS]{...}[TOOL_CALLS]name2[ARGS]{...}
                parts = re.split(start_escaped, text)
                # First part is pre-marker text, rest are tool call segments
                matches = [p for p in parts[1:] if p.strip()]

            for match in matches:
                try:
                    parsed = tool_parser(match.strip(), tools)
                    # MiniMax M2 parser returns a list when a single
                    # <minimax:tool_call> block contains multiple <invoke>s.
                    items = parsed if isinstance(parsed, list) else [parsed]
                    for p in items:
                        name = p.get("name", "")
                        arguments = p.get("arguments", {})
                        _built = _build_tool_call(name, arguments)
                        if _built is not None:
                            tool_calls.append(_built)
                except (
                    ValueError,
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                    SyntaxError,
                    TypeError,
                    # The parser is third-party code that decodes internally
                    # (glm47, kimi_k2 and qwen3_coder all call json.loads), so
                    # deep nesting surfaces here rather than at a decode site
                    # in this module (#2545).
                    *_DEEP_NEST_ERRORS,
                ) as primary_err:
                    # Gemma 4 only: try robust fallback that handles bare
                    # string values and colons in function names.
                    gemma4_handled = False
                    if tool_call_start == "<|tool_call>":
                        try:
                            parsed = _parse_gemma4_tool_call_fallback(
                                match.strip()
                            )
                            items = (
                                parsed if isinstance(parsed, list) else [parsed]
                            )
                            for p in items:
                                name = p.get("name", "")
                                arguments = p.get("arguments", {})
                                _built = _build_tool_call(name, arguments)
                                if _built is not None:
                                    tool_calls.append(_built)
                            gemma4_handled = True
                        except (
                            ValueError,
                            json.JSONDecodeError,
                            KeyError,
                            SyntaxError,
                            TypeError,
                        ):
                            pass

                    if gemma4_handled:
                        continue

                    # Per-match XML fallback: regex-only, no ast.literal_eval,
                    # recovers Qwen/GLM/Hermes-JSON formats. Prevents silent
                    # drop when the native parser raises (e.g. ast.literal_eval
                    # SyntaxError on non-Python-literal parameter values).
                    fb_wrapped = f"<tool_call>{match}</tool_call>"
                    _, fb_calls = _parse_xml_tool_calls(fb_wrapped, tools)
                    if fb_calls:
                        tool_calls.extend(fb_calls)
                        logger.warning(
                            "Native tool parser failed (%s: %s), "
                            "recovered via XML fallback. Match: %r",
                            type(primary_err).__name__,
                            primary_err,
                            match[:200],
                        )
                    else:
                        logger.warning(
                            "Native tool parser failed (%s: %s) and XML "
                            "fallback could not recover. Dropping match: %r",
                            type(primary_err).__name__,
                            primary_err,
                            match[:200],
                        )
                    continue

            if tool_calls:
                if tool_call_end:
                    cleaned_text = _strip_marker_spans(
                        cleaned_text, tool_call_start, tool_call_end
                    ).strip()
                else:
                    # One-sided: everything from first marker to end is tool calls
                    idx = cleaned_text.find(tool_call_start)
                    if idx >= 0:
                        cleaned_text = cleaned_text[:idx].strip()
                return cleaned_text, tool_calls

    # Fallback: parse XML <tool_call> tags (GLM, Qwen, generic formats)
    if "<tool_call>" in cleaned_text:
        return _parse_xml_tool_calls(cleaned_text, tools)

    # Fallback: namespaced tool_call tags (e.g. <minimax:tool_call>)
    ns_match = re.search(r"<([A-Za-z_][\w.-]*):tool_call>", cleaned_text)
    if ns_match:
        ns = ns_match.group(1)
        return _parse_namespaced_tool_calls(cleaned_text, ns, tools)

    # Fallback: Hermes-style tool calls (<|tool_call_start|>[func(args)]<|tool_call_end|>)
    if "<|tool_call_start|>" in cleaned_text:
        hermes_result = _parse_hermes_tool_calls(cleaned_text)
        if hermes_result[1] is not None:
            return hermes_result

    # Fallback: bracket tool call formats (from text-formatted history)
    if "[Calling tool:" in cleaned_text or "[Tool call:" in cleaned_text:
        return _parse_bracket_tool_calls(cleaned_text)

    # All parsing attempts exhausted. Strip known tool-call markers so raw
    # control markup never leaks into the API response.  Models whose markers
    # overlap with the generic ``<tool_call>`` tag already returned above via
    # Branch 2 (_parse_xml_tool_calls), so this only affects models with
    # unique markers (Gemma 4, Mistral, Pythonic, Kimi K2, Longcat, etc.).
    if getattr(tokenizer, "has_tool_calling", False):
        _start = getattr(tokenizer, "tool_call_start", None)
        _end = getattr(tokenizer, "tool_call_end", None)
        if _start and _end:
            stripped = _marker_payloads(cleaned_text, _start, _end)
            if stripped:
                logger.warning(
                    "Tool call markers found but parsing failed, "
                    "stripping markers. Raw content: %s",
                    stripped,
                )
            cleaned_text = _strip_marker_spans(cleaned_text, _start, _end).strip()
        elif _start:
            idx = cleaned_text.find(_start)
            if idx >= 0:
                logger.warning(
                    "Tool call start marker found but parsing failed, "
                    "stripping marker. Raw content: %s",
                    cleaned_text[idx:],
                )
                cleaned_text = cleaned_text[:idx].strip()

    # Strip Hermes markers if still present (models without has_tool_calling)
    if "<|tool_call_start|>" in cleaned_text:
        cleaned_text = _strip_marker_spans(
            cleaned_text, "<|tool_call_start|>", "<|tool_call_end|>"
        ).strip()

    return cleaned_text, None


def sanitize_tool_call_markup(text: str, tokenizer: Any) -> str:
    """Remove tool-call control markup while preserving surrounding prose."""
    if not text:
        return ""

    # Every caller sanitizes thinking-channel text; keep it byte-identical
    # with the streamed reasoning deltas, which do not consume DeepSeek
    # V4's separator either.
    stream_filter = ToolCallStreamFilter(tokenizer, consume_dsml_separator=False)
    cleaned = stream_filter.feed(text)
    cleaned += stream_filter.finish()
    return cleaned.strip()


def _extract_tool_names(tools: List) -> set:
    """Extract function names from OpenAI-format tool definitions."""
    names = set()
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function", {})
            if isinstance(func, dict):
                name = func.get("name")
                if name:
                    names.add(name)
    return names


def extract_tool_calls_with_thinking(
    thinking_content: str,
    regular_content: str,
    tokenizer: Any,
    tools: Optional[List] = None,
) -> ToolCallExtraction:
    """Extract tool calls while keeping a sanitized reasoning transcript.

    When tool calls are found in thinking content (not regular content),
    the ``tools`` parameter controls validation:

    * ``None`` (default) — no tools list was provided.  Thinking-embedded
      calls are kept only when ``regular_content`` is empty (the model
      produced no competing prose).  Otherwise they are dropped as
      potential hallucinated reasoning.
    * ``[]`` — "no tools allowed".  All thinking-embedded calls are
      dropped regardless of ``regular_content``.
    * Non-empty list — name matching is the sole discriminator.
      Calls whose name matches a provided tool are promoted regardless
      of whether regular text was also produced.
    """
    cleaned_text, tool_calls = parse_tool_calls(regular_content, tokenizer, tools)
    cleaned_thinking = sanitize_tool_call_markup(thinking_content, tokenizer)
    tool_calls_from_thinking = False

    if not tool_calls and thinking_content:
        _, tool_calls = parse_tool_calls(thinking_content, tokenizer, tools)
        tool_calls_from_thinking = bool(tool_calls)

        # Guard: validate thinking-embedded tool calls.
        #
        # Three cases:
        # 1. tools is None (not provided) AND regular text exists → drop.
        #    The call is unvalidated and could be hallucinated reasoning.
        # 2. tools is None AND no regular text → keep.  The model clearly
        #    intended a tool invocation (no competing prose).
        # 3. tools is a list (including empty) → name matching is the sole
        #    discriminator.  An empty list means "no tools allowed" so all
        #    calls are dropped.  A non-empty list filters by name, regardless
        #    of whether regular text was also produced.  The previous "regular
        #    text means just reasoning" heuristic was wrong for models
        #    (Qwen3-Coder) that genuinely place tool calls in thinking.
        # See https://github.com/jundot/omlx/issues/1392
        if tool_calls:
            if tools is None:
                if regular_content.strip():
                    tool_calls = None
                    tool_calls_from_thinking = False
            else:
                valid_names = _extract_tool_names(tools)
                tool_calls = [tc for tc in tool_calls if tc.function.name in valid_names]
                if not tool_calls:
                    tool_calls = None
                    tool_calls_from_thinking = False

    return ToolCallExtraction(
        cleaned_text=cleaned_text,
        tool_calls=tool_calls,
        cleaned_thinking=cleaned_thinking,
        tool_calls_from_thinking=tool_calls_from_thinking,
    )


def parse_tool_calls_with_thinking_fallback(
    thinking_content: str,
    regular_content: str,
    tokenizer: Any,
    tools: Optional[List] = None,
) -> Tuple[str, Optional[List[ToolCall]]]:
    """Parse tool calls from content, falling back to thinking if none found.

    Small reasoning models sometimes generate tool call XML inside <think>
    blocks instead of after </think>. This function first tries the normal
    content, then falls back to parsing from thinking content.

    Args:
        thinking_content: Text extracted from <think>...</think> blocks.
        regular_content: Text outside thinking blocks.
        tokenizer: mlx-lm's TokenizerWrapper.
        tools: Tool definitions for type conversion (optional).

    Returns:
        Tuple of (cleaned_text, tool_calls or None).
        cleaned_text comes from regular_content only (thinking text is
        never promoted to content).
    """
    result = extract_tool_calls_with_thinking(
        thinking_content,
        regular_content,
        tokenizer,
        tools,
    )
    return result.cleaned_text, result.tool_calls


class ToolCallStreamFilter:
    """Streaming filter that suppresses tool-call markup from content deltas.

    Detects known tool-call start envelopes during streaming and suppresses
    control markup from assistant-visible content. Supports tokenizer-defined
    delimiters, namespaced XML envelopes, and high-confidence bracket-format
    envelopes handled by ``parse_tool_calls``.

    Suppression is envelope-bounded: control markup is removed, then visible
    prose after a closed envelope continues streaming normally.

    Args:
        tokenizer: The model's tokenizer. Uses tokenizer-defined
            ``tool_call_start`` when available.
        consume_dsml_separator: Treat DeepSeek V4's ``"\n\n"`` separator as
            part of the DSML tool-call envelope, matching the reference
            decoder's stop token. Disable for filters watching a channel
            that never contains a separator-prefixed tool-call block (the
            thinking channel), where holding trailing newlines would flush
            them only after the channel closed.
    """

    def __init__(self, tokenizer: Any, *, consume_dsml_separator: bool = True):
        marker = getattr(tokenizer, "tool_call_start", None)
        marker_end = getattr(tokenizer, "tool_call_end", None)
        # Normalize None-like values but preserve empty strings.
        if marker is None:
            marker = ""
        if marker_end is None:
            marker_end = ""
        self._marker_pairs: List[Tuple[str, str]] = [
            ("]<]minimax[>[<tool_call>", "]<]minimax[>[</tool_call>"),
            ("<|tool_call_start|>", "<|tool_call_end|>"),
            ("<tool_call>", "</tool_call>"),
        ]
        self._suppress_after_markers: List[str] = []
        if marker:
            if marker_end:
                self._marker_pairs.insert(0, (marker, marker_end))
            else:
                # One-sided markers (e.g. Mistral "[TOOL_CALLS]" with no
                # end marker): suppress everything after the start marker.
                self._suppress_after_markers.append(marker)
        # DeepSeek V4's reference decoder consumes "\n\n<｜DSML｜tool_calls"
        # as one literal stop token, so the separator belongs to the envelope,
        # not to content. Register the separator-inclusive variant so the
        # earliest-match rule consumes it when present; the bare pair stays as
        # fallback for emissions that omit the separator.
        is_dsml_tool_marker = (
            marker == "<｜DSML｜tool_calls>" and marker_end == "</｜DSML｜tool_calls>"
        )
        if is_dsml_tool_marker and consume_dsml_separator:
            self._marker_pairs.insert(0, ("\n\n" + marker, marker_end))
        # Gemma 4 can emit a bare close token outside a matched tool-call
        # envelope. Do not apply this to XML-style closers like </tool_call>,
        # which may appear as literal prose.
        is_gemma4_tool_marker = (
            marker == "<|tool_call>" and marker_end == "<tool_call|>"
        )
        self._stray_close_markers: List[str] = (
            [marker_end] if is_gemma4_tool_marker else []
        )
        self._orphan_close_markers: List[str] = ["<|tool_call_end|>"]
        if marker_end and not self._is_xml_close_marker(marker_end):
            self._orphan_close_markers.append(marker_end)
        self._orphan_close_markers = list(dict.fromkeys(self._orphan_close_markers))
        self._namespaced_open_re = re.compile(r"<([A-Za-z_][\w.-]*):tool_call>")
        self._bracket_prefixes = ["[Calling tool:", "[Tool call:"]
        self._bracket_call_re = re.compile(
            r"^\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]",
            re.DOTALL,
        )
        self._buffer = ""
        self._suppressing_until: Optional[str] = None
        self._suppressing = False
        self._pending_envelope_parts: List[str] = []
        self._pending_start_marker: Optional[str] = None
        self._recovery_candidate = ""
        self._reset_json_scan()

    @staticmethod
    def _is_xml_close_marker(marker: str) -> bool:
        return marker.startswith("</") and marker.endswith(">")

    @property
    def active(self) -> bool:
        """Whether this filter should run for tool-enabled streams."""
        return True

    def take_recovery_candidate(self) -> str:
        """Return and clear an unterminated paired envelope captured at EOF.

        The caller must only surface this text after final tool parsing confirms
        that no structured tool call was recovered. This keeps valid tool calls
        hidden even when another parser can recover malformed outer markup.
        """
        candidate = self._recovery_candidate
        self._recovery_candidate = ""
        return candidate

    def _clear_pending_envelope(self) -> None:
        self._pending_envelope_parts = []
        self._pending_start_marker = None
        self._reset_json_scan()

    # -- Incremental JSON scan over a suppressed payload (#2507) ------------
    #
    # A close marker inside a JSON string argument must not end the envelope,
    # or the rest of the tool call is emitted as visible content mid-stream.
    # Deciding that needs to know where the JSON value really ends, but
    # re-decoding the accumulated payload on every chunk would be quadratic on
    # long tool calls. Model output is untrusted and attacker-influenceable, so
    # this module's rule is that parsing stays linear (see the bounds above
    # _GEMMA4_MAX_ARGS_LEN). This scanner therefore examines each character
    # exactly once, carrying bracket/string state across chunks. It covers
    # object payloads and ``[{...}]`` array payloads, matching the shapes the
    # non-streaming parser accepts from ``_json_value_end``.

    def _reset_json_scan(self) -> None:
        self._json_state = "undecided"
        self._json_depth = 0
        self._json_in_string = False
        self._json_escaped = False
        self._json_scan_off = 0
        self._json_complete_off = 0
        # Tail of already-consumed payload, kept so a `</function>` straddling
        # the pending/buffer boundary is still visible to the xml check.
        self._xml_prev_tail = ""
        # First payload characters, accumulated across buffer drains purely to
        # decide which dialect the payload is.
        self._payload_head = ""

    def _shift_json_scan(self, dropped: int, moved: str = "") -> None:
        """Rebase scan offsets after ``dropped`` chars leave the buffer front."""
        self._json_scan_off = max(0, self._json_scan_off - dropped)
        self._json_complete_off = max(0, self._json_complete_off - dropped)
        if moved:
            self._xml_prev_tail = (self._xml_prev_tail + moved)[-_XML_TAIL_KEEP:]

    def _advance_json_scan(self, buffer: str) -> None:
        """Consume buffer chars not yet seen, updating JSON-completion state."""
        i = self._json_scan_off
        n = len(buffer)

        if self._json_state == "undecided":
            # `{` settles JSON from a single character, but `<function=` needs
            # ten, and the buffer is drained into the pending envelope between
            # chunks, so those ten never coexist in it. Accumulate a small
            # persistent head to decide the dialect across drains.
            head_chunk: Optional[str] = None
            if self._payload_head:
                head_chunk = buffer[i:n]
            else:
                while i < n and buffer[i] in " \t\r\n":
                    i += 1
                if i >= n:
                    self._json_scan_off = i
                    return
                if buffer[i] == "{":
                    self._json_state = "scanning"
                elif buffer[i] == "[":
                    # A leading '[' is either a JSON array of calls ("[{")
                    # or the Hermes bracket dialect ([execute_code(...)]),
                    # which never parses as JSON and must not be treated as
                    # an unfinished value or the envelope is suppressed
                    # forever. The next non-space character settles it.
                    self._json_state = "array_head"
                    self._json_depth = 1
                    i += 1
                else:
                    head_chunk = buffer[i:n]

            if head_chunk is not None:
                self._payload_head = (self._payload_head + head_chunk)[
                    : len(_XML_FUNCTION_OPEN)
                ]
                self._json_scan_off = n
                head = self._payload_head
                if len(head) < len(_XML_FUNCTION_OPEN):
                    if _XML_FUNCTION_OPEN.startswith(head):
                        return  # still ambiguous, wait for more
                    self._json_state = "not_json"
                else:
                    self._json_state = (
                        "xml" if head == _XML_FUNCTION_OPEN else "not_json"
                    )

        if self._json_state == "array_head":
            while i < n and buffer[i] in " \t\r\n":
                i += 1
            if i >= n:
                self._json_scan_off = i
                return
            # The scan loop consumes the '{' itself, raising the depth above
            # the pending '[' so the array completes on its closing ']'.
            self._json_state = "scanning" if buffer[i] == "{" else "not_json"

        if self._json_state != "scanning":
            self._json_scan_off = n
            return

        while i < n:
            ch = buffer[i]
            if self._json_in_string:
                if self._json_escaped:
                    self._json_escaped = False
                elif ch == "\\":
                    self._json_escaped = True
                elif ch == '"':
                    self._json_in_string = False
            elif ch == '"':
                self._json_in_string = True
            elif ch in "{[":
                self._json_depth += 1
            elif ch in "}]":
                self._json_depth -= 1
                if self._json_depth == 0:
                    self._json_state = "complete"
                    self._json_complete_off = i + 1
                    i += 1
                    break
            i += 1
        self._json_scan_off = i

    def _find_suppression_end(self, buffer: str) -> int:
        """Index in ``buffer`` of the close marker that really ends the envelope.

        Streaming counterpart of ``_find_marker_span_end`` (#2507): a close
        marker sitting inside a JSON string argument must not end the
        envelope, or the rest of the tool call is emitted as visible content
        mid-stream.

        Decodes the accumulated payload at most ONCE per call rather than
        testing each candidate close marker.  Model output is untrusted, and a
        per-candidate loop would re-decode the payload for every embedded
        marker, which is quadratic on output that repeats the marker (the
        superlinear-work-on-model-output trap from #1854/#1905).  One decode
        settles where the JSON value ends; the real close marker is simply the
        first one at or after that point.

        Returns -1 to mean "no close marker yet", which makes the caller wait
        for more input.  An envelope whose JSON never completes stays
        suppressed and is handled at EOF by the existing unterminated-envelope
        recovery, rather than leaking its tail as content.
        """
        marker = self._suppressing_until
        if not marker:
            return -1

        self._advance_json_scan(buffer)

        if self._json_state == "xml":
            # qwen3_coder dialect: the envelope ends with </function> right
            # before the close marker, so a marker inside a parameter value is
            # not preceded by one. This is a local O(1) test per candidate,
            # unlike the non-streaming path which can also balance parameter
            # elements because it has the whole payload at once.
            search = 0
            while True:
                idx = buffer.find(marker, search)
                if idx < 0:
                    return -1
                # Only the characters immediately before the marker matter, so
                # look at a fixed window instead of slicing the whole buffer:
                # a per-candidate full slice would be quadratic when one chunk
                # carries many markers.
                window_start = idx - _XML_TAIL_KEEP
                if window_start <= 0:
                    seen = self._xml_prev_tail + buffer[:idx]
                else:
                    seen = buffer[window_start:idx]
                if seen.rstrip(" \t\r\n").endswith(_XML_FUNCTION_CLOSE):
                    return idx
                search = idx + len(marker)

        if self._json_state == "not_json":
            # Hermes brackets, GLM arg_key/arg_value, prose: historical
            # first-match behaviour, unchanged.
            return buffer.find(marker)
        if self._json_state != "complete":
            # Still undecided or mid-object: any close marker visible now sits
            # inside the payload, so wait for more input instead of closing
            # early. An envelope whose JSON never completes stays suppressed
            # and is handled at EOF by the unterminated-envelope recovery.
            return -1

        # The JSON object is closed, so its end is authoritative and markers
        # embedded inside it are ignored.
        return buffer.find(marker, self._json_complete_off)

    def _find_start_envelope(
        self,
        text: str,
        start: int = 0,
        cache: Optional[Dict[Any, Any]] = None,
    ) -> Optional[Tuple[int, int, Optional[str]]]:
        """Find earliest complete opening envelope at or after ``start``.

        Returns:
            tuple(index, consume_len, close_marker_or_none)
            - close_marker_or_none is a close marker to wait for, or ``None``
              when the whole envelope is already contained in consume_len.

        ``cache`` is used by the EOF unwind, whose ``start`` only moves
        forward over one fixed string: a cached hit at or after ``start`` is
        reused, a cached miss is permanent, and a hit that ``start`` has
        passed is recomputed from ``start``. That keeps repeated calls linear
        overall instead of rescanning the tail once per envelope.
        """
        miss = object()

        def lookup(key: Any, compute: Any) -> Optional[Tuple[int, int, Optional[str]]]:
            if cache is None:
                return compute()
            hit = cache.get(key, miss)
            if hit is miss or (hit is not None and hit[0] < start):
                hit = compute()
                cache[key] = hit
            return hit

        starts: List[Tuple[int, int, Optional[str]]] = []

        for marker, close in self._marker_pairs:

            def compute_pair(
                marker: str = marker, close: str = close
            ) -> Optional[Tuple[int, int, Optional[str]]]:
                idx = text.find(marker, start)
                return None if idx < 0 else (idx, len(marker), close)

            hit = lookup(("pair", marker), compute_pair)
            if hit is not None:
                starts.append(hit)

        for close in self._orphan_close_markers:

            def compute_orphan(
                close: str = close,
            ) -> Optional[Tuple[int, int, Optional[str]]]:
                close_idx = text.find(close, start)
                return None if close_idx < 0 else (close_idx, len(close), None)

            hit = lookup(("orphan", close), compute_orphan)
            if hit is not None:
                starts.append(hit)

        def compute_ns() -> Optional[Tuple[int, int, Optional[str]]]:
            ns_match = self._namespaced_open_re.search(text, start)
            if not ns_match:
                return None
            ns = ns_match.group(1)
            return (
                ns_match.start(),
                len(ns_match.group(0)),
                f"</{ns}:tool_call>",
            )

        hit = lookup("ns", compute_ns)
        if hit is not None:
            starts.append(hit)

        def compute_bracket() -> Optional[Tuple[int, int, Optional[str]]]:
            best: Optional[Tuple[int, int, Optional[str]]] = None
            for bp in self._bracket_prefixes:
                bracket_idx = text.find(bp, start)
                while bracket_idx >= 0:
                    if best is not None and bracket_idx >= best[0]:
                        break
                    bracket_candidate = text[bracket_idx:]
                    bracket_match = self._bracket_call_re.match(bracket_candidate)
                    if bracket_match:
                        best = (bracket_idx, bracket_match.end(), None)
                        break
                    bracket_idx = text.find(bp, bracket_idx + 1)
            return best

        hit = lookup("bracket", compute_bracket)
        if hit is not None:
            starts.append(hit)

        # One-sided markers: suppress from start marker to end of buffer.
        for sa_marker in self._suppress_after_markers:

            def compute_sa(
                sa_marker: str = sa_marker,
            ) -> Optional[Tuple[int, int, Optional[str]]]:
                idx = text.find(sa_marker, start)
                if idx < 0:
                    return None
                return (idx, len(text) - idx, "__suppress_permanently__")

            hit = lookup(("sa", sa_marker), compute_sa)
            if hit is not None:
                starts.append(hit)

        if not starts:
            return None
        return min(starts, key=lambda x: x[0])

    @staticmethod
    def _partial_prefix_len(text: str, marker: str) -> int:
        """Longest suffix of text that is a proper prefix of marker."""
        max_len = min(len(text), len(marker) - 1)
        for n in range(max_len, 0, -1):
            if text.endswith(marker[:n]):
                return n
        return 0

    @staticmethod
    def _could_be_partial_namespaced_open(candidate: str) -> bool:
        """Return True if candidate could prefix a namespaced <ns:tool_call> tag."""
        if not candidate.startswith("<"):
            return False
        if ">" in candidate:
            return False

        body = candidate[1:]
        if not body:
            return True
        if body.startswith("/"):
            return False

        if ":" not in body:
            return re.match(r"^[A-Za-z_][\w.-]*$", body) is not None

        ns, suffix = body.split(":", 1)
        if not re.match(r"^[A-Za-z_][\w.-]*$", ns):
            return False
        return "tool_call".startswith(suffix)

    def _partial_suffix_len(self, text: str) -> int:
        """Length of trailing suffix that might be an opening-marker prefix."""
        keep = 0
        for marker, _close in self._marker_pairs:
            keep = max(keep, self._partial_prefix_len(text, marker))

        last_lt = text.rfind("<")
        if last_lt >= 0:
            candidate = text[last_lt:]
            if self._could_be_partial_namespaced_open(candidate):
                keep = max(keep, len(candidate))

        # Partial prefix detection for bracket markers (e.g. "[", "[C",
        # "[Cal" could be start of "[Calling tool:" or "[Tool call:").
        for bp in self._bracket_prefixes:
            keep = max(keep, self._partial_prefix_len(text, bp))
        # Same for suppress-after markers (e.g. "[TOOL" for "[TOOL_CALLS]").
        for sa_marker in self._suppress_after_markers:
            keep = max(keep, self._partial_prefix_len(text, sa_marker))
        # Hold partial prefix of a stray-close marker so it reassembles before
        # the strip check — prevents the "hello<tool_call|" + ">" split leak.
        for close_marker in self._orphan_close_markers:
            keep = max(keep, self._partial_prefix_len(text, close_marker))

        bracket_idx = -1
        for bp in self._bracket_prefixes:
            idx = text.rfind(bp)
            if idx > bracket_idx:
                bracket_idx = idx
        if bracket_idx >= 0:
            bracket_candidate = text[bracket_idx:]
            # Hold unresolved bracket prefix until we can classify parseable
            # envelope vs literal prose.
            if "]" not in bracket_candidate:
                keep = max(keep, len(bracket_candidate))
                # Do not cap unresolved bracket candidates: capping can leak
                # raw control markup once the prefix grows past the cap.
                return keep

        # Cap retained suffix window to avoid unbounded buffering on malformed text.
        return min(keep, 128)

    def _should_drop_tail_at_finish(self, tail: str) -> bool:
        """Whether unresolved tail should be suppressed under strict mode."""
        if not tail:
            return False

        for marker, _close in self._marker_pairs:
            if marker.startswith(tail):
                # MiniMax M3 markers start with ``]``. A single closing
                # bracket at end-of-stream is much more likely to be literal
                # prose than an incomplete MiniMax control marker.
                if tail == "]":
                    continue
                # Newline-only tails are held back only because of the
                # separator-inclusive DeepSeek V4 marker; at end-of-stream
                # with no envelope following they are literal content.
                if not tail.strip("\n"):
                    continue
                return True

        for close_marker in self._orphan_close_markers:
            if close_marker.startswith(tail):
                return True

        # Drop unresolved bracket tool-call prefixes
        for bp in self._bracket_prefixes:
            if tail.startswith(bp):
                return True

        # Drop unresolved suppress-after marker prefixes
        for sa_marker in self._suppress_after_markers:
            if sa_marker.startswith(tail) or tail.startswith(sa_marker):
                return True

        if not tail.startswith("<"):
            return False
        if ">" in tail:
            return False

        body = tail[1:]
        if not body:
            return True
        if body.startswith("/"):
            return False

        if ":" not in body:
            # Preserve plain literal tails like "<alpha".
            return False

        ns, suffix = body.split(":", 1)
        if not re.match(r"^[A-Za-z_][\w.-]*$", ns):
            return False
        return "tool_call".startswith(suffix)

    def _sanitize_prefix_before_suppression(self, text: str) -> str:
        """Strip unresolved bracket-control prefixes while preserving prose."""
        if not any(bp in text for bp in self._bracket_prefixes):
            return text

        out: List[str] = []
        cursor = 0
        while cursor < len(text):
            bracket_idx = -1
            bracket_prefix = ""
            for bp in self._bracket_prefixes:
                idx = text.find(bp, cursor)
                if idx >= 0 and (bracket_idx < 0 or idx < bracket_idx):
                    bracket_idx = idx
                    bracket_prefix = bp
            if bracket_idx < 0:
                out.append(text[cursor:])
                break

            out.append(text[cursor:bracket_idx])
            after_prefix = bracket_idx + len(bracket_prefix)
            close_idx = text.find("]", after_prefix)
            if close_idx < 0:
                # Drop only the marker token; keep following prose.
                cursor = after_prefix
                continue

            # Preserve balanced literal bracket text that is not being suppressed.
            out.append(text[bracket_idx : close_idx + 1])
            cursor = close_idx + 1

        return "".join(out)

    def _unwind_withheld_at_eof(
        self, candidate: str, marker: str, start_marker: str
    ) -> str:
        """Single pass over text withheld at end of stream, first-marker split.

        ``candidate`` starts with the opening marker of an envelope whose
        payload scan never confirmed a structural end.  For such malformed
        payloads there is no reliable way to tell an embedded close marker
        from the real one, so match the historical (pre-#2507) behaviour:
        the FIRST close marker ends the envelope, which preserves prose the
        model resumed afterwards, including any later literal marker text.

        Runs as one forward scan with monotone offsets rather than re-feeding
        tails through ``feed``: a re-feed loop re-copies the remaining text
        once per malformed envelope, which is quadratic when output repeats
        the marker (the superlinear-work-on-model-output trap from
        #1854/#1905).

        Trailing prose after the last envelope is staged in ``self._buffer``
        so the caller's end-of-stream tail rules still apply to it.
        """
        out: List[str] = []
        cache: Dict[Any, Any] = {}
        # The close search must not match inside the opening marker itself.
        pos = min(len(start_marker), len(candidate))
        env_start = 0
        while True:
            if marker == "__suppress_permanently__":
                self._suppressing = True
                break
            # Same span primitive as the non-streaming parser: a valid JSON or
            # XML payload ends at its structural boundary (so an embedded
            # close marker cannot cut it, #2507), and a malformed payload
            # falls back to the historical first close marker.  Keeping the
            # two paths identical means the content shown at EOF always
            # matches what the final parse extracts.
            found = _find_marker_span_end(candidate, pos, marker)
            if found is None:
                withheld = candidate[env_start:]
                if withheld:
                    self._recovery_candidate = withheld
                    logger.warning(
                        "Unclosed tool-call envelope at end of stream; "
                        "withheld %d characters are available for content "
                        "recovery (start_marker=%.80r)",
                        len(withheld),
                        candidate[env_start : env_start + 80],
                    )
                break
            pos = found[1]
            # Between envelopes now: emit prose until the next opening
            # envelope, swallowing self-contained ones (bracket calls,
            # orphan closes) along the way.
            entered_envelope = False
            while True:
                nxt = self._find_start_envelope(candidate, pos, cache)
                if nxt is None:
                    self._buffer = candidate[pos:]
                    break
                idx, consume_len, close = nxt
                if idx > pos:
                    out.append(
                        self._sanitize_prefix_before_suppression(
                            candidate[pos:idx]
                        )
                    )
                env_start = idx
                pos = idx + consume_len
                if close is not None:
                    marker = close
                    entered_envelope = True
                    break
            if not entered_envelope:
                break

        result = "".join(out)
        for close in self._stray_close_markers:
            if close in result:
                result = result.replace(close, "")
        return result

    def feed(self, text: str) -> str:
        """Feed a content delta, return the portion safe to emit."""
        if self._suppressing or not text:
            return ""
        if not self.active:
            return text

        self._buffer += text
        out: List[str] = []

        while self._buffer:
            if self._suppressing_until == "__suppress_permanently__":
                self._suppressing = True
                self._suppressing_until = None
                self._buffer = ""
                break

            if self._suppressing_until is not None:
                end_idx = self._find_suppression_end(self._buffer)
                if end_idx < 0:
                    keep = self._partial_prefix_len(
                        self._buffer, self._suppressing_until
                    )
                    if keep:
                        moved = self._buffer[:-keep]
                        self._pending_envelope_parts.append(moved)
                        # Rebase the JSON scan: those chars left the buffer
                        # front but were already consumed by the scanner.
                        self._shift_json_scan(len(moved), moved)
                        self._buffer = self._buffer[-keep:]
                    else:
                        moved = self._buffer
                        self._pending_envelope_parts.append(moved)
                        self._shift_json_scan(len(moved), moved)
                        self._buffer = ""
                    break
                self._buffer = self._buffer[end_idx + len(self._suppressing_until) :]
                self._suppressing_until = None
                self._clear_pending_envelope()
                continue

            start = self._find_start_envelope(self._buffer)
            if start:
                idx, consume_len, close_marker = start
                opening_marker = self._buffer[idx : idx + consume_len]
                if idx > 0:
                    out.append(
                        self._sanitize_prefix_before_suppression(self._buffer[:idx])
                    )
                self._buffer = self._buffer[idx + consume_len :]
                if close_marker is not None:
                    self._suppressing_until = close_marker
                    if close_marker != "__suppress_permanently__":
                        # Recover the exact opening bytes, including dynamic
                        # namespace markers, if the matching close never arrives.
                        self._pending_envelope_parts = [opening_marker]
                        self._pending_start_marker = opening_marker
                        # Fresh envelope: start the payload scan from scratch.
                        self._reset_json_scan()
                continue

            keep = self._partial_suffix_len(self._buffer)
            if keep == 0:
                out.append(self._buffer)
                self._buffer = ""
                break
            if len(self._buffer) > keep:
                out.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            break

        result = "".join(out)
        for close in self._stray_close_markers:
            if close in result:
                result = result.replace(close, "")
        return result

    def finish(self) -> str:
        """Flush remaining safe buffer content.

        In clean-output strict mode, unresolved marker-like suffixes are dropped
        so partial control markup does not leak into user-visible text.
        """
        if self._suppressing:
            self._buffer = ""
            self._suppressing_until = None
            self._clear_pending_envelope()
            return ""

        # An envelope still suppressing at EOF means the payload scan never
        # confirmed where the value ends.  A literal close marker is then the
        # best evidence available, so trust it rather than withholding to EOF:
        # prose the model resumed after the envelope survives, as it did before
        # the #2507 span scanning.
        recovered = ""
        if self._suppressing_until is not None:
            marker = self._suppressing_until
            self._pending_envelope_parts.append(self._buffer)
            candidate = "".join(self._pending_envelope_parts)
            start_marker = self._pending_start_marker or ""
            self._buffer = ""
            self._suppressing_until = None
            self._clear_pending_envelope()
            recovered = self._unwind_withheld_at_eof(candidate, marker, start_marker)
            if self._suppressing:
                return recovered

        keep = self._partial_suffix_len(self._buffer)
        if keep >= len(self._buffer):
            tail = self._buffer
            self._buffer = ""
            if self._should_drop_tail_at_finish(tail):
                return recovered
            return recovered + tail

        if keep:
            buf = self._buffer[:-keep]
            tail = self._buffer[-keep:]
            if not self._should_drop_tail_at_finish(tail):
                buf += tail
        else:
            buf = self._buffer
        self._buffer = ""
        for close in self._stray_close_markers:
            if close in buf:
                buf = buf.replace(close, "")
        return recovered + buf


def convert_tools_for_template(tools: Optional[List]) -> Optional[List[dict]]:
    """
    Convert OpenAI tools format to format expected by tokenizer.apply_chat_template.

    OpenAI format:
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Template format (commonly used by models):
    [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Args:
        tools: List of ToolDefinition objects or dicts in OpenAI format

    Returns:
        List of tool definitions in template format, or None if no tools
    """
    if not tools:
        return None

    converted = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            tool_func = tool.get("function")
        else:
            tool_type = getattr(tool, "type", None)
            tool_func = getattr(tool, "function", None)

        if tool_type == "function" and tool_func:
            # Handle function as dict or Pydantic model
            if isinstance(tool_func, dict):
                func_name = tool_func.get("name", "")
                func_desc = tool_func.get("description", "")
                func_params = tool_func.get(
                    "parameters", {"type": "object", "properties": {}}
                )
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(
                    tool_func, "parameters", {"type": "object", "properties": {}}
                )

            if func_params is None:
                func_params = {"type": "object", "properties": {}}

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": _template_safe_description(func_desc),
                        "parameters": _copy_schema_with_template_defaults(
                            func_params, is_schema=False
                        ),
                    },
                }
            )

    return converted if converted else None


# Parameter names that collide with JSON Schema keywords.
# Gemma 4 confuses these with schema-level fields and drops them from
# tool call output.  We rename them before the chat template and restore
# them after parsing the model's response.
_GEMMA4_COLLIDING_PARAMS = {"description"}
_GEMMA4_RENAME_PREFIX = "param_"


def enrich_tool_params_for_gemma4(tools: list[dict]) -> list[dict]:
    """Fix tool schemas for Gemma 4 models.

    1. Renames parameters whose names collide with JSON Schema keywords
       (e.g. ``description`` -> ``param_description``) so Gemma 4 doesn't
       confuse them with schema-level fields.
    2. Adds explicit descriptions to required parameters that lack them.

    Use :func:`restore_gemma4_param_names` on tool call arguments to
    reverse the renaming before returning them to the caller.
    """
    enriched = []
    for tool in tools:
        tool = dict(tool)
        func = dict(tool.get("function", {}))
        params = func.get("parameters", {})
        if isinstance(params, dict) and "properties" in params:
            params = dict(params)
            old_props = params.get("properties", {})
            required = list(params.get("required", []))
            new_props = {}
            new_required = []
            for pname, pdef in old_props.items():
                pdef = dict(pdef)
                if pname in _GEMMA4_COLLIDING_PARAMS:
                    new_name = _GEMMA4_RENAME_PREFIX + pname
                else:
                    new_name = pname
                if not pdef.get("description"):
                    label = "REQUIRED. " if pname in required else ""
                    pdef["description"] = (
                        f"{label}The '{pname}' value"
                        f" (type: {pdef.get('type', 'string')})"
                    )
                new_props[new_name] = pdef
                new_required.append(new_name if pname in required else None)
            params["properties"] = new_props
            params["required"] = [r for r in new_required if r]
            func["parameters"] = params
        tool["function"] = func
        enriched.append(tool)
    return enriched


def restore_gemma4_param_names(arguments: dict) -> dict:
    """Reverse the parameter renaming done by :func:`enrich_tool_params_for_gemma4`."""
    restored = {}
    for k, v in arguments.items():
        if k.startswith(_GEMMA4_RENAME_PREFIX):
            original = k[len(_GEMMA4_RENAME_PREFIX):]
            if original in _GEMMA4_COLLIDING_PARAMS:
                restored[original] = v
                continue
        restored[k] = v
    return restored


def format_tool_call_for_message(tool_call: ToolCall) -> dict:
    """
    Format a ToolCall object for inclusion in a message.

    Args:
        tool_call: ToolCall object

    Returns:
        Dict representation suitable for message content
    """
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================


def validate_json_schema(
    data: Any, schema: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON data against a JSON Schema.

    Args:
        data: The JSON data to validate (dict, list, etc.)
        schema: JSON Schema specification

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if data matches schema
        - error_message: Error description if invalid, None if valid
    """
    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, str(e.message)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output text.

    Tries multiple strategies:
    1. Parse entire text as JSON
    2. Extract JSON from markdown code blocks
    3. Find JSON object/array in text

    Args:
        text: Raw model output text

    Returns:
        Parsed JSON data, or None if no valid JSON found
    """
    text = text.strip()

    # Strategy 1: Try to parse entire text as JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, *_DEEP_NEST_ERRORS):
        pass

    # Strategy 2: Extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except (json.JSONDecodeError, *_DEEP_NEST_ERRORS):
            continue

    # Strategy 3: Find JSON object or array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r"(\{[\s\S]*\})",  # Object
        r"(\[[\s\S]*\])",  # Array
    ]
    for pattern in json_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, *_DEEP_NEST_ERRORS):
                continue

    return None


def parse_json_output(
    text: str, response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
) -> Tuple[str, Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Parse JSON from model output when response_format is set.

    Args:
        text: Raw model output text
        response_format: ResponseFormat specification (optional)
            - If type="json_object", extracts any valid JSON
            - If type="json_schema", extracts and validates against schema

    Returns:
        Tuple of (cleaned_text, parsed_json, is_valid, error_message)
        - cleaned_text: Original text (preserved for reference)
        - parsed_json: Extracted JSON data, or None if extraction failed
        - is_valid: True if JSON is valid (and matches schema if specified)
        - error_message: Error description if invalid, None if valid
    """
    # Handle None or text format - just return original
    if response_format is None:
        return text, None, True, None

    # Normalize response_format to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    # text format - no JSON extraction
    if format_type == "text":
        return text, None, True, None

    # json_object or json_schema - extract JSON
    parsed = extract_json_from_text(text)

    if parsed is None:
        return text, None, False, "Failed to extract valid JSON from output"

    # json_object - just verify it's valid JSON (already done by extraction)
    if format_type == "json_object":
        return text, parsed, True, None

    # json_schema - validate against schema
    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema") or {}
        schema = json_schema_spec.get("schema", {})

        if schema:
            is_valid, error = validate_json_schema(parsed, schema)
            if not is_valid:
                return text, parsed, False, f"JSON Schema validation failed: {error}"

        return text, parsed, True, None

    # Unknown format type - treat as text
    return text, None, True, None


def build_json_system_prompt(
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Build a system prompt instruction for JSON output.

    For models without native JSON mode support, this adds instructions
    to the prompt to encourage proper JSON formatting.

    Args:
        response_format: ResponseFormat specification

    Returns:
        System prompt instruction string, or None if not needed
    """
    if response_format is None:
        return None

    # Normalize to dict
    if isinstance(response_format, ResponseFormat):
        rf_dict = {"type": response_format.type, "json_schema": None}
        if response_format.json_schema:
            rf_dict["json_schema"] = {
                "name": response_format.json_schema.name,
                "description": response_format.json_schema.description,
                "schema": response_format.json_schema.schema_,
                "strict": response_format.json_schema.strict,
            }
    else:
        rf_dict = response_format

    format_type = rf_dict.get("type", "text")

    if format_type == "text":
        return None

    if format_type == "json_object":
        return (
            "You must respond with valid JSON only. "
            "Do not include any explanation or text outside the JSON object."
        )

    if format_type == "json_schema":
        json_schema_spec = rf_dict.get("json_schema") or {}
        schema = json_schema_spec.get("schema", {})
        name = json_schema_spec.get("name", "response")
        description = json_schema_spec.get("description", "")

        prompt = f"You must respond with valid JSON matching the '{name}' schema."
        if description:
            prompt += f" {description}"
        prompt += (
            f"\n\nJSON Schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n\n"
            "Respond with only the JSON object, no additional text or explanation."
        )
        return prompt

    return None
