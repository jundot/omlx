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

import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import validate, ValidationError

from .openai_models import FunctionCall, ResponseFormat, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallExtraction:
    """Parsed tool-call result plus sanitized reasoning text."""

    cleaned_text: str
    tool_calls: Optional[List[ToolCall]]
    cleaned_thinking: str
    tool_calls_from_thinking: bool = False


def _parse_xml_tool_calls(text: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Fallback parser for XML-based tool call formats.

    Handles models that use <tool_call>...</tool_call> XML format, including:
    - GLM format: <tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    - Qwen/Llama format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    - Generic JSON: <tool_call>{"name": ..., "arguments": ...}</tool_call>

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        content = match.strip()
        try:
            # Try JSON format first: {"name": "func", "arguments": {...}}
            parsed = json.loads(content)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False)
                        if isinstance(arguments, dict)
                        else str(arguments),
                    ),
                )
            )
            continue
        except (json.JSONDecodeError, AttributeError):
            pass

        # Qwen/Llama format: <function=name><parameter=key>value</parameter></function>
        func_match = re.match(r"<function=(\w+)>(.*?)</function>", content, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            params_text = func_match.group(2)
            arguments = {}
            for pm in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_text, re.DOTALL
            ):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    arguments[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = val
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
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
            arguments = {}
            for k, v in zip(arg_keys, arg_values):
                # Try to parse JSON values (arrays, objects, numbers, booleans)
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

    if not tool_calls:
        return text, None

    # Remove tool call tags from text
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


def _parse_namespaced_tool_calls(
    text: str, namespace: str
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse namespaced tool call tags like <minimax:tool_call>...</minimax:tool_call>.

    Handles the <invoke name="func"><parameter name="key">value</parameter></invoke>
    format used by MiniMax and similar models.

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
            arguments = {}
            for pm in re.finditer(
                r'<parameter\s+name="([^"]+)">(.*?)</parameter>', params_text, re.DOTALL
            ):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    arguments[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = val
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=func_name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )

    if not tool_calls:
        return text, None

    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
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
        except (json.JSONDecodeError, ValueError):
            arguments = {"raw": args_str}
        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )
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

def _gemma4_args_to_json_robust(args_str: str) -> dict:
    """Convert Gemma 4 tool call args to a Python dict.

    Handles the common failure cases that mlx-lm's parser cannot:
    - Bare string values without ``<|"|>`` delimiters (e.g. ``{location: Tokyo}``)
    - Spaces after commas in key-value pairs
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
    except json.JSONDecodeError:
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
        except (json.JSONDecodeError, ValueError):
            return f": {json.dumps(value)}{suffix}"

    text = regex.sub(
        r"(:\s*)([^\",\[\]{}\s][^,}]*?)(\s*[,}])", _quote_bare, text
    )
    return json.loads(text)


def _parse_gemma4_tool_call_fallback(text: str) -> Union[dict, list]:
    """Robust fallback parser for Gemma 4 ``call:name{args}`` format.

    Activated only for Gemma 4 models (guarded by ``tool_call_start`` check).
    Extends mlx-lm's parser to handle:
    - Bare string values without ``<|"|>`` delimiters
    - Colons / dots / hyphens in function names
    """
    import regex

    pattern = regex.compile(
        r"call:([\w:.-]+)(\{(?:[^{}]|(?2))*\})", regex.DOTALL
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError("No function call found in Gemma 4 format")

    results = []
    for match in matches:
        func_name = match.group(1)
        args_str = match.group(2)

        # Try standard JSON first (model may emit valid JSON args)
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = _gemma4_args_to_json_robust(args_str)

        results.append({"name": func_name, "arguments": arguments})

    return results[0] if len(results) == 1 else results


def parse_tool_calls(
    text: str,
    tokenizer: Any,
    tools: Optional[List] = None,
) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse tool calls from model output.

    Uses mlx-lm's TokenizerWrapper tool parser if available, otherwise
    falls back to generic XML tool call parsing for models like GLM.

    Args:
        text: Raw model output text
        tokenizer: mlx-lm's TokenizerWrapper (required)
        tools: Tool definitions for type conversion (optional)

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
        - cleaned_text: Text with tool call tags and thinking tags removed
        - tool_calls: List of ToolCall objects, or None if no tool calls found
    """
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
                # Paired markers (e.g. <tool_call>...</tool_call>)
                end_escaped = re.escape(tool_call_end)
                pattern = rf"{start_escaped}(.*?){end_escaped}"
                matches = re.findall(pattern, text, re.DOTALL)
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
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                type="function",
                                function=FunctionCall(
                                    name=name,
                                    arguments=json.dumps(arguments, ensure_ascii=False)
                                    if isinstance(arguments, dict)
                                    else str(arguments),
                                ),
                            )
                        )
                except (
                    ValueError,
                    json.JSONDecodeError,
                    AttributeError,
                    KeyError,
                    SyntaxError,
                    TypeError,
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
                                tool_calls.append(
                                    ToolCall(
                                        id=f"call_{uuid.uuid4().hex[:8]}",
                                        type="function",
                                        function=FunctionCall(
                                            name=name,
                                            arguments=json.dumps(
                                                arguments, ensure_ascii=False
                                            )
                                            if isinstance(arguments, dict)
                                            else str(arguments),
                                        ),
                                    )
                                )
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
                    _, fb_calls = _parse_xml_tool_calls(fb_wrapped)
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
                    cleaned_text = re.sub(
                        rf"{start_escaped}.*?{re.escape(tool_call_end)}",
                        "",
                        cleaned_text,
                        flags=re.DOTALL,
                    ).strip()
                else:
                    # One-sided: everything from first marker to end is tool calls
                    idx = cleaned_text.find(tool_call_start)
                    if idx >= 0:
                        cleaned_text = cleaned_text[:idx].strip()
                return cleaned_text, tool_calls

    # Fallback: parse XML <tool_call> tags (GLM, Qwen, generic formats)
    if "<tool_call>" in cleaned_text:
        return _parse_xml_tool_calls(cleaned_text)

    # Fallback: namespaced tool_call tags (e.g. <minimax:tool_call>)
    ns_match = re.search(r"<([A-Za-z_][\w.-]*):tool_call>", cleaned_text)
    if ns_match:
        ns = ns_match.group(1)
        return _parse_namespaced_tool_calls(cleaned_text, ns)

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
            s_esc = re.escape(_start)
            e_esc = re.escape(_end)
            stripped = re.findall(
                rf"{s_esc}(.*?){e_esc}", cleaned_text, flags=re.DOTALL
            )
            if stripped:
                logger.warning(
                    "Tool call markers found but parsing failed, "
                    "stripping markers. Raw content: %s",
                    stripped,
                )
            cleaned_text = re.sub(
                rf"{s_esc}.*?{e_esc}", "", cleaned_text, flags=re.DOTALL
            ).strip()
        elif _start:
            idx = cleaned_text.find(_start)
            if idx >= 0:
                logger.warning(
                    "Tool call start marker found but parsing failed, "
                    "stripping marker. Raw content: %s",
                    cleaned_text[idx:],
                )
                cleaned_text = cleaned_text[:idx].strip()

    return cleaned_text, None


def sanitize_tool_call_markup(text: str, tokenizer: Any) -> str:
    """Remove tool-call control markup while preserving surrounding prose."""
    if not text:
        return ""

    stream_filter = ToolCallStreamFilter(tokenizer)
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
    """Extract tool calls while keeping a sanitized reasoning transcript."""
    cleaned_text, tool_calls = parse_tool_calls(regular_content, tokenizer, tools)
    cleaned_thinking = sanitize_tool_call_markup(thinking_content, tokenizer)
    tool_calls_from_thinking = False

    if not tool_calls and thinking_content:
        _, tool_calls = parse_tool_calls(thinking_content, tokenizer, tools)
        tool_calls_from_thinking = bool(tool_calls)

        # Guard 1: if model produced regular text, the tool call in thinking
        # is just reasoning, not an actual invocation request.
        if tool_calls and regular_content.strip():
            tool_calls = None
            tool_calls_from_thinking = False

        # Guard 2: only keep tool calls whose name matches a provided tool.
        if tool_calls and tools:
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
    """

    def __init__(self, tokenizer: Any):
        marker = getattr(tokenizer, "tool_call_start", None)
        marker_end = getattr(tokenizer, "tool_call_end", None)
        # Normalize None-like values but preserve empty strings.
        if marker is None:
            marker = ""
        if marker_end is None:
            marker_end = ""
        self._marker_pairs: List[Tuple[str, str]] = [("<tool_call>", "</tool_call>")]
        self._suppress_after_markers: List[str] = []
        if marker:
            if marker_end:
                self._marker_pairs.insert(0, (marker, marker_end))
            else:
                # One-sided markers (e.g. Mistral "[TOOL_CALLS]" with no
                # end marker): suppress everything after the start marker.
                self._suppress_after_markers.append(marker)
        self._namespaced_open_re = re.compile(r"<([A-Za-z_][\w.-]*):tool_call>")
        self._bracket_prefixes = ["[Calling tool:", "[Tool call:"]
        self._bracket_call_re = re.compile(
            r"^\[(?:Calling tool|Tool call):\s*([A-Za-z_][\w.-]*)(?:\(({.*?})\))?\]",
            re.DOTALL,
        )
        self._buffer = ""
        self._suppressing_until: Optional[str] = None
        self._suppressing = False

    @property
    def active(self) -> bool:
        """Whether this filter should run for tool-enabled streams."""
        return True

    def _find_start_envelope(
        self, text: str
    ) -> Optional[Tuple[int, int, Optional[str]]]:
        """Find earliest complete opening envelope.

        Returns:
            tuple(index, consume_len, close_marker_or_none)
            - close_marker_or_none is a close marker to wait for, or ``None``
              when the whole envelope is already contained in consume_len.
        """
        starts: List[Tuple[int, int, Optional[str]]] = []

        for marker, close in self._marker_pairs:
            idx = text.find(marker)
            if idx >= 0:
                starts.append((idx, len(marker), close))

        ns_match = self._namespaced_open_re.search(text)
        if ns_match:
            ns = ns_match.group(1)
            starts.append(
                (ns_match.start(), len(ns_match.group(0)), f"</{ns}:tool_call>")
            )

        for bp in self._bracket_prefixes:
            bracket_idx = text.find(bp)
            while bracket_idx >= 0:
                bracket_candidate = text[bracket_idx:]
                bracket_match = self._bracket_call_re.match(bracket_candidate)
                if bracket_match:
                    starts.append((bracket_idx, bracket_match.end(), None))
                bracket_idx = text.find(bp, bracket_idx + 1)

        # One-sided markers: suppress from start marker to end of buffer.
        for sa_marker in self._suppress_after_markers:
            idx = text.find(sa_marker)
            if idx >= 0:
                starts.append((idx, len(text) - idx, "__suppress_permanently__"))

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
                end_idx = self._buffer.find(self._suppressing_until)
                if end_idx < 0:
                    keep = self._partial_prefix_len(
                        self._buffer, self._suppressing_until
                    )
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[end_idx + len(self._suppressing_until) :]
                self._suppressing_until = None
                continue

            start = self._find_start_envelope(self._buffer)
            if start:
                idx, consume_len, close_marker = start
                if idx > 0:
                    out.append(
                        self._sanitize_prefix_before_suppression(self._buffer[:idx])
                    )
                self._buffer = self._buffer[idx + consume_len :]
                if close_marker is not None:
                    self._suppressing_until = close_marker
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

        return "".join(out)

    def finish(self) -> str:
        """Flush remaining safe buffer content.

        In clean-output strict mode, unresolved marker-like suffixes are dropped
        so partial control markup does not leak into user-visible text.
        """
        if self._suppressing or self._suppressing_until is not None:
            self._buffer = ""
            self._suppressing_until = None
            return ""

        keep = self._partial_suffix_len(self._buffer)
        if keep >= len(self._buffer):
            tail = self._buffer
            self._buffer = ""
            if self._should_drop_tail_at_finish(tail):
                return ""
            return tail

        if keep:
            buf = self._buffer[:-keep]
        else:
            buf = self._buffer
        self._buffer = ""
        return buf


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

            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": func_desc,
                        "parameters": func_params,
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
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
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
            except json.JSONDecodeError:
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
        json_schema_spec = rf_dict.get("json_schema", {})
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
        json_schema_spec = rf_dict.get("json_schema", {})
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
