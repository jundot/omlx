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
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import validate, ValidationError

from .openai_models import FunctionCall, ResponseFormat, ToolCall, ToolDefinition


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
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        content = match.strip()
        try:
            # Try JSON format first: {"name": "func", "arguments": {...}}
            parsed = json.loads(content)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False)
                        if isinstance(arguments, dict) else str(arguments),
                ),
            ))
            continue
        except (json.JSONDecodeError, AttributeError):
            pass

        # Qwen/Llama format: <function=name><parameter=key>value</parameter></function>
        func_match = re.match(r'<function=(\w+)>(.*?)</function>', content, re.DOTALL)
        if func_match:
            func_name = func_match.group(1)
            params_text = func_match.group(2)
            arguments = {}
            for pm in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', params_text, re.DOTALL):
                key = pm.group(1)
                val = pm.group(2).strip()
                try:
                    arguments[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    arguments[key] = val
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            ))
            continue

        # GLM XML format: func_name<arg_key>k</arg_key><arg_value>v</arg_value>...
        arg_keys = re.findall(r'<arg_key>(.*?)</arg_key>', content)
        arg_values = re.findall(r'<arg_value>(.*?)</arg_value>', content, re.DOTALL)
        if arg_keys:
            # Function name is the text before the first <arg_key>
            name_match = re.match(r'^(.*?)<arg_key>', content, re.DOTALL)
            func_name = name_match.group(1).strip() if name_match else content.split('<')[0].strip()
            arguments = {}
            for k, v in zip(arg_keys, arg_values):
                # Try to parse JSON values (arrays, objects, numbers, booleans)
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            ))

    if not tool_calls:
        return text, None

    # Remove tool call tags from text
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


def _parse_namespaced_tool_calls(text: str, namespace: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Parse namespaced tool call tags like <minimax:tool_call>...</minimax:tool_call>.

    Handles the <invoke name="func"><parameter name="key">value</parameter></invoke>
    format used by MiniMax and similar models.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    tag_start = f'<{namespace}:tool_call>'
    tag_end = f'</{namespace}:tool_call>'
    pattern = re.escape(tag_start) + r'(.*?)' + re.escape(tag_end)
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
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            ))

    if not tool_calls:
        return text, None

    cleaned = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


def _parse_bracket_tool_calls(text: str) -> Tuple[str, Optional[List[ToolCall]]]:
    """
    Fallback parser for [Calling tool: name(args)] format.

    This format can appear when models mimic text-formatted tool calls
    from conversation history. Parses the JSON arguments from within
    the parentheses.

    Returns:
        Tuple of (cleaned_text, tool_calls or None)
    """
    tool_calls = []
    pattern = r'\[Calling tool:\s*(\w+)\(({.*?})\)\]'
    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1)
        args_str = match.group(2)
        try:
            arguments = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            arguments = {"raw": args_str}
        tool_calls.append(ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=FunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        ))

    if not tool_calls:
        return text, None

    cleaned = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return cleaned, tool_calls


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
        r'<think>.*?</think>',
        '',
        cleaned_text,
        flags=re.DOTALL
    ).strip()

    # Try mlx-lm's native tool parser first
    if getattr(tokenizer, 'has_tool_calling', False):
        tool_call_start = tokenizer.tool_call_start
        tool_call_end = tokenizer.tool_call_end
        tool_parser = tokenizer.tool_parser

        if all([tool_call_start, tool_call_end, tool_parser]):
            tool_calls = []
            start_escaped = re.escape(tool_call_start)
            end_escaped = re.escape(tool_call_end)
            pattern = rf'{start_escaped}(.*?){end_escaped}'

            matches = re.findall(pattern, text, re.DOTALL)

            for match in matches:
                try:
                    parsed = tool_parser(match.strip(), tools)
                    name = parsed.get('name', '')
                    arguments = parsed.get('arguments', {})
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=FunctionCall(
                            name=name,
                            arguments=json.dumps(arguments, ensure_ascii=False)
                                if isinstance(arguments, dict) else str(arguments)
                        )
                    ))
                except (ValueError, json.JSONDecodeError, AttributeError, KeyError):
                    continue

            if tool_calls:
                cleaned_text = re.sub(
                    rf'{start_escaped}.*?{end_escaped}',
                    '',
                    cleaned_text,
                    flags=re.DOTALL
                ).strip()
                return cleaned_text, tool_calls

    # Fallback: parse XML <tool_call> tags (GLM, Qwen, generic formats)
    if '<tool_call>' in cleaned_text:
        return _parse_xml_tool_calls(cleaned_text)

    # Fallback: namespaced tool_call tags (e.g. <minimax:tool_call>)
    ns_match = re.search(r'<(\w+):tool_call>', cleaned_text)
    if ns_match:
        ns = ns_match.group(1)
        return _parse_namespaced_tool_calls(cleaned_text, ns)

    # Fallback: [Calling tool: name(args)] format (from text-formatted history)
    if '[Calling tool:' in cleaned_text:
        return _parse_bracket_tool_calls(cleaned_text)

    return cleaned_text, None


def convert_tools_for_template(
    tools: Optional[List]
) -> Optional[List[dict]]:
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
                func_params = tool_func.get("parameters", {"type": "object", "properties": {}})
            else:
                func_name = getattr(tool_func, "name", "")
                func_desc = getattr(tool_func, "description", "")
                func_params = getattr(tool_func, "parameters", {"type": "object", "properties": {}})

            converted.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_desc,
                    "parameters": func_params
                }
            })

    return converted if converted else None


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
        }
    }


# =============================================================================
# Structured Output (JSON Schema) Utilities
# =============================================================================

def validate_json_schema(
    data: Any,
    schema: Dict[str, Any]
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
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Find JSON object or array in text
    # Look for { ... } or [ ... ]
    json_patterns = [
        r'(\{[\s\S]*\})',  # Object
        r'(\[[\s\S]*\])',  # Array
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
    text: str,
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
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
        rf_dict = {
            "type": response_format.type,
            "json_schema": None
        }
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
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
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
        rf_dict = {
            "type": response_format.type,
            "json_schema": None
        }
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
