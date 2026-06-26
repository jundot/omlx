"""MiniMax M3 XML-style tool-call parser."""

import json
import re

tool_call_start = "]<]minimax[>[<tool_call>"
tool_call_end = "]<]minimax[>[</tool_call>"

_NS_TOKEN = "]<]minimax[>["
_ELEMENT_START = f"{_NS_TOKEN}<"
_ELEMENT_END_START = f"{_NS_TOKEN}</"
_TAG_NAME_RE = re.compile(r"[A-Za-z_$][\w:.$-]*")
_INVOKE_RE = re.compile(
    rf"{re.escape(_NS_TOKEN)}<invoke\b(?P<attrs>[^>]*)>"
    rf"(?P<body>.*?){re.escape(_NS_TOKEN)}</invoke>",
    flags=re.DOTALL,
)
_NAME_RE = re.compile(
    r"""\bname\s*=\s*(?:(["'])(?P<quoted>.*?)\1|(?P<bare>[^\s>]+))""",
    flags=re.DOTALL,
)
_MISSING = object()
_MIXED_TEXT_FIELD = "$text"


def _get_field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _schema_kinds(schema):
    if not isinstance(schema, dict):
        return []

    type_value = schema.get("type")
    if isinstance(type_value, str):
        kinds = [type_value]
    elif isinstance(type_value, list):
        kinds = [kind for kind in type_value if isinstance(kind, str)]
    else:
        kinds = []

    composite_seen = False
    if not kinds:
        for key in ("anyOf", "oneOf"):
            options = schema.get(key)
            if isinstance(options, list):
                composite_seen = True
                for option in options:
                    kinds.extend(_schema_kinds(option))
                if kinds:
                    break

    if not kinds:
        if composite_seen:
            kinds = ["object"]
        elif "enum" in schema:
            kinds = ["string"]
        elif "items" in schema:
            kinds = ["array"]
        elif "properties" in schema or "additionalProperties" in schema:
            kinds = ["object"]

    aliases = {
        "str": "string",
        "text": "string",
        "varchar": "string",
        "char": "string",
        "enum": "string",
        "int": "integer",
        "bool": "boolean",
        "binary": "boolean",
        "float": "number",
        "double": "number",
        "dict": "object",
        "map": "object",
        "arr": "array",
        "list": "array",
        "sequence": "array",
    }
    supported = {"string", "integer", "number", "boolean", "object", "array", "null"}
    normalized = []
    for kind in kinds:
        kind = kind.strip().lower()
        if kind.startswith(("int", "uint", "long", "short", "unsigned")):
            kind = "integer"
        elif kind.startswith(("num", "float")):
            kind = "number"
        elif kind.startswith("dict"):
            kind = "object"
        elif kind.startswith("list"):
            kind = "array"
        kind = aliases.get(kind, kind)
        if kind in supported:
            normalized.append(kind)
    return normalized


def _schema_has_kind(schema, kind):
    return kind in _schema_kinds(schema)


def _schema_property(schema, name):
    if not _schema_has_kind(schema, "object"):
        return None
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict) and name in properties:
        return properties[name]
    additional = (
        schema.get("additionalProperties") if isinstance(schema, dict) else None
    )
    return additional if isinstance(additional, dict) else None


def _schema_items(schema):
    if not _schema_has_kind(schema, "array"):
        return None
    items = schema.get("items") if isinstance(schema, dict) else None
    return items if isinstance(items, dict) else None


def _tool_schema(tools, name):
    for tool in tools or []:
        function = _get_field(tool, "function", tool)
        if _get_field(function, "name") != name:
            continue
        parameters = _get_field(function, "parameters", None)
        return parameters if isinstance(parameters, dict) else None
    return None


def _coerce_scalar_with_schema(text, schema):
    if not isinstance(schema, dict):
        return _MISSING

    if text.lower() == "null":
        return None

    for kind in _schema_kinds(schema):
        if kind == "string":
            return text
        if kind == "integer":
            try:
                return int(text)
            except ValueError:
                continue
        if kind == "number":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                try:
                    return float(text)
                except ValueError:
                    continue
        if kind == "boolean":
            lowered = text.strip().lower()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
        if kind == "object":
            if text == "":
                return {}
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        if kind == "array":
            if text == "":
                return []
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                return value
        if kind == "null" and text.lower() == "null":
            return None

    return _MISSING


def _coerce_scalar(text, schema=None, schema_context=False):
    text = (text or "").strip()
    schema_value = _coerce_scalar_with_schema(text, schema)
    if schema_value is not _MISSING:
        return schema_value
    if text == "":
        return ""
    if schema_context:
        return None if text.lower() == "null" else text
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def _insert_value(result, repeated_tags, tag, value):
    if tag in result:
        if tag not in repeated_tags:
            result[tag] = [result[tag]]
            repeated_tags.add(tag)
        result[tag].append(value)
    else:
        result[tag] = value


def _nodes_to_object(children, schema=None, schema_context=False):
    result = {}
    repeated_tags = set()
    for child in children:
        child_schema = _schema_property(schema, child["tag"])
        _insert_value(
            result,
            repeated_tags,
            child["tag"],
            _node_to_value(
                child,
                child_schema,
                schema_context=schema_context or isinstance(schema, dict),
            ),
        )
    return result


def _mixed_text_tag(children):
    tag = _MIXED_TEXT_FIELD
    child_tags = {child["tag"] for child in children}
    while tag in child_tags:
        tag = f"${tag}"
    return tag


def _node_to_value(node, schema=None, schema_context=False):
    children = node["children"]
    if not children:
        return _coerce_scalar(
            "".join(node["text"]), schema, schema_context=schema_context
        )
    mixed_text = "".join(node["text"])
    if mixed_text.strip():
        children = [
            *children,
            {"tag": _mixed_text_tag(children), "children": [], "text": [mixed_text]},
        ]
    if _schema_has_kind(schema, "array"):
        item_schema = _schema_items(schema)
        return [
            _node_to_value(
                child,
                item_schema,
                schema_context=schema_context or isinstance(schema, dict),
            )
            for child in children
        ]
    if _schema_has_kind(schema, "object"):
        return _nodes_to_object(children, schema, schema_context=True)
    if all(child["tag"] == "item" for child in children):
        return [_node_to_value(child) for child in children]
    return _nodes_to_object(children, schema, schema_context=schema_context)


def _parse_open_tag(body, pos):
    if not body.startswith(_ELEMENT_START, pos) or body.startswith(
        _ELEMENT_END_START, pos
    ):
        raise ValueError("MiniMax M3 tool invocation expected an opening tag.")

    name_start = pos + len(_ELEMENT_START)
    name_end = body.find(">", name_start)
    if name_end < 0:
        raise ValueError("MiniMax M3 tool invocation has an unclosed opening tag.")

    tag = body[name_start:name_end]
    if not _TAG_NAME_RE.fullmatch(tag):
        raise ValueError(f"MiniMax M3 tool invocation has invalid tag {tag!r}.")
    return tag, name_end + 1


def _parse_element(body, pos):
    tag, pos = _parse_open_tag(body, pos)
    close_tag = f"{_ELEMENT_END_START}{tag}>"
    node = {"tag": tag, "children": [], "text": []}

    while True:
        next_ns = body.find(_NS_TOKEN, pos)
        if next_ns < 0:
            node["text"].append(body[pos:])
            raise ValueError(f"MiniMax M3 tool invocation has unclosed {tag!r}.")

        node["text"].append(body[pos:next_ns])
        pos = next_ns

        if body.startswith(close_tag, pos):
            return node, pos + len(close_tag)
        if body.startswith(_ELEMENT_START, pos) and not body.startswith(
            _ELEMENT_END_START, pos
        ):
            child, pos = _parse_element(body, pos)
            node["children"].append(child)
            continue
        raise ValueError(
            "MiniMax M3 tool invocation has an unexpected namespace marker."
        )


def _parse_nodes(body):
    children = []
    pos = 0
    length = len(body)

    while pos < length:
        whitespace = re.match(r"\s*", body[pos:])
        if whitespace:
            pos += whitespace.end()
        if pos >= length:
            break

        if body.startswith(_ELEMENT_START, pos) and not body.startswith(
            _ELEMENT_END_START, pos
        ):
            child, pos = _parse_element(body, pos)
            children.append(child)
            continue
        if body.startswith(_NS_TOKEN, pos):
            raise ValueError(
                "MiniMax M3 tool invocation has an unexpected namespace marker."
            )
        break

    return children


def _parse_invocation(attrs, body, tools=None):
    name_match = _NAME_RE.search(attrs)
    name = (
        name_match.group("quoted") or name_match.group("bare") if name_match else None
    )
    if not name:
        raise ValueError("MiniMax M3 tool invocation is missing a name.")
    schema = _tool_schema(tools, name)
    schema_context = bool(tools) or isinstance(schema, dict)
    arguments = {}
    repeated_tags = set()
    for child in _parse_nodes(body):
        child_schema = _schema_property(schema, child["tag"])
        _insert_value(
            arguments,
            repeated_tags,
            child["tag"],
            _node_to_value(child, child_schema, schema_context=schema_context),
        )
    return {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}


def parse_tool_call(text, tools=None):
    xml_text = text.strip()
    invocations = list(_INVOKE_RE.finditer(xml_text))
    if not invocations:
        raise ValueError("No MiniMax M3 tool invocation found.")

    parsed = []
    for invocation in invocations:
        parsed.append(
            _parse_invocation(
                invocation.group("attrs"), invocation.group("body"), tools
            )
        )
    return parsed if len(parsed) > 1 else parsed[0]
