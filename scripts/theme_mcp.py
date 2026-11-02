#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tiny stdio MCP adapter for the oMLX external theme API.

Run it from an MCP client:

    OMLX_API_KEY=... OMLX_THEME_URL=http://127.0.0.1:8000 \
        python3 scripts/theme_mcp.py

It only forwards to /admin/api/theme; it does not store anything locally or
read images. The agent still plans colors externally.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("OMLX_THEME_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("OMLX_API_KEY", "")
THEME_URL = f"{BASE_URL}/admin/api/theme"

COLORS = {"type": "object", "additionalProperties": {"type": "string"}}
THEME = {"light": COLORS, "dark": COLORS}
FLOOR = "Primary/secondary text, links and button text must reach 4.5:1."

# name: (description, extra input properties, required, HTTP method, path)
TOOLS = {
    "theme_schema": (
        "Read the token contract: groups, roles, defaults and contrast rules.",
        {}, [], "GET", "/schema",
    ),
    "theme_get": (
        "Read the live theme, revision, resolved palette, draft and history.",
        {}, [], "GET", "",
    ),
    "theme_validate": (
        f"Dry-run light/dark token overrides; returns structured errors. {FLOOR}",
        THEME, ["light", "dark"], "POST", "/validate",
    ),
    "theme_recipe": (
        "Expand background/text/accent seeds (surface and status colors "
        "optional) into all 22 tokens without saving.",
        {"light": {"type": "object"}, "dark": {"type": "object"}},
        ["light", "dark"], "POST", "/recipe",
    ),
    "theme_draft": (
        "Store a proposal for the Custom Theme page; the live theme is "
        f"unchanged. {FLOOR}",
        {**THEME, "note": {"type": "string"}, "source": {"type": "string"}},
        ["light", "dark"], "POST", "/draft",
    ),
    "theme_apply": (
        "Apply a theme. Only call after the user explicitly approves; needs "
        f"the current revision. {FLOOR} A lower pair is refused, not saved.",
        {**THEME, "revision": {"type": "integer"}},
        ["revision", "light", "dark"], "PUT", "",
    ),
}


def _request(method: str, path: str, body: dict | None) -> tuple[int, dict]:
    request = urllib.request.Request(
        THEME_URL + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"ok": False, "error": raw}


def call_tool(name: str, arguments: dict) -> tuple[dict, bool]:
    if name not in TOOLS:
        return {"ok": False, "error": f"unknown tool {name}"}, True
    _, properties, _, method, path = TOOLS[name]
    body = None
    if method != "GET":
        body = {key: arguments[key] for key in properties if key in arguments}
        if name == "theme_apply":
            body["source"] = "mcp"
    status, payload = _request(method, path, body)
    return payload, status >= 400


def tool_list() -> list[dict]:
    return [
        {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        for name, (description, properties, required, _, _) in TOOLS.items()
    ]


def respond(request_id, result=None, error=None) -> None:
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            respond(
                request_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "omlx-theme", "version": "1.0.0"},
                },
            )
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            respond(request_id, {})
        elif method == "tools/list":
            respond(request_id, {"tools": tool_list()})
        elif method == "tools/call":
            params = message.get("params") or {}
            payload, is_error = call_tool(
                params.get("name", ""), params.get("arguments") or {}
            )
            text = json.dumps(payload, indent=2)
            respond(
                request_id,
                {"content": [{"type": "text", "text": text}], "isError": is_error},
            )
        elif request_id is not None:
            error = {"code": -32601, "message": f"unknown method {method}"}
            respond(request_id, error=error)


if __name__ == "__main__":
    main()
