# SPDX-License-Identifier: Apache-2.0
"""Server-side WebSearch support for the Anthropic Messages API.

Claude Code implements its built-in WebSearch tool by sending Anthropic's
server-side ``web_search_20250305`` tool to the model backend. The backend is
expected to execute the search and return ``server_tool_use`` /
``web_search_tool_result`` content blocks. oMLX cannot delegate that to
Anthropic, so the normal tool conversion layer drops the tool.

This module fulfils Claude Code's nested search request with the search
provider configured under Settings -> Integrations -> Web Search. It only
handles the specific request shape Claude Code uses (the "Perform a web
search for the query: ..." prompt); ordinary chat requests are left untouched.

The result count, snippet/full-page mode, and per-page character limit are
owned by the existing Web Search settings and are not duplicated here.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import secrets
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from fastapi.responses import JSONResponse, StreamingResponse

if TYPE_CHECKING:
    from .anthropic_models import MessagesRequest

logger = logging.getLogger(__name__)

SEARCH_PROMPT_PREFIX = "Perform a web search for the query:"
REQUEST_ID_HEADER = "X-oMLX-Request-ID"


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Pydantic model, dict, or similar object."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        value = getattr(obj, name)
    except AttributeError:
        value = None
    else:
        if value is not None:
            return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict) and name in extra:
        return extra[name]
    return default


def _block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        try:
            return block.model_dump()
        except Exception:
            return {}
    return {}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            data = _block_dict(block)
            if data.get("type") == "text" and isinstance(data.get("text"), str):
                parts.append(data["text"])
        return "\n".join(parts)
    return ""


def find_web_search_tool(request: Any) -> Any | None:
    """Return the Anthropic server-side web search tool, if present."""
    for tool in _get(request, "tools", None) or []:
        tool_type = _get(tool, "type", None)
        if isinstance(tool_type, str) and tool_type.startswith("web_search_"):
            return tool
        if (
            _get(tool, "name", None) == "web_search"
            and _get(tool, "input_schema", None) is None
        ):
            return tool
    return None


def extract_query(request: Any) -> str:
    """Extract Claude Code's nested search query.

    Claude Code sends a single user message:
        Perform a web search for the query: <query>

    A ``server_tool_use`` block is also accepted for replay/robustness. No
    generic "last user text" fallback is used, so an unrelated chat request
    carrying a server tool is never turned into a search.
    """
    messages = _get(request, "messages", None) or []

    for message in reversed(messages):
        if _get(message, "role") != "user":
            continue
        text = _content_text(_get(message, "content", None))
        match = re.search(
            rf"{re.escape(SEARCH_PROMPT_PREFIX)}\s*(.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match and match.group(1).strip():
            return match.group(1).strip()

    for message in messages:
        for block in _get(message, "content", None) or []:
            data = _block_dict(block)
            if data.get("type") != "server_tool_use":
                continue
            if data.get("name") != "web_search":
                continue
            query = (data.get("input") or {}).get("query")
            if isinstance(query, str) and query.strip():
                return query.strip()

    return ""


# ---------------------------------------------------------------------------
# Domain filtering
# ---------------------------------------------------------------------------


def _normalize_domains(value: Any) -> list[tuple[str, str]]:
    """Normalize Anthropic domain filters to (host, path-prefix) pairs.

    Anthropic accepts bare domains with an optional path, for example
    ``example.com`` or ``example.com/blog``. Bare domains match the whole
    host; a path prefix additionally constrains the URL path.
    """
    if not isinstance(value, (list, tuple)):
        return []
    domains: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip().lower().lstrip("*.")
        candidate = item if "://" in item else f"http://{item}"
        try:
            parsed = urlparse(candidate)
        except ValueError:
            continue
        host = (parsed.hostname or "").lstrip("*.").rstrip(".")
        if not host:
            continue
        domains.append((host, (parsed.path or "").rstrip("/")))
    return domains


def _host_matches(hostname: str, domain: str) -> bool:
    host = (hostname or "").lower().rstrip(".")
    target = (domain or "").lower().lstrip("*.").rstrip(".")
    if not host or not target:
        return False
    return host == target or host.endswith("." + target)


def _domain_matches(hostname: str, path: str, domain: tuple[str, str]) -> bool:
    host, prefix = domain
    if not _host_matches(hostname, host):
        return False
    if not prefix:
        return True
    normalized = (path or "/").rstrip("/")
    return normalized == prefix or normalized.startswith(prefix + "/")


def filter_results(results: Any, allowed: Any, blocked: Any) -> list[dict[str, Any]]:
    """Apply Claude Code's allowed_domains/blocked_domains filters."""
    allowed_domains = _normalize_domains(allowed)
    blocked_domains = _normalize_domains(blocked)
    filtered: list[dict[str, Any]] = []

    for result in results or []:
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        if not isinstance(url, str):
            continue
        try:
            parsed = urlparse(url)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            continue
        if any(
            _domain_matches(parsed.hostname, parsed.path, domain)
            for domain in blocked_domains
        ):
            continue
        if allowed_domains and not any(
            _domain_matches(parsed.hostname, parsed.path, domain)
            for domain in allowed_domains
        ):
            continue
        filtered.append(result)

    return filtered


# ---------------------------------------------------------------------------
# Anthropic content blocks
# ---------------------------------------------------------------------------


def _opaque_content(result: dict[str, Any]) -> str:
    """Opaque encrypted_content value.

    Claude Code treats this as an opaque string and does not decrypt it.
    Encode only the result metadata: including the full fetched page text
    here would bloat the SSE response in full-content mode without helping
    the client.
    """
    payload = {
        "title": result.get("title"),
        "url": result.get("url"),
        "snippet": result.get("snippet"),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _summarize(query: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return f'No web results found for "{query}".'

    lines = [f'Search results for "{query}":']
    for index, result in enumerate(results[:10], 1):
        title = str(result.get("title") or result.get("url") or "").strip()
        lines.append(f"{index}. {title}")
        lines.append(f"   {result.get('url', '')}")
        detail = result.get("content") or result.get("snippet") or ""
        detail = re.sub(r"\s+", " ", str(detail)).strip()
        if detail:
            # No extra truncation here: web_search_content_max_chars /
            # web_search_content_truncate already bound each page upstream.
            lines.append(f"   {detail}")
    return "\n".join(lines)


def _map_error_code(payload: dict[str, Any] | None) -> str:
    code = ((payload or {}).get("error") or {}).get("code")
    if code in ("rate_limited", "too_many_requests"):
        return "too_many_requests"
    if code in ("invalid_query", "invalid_input", "invalid_arguments"):
        return "invalid_tool_input"
    if code in ("query_too_long", "request_too_large", "max_uses_exceeded"):
        return code
    return "unavailable"


def build_blocks(
    query: str, payload: dict[str, Any] | None, tool: Any
) -> dict[str, Any]:
    """Build the server_tool_use + web_search_tool_result pair."""
    tool_use_id = f"srvtoolu_{secrets.token_hex(12)}"
    server_tool_use = {
        "type": "server_tool_use",
        "id": tool_use_id,
        "name": "web_search",
        "input": {"query": query},
    }

    if not payload or payload.get("ok") is not True:
        message = ((payload or {}).get("error") or {}).get(
            "message"
        ) or "oMLX web search provider failed"
        result_block = {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": {
                "type": "web_search_tool_result_error",
                "error_code": _map_error_code(payload),
            },
        }
        return {
            "server_tool_use": server_tool_use,
            "result_block": result_block,
            "summary": f'Web search failed for "{query}": {message}',
            "search_count": 1,
            "provider": None,
        }

    results = filter_results(
        payload.get("results"),
        _get(tool, "allowed_domains", None),
        _get(tool, "blocked_domains", None),
    )
    result_block = {
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {
                "type": "web_search_result",
                "title": str(result.get("title") or result.get("url") or "")[:512],
                "url": result.get("url"),
                "encrypted_content": _opaque_content(result),
                "page_age": None,
            }
            for result in results
        ],
    }
    return {
        "server_tool_use": server_tool_use,
        "result_block": result_block,
        "summary": _summarize(query, results),
        "search_count": 1,
        "provider": payload.get("provider"),
    }


# ---------------------------------------------------------------------------
# Anthropic response builders
# ---------------------------------------------------------------------------


def _usage(input_tokens: int, output_tokens: int, searches: int) -> dict[str, Any]:
    return {
        "input_tokens": max(0, int(input_tokens)),
        "output_tokens": max(0, int(output_tokens)),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "server_tool_use": {"web_search_requests": max(0, int(searches))},
    }


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def _format_sse(event_type: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n"


def _new_message_id() -> str:
    return f"msg_{secrets.token_hex(12)}"


def build_message(
    model: str, blocks: dict[str, Any], input_tokens: int
) -> dict[str, Any]:
    """Build a non-streaming Anthropic Messages response."""
    return {
        "id": _new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            blocks["server_tool_use"],
            blocks["result_block"],
            {"type": "text", "text": blocks["summary"]},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _usage(
            input_tokens,
            _estimate_tokens(blocks["summary"]),
            blocks["search_count"],
        ),
    }


async def stream_events(
    model: str, blocks: dict[str, Any], input_tokens: int
) -> AsyncIterator[str]:
    """Yield Claude Code's expected Anthropic SSE sequence."""
    message_id = _new_message_id()
    query = blocks["server_tool_use"]["input"]["query"]
    output_tokens = _estimate_tokens(blocks["summary"])

    yield _format_sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": _usage(input_tokens, 1, 0),
            },
        },
    )

    yield _format_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                **blocks["server_tool_use"],
                "input": {},
            },
        },
    )
    yield _format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps({"query": query}, ensure_ascii=False),
            },
        },
    )
    yield _format_sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    yield _format_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": blocks["result_block"],
        },
    )
    yield _format_sse("content_block_stop", {"type": "content_block_stop", "index": 1})

    yield _format_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield _format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": blocks["summary"]},
        },
    )
    yield _format_sse("content_block_stop", {"type": "content_block_stop", "index": 2})

    yield _format_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": _usage(input_tokens, output_tokens, blocks["search_count"]),
        },
    )
    yield _format_sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def maybe_handle_anthropic_web_search(
    request: MessagesRequest, settings: Any
) -> Any | None:
    """Handle a Claude Code WebSearch request, or return None to continue.

    Returning None lets /v1/messages proceed normally. This is important for
    requests that merely mention a server-side tool but are not Claude Code's
    nested search call.
    """
    tool = find_web_search_tool(request)
    if tool is None:
        return None

    query = extract_query(request)
    if not query:
        return None

    integrations = (
        getattr(settings, "integrations", None) if settings is not None else None
    )
    if integrations is None:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {
                "code": "unexpected_failure",
                "message": "Server settings are unavailable.",
            },
        }
    else:
        try:
            # Imported lazily so this module stays importable in unit tests
            # that do not need the HTTP search stack.
            from ..websearch import run_web_search

            payload = await run_web_search(query, integrations)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Anthropic web_search failed")
            payload = {
                "ok": False,
                "error": {"code": "request_failed", "message": str(exc)},
            }

    blocks = build_blocks(query, payload, tool)

    try:
        request_json = json.dumps(
            request.model_dump(), ensure_ascii=False, separators=(",", ":")
        )
        input_tokens = max(1, len(request_json) // 4)
    except Exception:
        input_tokens = 1

    provider = blocks.get("provider")
    results = (payload or {}).get("results") or []
    logger.info(
        "Anthropic web_search: provider=%s results=%d ok=%s",
        provider,
        len(results),
        bool((payload or {}).get("ok")),
    )
    logger.debug("Anthropic web_search query=%r", query)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        REQUEST_ID_HEADER: str(uuid.uuid4()),
    }
    model = _get(request, "model", "unknown") or "unknown"

    if _get(request, "stream", False):
        return StreamingResponse(
            stream_events(model, blocks, input_tokens),
            media_type="text/event-stream",
            headers=headers,
        )

    return JSONResponse(build_message(model, blocks, input_tokens), headers=headers)
