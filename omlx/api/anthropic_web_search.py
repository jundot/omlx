# SPDX-License-Identifier: Apache-2.0
"""Answer Claude Code's WebSearch call with the configured web search provider.

Claude Code runs WebSearch as a nested /v1/messages request that carries the
server-side web_search tool and one "Perform a web search for the query: ..."
user message. The model is not needed for that request.
"""

import json
import secrets
from typing import Any
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse, StreamingResponse

from ..websearch import run_web_search
from .anthropic_models import ContentBlockText, MessagesRequest
from .anthropic_utils import format_sse_event

SEARCH_PROMPT_PREFIX = "Perform a web search for the query:"

_ERROR_CODES = {
    "rate_limited": "too_many_requests",
    "invalid_arguments": "invalid_tool_input",
}


def _search_query(request: MessagesRequest) -> str:
    message = request.messages[-1]
    if message.role != "user":
        return ""
    if isinstance(message.content, str):
        text = message.content
    else:
        text = "".join(
            block.text
            for block in message.content
            if isinstance(block, ContentBlockText)
        )
    text = text.strip()
    if not text.startswith(SEARCH_PROMPT_PREFIX):
        return ""
    return text[len(SEARCH_PROMPT_PREFIX) :].strip()


def _domain_rules(domains: Any) -> list[tuple[str, str]]:
    rules = []
    for domain in domains or []:
        host, _, path = domain.strip().lower().partition("/")
        rules.append((host, f"/{path}" if path else ""))
    return rules


def _matches(url: str, rules: list[tuple[str, str]]) -> bool:
    parts = urlsplit(url)
    host = parts.hostname or ""
    return any(
        (host == rule_host or host.endswith("." + rule_host))
        and parts.path.startswith(rule_path)
        for rule_host, rule_path in rules
    )


async def _search_content(
    query: str, tool: Any, integrations: Any
) -> list[dict[str, Any]]:
    allowed = _domain_rules(getattr(tool, "allowed_domains", None))
    blocked = _domain_rules(getattr(tool, "blocked_domains", None))
    # Providers return only a few results, so a post-filter alone is mostly empty.
    search_query = f"{query} site:{allowed[0][0]}" if len(allowed) == 1 else query
    payload = await run_web_search(search_query, integrations)

    tool_use_id = f"srvtoolu_{secrets.token_hex(12)}"
    content: list[dict[str, Any]] = [
        {
            "type": "server_tool_use",
            "id": tool_use_id,
            "name": "web_search",
            "input": {"query": query},
        }
    ]
    if not payload.get("ok"):
        error = payload.get("error") or {}
        result: Any = {
            "type": "web_search_tool_result_error",
            "error_code": _ERROR_CODES.get(error.get("code"), "unavailable"),
        }
        text = f"Web search failed: {error.get('message', '')}"
    else:
        results = [
            r
            for r in payload["results"]
            if not _matches(r["url"], blocked)
            and (not allowed or _matches(r["url"], allowed))
        ]
        result = [
            {
                "type": "web_search_result",
                "title": r["title"],
                "url": r["url"],
                "encrypted_content": "",
                "page_age": None,
            }
            for r in results
        ]
        # Claude Code keeps only title and url from result blocks.
        text = "\n\n".join(
            f"{r['title']}\n{r['url']}\n{r.get('content') or r['snippet']}"
            for r in results
        )
    content.append(
        {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
        }
    )
    content.append({"type": "text", "text": text or "No results found."})
    return content


async def _stream(message: dict[str, Any]):
    yield format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {**message, "content": [], "stop_reason": None},
        },
    )
    for index, block in enumerate(message["content"]):
        delta = None
        if block["type"] == "server_tool_use":
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
            block = {**block, "input": {}}
        elif block["type"] == "text":
            delta = {"type": "text_delta", "text": block["text"]}
            block = {"type": "text", "text": ""}
        yield format_sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": block},
        )
        if delta is not None:
            yield format_sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )
        yield format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": index}
        )
    yield format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": message["usage"],
        },
    )
    yield format_sse_event("message_stop", {"type": "message_stop"})


async def maybe_handle_anthropic_web_search(
    request: MessagesRequest, settings: Any
) -> JSONResponse | StreamingResponse | None:
    """Return the search response for Claude Code's WebSearch call, else None."""
    tool = next(
        (t for t in request.tools or [] if (t.type or "").startswith("web_search_")),
        None,
    )
    if tool is None or not request.messages:
        return None
    query = _search_query(request)
    if not query:
        return None

    message = {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": await _search_content(
            query, tool, getattr(settings, "integrations", None)
        ),
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "server_tool_use": {"web_search_requests": 1},
        },
    }
    if request.stream:
        return StreamingResponse(
            _stream(message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(message)
