# SPDX-License-Identifier: Apache-2.0
"""
Web search / page fetch API routes for the admin chat UI.

- POST /v1/web/search - Search with the configured provider
- POST /v1/web/fetch - Fetch a public page as markdown

Both endpoints always answer HTTP 200 with an {"ok": bool, ...} payload;
failures ride in the payload so the chat model can read and relay them.
The chat page is the only intended caller — these tools are never merged
into /v1/chat/completions tool lists.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..websearch import (
    failure_payload,
    fetch_content_chars,
    run_fetch_url,
    run_web_search,
)

router = APIRouter(prefix="/v1/web", tags=["web"])


# Callback function to get global settings (set by server.py)
_get_global_settings = None


def set_global_settings_getter(getter):
    """
    Set the callback function to get the live GlobalSettings instance.

    Args:
        getter: A callable that returns the GlobalSettings instance or None
    """
    global _get_global_settings
    _get_global_settings = getter


class WebSearchRequest(BaseModel):
    # Defaults keep blank input as a readable payload error instead of a
    # 422 the model cannot interpret.
    query: str = ""


class WebFetchRequest(BaseModel):
    url: str = ""


@router.post("/search")
async def web_search(request: WebSearchRequest) -> dict[str, Any]:
    """Run a web search and return an {"ok": bool, ...} payload."""
    settings = _get_global_settings() if _get_global_settings else None
    if settings is None:
        return failure_payload(
            "unexpected_failure", "Server settings are unavailable."
        )
    return await run_web_search(request.query, settings.integrations)


@router.post("/fetch")
async def web_fetch(request: WebFetchRequest) -> dict[str, Any]:
    """Fetch a public page and return an {"ok": bool, ...} payload."""
    settings = _get_global_settings() if _get_global_settings else None
    integrations = settings.integrations if settings is not None else None
    if integrations is not None:
        return await run_fetch_url(
            request.url,
            max_chars=fetch_content_chars(integrations),
            truncate=bool(
                getattr(integrations, "web_search_content_truncate", True)
            ),
        )
    return await run_fetch_url(request.url)
