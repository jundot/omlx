# SPDX-License-Identifier: Apache-2.0
"""
Built-in web search and page fetch for the admin chat UI.

The chat page advertises two client-side tools (web_search / fetch_url)
and executes them through the /v1/web endpoints, which land here. These
tools are never merged into /v1/chat/completions requests, so external
API clients are unaffected.

Providers:
- duckduckgo (default): ddgs library, no credentials required. Note that
  ddgs performs HTTP through primp, which may not honor the proxy
  environment variables applied from NetworkSettings; the brave and
  searxng paths use httpx and inherit them.
- brave: Brave Search API with a user-supplied subscription token.
- searxng: an external SearXNG instance reachable via a base URL. The
  instance must have the JSON output format enabled.

Every entry point returns a payload dict instead of raising: successes
as {"ok": True, ...} and failures as {"ok": False, "error": {...}} so
the chat model can read the outcome and guide the user.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import urllib.parse
from typing import Any

import httpx

from .api.markitdown import convert_html_to_markdown

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ddgs", "ddgs_custom", "duckduckgo", "brave", "searxng")

# Text backends the pinned ddgs 9.14.1 actually registers (bing/google
# exist as modules but are disabled in this version).
DDGS_TEXT_BACKENDS = (
    "brave",
    "duckduckgo",
    "grokipedia",
    "mojeek",
    "wikipedia",
    "yahoo",
    "yandex",
)

# Defaults and hard caps. Result count and page-content truncation are
# user-tunable through IntegrationSettings; the rest stays fixed.
DEFAULT_MAX_RESULTS = 3
MAX_RESULTS_CAP = 10
MAX_QUERY_CHARS = 300
MAX_URL_CHARS = 2048
MAX_TITLE_CHARS = 160
MAX_SNIPPET_CHARS = 500
HTTP_TIMEOUT_SECONDS = 20.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
DEFAULT_FETCH_CONTENT_CHARS = 20_000

_SETTINGS_HINT = "Dashboard -> Settings -> Integrations -> Web Search"

_FETCHABLE_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "application/json",
}
_HTML_CONTENT_TYPES = {"", "text/html", "application/xhtml+xml"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_USER_AGENT = "omlx-web/1.0"


class WebSearchError(Exception):
    """Tool failure carrying a stable error code the chat model can act on."""

    def __init__(self, code: str, message: str, needs_user_action: bool = False):
        super().__init__(message)
        self.code = code
        self.needs_user_action = needs_user_action


def failure_payload(
    code: str, message: str, needs_user_action: bool = False
) -> dict[str, Any]:
    """Build the {"ok": False, ...} payload returned to the model."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "user_action_required": needs_user_action,
        },
    }


def _error_from_status(status: int, source: str) -> WebSearchError:
    if status == 401:
        return WebSearchError(
            "invalid_authentication",
            f"{source} rejected the configured credential (HTTP 401). "
            f"Check the key in {_SETTINGS_HINT}.",
            True,
        )
    if status == 402:
        return WebSearchError(
            "insufficient_funds",
            f"{source} reported the account is out of credits (HTTP 402).",
            True,
        )
    if status == 403:
        return WebSearchError(
            "plan_access",
            f"{source} denied access for the current plan (HTTP 403).",
            True,
        )
    if status == 429:
        return WebSearchError(
            "rate_limited", f"{source} rate limited the request (HTTP 429)."
        )
    if status >= 500:
        return WebSearchError(
            "provider_unavailable", f"{source} is unavailable (HTTP {status})."
        )
    return WebSearchError("request_failed", f"{source} returned HTTP {status}.")


def sanitize_result(title: Any, url: Any, snippet: Any) -> dict[str, str] | None:
    """Normalize one raw provider row; return None to drop it entirely.

    Only http(s) URLs with a host and no embedded credentials survive, so
    javascript:/file:/data: links and credential-bearing URLs from a
    malicious page never reach the model.
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or len(url) > MAX_URL_CHARS:
        return None
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    if not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    clean_title = str(title or "").strip()[:MAX_TITLE_CHARS] or parts.hostname
    clean_snippet = str(snippet or "").strip()[:MAX_SNIPPET_CHARS]
    return {"title": clean_title, "url": url, "snippet": clean_snippet}


class SearchProvider:
    """One search backend. search() returns raw rows or raises WebSearchError."""

    name: str = ""

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        raise NotImplementedError


def _ddgs_text(query: str, max_results: int, backend: str) -> list[dict[str, Any]]:
    # Lazy import keeps ddgs (and its primp native extension) off the
    # server boot path; same precedent as the markitdown converter.
    from ddgs import DDGS

    with DDGS(timeout=int(HTTP_TIMEOUT_SECONDS)) as client:
        return client.text(query, max_results=max_results, backend=backend)


def _map_ddgs_error(exc: Exception) -> WebSearchError:
    try:
        from ddgs.exceptions import (
            DDGSException,
            RatelimitException,
            TimeoutException,
        )
    except ImportError:
        return WebSearchError(
            "provider_unavailable", "The ddgs package is not installed."
        )
    if isinstance(exc, RatelimitException):
        return WebSearchError(
            "rate_limited", "DuckDuckGo rate limited the search; retry later."
        )
    if isinstance(exc, (TimeoutException, DDGSException)):
        return WebSearchError(
            "provider_unavailable", f"DuckDuckGo search failed: {exc}"
        )
    return WebSearchError(
        "unexpected_failure", f"DuckDuckGo search failed: {type(exc).__name__}"
    )


class DdgsProvider(SearchProvider):
    """ddgs metasearch. backend is "auto", one engine, or a csv of engines."""

    def __init__(self, name: str, backend: str):
        self.name = name
        self._backend = backend

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        try:
            rows = await asyncio.to_thread(
                _ddgs_text, query, max_results, self._backend
            )
        except Exception as exc:
            # ddgs raises instead of returning an empty list; a zero-hit
            # query is a valid outcome, not a provider failure.
            if "No results found" in str(exc):
                return []
            raise _map_ddgs_error(exc) from exc
        return [
            {
                "title": row.get("title"),
                "url": row.get("href"),
                "snippet": row.get("body"),
            }
            for row in rows
            if isinstance(row, dict)
        ]


class BraveProvider(SearchProvider):
    name = "brave"

    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self, api_key: str, transport: httpx.AsyncBaseTransport | None = None
    ):
        self._api_key = (api_key or "").strip()
        self._transport = transport

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self._api_key:
            raise WebSearchError(
                "missing_api_key",
                f"No Brave Search API key is configured. Add one in {_SETTINGS_HINT}.",
                True,
            )
        response = await _provider_get(
            self.ENDPOINT,
            params={"q": query, "count": max_results},
            headers={
                "X-Subscription-Token": self._api_key,
                "Accept": "application/json",
            },
            source="Brave Search",
            transport=self._transport,
        )
        data = _parse_json(response, "Brave Search")
        web = data.get("web") if isinstance(data, dict) else None
        rows = web.get("results") if isinstance(web, dict) else None
        return [
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "snippet": row.get("description"),
            }
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, dict)
        ]


class SearXNGProvider(SearchProvider):
    name = "searxng"

    def __init__(
        self, base_url: str, transport: httpx.AsyncBaseTransport | None = None
    ):
        self._base_url = (base_url or "").strip().rstrip("/")
        self._transport = transport

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if not self._base_url:
            raise WebSearchError(
                "missing_api_key",
                f"No SearXNG instance URL is configured. Add one in {_SETTINGS_HINT}.",
                True,
            )
        response = await _provider_get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json"},
            headers={"Accept": "application/json"},
            source="SearXNG",
            transport=self._transport,
        )
        data = _parse_json(response, "SearXNG")
        rows = data.get("results") if isinstance(data, dict) else None
        return [
            {
                "title": row.get("title"),
                "url": row.get("url"),
                "snippet": row.get("content"),
            }
            for row in (rows if isinstance(rows, list) else [])
            if isinstance(row, dict)
        ]


async def _provider_get(
    url: str,
    params: dict[str, Any],
    headers: dict[str, str],
    source: str,
    transport: httpx.AsyncBaseTransport | None,
) -> httpx.Response:
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS, transport=transport
    ) as client:
        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise WebSearchError(
                "provider_unavailable", f"{source} timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise WebSearchError(
                "request_failed", f"{source} request failed: {exc}"
            ) from exc
    if response.status_code != 200:
        raise _error_from_status(response.status_code, source)
    return response


def _parse_json(response: httpx.Response, source: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise WebSearchError(
            "invalid_response", f"{source} returned malformed JSON."
        ) from exc


def normalize_ddgs_backends(raw: str) -> list[str]:
    """Parse a comma-separated backend list, keeping only known engines."""
    seen: list[str] = []
    for part in (raw or "").split(","):
        name = part.strip().lower()
        if name and name in DDGS_TEXT_BACKENDS and name not in seen:
            seen.append(name)
    return seen


def clamp_max_results(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS
    return min(max(count, 1), MAX_RESULTS_CAP)


def build_provider(
    provider: str,
    brave_api_key: str = "",
    searxng_url: str = "",
    ddgs_backends: str = "",
    transport: httpx.AsyncBaseTransport | None = None,
) -> SearchProvider:
    """Build a provider from explicit values (settings or a pending form)."""
    normalized = (provider or "").strip().lower() or "ddgs"
    if normalized not in SUPPORTED_PROVIDERS:
        raise WebSearchError(
            "invalid_arguments", f"Unknown web search provider {normalized!r}."
        )
    if normalized == "brave":
        return BraveProvider(brave_api_key, transport=transport)
    if normalized == "searxng":
        return SearXNGProvider(searxng_url, transport=transport)
    if normalized == "duckduckgo":
        return DdgsProvider("duckduckgo", "duckduckgo")
    if normalized == "ddgs_custom":
        backends = normalize_ddgs_backends(ddgs_backends)
        if not backends:
            raise WebSearchError(
                "missing_api_key",
                "No DDGS search engines are selected. "
                f"Pick at least one in {_SETTINGS_HINT}.",
                True,
            )
        return DdgsProvider("ddgs_custom", ",".join(backends))
    # "ddgs" total: every engine ddgs knows, duckduckgo included.
    return DdgsProvider("ddgs", "auto")


async def _attach_page_contents(
    results: list[dict[str, Any]],
    transport: httpx.AsyncBaseTransport | None,
    max_chars: int,
    truncate: bool,
) -> None:
    """Full-content mode: fetch each result page and inline its text."""

    async def fill(result: dict[str, Any]) -> None:
        payload = await run_fetch_url(
            result["url"], transport=transport, max_chars=max_chars,
            truncate=truncate,
        )
        if payload.get("ok"):
            result["content"] = payload["content"]
            result["content_truncated"] = payload["truncated"]
        else:
            result["content_error"] = payload["error"]["message"]

    await asyncio.gather(*(fill(result) for result in results))


async def _search_with_provider(
    provider_name: str,
    query: str,
    brave_api_key: str,
    searxng_url: str,
    ddgs_backends: str,
    max_results: int,
    content_mode: str,
    content_max_chars: int,
    content_truncate: bool,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    trimmed = (query or "").strip()[:MAX_QUERY_CHARS]
    if not trimmed:
        return failure_payload(
            "invalid_arguments", "web_search needs a non-empty 'query' string."
        )
    try:
        provider = build_provider(
            provider_name,
            brave_api_key=brave_api_key,
            searxng_url=searxng_url,
            ddgs_backends=ddgs_backends,
            transport=transport,
        )
        rows = await provider.search(trimmed, max_results)
    except WebSearchError as exc:
        return failure_payload(exc.code, str(exc), exc.needs_user_action)
    except Exception:
        logger.exception("web_search failed unexpectedly")
        return failure_payload(
            "unexpected_failure", "web_search hit an unexpected server error."
        )
    results: list[dict[str, Any]] = []
    for row in rows:
        cleaned = sanitize_result(
            row.get("title"), row.get("url"), row.get("snippet")
        )
        if cleaned is not None:
            results.append(cleaned)
        if len(results) >= max_results:
            break
    if content_mode == "full" and results:
        try:
            await _attach_page_contents(
                results, transport, content_max_chars, content_truncate
            )
        except Exception:
            # Content enrichment must never sink the search itself.
            logger.exception("web_search full-content fetch failed")
    return {"ok": True, "provider": provider.name, "results": results}


async def run_web_search(
    query: str,
    integrations: Any,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Search with the provider and options configured in IntegrationSettings."""
    return await _search_with_provider(
        getattr(integrations, "web_search_provider", ""),
        query,
        getattr(integrations, "web_search_brave_api_key", ""),
        getattr(integrations, "web_search_searxng_url", ""),
        getattr(integrations, "web_search_ddgs_backends", ""),
        clamp_max_results(getattr(integrations, "web_search_max_results", None)),
        getattr(integrations, "web_search_content_mode", "snippet"),
        fetch_content_chars(integrations),
        bool(getattr(integrations, "web_search_content_truncate", True)),
        transport,
    )


def fetch_content_chars(integrations: Any) -> int:
    try:
        chars = int(getattr(integrations, "web_search_content_max_chars", 0))
    except (TypeError, ValueError):
        chars = 0
    return chars if chars > 0 else DEFAULT_FETCH_CONTENT_CHARS


async def run_web_search_test(
    provider: str,
    brave_api_key: str = "",
    searxng_url: str = "",
    ddgs_backends: str = "",
    max_results: int = DEFAULT_MAX_RESULTS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Validate pending settings-form values with one real search.

    Always runs in snippet mode, but honors the pending result count. The test
    checks connectivity and credentials without fetching full page content.
    """
    return await _search_with_provider(
        provider,
        "omlx web search connectivity check",
        brave_api_key,
        searxng_url,
        ddgs_backends,
        clamp_max_results(max_results),
        "snippet",
        DEFAULT_FETCH_CONTENT_CHARS,
        True,
        transport,
    )


async def _resolve_host(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def _assert_public_host(host: str) -> None:
    """SSRF guard: refuse hosts that resolve to non-global addresses.

    Blocks loopback, RFC1918, link-local (169.254.169.254 metadata),
    IPv6 ULA, and IPv4-mapped variants so the fetch tool cannot be
    pointed at localhost admin endpoints or LAN services. The window
    between this resolution and httpx's own connect is accepted for a
    local single-user server (no DNS rebinding defense).
    """
    try:
        addresses = await _resolve_host(host)
    except (socket.gaierror, OSError) as exc:
        raise WebSearchError(
            "request_failed", f"Could not resolve host {host!r}."
        ) from exc
    if not addresses:
        raise WebSearchError("request_failed", f"Could not resolve host {host!r}.")
    for raw in addresses:
        try:
            ip = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError as exc:
            raise WebSearchError(
                "request_failed", f"Could not resolve host {host!r}."
            ) from exc
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
        if not ip.is_global:
            raise WebSearchError(
                "invalid_arguments",
                "Fetching private or local network addresses is not allowed.",
            )


def _validate_fetch_url(url: str) -> urllib.parse.SplitResult:
    candidate = (url or "").strip()
    if not candidate or len(candidate) > MAX_URL_CHARS:
        raise WebSearchError(
            "invalid_arguments",
            f"fetch_url needs an http(s) URL up to {MAX_URL_CHARS} characters.",
        )
    try:
        parts = urllib.parse.urlsplit(candidate)
    except ValueError as exc:
        raise WebSearchError(
            "invalid_arguments", "fetch_url could not parse the URL."
        ) from exc
    if parts.scheme not in ("http", "https"):
        raise WebSearchError(
            "invalid_arguments", "Only http and https URLs can be fetched."
        )
    if not parts.hostname:
        raise WebSearchError("invalid_arguments", "The URL has no host.")
    if parts.username or parts.password:
        raise WebSearchError(
            "invalid_arguments", "URLs with embedded credentials are not allowed."
        )
    return parts


async def run_fetch_url(
    url: str,
    transport: httpx.AsyncBaseTransport | None = None,
    max_chars: int = DEFAULT_FETCH_CONTENT_CHARS,
    truncate: bool = True,
) -> dict[str, Any]:
    """Fetch one public page and return it as markdown/text.

    Character truncation follows the caller's settings; the 2 MB raw
    body cap always applies regardless.
    """
    content_type = ""
    body = b""
    body_truncated = False
    current = (url or "").strip()
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            transport=transport,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            # Redirects are followed by hand so every hop goes through
            # URL validation and the SSRF guard again.
            for _ in range(MAX_REDIRECTS + 1):
                parts = _validate_fetch_url(current)
                await _assert_public_host(parts.hostname)
                async with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise WebSearchError(
                                "invalid_response",
                                "The server redirected without a Location header.",
                            )
                        current = str(httpx.URL(current).join(location))
                        continue
                    if response.status_code != 200:
                        raise _error_from_status(
                            response.status_code, "The web server"
                        )
                    content_type = (
                        (response.headers.get("content-type") or "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type and content_type not in _FETCHABLE_CONTENT_TYPES:
                        raise WebSearchError(
                            "invalid_response",
                            f"Content type {content_type!r} is not supported; "
                            "only HTML and plain text pages can be fetched.",
                        )
                    chunks = bytearray()
                    async for chunk in response.aiter_bytes():
                        chunks.extend(chunk)
                        if len(chunks) >= MAX_RESPONSE_BYTES:
                            body_truncated = True
                            break
                    body = bytes(chunks[:MAX_RESPONSE_BYTES])
                    break
            else:
                raise WebSearchError(
                    "request_failed", f"Gave up after {MAX_REDIRECTS} redirects."
                )
    except WebSearchError as exc:
        return failure_payload(exc.code, str(exc), exc.needs_user_action)
    except httpx.TimeoutException:
        return failure_payload(
            "request_failed",
            f"The page did not respond within {int(HTTP_TIMEOUT_SECONDS)} seconds.",
        )
    except httpx.HTTPError as exc:
        return failure_payload("request_failed", f"Fetching the page failed: {exc}")
    except Exception:
        logger.exception("fetch_url failed unexpectedly")
        return failure_payload(
            "unexpected_failure", "fetch_url hit an unexpected server error."
        )

    if content_type in _HTML_CONTENT_TYPES:
        try:
            text = await asyncio.to_thread(convert_html_to_markdown, body, current)
        except Exception:
            logger.exception("fetch_url markdown conversion failed")
            return failure_payload(
                "invalid_response",
                "The page could not be converted to readable text.",
            )
    else:
        text = body.decode("utf-8", errors="replace")

    if truncate and max_chars > 0:
        truncated = body_truncated or len(text) > max_chars
        text = text[:max_chars]
    else:
        truncated = body_truncated
    return {
        "ok": True,
        "url": current,
        "content": text,
        "truncated": truncated,
    }
