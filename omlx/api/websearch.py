# SPDX-License-Identifier: Apache-2.0
"""
Built-in web search for oMLX.

This module turns a user query into clean, LLM-friendly context by:
  1. Searching the web with a pluggable provider (DuckDuckGo-lite by default,
     AnySearch, or a self-hosted SearXNG instance).
  2. Fetching + extracting the top result pages into clean markdown
     (via the provider's own extractor when available, otherwise a local
     readability/lxml pass; Jina Reader is used as an optional fallback).
  3. Assembling a compact, citation-tagged context block that the chat route
     can inject ahead of the user's question.

Design goals:
  * Local-first: the default provider (DuckDuckGo HTML) needs no API key and
    no third-party AI service. SearXNG is fully self-hosted.
  * Cloud-optional: AnySearch works anonymously (rate-limited) with no key,
    and upgrades when an API key is supplied.
  * Cheap to call: every network call is bounded by a timeout; failures are
    caught and degrade gracefully (we never break the chat request).

This file must stay dependency-light: ``httpx`` and ``requests`` are already
in oMLX's dependencies. ``readability-lxml`` / ``lxml`` / ``trafilatura`` are
optional; if absent we fall back to a small built-in HTML-to-text scrubber.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public configuration object
# ---------------------------------------------------------------------------

# Provider identifiers accepted by the chat route's ``web_search`` field.
PROVIDER_DUCKDUCKGO = "duckduckgo"
PROVIDER_ANYSEARCH = "anysearch"
PROVIDER_SEARXNG = "searxng"

DEFAULT_PROVIDER = PROVIDER_DUCKDUCKGO
DEFAULT_TOP_K = 3
DEFAULT_MAX_CHARS_PER_PAGE = 4000
DEFAULT_SEARXNG_URL = "http://localhost:8080"

# AnySearch JSON-RPC endpoint (MCP-compatible). Anonymous access is allowed;
# an Authorization bearer token only raises the rate limit / quota.
ANYSEARCH_ENDPOINT = "https://api.anysearch.com/mcp"
ANYSEARCH_CLIENT_HEADER = "omlx/1.0"
# Jina Reader turns any URL into clean markdown. Optional cloud fallback.
JINA_READER_ENDPOINT = "https://r.jina.ai/"


@dataclass
class WebSearchConfig:
    """Per-request web search configuration, supplied by the chat UI."""

    enabled: bool = False
    provider: str = DEFAULT_PROVIDER
    api_key: str = ""
    searxng_url: str = DEFAULT_SEARXNG_URL
    top_k: int = DEFAULT_TOP_K
    max_chars_per_page: int = DEFAULT_MAX_CHARS_PER_PAGE
    # When True, also fetch+extract page bodies (more context, slower).
    fetch_pages: bool = True

    @classmethod
    def from_request(cls, raw: Any) -> "WebSearchConfig":
        """Build a config from the ``web_search`` field of a chat request.

        The field may be a bool (just "on/off") or a dict with overrides.
        """
        if raw is None or raw is False:
            return cls(enabled=False)
        if raw is True:
            return cls(enabled=True)
        if not isinstance(raw, dict):
            return cls(enabled=False)
        cfg = cls(enabled=bool(raw.get("enabled", True)))
        if "provider" in raw:
            cfg.provider = str(raw["provider"])
        if "api_key" in raw:
            cfg.api_key = str(raw["api_key"] or "")
        if "searxng_url" in raw:
            cfg.searxng_url = str(raw["searxng_url"] or DEFAULT_SEARXNG_URL)
        if "top_k" in raw:
            try:
                cfg.top_k = max(1, min(int(raw["top_k"]), 10))
            except (TypeError, ValueError):
                pass
        if "max_chars_per_page" in raw:
            try:
                cfg.max_chars_per_page = max(500, int(raw["max_chars_per_page"]))
            except (TypeError, ValueError):
                pass
        if "fetch_pages" in raw:
            cfg.fetch_pages = bool(raw["fetch_pages"])
        return cfg


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str = ""


@dataclass
class WebSearchResult:
    provider: str
    query: str
    hits: List[SearchHit] = field(default_factory=list)
    context: str = ""
    elapsed_ms: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


async def _search_duckduckgo(query: str, top_k: int, timeout: float) -> List[SearchHit]:
    """Anonymous DuckDuckGo HTML endpoint. No API key, no third-party AI."""
    url = "https://html.duckduckgo.com/html/"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.post(
            url,
            data={"q": query},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        body = resp.text

    hits: List[SearchHit] = []
    # DuckDuckGo HTML returns result blocks with class "result__a" (title+link)
    # and "result__snippet" (summary). They appear in document order.
    link_re = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.S)

    raw_links = link_re.findall(body)
    snippets = [re.sub("<[^>]+>", "", s) for s in snippet_re.findall(body)]

    for i, (href, title_html) in enumerate(raw_links[:top_k]):
        # DDG wraps the real URL in a 302 redirect; decode it.
        real_url = _decode_ddg_redirect(href)
        title = _strip_tags(title_html).strip()
        snippet = _strip_tags(snippets[i]).strip() if i < len(snippets) else ""
        if real_url:
            hits.append(SearchHit(title=title, url=real_url, snippet=snippet))
    return hits


def _decode_ddg_redirect(href: str) -> str:
    """DuckDuckGo HTML links are 302 redirects like
    /l/?uddg=<urlencoded>&rut=... — extract the real destination."""
    m = re.search(r"uddg=([^&]+)", href)
    if m:
        from urllib.parse import unquote

        return unquote(m.group(1))
    # Some links are already absolute.
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return ""


async def _search_searxng(
    query: str, top_k: int, searxng_url: str, timeout: float
) -> List[SearchHit]:
    """Self-hosted SearXNG JSON endpoint. Fully local aggregation."""
    base = searxng_url.rstrip("/")
    url = f"{base}/search"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(
            url,
            params={"q": query, "format": "json", "num_results": top_k},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

    hits: List[SearchHit] = []
    for item in (data.get("results") or [])[:top_k]:
        hits.append(
            SearchHit(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "") or "",
            )
        )
    return hits


async def _search_anysearch(
    query: str, top_k: int, api_key: str, timeout: float
) -> List[SearchHit]:
    """AnySearch JSON-RPC (MCP-compatible). Anonymous access is allowed."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search",
            "arguments": {"query": query, "max_results": min(top_k, 10)},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Anysearch-Client": ANYSEARCH_CLIENT_HEADER,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(ANYSEARCH_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        raise RuntimeError(data["error"].get("message", "AnySearch error"))

    text = ""
    for item in data.get("result", {}).get("content", []):
        if item.get("type") == "text":
            text = item.get("text", "")
            break

    return _parse_anysearch_markdown(text, top_k)


def _parse_anysearch_markdown(text: str, top_k: int) -> List[SearchHit]:
    """AnySearch returns markdown like:

    ### 1. Title
    - **URL**: https://...
    - snippet text
    """
    hits: List[SearchHit] = []
    # Split on "### N." headings.
    blocks = re.split(r"\n###\s+\d+\.\s*", text)
    for block in blocks[1:top_k + 1]:
        lines = block.splitlines()
        title = lines[0].strip() if lines else ""
        url = ""
        snippet_lines: List[str] = []
        for line in lines[1:]:
            um = re.search(r"\*\*URL\*\*:\s*(\S+)", line)
            if um:
                url = um.group(1).strip()
            elif line.strip().startswith("-") or line.strip():
                clean = re.sub(r"^[-*]\s*", "", line).strip()
                if clean:
                    snippet_lines.append(clean)
        if url:
            hits.append(
                SearchHit(
                    title=title,
                    url=url,
                    snippet=" ".join(snippet_lines)[:400],
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Page extraction (clean markdown for the model)
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _strip_tags(s: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", s)).strip()


async def _extract_page(
    url: str, max_chars: int, timeout: float, provider: str = "", api_key: str = ""
) -> str:
    """Return clean text for a URL.

    Order of preference:
      1. Provider-native extractor (AnySearch ``extract`` — cleanest, it is
         purpose-built for LLM consumption).
      2. Local readability/lxml if the user installed it (fully local).
      3. Jina Reader (cloud, but very clean markdown; needs a round trip).
      4. Built-in HTML scrubber (last resort, fully local).

    We never raise; on failure we return "".
    """
    text = ""
    if provider == PROVIDER_ANYSEARCH:
        text = await _extract_anysearch(url, api_key, timeout)
    if not text:
        text = _extract_with_readability(url, timeout)
    if not text:
        text = await _extract_with_jina(url, timeout)
    if not text:
        text = await _extract_raw(url, timeout)
    if text:
        text = _MD_LINK_RE.sub(r"\1", text)  # drop link targets, keep labels
        text = _WS_RE.sub(" ", text).strip()
    return text[:max_chars]


async def _extract_anysearch(url: str, api_key: str, timeout: float) -> str:
    """Use AnySearch's native ``extract`` tool — purpose-built clean markdown."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "extract", "arguments": {"url": url}},
    }
    headers = {
        "Content-Type": "application/json",
        "X-Anysearch-Client": ANYSEARCH_CLIENT_HEADER,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(ANYSEARCH_ENDPOINT, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        if "error" in data:
            return ""
        for item in data.get("result", {}).get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")
    except Exception as e:  # pragma: no cover
        logger.debug("AnySearch extract failed: %s", e)
    return ""


def _extract_with_readability(url: str, timeout: float) -> str:
    try:
        from readability import Document  # type: ignore
        import requests  # type: ignore

        resp = requests.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        doc = Document(resp.text)
        return _strip_tags(doc.summary())
    except Exception as e:  # pragma: no cover - optional dependency
        logger.debug("readability extraction unavailable: %s", e)
        return ""


async def _extract_with_jina(url: str, timeout: float) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(JINA_READER_ENDPOINT + url)
            if resp.status_code == 200:
                return resp.text
    except Exception as e:  # pragma: no cover
        logger.debug("Jina extraction failed: %s", e)
    return ""


async def _extract_raw(url: str, timeout: float) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            return _strip_tags(resp.text)
    except Exception as e:  # pragma: no cover
        logger.debug("raw extraction failed: %s", e)
    return ""


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

async def run_web_search(cfg: WebSearchConfig, query: str) -> WebSearchResult:
    """Search + (optionally) fetch pages, returning a ready-to-inject context."""
    start = time.perf_counter()
    result = WebSearchResult(provider=cfg.provider, query=query)
    if not query.strip():
        return result

    try:
        if cfg.provider == PROVIDER_DUCKDUCKGO:
            result.hits = await _search_duckduckgo(query, cfg.top_k, timeout=15.0)
        elif cfg.provider == PROVIDER_SEARXNG:
            result.hits = await _search_searxng(
                query, cfg.top_k, cfg.searxng_url, timeout=15.0
            )
        elif cfg.provider == PROVIDER_ANYSEARCH:
            result.hits = await _search_anysearch(
                query, cfg.top_k, cfg.api_key, timeout=30.0
            )
        else:
            logger.warning("Unknown web_search provider %r; falling back to DuckDuckGo", cfg.provider)
            result.hits = await _search_duckduckgo(query, cfg.top_k, timeout=15.0)
    except Exception as e:
        result.error = f"search failed: {e}"
        logger.warning("Web search error: %s", e)
        result.elapsed_ms = int((time.perf_counter() - start) * 1000)
        return result

    if not result.hits:
        if cfg.provider == PROVIDER_DUCKDUCKGO:
            result.error = ("DuckDuckGo 暂不可用（返回空结果），请在设置中改用 AnySearch 或 SearXNG。")
        else:
            result.error = f"{cfg.provider} 搜索未返回结果，请检查配置或换用其他搜索方案。"
        logger.warning("Web search returned no results for provider %r", cfg.provider)
        result.elapsed_ms = int((time.perf_counter() - start) * 1000)
        return result

    if cfg.fetch_pages and result.hits:
        try:
            # Fetch page bodies concurrently instead of one-by-one so the
            # wall-clock cost is bounded by the slowest page, not the sum.
            async def _fetch_one(hit):
                body = await _extract_page(
                    hit.url, cfg.max_chars_per_page, timeout=20.0,
                    provider=cfg.provider, api_key=cfg.api_key,
                )
                return (hit, body) if body else None

            pages = [p for p in await asyncio.gather(
                *[_fetch_one(h) for h in result.hits]
            ) if p is not None]
            result.context = _assemble_context(result.hits, pages)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Page fetch failed, using snippets only: %s", e)
            result.context = _assemble_context(result.hits, [])
    else:
        result.context = _assemble_context(result.hits, [])

    result.elapsed_ms = int((time.perf_counter() - start) * 1000)
    return result


def _assemble_context(
    hits: List[SearchHit], pages: List[tuple[SearchHit, str]]
) -> str:
    """Build a compact, citation-tagged context block for the model."""
    if not hits:
        return ""
    lines = [
        "以下是联网检索到的参考信息（已整理为适合阅读的内容，附带来源链接）：",
        "",
    ]
    page_map = {h.url: body for h, body in pages}
    for i, hit in enumerate(hits, 1):
        lines.append(f"[{i}] {hit.title}")
        lines.append(f"来源: {hit.url}")
        body = page_map.get(hit.url)
        if body:
            lines.append(body)
        elif hit.snippet:
            lines.append(hit.snippet)
        lines.append("")
    return "\n".join(lines).strip()


def build_web_search_context_block(result: WebSearchResult) -> str:
    """Convenience wrapper used by the chat route to inject context."""
    return result.context
