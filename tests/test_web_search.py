# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/websearch.py and omlx/api/websearch_routes.py.

Covers result sanitization, provider adapters (ddgs is monkeypatched,
brave/searxng use httpx.MockTransport), backend selection for the ddgs
providers, the payload contract of run_web_search / run_fetch_url
(including full-content mode and configurable truncation), the SSRF
guard, and the /v1/web HTTP layer.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import omlx.websearch as websearch
from omlx.api import websearch_routes
from omlx.settings import IntegrationSettings
from omlx.websearch import (
    BraveProvider,
    SearXNGProvider,
    WebSearchError,
    build_provider,
    clamp_max_results,
    normalize_ddgs_backends,
    run_fetch_url,
    run_web_search,
    run_web_search_test,
    sanitize_result,
)

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address so the SSRF guard passes."""

    async def resolve(host):
        return [PUBLIC_IP]

    monkeypatch.setattr(websearch, "_resolve_host", resolve)


def make_integrations(provider="ddgs", brave_key="", searxng_url="", **overrides):
    return IntegrationSettings(
        web_search_provider=provider,
        web_search_brave_api_key=brave_key,
        web_search_searxng_url=searxng_url,
        **overrides,
    )


def fake_rows(count=1):
    return [
        {"title": f"T{i}", "href": f"https://example.com/{i}", "body": f"B{i}"}
        for i in range(count)
    ]


class TestSanitizeResult:
    def test_valid_result_passes_through(self):
        result = sanitize_result("MLX", "https://example.com/mlx", "snippet text")
        assert result == {
            "title": "MLX",
            "url": "https://example.com/mlx",
            "snippet": "snippet text",
        }

    def test_non_http_schemes_dropped(self):
        for url in (
            "javascript:alert(1)",
            "file:///etc/passwd",
            "data:text/html,hi",
            "ftp://example.com/x",
        ):
            assert sanitize_result("t", url, "s") is None

    def test_missing_or_bad_url_dropped(self):
        assert sanitize_result("t", None, "s") is None
        assert sanitize_result("t", "", "s") is None
        assert sanitize_result("t", "https://", "s") is None
        assert sanitize_result("t", 123, "s") is None

    def test_overlong_url_dropped(self):
        url = "https://example.com/" + "a" * websearch.MAX_URL_CHARS
        assert sanitize_result("t", url, "s") is None

    def test_embedded_credentials_dropped(self):
        assert sanitize_result("t", "https://user:pw@example.com/", "s") is None
        assert sanitize_result("t", "https://token@example.com/", "s") is None

    def test_title_falls_back_to_host(self):
        result = sanitize_result("", "https://example.com/x", "s")
        assert result["title"] == "example.com"

    def test_title_and_snippet_truncated(self):
        result = sanitize_result("T" * 500, "https://example.com/", "S" * 2000)
        assert len(result["title"]) == websearch.MAX_TITLE_CHARS
        assert len(result["snippet"]) == websearch.MAX_SNIPPET_CHARS


class TestSettingHelpers:
    def test_normalize_ddgs_backends(self):
        assert normalize_ddgs_backends("") == []
        assert normalize_ddgs_backends("brave, yahoo") == ["brave", "yahoo"]
        assert normalize_ddgs_backends("Brave,BRAVE,unknown") == ["brave"]

    def test_clamp_max_results(self):
        assert clamp_max_results(None) == websearch.DEFAULT_MAX_RESULTS
        assert clamp_max_results("x") == websearch.DEFAULT_MAX_RESULTS
        assert clamp_max_results(0) == 1
        assert clamp_max_results(99) == websearch.MAX_RESULTS_CAP
        assert clamp_max_results(5) == 5


class TestBuildProvider:
    def test_default_is_ddgs_auto(self):
        provider = build_provider("")
        assert provider.name == "ddgs"
        assert provider._backend == "auto"
        assert build_provider(None).name == "ddgs"

    def test_duckduckgo_is_strict_backend(self):
        provider = build_provider("duckduckgo")
        assert provider.name == "duckduckgo"
        assert provider._backend == "duckduckgo"

    def test_ddgs_custom_uses_selected_backends(self):
        provider = build_provider("ddgs_custom", ddgs_backends="yahoo, mojeek")
        assert provider.name == "ddgs_custom"
        assert provider._backend == "yahoo,mojeek"

    def test_ddgs_custom_without_backends_raises(self):
        with pytest.raises(WebSearchError) as exc_info:
            build_provider("ddgs_custom", ddgs_backends="")
        assert exc_info.value.code == "missing_api_key"
        assert exc_info.value.needs_user_action is True

    def test_known_providers(self):
        assert build_provider("brave", brave_api_key="k").name == "brave"
        assert build_provider("searxng", searxng_url="http://x").name == "searxng"

    def test_unknown_provider_raises(self):
        with pytest.raises(WebSearchError) as exc_info:
            build_provider("bing")
        assert exc_info.value.code == "invalid_arguments"


class TestDdgsProvider:
    async def test_rows_are_remapped_and_backend_passed(self, monkeypatch):
        seen = {}

        def fake_text(query, max_results, backend):
            seen.update(query=query, max_results=max_results, backend=backend)
            return fake_rows(1)

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        rows = await build_provider("duckduckgo").search("mlx", 4)
        assert seen == {"query": "mlx", "max_results": 4, "backend": "duckduckgo"}
        assert rows == [
            {"title": "T0", "url": "https://example.com/0", "snippet": "B0"}
        ]

    async def test_auto_backend_for_total(self, monkeypatch):
        seen = {}

        def fake_text(query, max_results, backend):
            seen["backend"] = backend
            return []

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        await build_provider("ddgs").search("mlx", 3)
        assert seen["backend"] == "auto"

    async def test_no_results_exception_is_empty_list(self, monkeypatch):
        from ddgs.exceptions import DDGSException

        def fake_text(query, max_results, backend):
            raise DDGSException("No results found.")

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        assert await build_provider("ddgs").search("mlx", 3) == []

    async def test_ratelimit_maps_to_rate_limited(self, monkeypatch):
        from ddgs.exceptions import RatelimitException

        def fake_text(query, max_results, backend):
            raise RatelimitException("429")

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        with pytest.raises(WebSearchError) as exc_info:
            await build_provider("ddgs").search("mlx", 3)
        assert exc_info.value.code == "rate_limited"

    async def test_ddgs_exception_maps_to_provider_unavailable(self, monkeypatch):
        from ddgs.exceptions import DDGSException

        def fake_text(query, max_results, backend):
            raise DDGSException("boom")

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        with pytest.raises(WebSearchError) as exc_info:
            await build_provider("ddgs").search("mlx", 3)
        assert exc_info.value.code == "provider_unavailable"


class TestBraveProvider:
    def _transport(self, handler):
        return httpx.MockTransport(handler)

    async def test_request_shape_and_parsing(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["token"] = request.headers.get("X-Subscription-Token")
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "T",
                                "url": "https://example.com/",
                                "description": "D",
                            }
                        ]
                    }
                },
            )

        provider = BraveProvider("key123", transport=self._transport(handler))
        rows = await provider.search("mlx", 5)
        assert seen["token"] == "key123"
        assert "q=mlx" in seen["url"]
        assert "count=5" in seen["url"]
        assert rows == [
            {"title": "T", "url": "https://example.com/", "snippet": "D"}
        ]

    async def test_empty_key_fails_without_request(self):
        def handler(request):
            raise AssertionError("no request expected without a key")

        provider = BraveProvider("", transport=self._transport(handler))
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search("mlx", 3)
        assert exc_info.value.code == "missing_api_key"
        assert exc_info.value.needs_user_action is True

    @pytest.mark.parametrize(
        "status,code",
        [
            (401, "invalid_authentication"),
            (402, "insufficient_funds"),
            (403, "plan_access"),
            (429, "rate_limited"),
            (500, "provider_unavailable"),
            (418, "request_failed"),
        ],
    )
    async def test_http_status_mapping(self, status, code):
        provider = BraveProvider(
            "k", transport=self._transport(lambda request: httpx.Response(status))
        )
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search("mlx", 3)
        assert exc_info.value.code == code

    async def test_malformed_json_maps_to_invalid_response(self):
        provider = BraveProvider(
            "k",
            transport=self._transport(
                lambda request: httpx.Response(200, text="not json")
            ),
        )
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search("mlx", 3)
        assert exc_info.value.code == "invalid_response"


class TestSearXNGProvider:
    async def test_request_shape_and_parsing(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "T",
                            "url": "https://example.com/",
                            "content": "C",
                        }
                    ]
                },
            )

        provider = SearXNGProvider(
            "http://searx.local:8080/", transport=httpx.MockTransport(handler)
        )
        rows = await provider.search("mlx", 3)
        assert seen["url"].startswith("http://searx.local:8080/search?")
        assert "format=json" in seen["url"]
        assert rows == [
            {"title": "T", "url": "https://example.com/", "snippet": "C"}
        ]

    async def test_empty_url_fails(self):
        provider = SearXNGProvider("")
        with pytest.raises(WebSearchError) as exc_info:
            await provider.search("mlx", 3)
        assert exc_info.value.code == "missing_api_key"


class TestRunWebSearch:
    async def test_blank_query_is_payload_error(self):
        payload = await run_web_search("   ", make_integrations())
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_arguments"

    async def test_success_payload_shape(self, monkeypatch):
        monkeypatch.setattr(
            websearch,
            "_ddgs_text",
            lambda query, max_results, backend: [
                {"title": "T", "href": "https://example.com/", "body": "B"},
                {"title": "bad", "href": "javascript:x", "body": "dropped"},
            ],
        )
        payload = await run_web_search("mlx", make_integrations())
        assert payload["ok"] is True
        assert payload["provider"] == "ddgs"
        assert [r["url"] for r in payload["results"]] == ["https://example.com/"]
        assert "content" not in payload["results"][0]

    async def test_provider_error_becomes_failure_payload(self):
        payload = await run_web_search("mlx", make_integrations(provider="brave"))
        assert payload["ok"] is False
        assert payload["error"]["code"] == "missing_api_key"
        assert payload["error"]["user_action_required"] is True

    async def test_unexpected_error_is_caught(self, monkeypatch):
        # Raise from the provider itself so the error bypasses
        # _map_ddgs_error and lands in the generic handler.
        async def broken_search(self, query, max_results):
            raise RuntimeError("boom")

        monkeypatch.setattr(websearch.DdgsProvider, "search", broken_search)
        payload = await run_web_search("mlx", make_integrations())
        assert payload["ok"] is False
        assert payload["error"]["code"] == "unexpected_failure"

    async def test_max_results_setting_caps_results(self, monkeypatch):
        monkeypatch.setattr(
            websearch, "_ddgs_text", lambda q, n, b: fake_rows(10)
        )
        payload = await run_web_search(
            "mlx", make_integrations(web_search_max_results=5)
        )
        assert len(payload["results"]) == 5

    async def test_max_results_setting_clamped(self, monkeypatch):
        seen = {}

        def fake_text(query, max_results, backend):
            seen["max_results"] = max_results
            return []

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        await run_web_search("mlx", make_integrations(web_search_max_results=99))
        assert seen["max_results"] == websearch.MAX_RESULTS_CAP

    async def test_query_is_truncated(self, monkeypatch):
        seen = {}

        def fake_text(query, max_results, backend):
            seen["query"] = query
            return []

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        await run_web_search("q" * 1000, make_integrations())
        assert len(seen["query"]) == websearch.MAX_QUERY_CHARS

    async def test_full_mode_attaches_page_content(self, monkeypatch):
        monkeypatch.setattr(
            websearch, "_ddgs_text", lambda q, n, b: fake_rows(2)
        )
        fetched = []

        async def fake_fetch(url, transport=None, max_chars=0, truncate=True):
            fetched.append((url, max_chars, truncate))
            if url.endswith("/1"):
                return websearch.failure_payload("request_failed", "dead link")
            return {"ok": True, "url": url, "content": "PAGE", "truncated": True}

        monkeypatch.setattr(websearch, "run_fetch_url", fake_fetch)
        payload = await run_web_search(
            "mlx",
            make_integrations(
                web_search_content_mode="full",
                web_search_content_max_chars=1234,
                web_search_content_truncate=False,
            ),
        )
        assert payload["ok"] is True
        assert payload["results"][0]["content"] == "PAGE"
        assert payload["results"][0]["content_truncated"] is True
        assert payload["results"][1]["content_error"] == "dead link"
        assert all(m == 1234 and t is False for _, m, t in fetched)

    async def test_run_web_search_test_uses_pending_values(self):
        payload = await run_web_search_test("brave", brave_api_key="")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "missing_api_key"

    async def test_run_web_search_test_custom_without_backends(self):
        payload = await run_web_search_test("ddgs_custom", ddgs_backends="")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "missing_api_key"

    async def test_run_web_search_test_uses_pending_max_results(self, monkeypatch):
        monkeypatch.setattr(
            websearch, "_ddgs_text", lambda query, max_results, backend: fake_rows(10)
        )
        payload = await run_web_search_test("duckduckgo", max_results=7)
        assert len(payload["results"]) == 7


class TestSsrfGuard:
    @pytest.mark.parametrize(
        "address",
        [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.5",
            "172.16.0.9",
            "169.254.169.254",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fd00::1",
            "::ffff:10.0.0.1",
        ],
    )
    async def test_non_global_addresses_blocked(self, monkeypatch, address):
        async def resolve(host):
            return [address]

        monkeypatch.setattr(websearch, "_resolve_host", resolve)
        with pytest.raises(WebSearchError):
            await websearch._assert_public_host("example.com")

    async def test_public_address_allowed(self, monkeypatch):
        async def resolve(host):
            return [PUBLIC_IP]

        monkeypatch.setattr(websearch, "_resolve_host", resolve)
        await websearch._assert_public_host("example.com")

    async def test_unresolvable_host_is_request_failed(self, monkeypatch):
        import socket

        async def resolve(host):
            raise socket.gaierror("nope")

        monkeypatch.setattr(websearch, "_resolve_host", resolve)
        with pytest.raises(WebSearchError) as exc_info:
            await websearch._assert_public_host("nope.invalid")
        assert exc_info.value.code == "request_failed"


class TestFetchUrl:
    async def test_html_is_converted(self, monkeypatch, public_dns):
        monkeypatch.setattr(
            websearch,
            "convert_html_to_markdown",
            lambda data, url=None: "converted markdown",
        )
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=b"<p>hi</p>"
            )
        )
        payload = await run_fetch_url("https://example.com/", transport=transport)
        assert payload == {
            "ok": True,
            "url": "https://example.com/",
            "content": "converted markdown",
            "truncated": False,
        }

    async def test_real_markdown_conversion(self, public_dns):
        html = b"<html><body><h1>Title</h1><p>Body text.</p></body></html>"
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/html"}, content=html
            )
        )
        payload = await run_fetch_url("https://example.com/", transport=transport)
        assert payload["ok"] is True
        assert "Title" in payload["content"]
        assert "Body text." in payload["content"]

    async def test_plain_text_passthrough(self, public_dns):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"raw text"
            )
        )
        payload = await run_fetch_url("https://example.com/x", transport=transport)
        assert payload["content"] == "raw text"

    async def test_redirect_chain_is_followed(self, public_dns, monkeypatch):
        monkeypatch.setattr(
            websearch, "convert_html_to_markdown", lambda data, url=None: "ok"
        )
        calls = []

        def handler(request):
            calls.append(str(request.url))
            if len(calls) == 1:
                return httpx.Response(
                    302, headers={"location": "https://example.com/final"}
                )
            return httpx.Response(
                200, headers={"content-type": "text/html"}, content=b"<p>x</p>"
            )

        payload = await run_fetch_url(
            "https://example.com/start", transport=httpx.MockTransport(handler)
        )
        assert calls == [
            "https://example.com/start",
            "https://example.com/final",
        ]
        assert payload["ok"] is True
        assert payload["url"] == "https://example.com/final"

    async def test_too_many_redirects(self, public_dns):
        def handler(request):
            return httpx.Response(
                301, headers={"location": "https://example.com/loop"}
            )

        payload = await run_fetch_url(
            "https://example.com/", transport=httpx.MockTransport(handler)
        )
        assert payload["ok"] is False
        assert payload["error"]["code"] == "request_failed"

    async def test_redirect_to_private_address_blocked(self, monkeypatch):
        resolutions = {"example.com": PUBLIC_IP, "internal.lan": "10.0.0.5"}

        async def resolve(host):
            return [resolutions[host]]

        monkeypatch.setattr(websearch, "_resolve_host", resolve)

        def handler(request):
            return httpx.Response(
                302, headers={"location": "http://internal.lan/admin"}
            )

        payload = await run_fetch_url(
            "https://example.com/", transport=httpx.MockTransport(handler)
        )
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_arguments"

    async def test_direct_private_url_blocked(self, monkeypatch):
        async def resolve(host):
            return ["169.254.169.254"]

        monkeypatch.setattr(websearch, "_resolve_host", resolve)
        payload = await run_fetch_url("http://169.254.169.254/latest/meta-data")
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_arguments"

    async def test_unsupported_content_type_rejected(self, public_dns):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "image/png"}, content=b"\x89PNG"
            )
        )
        payload = await run_fetch_url("https://example.com/a.png", transport=transport)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_response"

    async def test_oversized_body_is_truncated(self, public_dns):
        big = b"x" * (websearch.MAX_RESPONSE_BYTES + 1024)
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=big
            )
        )
        payload = await run_fetch_url("https://example.com/", transport=transport)
        assert payload["ok"] is True
        assert payload["truncated"] is True
        assert len(payload["content"]) == websearch.DEFAULT_FETCH_CONTENT_CHARS

    async def test_char_budget_is_configurable(self, public_dns):
        text = b"y" * 500
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=text
            )
        )
        payload = await run_fetch_url(
            "https://example.com/", transport=transport, max_chars=100
        )
        assert payload["truncated"] is True
        assert len(payload["content"]) == 100

    async def test_truncation_can_be_disabled(self, public_dns):
        text = b"y" * (websearch.DEFAULT_FETCH_CONTENT_CHARS + 500)
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=text
            )
        )
        payload = await run_fetch_url(
            "https://example.com/", transport=transport, truncate=False
        )
        assert payload["truncated"] is False
        assert len(payload["content"]) == len(text)

    async def test_invalid_url_rejected(self):
        for url in ("", "ftp://example.com/", "https://user:pw@example.com/"):
            payload = await run_fetch_url(url)
            assert payload["ok"] is False
            assert payload["error"]["code"] == "invalid_arguments"

    async def test_non_200_status_mapped(self, public_dns):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(404, content=b"")
        )
        payload = await run_fetch_url("https://example.com/", transport=transport)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "request_failed"


class TestWebRoutes:
    @pytest.fixture
    def app_client(self):
        """TestClient mounting only the /v1/web router."""
        app = FastAPI()
        app.include_router(websearch_routes.router)
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def reset_settings_getter(self):
        original = websearch_routes._get_global_settings
        websearch_routes._get_global_settings = None
        yield
        websearch_routes._get_global_settings = original

    def _install_settings(self, integrations):
        class Settings:
            pass

        settings = Settings()
        settings.integrations = integrations
        websearch_routes.set_global_settings_getter(lambda: settings)

    def test_search_success(self, app_client, monkeypatch):
        monkeypatch.setattr(
            websearch,
            "_ddgs_text",
            lambda query, max_results, backend: [
                {"title": "T", "href": "https://example.com/", "body": "B"}
            ],
        )
        self._install_settings(make_integrations())
        response = app_client.post("/v1/web/search", json={"query": "mlx"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["results"][0]["url"] == "https://example.com/"

    def test_search_blank_query_is_payload_error(self, app_client):
        self._install_settings(make_integrations())
        response = app_client.post("/v1/web/search", json={})
        assert response.status_code == 200
        assert response.json()["error"]["code"] == "invalid_arguments"

    def test_search_without_settings_getter(self, app_client):
        response = app_client.post("/v1/web/search", json={"query": "mlx"})
        assert response.status_code == 200
        assert response.json()["error"]["code"] == "unexpected_failure"

    def test_fetch_invalid_url_is_payload_error(self, app_client):
        response = app_client.post("/v1/web/fetch", json={"url": "ftp://x/"})
        assert response.status_code == 200
        assert response.json()["error"]["code"] == "invalid_arguments"

    def test_fetch_honors_truncation_settings(self, app_client, monkeypatch):
        seen = {}

        async def fake_fetch(url, transport=None, max_chars=0, truncate=True):
            seen.update(max_chars=max_chars, truncate=truncate)
            return {"ok": True, "url": url, "content": "", "truncated": False}

        monkeypatch.setattr(websearch_routes, "run_fetch_url", fake_fetch)
        self._install_settings(
            make_integrations(
                web_search_content_max_chars=777,
                web_search_content_truncate=False,
            )
        )
        response = app_client.post(
            "/v1/web/fetch", json={"url": "https://example.com/"}
        )
        assert response.status_code == 200
        assert seen == {"max_chars": 777, "truncate": False}

    def test_fetch_private_address_blocked(self, app_client, monkeypatch):
        async def resolve(host):
            return ["127.0.0.1"]

        monkeypatch.setattr(websearch, "_resolve_host", resolve)
        response = app_client.post(
            "/v1/web/fetch", json={"url": "http://localhost:8000/admin"}
        )
        assert response.status_code == 200
        assert response.json()["error"]["code"] == "invalid_arguments"


class TestAdminWebSearchTest:
    def _client(self):
        from omlx.admin import routes as admin_routes
        from omlx.admin.auth import require_admin

        app = FastAPI()
        app.include_router(admin_routes.router)
        app.dependency_overrides[require_admin] = lambda: True
        return TestClient(app)

    def test_pending_values_are_used_and_not_saved(self, monkeypatch):
        seen = {}

        def fake_text(query, max_results, backend):
            seen["query"] = query
            seen["backend"] = backend
            seen["max_results"] = max_results
            return [{"title": "T", "href": "https://example.com/", "body": "B"}]

        monkeypatch.setattr(websearch, "_ddgs_text", fake_text)
        response = self._client().post(
            "/admin/api/web-search/test",
            json={
                "provider": "ddgs_custom",
                "ddgs_backends": "yahoo,mojeek",
                "max_results": 7,
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert seen["backend"] == "yahoo,mojeek"
        assert seen["max_results"] == 7

    def test_dashboard_posts_pending_max_results(self):
        root = Path(__file__).resolve().parents[1]
        javascript = (root / "omlx/admin/static/js/dashboard.js").read_text()
        test_method = javascript.split("async testWebSearch()", 1)[1].split(
            "async saveLanguage", 1
        )[0]
        assert (
            "max_results: this.globalSettings.integrations.web_search_max_results"
            in test_method
        )

    def test_failure_payload_passes_through(self):
        response = self._client().post(
            "/admin/api/web-search/test",
            json={"provider": "brave", "brave_api_key": ""},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is False
        assert payload["error"]["code"] == "missing_api_key"
