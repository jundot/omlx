# SPDX-License-Identifier: Apache-2.0
"""Tests for Anthropic server-side WebSearch support."""

import base64
import json
from types import SimpleNamespace

from fastapi.responses import JSONResponse, StreamingResponse

from omlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicTool,
    ContentBlockText,
    ContentBlockUnknown,
    MessagesRequest,
)
from omlx.api.anthropic_web_search import (
    SEARCH_PROMPT_PREFIX,
    _map_error_code,
    _opaque_content,
    build_blocks,
    build_message,
    extract_query,
    filter_results,
    find_web_search_tool,
    maybe_handle_anthropic_web_search,
    stream_events,
)


def make_request(
    messages,
    tools=None,
    stream=False,
    model="test-model",
):
    return MessagesRequest(
        model=model,
        max_tokens=1024,
        messages=messages,
        tools=tools,
        stream=stream,
    )


def web_search_tool(**overrides):
    payload = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 8,
    }
    payload.update(overrides)
    return AnthropicTool(**payload)


def success_payload():
    return {
        "ok": True,
        "provider": "ddgs",
        "results": [
            {
                "title": "oMLX",
                "url": "https://example.com/omlx",
                "snippet": "oMLX is an MLX inference server.",
            },
            {
                "title": "Blocked",
                "url": "https://blocked.example.net/page",
                "snippet": "should be filtered",
            },
        ],
    }


class TestFindWebSearchTool:
    def test_finds_server_tool(self):
        request = make_request(
            [AnthropicMessage(role="user", content="hi")],
            tools=[web_search_tool()],
        )
        tool = find_web_search_tool(request)
        assert tool is not None
        assert tool.type == "web_search_20250305"

    def test_ignores_regular_tool(self):
        request = make_request(
            [AnthropicMessage(role="user", content="hi")],
            tools=[
                AnthropicTool(
                    name="Read",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
        )
        assert find_web_search_tool(request) is None

    def test_no_tools(self):
        request = make_request([AnthropicMessage(role="user", content="hi")])
        assert find_web_search_tool(request) is None

    def test_client_tool_with_empty_schema_is_not_server_tool(self):
        request = make_request(
            [AnthropicMessage(role="user", content="hi")],
            tools=[AnthropicTool(name="web_search", input_schema={})],
        )
        assert find_web_search_tool(request) is None


class TestExtractQuery:
    def test_string_content(self):
        request = make_request(
            [
                AnthropicMessage(
                    role="user",
                    content=f"{SEARCH_PROMPT_PREFIX} oMLX MLX server",
                )
            ]
        )
        assert extract_query(request) == "oMLX MLX server"

    def test_block_content(self):
        request = make_request(
            [
                AnthropicMessage(
                    role="user",
                    content=[ContentBlockText(text=f"{SEARCH_PROMPT_PREFIX} 中文查询")],
                )
            ]
        )
        assert extract_query(request) == "中文查询"

    def test_server_tool_use_fallback(self):
        request = make_request(
            [
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockUnknown(
                            type="server_tool_use",
                            name="web_search",
                            input={"query": "from tool block"},
                        )
                    ],
                )
            ]
        )
        assert extract_query(request) == "from tool block"

    def test_does_not_hijack_normal_chat(self):
        request = make_request(
            [AnthropicMessage(role="user", content="hello, how are you?")]
        )
        assert extract_query(request) == ""


class TestFilterResults:
    def test_allowed_and_blocked(self):
        results = [
            {"url": "https://example.com/a"},
            {"url": "https://sub.example.com/b"},
            {"url": "https://blocked.com/c"},
            {"url": "ftp://example.com/d"},
            {"url": "not a url"},
        ]
        filtered = filter_results(results, ["example.com"], ["blocked.com"])
        assert [item["url"] for item in filtered] == [
            "https://example.com/a",
            "https://sub.example.com/b",
        ]

    def test_no_domains_keeps_all_valid(self):
        results = [
            {"url": "https://a.example/x"},
            {"url": "https://b.example/y"},
        ]
        assert len(filter_results(results, None, None)) == 2

    def test_path_scoped_allowed_domain(self):
        results = [
            {"url": "https://example.com/blog/post"},
            {"url": "https://example.com/blog"},
            {"url": "https://example.com/blogger"},
            {"url": "https://example.com/other"},
        ]
        filtered = filter_results(results, ["example.com/blog"], [])
        assert [item["url"] for item in filtered] == [
            "https://example.com/blog/post",
            "https://example.com/blog",
        ]

    def test_path_scoped_blocked_domain(self):
        results = [
            {"url": "https://example.com/blog/post"},
            {"url": "https://example.com/other"},
        ]
        filtered = filter_results(results, [], ["example.com/blog"])
        assert [item["url"] for item in filtered] == [
            "https://example.com/other",
        ]


class TestBuildBlocks:
    def test_success(self):
        blocks = build_blocks(
            "oMLX",
            success_payload(),
            web_search_tool(allowed_domains=["example.com"]),
        )
        assert blocks["server_tool_use"]["type"] == "server_tool_use"
        assert blocks["server_tool_use"]["name"] == "web_search"
        assert blocks["server_tool_use"]["input"] == {"query": "oMLX"}
        assert blocks["result_block"]["type"] == "web_search_tool_result"
        assert blocks["result_block"]["tool_use_id"] == blocks["server_tool_use"]["id"]
        content = blocks["result_block"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "web_search_result"
        assert content[0]["url"] == "https://example.com/omlx"
        assert content[0]["encrypted_content"]
        assert "oMLX" in blocks["summary"]

    def test_provider_error(self):
        blocks = build_blocks(
            "oMLX",
            {
                "ok": False,
                "error": {"code": "rate_limited", "message": "slow down"},
            },
            web_search_tool(),
        )
        assert blocks["result_block"]["content"] == {
            "type": "web_search_tool_result_error",
            "error_code": "too_many_requests",
        }
        assert "slow down" in blocks["summary"]

    def test_error_codes_match_anthropic(self):
        assert (
            _map_error_code({"error": {"code": "invalid_query"}})
            == "invalid_tool_input"
        )
        assert (
            _map_error_code({"error": {"code": "invalid_arguments"}})
            == "invalid_tool_input"
        )
        assert (
            _map_error_code({"error": {"code": "query_too_long"}}) == "query_too_long"
        )
        assert (
            _map_error_code({"error": {"code": "request_too_large"}})
            == "request_too_large"
        )
        assert (
            _map_error_code({"error": {"code": "rate_limited"}}) == "too_many_requests"
        )

    def test_summary_keeps_full_content(self):
        content = "A" * 5000
        blocks = build_blocks(
            "oMLX",
            {
                "ok": True,
                "provider": "mock",
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                        "content": content,
                    }
                ],
            },
            web_search_tool(),
        )
        assert content in blocks["summary"]

    def test_opaque_content_excludes_page_body(self):
        encoded = _opaque_content(
            {
                "title": "Example",
                "url": "https://example.com",
                "snippet": "short",
                "content": "X" * 20000,
            }
        )
        decoded = base64.urlsafe_b64decode(encoded + "===").decode("utf-8")
        assert "example.com" in decoded
        assert "XXXX" not in decoded


class TestResponseBuilders:
    def test_build_message(self):
        blocks = build_blocks("oMLX", success_payload(), web_search_tool())
        message = build_message("test-model", blocks, 42)
        assert message["type"] == "message"
        assert message["role"] == "assistant"
        assert message["content"][0]["type"] == "server_tool_use"
        assert message["content"][1]["type"] == "web_search_tool_result"
        assert message["content"][2]["type"] == "text"
        assert message["usage"]["server_tool_use"]["web_search_requests"] == 1

    async def test_stream_events(self):
        blocks = build_blocks("oMLX", success_payload(), web_search_tool())
        events = [event async for event in stream_events("test-model", blocks, 42)]
        parsed = []
        for event in events:
            lines = event.strip().split("\n")
            event_type = lines[0].removeprefix("event: ")
            data = lines[1].removeprefix("data: ")
            parsed.append((event_type, json.loads(data)))

        assert [item[0] for item in parsed] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert parsed[1][1]["content_block"]["type"] == "server_tool_use"
        assert parsed[4][1]["content_block"]["type"] == "web_search_tool_result"
        assert (
            parsed[4][1]["content_block"]["content"][0]["type"] == "web_search_result"
        )
        assert parsed[7][1]["delta"]["type"] == "text_delta"
        assert parsed[9][1]["usage"]["server_tool_use"]["web_search_requests"] == 1


class TestMaybeHandle:
    async def test_returns_none_without_server_tool(self):
        request = make_request([AnthropicMessage(role="user", content="hello")])
        assert await maybe_handle_anthropic_web_search(request, None) is None

    async def test_returns_none_without_search_prompt(self):
        request = make_request(
            [AnthropicMessage(role="user", content="hello")],
            tools=[web_search_tool()],
        )
        assert await maybe_handle_anthropic_web_search(request, None) is None

    async def test_streaming_response(self, monkeypatch):
        async def fake_search(query, integrations):
            assert query == "oMLX MLX server"
            return success_payload()

        monkeypatch.setattr("omlx.websearch.run_web_search", fake_search)

        request = make_request(
            [
                AnthropicMessage(
                    role="user",
                    content=f"{SEARCH_PROMPT_PREFIX} oMLX MLX server",
                )
            ],
            tools=[web_search_tool()],
            stream=True,
        )
        response = await maybe_handle_anthropic_web_search(
            request, SimpleNamespace(integrations=object())
        )
        assert isinstance(response, StreamingResponse)
        chunks = [chunk async for chunk in response.body_iterator]
        body = "".join(
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in chunks
        )
        assert "server_tool_use" in body
        assert "web_search_tool_result" in body
        assert "https://example.com/omlx" in body

    async def test_non_streaming_response(self, monkeypatch):
        async def fake_search(query, integrations):
            return success_payload()

        monkeypatch.setattr("omlx.websearch.run_web_search", fake_search)

        request = make_request(
            [
                AnthropicMessage(
                    role="user",
                    content=f"{SEARCH_PROMPT_PREFIX} oMLX",
                )
            ],
            tools=[web_search_tool()],
            stream=False,
        )
        response = await maybe_handle_anthropic_web_search(
            request, SimpleNamespace(integrations=object())
        )
        assert isinstance(response, JSONResponse)
        payload = json.loads(response.body)
        assert payload["content"][0]["type"] == "server_tool_use"
        assert payload["content"][1]["type"] == "web_search_tool_result"
        assert payload["content"][2]["type"] == "text"

    async def test_missing_settings_returns_error_block(self):
        request = make_request(
            [
                AnthropicMessage(
                    role="user",
                    content=f"{SEARCH_PROMPT_PREFIX} oMLX",
                )
            ],
            tools=[web_search_tool()],
            stream=False,
        )
        response = await maybe_handle_anthropic_web_search(request, None)
        payload = json.loads(response.body)
        assert (
            payload["content"][1]["content"]["type"] == "web_search_tool_result_error"
        )
