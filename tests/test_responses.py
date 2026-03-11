# SPDX-License-Identifier: Apache-2.0
"""Tests for OpenAI Responses API models and utilities."""

import json

import pytest

from omlx.api.responses_models import (
    InputItem,
    OutputContent,
    OutputItem,
    ResponseObject,
    ResponsesRequest,
    ResponsesTool,
    ResponseUsage,
    TextConfig,
    TextFormatConfig,
)
from omlx.api.responses_utils import (
    ResponseStore,
    build_function_call_output_item,
    build_message_output_item,
    build_response_usage,
    convert_responses_input_to_messages,
    convert_responses_tools,
    convert_stored_response_to_messages,
    format_sse_event,
)
from omlx.api.shared_models import IDPrefix, generate_id


# =============================================================================
# ID Generation Tests
# =============================================================================


class TestIDGeneration:
    def test_response_id_prefix(self):
        rid = generate_id(IDPrefix.RESPONSE)
        assert rid.startswith("resp_")
        assert len(rid) == 29  # "resp_" + 24 hex chars

    def test_function_call_id_prefix(self):
        fid = generate_id(IDPrefix.FUNCTION_CALL)
        assert fid.startswith("fc_")
        assert len(fid) == 11  # "fc_" + 8 hex chars

    def test_unique_ids(self):
        ids = {generate_id(IDPrefix.RESPONSE) for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# Input Conversion Tests
# =============================================================================


class TestConvertResponsesInput:
    def test_string_input(self):
        messages = convert_responses_input_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello world"}

    def test_none_input(self):
        messages = convert_responses_input_to_messages(None)
        assert messages == []

    def test_instructions_prepended(self):
        messages = convert_responses_input_to_messages(
            "Hi", instructions="You are helpful"
        )
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hi"}

    def test_message_items(self):
        items = [
            InputItem(type="message", role="user", content="Hello"),
            InputItem(type="message", role="assistant", content="Hi there"),
            InputItem(type="message", role="user", content="How are you?"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["content"] == "How are you?"

    def test_easy_input_message_no_type(self):
        """EasyInputMessage has no type field — just role and content."""
        items = [
            InputItem(role="user", content="Hello"),
            InputItem(role="assistant", content="Hi"),
        ]
        assert items[0].type is None  # No type set
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_easy_input_message_from_dict(self):
        """Verify Pydantic parses EasyInputMessage dict (no type field)."""
        raw = {"role": "user", "content": "Hello from dict"}
        item = InputItem(**raw)
        assert item.type is None
        assert item.role == "user"
        messages = convert_responses_input_to_messages([item])
        assert messages[0]["content"] == "Hello from dict"

    def test_developer_role_maps_to_system(self):
        items = [
            InputItem(type="message", role="developer", content="Be concise"),
            InputItem(type="message", role="user", content="Hi"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be concise"
        assert messages[1]["role"] == "user"

    def test_developer_role_no_type(self):
        """EasyInputMessage with developer role (no type field)."""
        items = [InputItem(role="developer", content="Be brief")]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["role"] == "system"

    def test_instructions_and_developer_merged(self):
        """instructions + developer role in input → single system message."""
        items = [
            InputItem(role="developer", content="Working dir: /foo"),
            InputItem(role="user", content="Hello"),
        ]
        messages = convert_responses_input_to_messages(
            items, instructions="You are helpful"
        )
        # Should be exactly 2: one merged system + one user
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert "Working dir: /foo" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_multiple_developer_messages_merged(self):
        """Multiple developer messages all merge into one system message."""
        items = [
            InputItem(role="developer", content="Rule 1"),
            InputItem(role="developer", content="Rule 2"),
            InputItem(role="user", content="Hi"),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Rule 1" in messages[0]["content"]
        assert "Rule 2" in messages[0]["content"]

    def test_function_call_items(self):
        items = [
            InputItem(type="message", role="user", content="What's the weather?"),
            InputItem(
                type="function_call",
                call_id="call_abc",
                name="get_weather",
                arguments='{"location": "Paris"}',
            ),
            InputItem(
                type="function_call_output",
                call_id="call_abc",
                output='{"temp": 22}',
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        # function_call → assistant with tool_calls
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 1
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # arguments should be parsed as dict (not string) for Jinja2 chat templates
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"location": "Paris"}
        # function_call_output → tool message
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_abc"
        assert messages[2]["content"] == '{"temp": 22}'

    def test_multiple_function_calls_grouped(self):
        """Multiple consecutive function_calls should be grouped into one assistant message."""
        items = [
            InputItem(type="message", role="user", content="Do two things"),
            InputItem(
                type="function_call",
                call_id="call_1",
                name="fn_a",
                arguments="{}",
            ),
            InputItem(
                type="function_call",
                call_id="call_2",
                name="fn_b",
                arguments="{}",
            ),
            InputItem(
                type="function_call_output",
                call_id="call_1",
                output="result_a",
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        # user, assistant (2 tool_calls), tool (output for call_1)
        assert len(messages) == 3
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 2

    def test_content_parts_array(self):
        items = [
            InputItem(
                type="message",
                role="user",
                content=[
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ],
            ),
        ]
        messages = convert_responses_input_to_messages(items)
        assert messages[0]["content"] == "Hello\nWorld"

    def test_previous_messages_prepended(self):
        prev = [{"role": "assistant", "content": "Previous answer"}]
        messages = convert_responses_input_to_messages(
            "Follow up", previous_messages=prev
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"

    def test_instructions_with_previous(self):
        prev = [{"role": "assistant", "content": "Previous"}]
        messages = convert_responses_input_to_messages(
            "New question",
            instructions="System prompt",
            previous_messages=prev,
        )
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"


# =============================================================================
# Tool Conversion Tests
# =============================================================================


class TestConvertResponsesTools:
    def test_none_tools(self):
        assert convert_responses_tools(None) is None

    def test_empty_tools(self):
        assert convert_responses_tools([]) is None

    def test_single_tool(self):
        tools = [
            ResponsesTool(
                type="function",
                name="get_weather",
                description="Get weather info",
                parameters={"type": "object", "properties": {"location": {"type": "string"}}},
            )
        ]
        result = convert_responses_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"
        assert "parameters" in result[0]["function"]

    def test_strict_field(self):
        tools = [
            ResponsesTool(name="fn", strict=True)
        ]
        result = convert_responses_tools(tools)
        assert result[0]["function"]["strict"] is True

    def test_minimal_tool(self):
        tools = [ResponsesTool(name="simple_fn")]
        result = convert_responses_tools(tools)
        assert result[0]["function"]["name"] == "simple_fn"
        assert "description" not in result[0]["function"]
        assert "parameters" not in result[0]["function"]

    def test_non_function_tools_skipped(self):
        """Non-function tool types (local_shell, mcp, etc.) should be skipped."""
        tools = [
            ResponsesTool(type="local_shell"),
            ResponsesTool(type="function", name="fn_a"),
            ResponsesTool(type="mcp"),
        ]
        result = convert_responses_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "fn_a"

    def test_only_non_function_tools_returns_none(self):
        tools = [ResponsesTool(type="local_shell")]
        result = convert_responses_tools(tools)
        assert result is None


# =============================================================================
# Response Building Tests
# =============================================================================


class TestBuildOutputItems:
    def test_message_output_item(self):
        item = build_message_output_item("Hello!")
        assert item.type == "message"
        assert item.role == "assistant"
        assert item.content[0].text == "Hello!"
        assert item.content[0].type == "output_text"
        assert item.status == "completed"
        assert item.id.startswith("msg_")

    def test_function_call_output_item(self):
        item = build_function_call_output_item(
            name="get_weather",
            arguments='{"location": "Paris"}',
            call_id="call_abc",
        )
        assert item.type == "function_call"
        assert item.name == "get_weather"
        assert item.call_id == "call_abc"
        assert item.arguments == '{"location": "Paris"}'
        assert item.id.startswith("fc_")

    def test_response_usage(self):
        usage = build_response_usage(100, 50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150


class TestResponseObject:
    def test_defaults(self):
        resp = ResponseObject(model="test-model")
        assert resp.object == "response"
        assert resp.id.startswith("resp_")
        assert resp.status == "completed"
        assert resp.output == []

    def test_with_output(self):
        msg = build_message_output_item("Hi")
        resp = ResponseObject(
            model="test-model",
            output=[msg],
            usage=build_response_usage(10, 5),
        )
        assert len(resp.output) == 1
        assert resp.usage.total_tokens == 15


# =============================================================================
# SSE Event Formatting Tests
# =============================================================================


class TestFormatSSEEvent:
    def test_dict_data(self):
        result = format_sse_event("response.created", {"id": "resp_123"})
        assert result.startswith("event: response.created\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data_line = result.split("\n")[1]
        parsed = json.loads(data_line.removeprefix("data: "))
        assert parsed["id"] == "resp_123"

    def test_string_data(self):
        result = format_sse_event("response.output_text.delta", '{"delta": "hi"}')
        assert "event: response.output_text.delta\n" in result
        assert 'data: {"delta": "hi"}\n' in result

    def test_pydantic_model(self):
        usage = ResponseUsage(input_tokens=10, output_tokens=5)
        result = format_sse_event("test", usage)
        assert "event: test\n" in result
        parsed = json.loads(result.split("\n")[1].removeprefix("data: "))
        assert parsed["input_tokens"] == 10


# =============================================================================
# Response Store Tests
# =============================================================================


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"id": "resp_1", "output": []})
        assert store.get("resp_1") is not None
        assert store.get("resp_1")["id"] == "resp_1"

    def test_get_nonexistent(self):
        store = ResponseStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = ResponseStore()
        store.put("resp_1", {"id": "resp_1"})
        assert store.delete("resp_1") is True
        assert store.get("resp_1") is None

    def test_delete_nonexistent(self):
        store = ResponseStore()
        assert store.delete("nonexistent") is False

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1"})
        store.put("resp_2", {"id": "resp_2"})
        store.put("resp_3", {"id": "resp_3"})
        # This should evict resp_1 (oldest)
        store.put("resp_4", {"id": "resp_4"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert store.get("resp_4") is not None
        assert len(store) == 3

    def test_lru_access_refreshes(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1"})
        store.put("resp_2", {"id": "resp_2"})
        store.put("resp_3", {"id": "resp_3"})
        # Access resp_1 to refresh it
        store.get("resp_1")
        # Now resp_2 should be evicted (oldest after refresh)
        store.put("resp_4", {"id": "resp_4"})
        assert store.get("resp_1") is not None  # Refreshed, still here
        assert store.get("resp_2") is None  # Evicted
        assert store.get("resp_4") is not None

    def test_update_existing(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"id": "resp_1", "v": 1})
        store.put("resp_1", {"id": "resp_1", "v": 2})
        assert store.get("resp_1")["v"] == 2
        assert len(store) == 1


# =============================================================================
# Previous Response Conversion Tests
# =============================================================================


class TestConvertStoredResponse:
    def test_message_output(self):
        stored = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello!"}
                    ],
                }
            ]
        }
        messages = convert_stored_response_to_messages(stored)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hello!"

    def test_function_call_output(self):
        stored = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            ]
        }
        messages = convert_stored_response_to_messages(stored)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        # arguments should be parsed as dict for Jinja2 chat templates
        assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"location": "Paris"}

    def test_empty_output(self):
        stored = {"output": []}
        messages = convert_stored_response_to_messages(stored)
        assert messages == []


# =============================================================================
# Request Model Tests
# =============================================================================


class TestResponsesRequest:
    def test_minimal_request(self):
        req = ResponsesRequest(model="test-model")
        assert req.model == "test-model"
        assert req.input is None
        assert req.stream is False
        assert req.store is None

    def test_string_input(self):
        req = ResponsesRequest(model="test", input="Hello")
        assert req.input == "Hello"

    def test_item_input(self):
        req = ResponsesRequest(
            model="test",
            input=[InputItem(type="message", role="user", content="Hi")],
        )
        assert len(req.input) == 1
        assert req.input[0].type == "message"

    def test_with_tools(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            tools=[ResponsesTool(name="fn_a", description="Does A")],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "fn_a"

    def test_text_format(self):
        req = ResponsesRequest(
            model="test",
            input="Hi",
            text=TextConfig(format=TextFormatConfig(type="json_object")),
        )
        assert req.text.format.type == "json_object"

    def test_codex_fields(self):
        """Codex CLI sends include, service_tier, prompt_cache_key, etc."""
        req = ResponsesRequest(
            model="test",
            input=[InputItem(role="user", content="Hi")],
            stream=True,
            store=False,
            include=["reasoning.encrypted_content"],
            service_tier="auto",
            prompt_cache_key="test-key",
            reasoning={"effort": "high", "summary": "auto"},
            parallel_tool_calls=True,
        )
        assert req.include == ["reasoning.encrypted_content"]
        assert req.service_tier == "auto"
        assert req.prompt_cache_key == "test-key"
        assert req.reasoning["effort"] == "high"

    def test_extra_fields_allowed(self):
        """Unknown fields should not cause validation errors."""
        req = ResponsesRequest(
            model="test",
            input="Hi",
            some_future_field="value",
            another_field=42,
        )
        assert req.model == "test"

    def test_input_item_extra_fields(self):
        """InputItem should accept unknown fields for forward compatibility."""
        item = InputItem(
            type="message",
            role="user",
            content="Hi",
            phase="final_answer",
        )
        assert item.role == "user"

    def test_request_from_codex_like_json(self):
        """Parse a request body similar to what Codex CLI sends."""
        raw = {
            "model": "Qwen3.5-122B-A10B-4bit",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "Hello!"},
            ],
            "tools": [
                {"type": "local_shell"},
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                    "strict": True,
                },
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
        req = ResponsesRequest(**raw)
        assert req.model == "Qwen3.5-122B-A10B-4bit"
        assert req.stream is True
        assert len(req.input) == 1
        assert req.input[0].role == "user"
        assert req.input[0].type is None  # EasyInputMessage
        assert len(req.tools) == 2
        assert req.tools[0].type == "local_shell"
        assert req.tools[1].name == "read_file"
