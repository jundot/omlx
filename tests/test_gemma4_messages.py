# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 message extraction."""

from __future__ import annotations

from omlx.adapter.gemma4 import extract_gemma4_messages
from omlx.api.openai_models import Message


def _tool_call_dict(id: str, name: str, args: str = "{}") -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


def _assistant_with_calls(*calls) -> Message:
    return Message(role="assistant", content="", tool_calls=list(calls))


def _tool_result(id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=id)


class TestExtractGemma4Messages:
    def test_plain_messages_pass_through(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = extract_gemma4_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_tool_result_folded_onto_model_turn(self):
        """Single tool result becomes a model turn with tool_responses."""
        messages = [
            Message(role="user", content="What's the weather?"),
            _assistant_with_calls(_tool_call_dict("c1", "get_weather")),
            _tool_result("c1", "sunny"),
        ]
        result = extract_gemma4_messages(messages)
        # user + assistant(tool_calls) + assistant(tool_responses)
        assert len(result) == 3
        tr_msg = result[2]
        assert tr_msg["role"] == "assistant"
        assert tr_msg["content"] == ""
        assert tr_msg["tool_responses"] == [
            {"name": "get_weather", "response": "sunny"}
        ]

    def test_function_name_resolved_from_tool_call_id(self):
        messages = [
            _assistant_with_calls(
                _tool_call_dict("c1", "search"),
                _tool_call_dict("c2", "calculate"),
            ),
            _tool_result("c2", "42"),
            _tool_result("c1", "results"),
        ]
        result = extract_gemma4_messages(messages)
        tr_msg = result[-1]
        names = {tr["name"] for tr in tr_msg["tool_responses"]}
        assert names == {"calculate", "search"}

    def test_multiple_tool_results_batched(self):
        """Multiple consecutive tool results land in a single tool_responses turn."""
        messages = [
            _assistant_with_calls(
                _tool_call_dict("c1", "fn_a"),
                _tool_call_dict("c2", "fn_b"),
            ),
            _tool_result("c1", "result_a"),
            _tool_result("c2", "result_b"),
        ]
        result = extract_gemma4_messages(messages)
        tr_msg = result[-1]
        assert len(tr_msg["tool_responses"]) == 2
        assert tr_msg["tool_responses"][0] == {"name": "fn_a", "response": "result_a"}
        assert tr_msg["tool_responses"][1] == {"name": "fn_b", "response": "result_b"}

    def test_json_response_parsed_to_dict(self):
        """JSON-parseable tool result content becomes a dict."""
        messages = [
            _assistant_with_calls(_tool_call_dict("c1", "fn")),
            _tool_result("c1", '{"value": 42}'),
        ]
        result = extract_gemma4_messages(messages)
        response = result[-1]["tool_responses"][0]["response"]
        assert response == {"value": 42}

    def test_non_json_response_stays_string(self):
        messages = [
            _assistant_with_calls(_tool_call_dict("c1", "fn")),
            _tool_result("c1", "plain text result"),
        ]
        result = extract_gemma4_messages(messages)
        assert result[-1]["tool_responses"][0]["response"] == "plain text result"

    def test_orphaned_tool_result_fallback_to_tool_call_id_as_name(self):
        """Tool result with no preceding assistant turn uses tool_call_id as name."""
        messages = [_tool_result("call_xyz", "orphaned")]
        result = extract_gemma4_messages(messages)
        assert result[0]["tool_responses"][0]["name"] == "call_xyz"

    def test_tool_calls_preserved_on_assistant_turn(self):
        """Assistant turn tool_calls are kept so the template renders them."""
        messages = [
            _assistant_with_calls(_tool_call_dict("c1", "do_thing", '{"x": 1}'))
        ]
        result = extract_gemma4_messages(messages)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "do_thing"
        # arguments should be parsed to dict
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}

    def test_multi_turn_agentic_conversation(self):
        """Full agentic loop: user → tool call → result → follow-up answer."""
        messages = [
            Message(role="user", content="Look it up"),
            _assistant_with_calls(_tool_call_dict("c1", "search")),
            _tool_result("c1", "found it"),
            Message(role="assistant", content="Here is what I found."),
        ]
        result = extract_gemma4_messages(messages)
        assert result[0] == {"role": "user", "content": "Look it up"}
        assert "tool_calls" in result[1]
        assert result[2]["tool_responses"][0]["name"] == "search"
        assert result[3] == {"role": "assistant", "content": "Here is what I found."}

    def test_system_message_preserved(self):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        result = extract_gemma4_messages(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_developer_role_normalised_to_system(self):
        messages = [Message(role="developer", content="Be concise.")]
        result = extract_gemma4_messages(messages)
        assert result[0]["role"] == "system"

    def test_image_url_preserved_in_user_message(self):
        """User messages with image_url parts keep them for VLM processing."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            ),
        ]
        result = extract_gemma4_messages(messages)
        assert isinstance(result[0]["content"], list)
        types = [p["type"] for p in result[0]["content"]]
        assert "image_url" in types
        assert "text" in types

    def test_text_only_content_list_flattened(self):
        """User messages with text-only content list are flattened to string."""
        messages = [
            Message(
                role="user",
                content=[{"type": "text", "text": "hello world"}],
            ),
        ]
        result = extract_gemma4_messages(messages)
        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "hello world"

    def test_consecutive_user_with_image_merges_without_crash(self):
        """Consecutive user messages where one has images should not crash.

        Regression test for GitHub issue #671.
        """
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            ),
            Message(role="user", content="What is it?"),
        ]
        result = extract_gemma4_messages(messages)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        types = [p["type"] for p in content]
        assert "image_url" in types
