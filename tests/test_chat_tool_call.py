# SPDX-License-Identifier: Apache-2.0
"""Tests for chat MCP tool call loop (chat.html streamResponse changes)."""
import json


class TestChatToolCallMessageFiltering:
    """Test the messagesForApi filtering logic (Python equivalent of the JS)."""

    @staticmethod
    def build_messages_for_api(messages):
        """Replicate the messagesForApi logic from streamResponse in chat.html."""
        valid_roles = {"user", "assistant", "tool", "system"}
        result = []
        for msg in messages:
            if msg["role"] not in valid_roles:
                continue
            m = {"role": msg["role"], "content": msg.get("content")}
            if msg.get("tool_calls"):
                m["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            result.append(m)
        return result

    def test_filters_tool_call_indicator_messages(self):
        """tool_call role messages must not be sent to the API."""
        messages = [
            {"role": "user", "content": "Who is X?"},
            {"role": "tool_call", "content": "tavily__tavily_search…", "_ui": True},
            {"role": "assistant", "content": "X is...", "tool_calls": None},
        ]
        api_msgs = self.build_messages_for_api(messages)
        roles = [m["role"] for m in api_msgs]
        assert "tool_call" not in roles
        assert roles == ["user", "assistant"]

    def test_passes_tool_calls_and_tool_call_id(self):
        """Assistant tool_calls and tool result tool_call_id must be preserved."""
        tc = [{"id": "tc_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
        messages = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": None, "tool_calls": tc, "_ui": False},
            {"role": "tool", "tool_call_id": "tc_1", "content": "result...", "_ui": False},
        ]
        api_msgs = self.build_messages_for_api(messages)
        assert len(api_msgs) == 3
        assert api_msgs[1]["tool_calls"] == tc
        assert api_msgs[2]["tool_call_id"] == "tc_1"

    def test_normal_conversation_unchanged(self):
        """Normal user/assistant conversation with no tools is unaffected."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        api_msgs = self.build_messages_for_api(messages)
        assert len(api_msgs) == 2
        assert api_msgs[0] == {"role": "user", "content": "Hello"}
        assert api_msgs[1] == {"role": "assistant", "content": "Hi there"}


class TestChatToolCallAccumulation:
    """Test streaming tool_call chunk accumulation (Python equivalent of the JS)."""

    @staticmethod
    def accumulate_tool_calls(deltas):
        """Replicate the toolCallsMap accumulation logic from streamResponse."""
        tool_calls_map = {}
        for delta in deltas:
            if not delta.get("tool_calls"):
                continue
            for tc in delta["tool_calls"]:
                i = tc.get("index", 0)
                if i not in tool_calls_map:
                    tool_calls_map[i] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                if tc.get("id"):
                    tool_calls_map[i]["id"] = tc["id"]
                if tc.get("function", {}).get("name"):
                    tool_calls_map[i]["function"]["name"] += tc["function"]["name"]
                if tc.get("function", {}).get("arguments"):
                    tool_calls_map[i]["function"]["arguments"] += tc["function"]["arguments"]
        return list(tool_calls_map.values())

    def test_single_tool_call(self):
        """A single tool call split across multiple chunks is assembled correctly."""
        deltas = [
            {"tool_calls": [{"index": 0, "id": "tc_1", "function": {"name": "tavily__tavily_search"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"que'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": 'ry":"test"}'}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "tavily__tavily_search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "test"}

    def test_multiple_parallel_tool_calls(self):
        """Multiple tool calls with different indices are accumulated separately."""
        deltas = [
            {"tool_calls": [{"index": 0, "id": "tc_1", "function": {"name": "search"}}]},
            {"tool_calls": [{"index": 1, "id": "tc_2", "function": {"name": "extract"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"a"}'}}]},
            {"tool_calls": [{"index": 1, "function": {"arguments": '{"urls":["http://x"]}'}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "search"
        assert result[1]["function"]["name"] == "extract"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "a"}
        assert json.loads(result[1]["function"]["arguments"]) == {"urls": ["http://x"]}

    def test_no_tool_calls(self):
        """Deltas with no tool_calls produce empty list."""
        deltas = [
            {"content": "Hello"},
            {"content": " world"},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert result == []

    def test_missing_index_defaults_to_zero(self):
        """A tool_call chunk without an index field defaults to index 0."""
        deltas = [
            {"tool_calls": [{"id": "tc_1", "function": {"name": "t", "arguments": "{}"}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"
